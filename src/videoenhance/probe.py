"""Read video metadata with ffprobe.

The pipeline needs four things from the source file before it can plan any
work: the resolution, the frame rate, the frame count and whether there is an
audio stream to carry over. It also needs to know whether the file is variable
frame rate, because extracting VFR frames and re-encoding them at a constant
rate is what makes the audio drift out of sync.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .config import Binaries, binaries
from .process import ToolError, run

logger = logging.getLogger(__name__)


class ProbeError(RuntimeError):
    """ffprobe produced output this module could not interpret."""


@dataclass(frozen=True)
class VideoInfo:
    """What ffprobe reports about the first video stream of a file."""

    path: Path
    width: int
    height: int
    r_frame_rate: Fraction
    avg_frame_rate: Fraction
    nb_frames: int
    duration: float
    has_audio: bool
    codec: str = ""

    @property
    def fps(self) -> float:
        """Nominal frame rate as a float, for display and arithmetic."""
        return float(self.r_frame_rate)

    @property
    def is_vfr(self) -> bool:
        """Whether the file looks variable frame rate.

        ffprobe reports ``r_frame_rate`` as the stream's nominal rate and
        ``avg_frame_rate`` as the rate actually observed; when they disagree the
        file is almost certainly VFR, which is common in phone recordings.
        """
        if not self.avg_frame_rate or not self.r_frame_rate:
            return False
        return self.r_frame_rate != self.avg_frame_rate

    @property
    def pixels(self) -> int:
        """Pixels per frame."""
        return self.width * self.height

    def describe(self) -> str:
        """A short human-readable summary, used by ``--probe-only``."""
        audio = "yes" if self.has_audio else "no"
        rate = f"{self.fps:g} fps"
        if self.is_vfr:
            rate += f" (VFR: nominal {self.r_frame_rate}, average {self.avg_frame_rate})"
        lines = [
            f"file:       {self.path}",
            f"resolution: {self.width}x{self.height}",
            f"frame rate: {rate}",
            f"frames:     {self.nb_frames}" + ("" if self.nb_frames else " (not reported)"),
            f"duration:   {self.duration:.3f}s",
            f"audio:      {audio}",
        ]
        if self.codec:
            lines.insert(2, f"codec:      {self.codec}")
        return "\n".join(lines)


def _parse_fraction(value: object) -> Fraction:
    """Parse an ffprobe rate such as ``"30000/1001"``; ``0/0`` becomes 0."""
    if value in (None, "", "0/0"):
        return Fraction(0)
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return Fraction(0)


def _ffprobe_json(args: list[object], bins: Binaries) -> dict:
    cmd = [bins.ffprobe, "-v", "error", *args, "-of", "json"]
    completed = run(cmd)
    try:
        return json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ProbeError(f"could not parse ffprobe output for {cmd[-1]}: {exc}") from exc


def has_audio_stream(path: Path, bins: Binaries | None = None) -> bool:
    """Whether ``path`` carries at least one audio stream."""
    bins = bins or binaries()
    data = _ffprobe_json(
        ["-select_streams", "a", "-show_entries", "stream=index", str(path)], bins
    )
    return bool(data.get("streams"))


def probe(path: Path | str, bins: Binaries | None = None) -> VideoInfo:
    """Describe the first video stream of ``path``.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ProbeError: the file has no video stream, or ffprobe output was
            unreadable.
        ToolError: ffprobe itself failed.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"input video not found: {path}")

    bins = bins or binaries()
    data = _ffprobe_json(
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name"
            ":format=duration",
            str(path),
        ],
        bins,
    )

    streams = data.get("streams") or []
    if not streams:
        raise ProbeError(f"no video stream found in {path}")
    stream = streams[0]

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if not width or not height:
        raise ProbeError(f"ffprobe did not report a resolution for {path}")

    r_rate = _parse_fraction(stream.get("r_frame_rate"))
    avg_rate = _parse_fraction(stream.get("avg_frame_rate"))
    if not r_rate:
        # Some containers only report the average rate; it is better than nothing.
        r_rate = avg_rate
    if not r_rate:
        raise ProbeError(f"ffprobe did not report a frame rate for {path}")

    duration = float(
        stream.get("duration") or (data.get("format") or {}).get("duration") or 0.0
    )

    nb_frames = int(stream.get("nb_frames") or 0)
    if not nb_frames and duration:
        # Containers such as Matroska often omit nb_frames; estimate it rather
        # than counting, which would mean decoding the whole file.
        nb_frames = math.floor(duration * float(r_rate))
        logger.debug("nb_frames missing for %s, estimated %d", path, nb_frames)

    info = VideoInfo(
        path=path,
        width=width,
        height=height,
        r_frame_rate=r_rate,
        avg_frame_rate=avg_rate or r_rate,
        nb_frames=nb_frames,
        duration=duration,
        has_audio=has_audio_stream(path, bins),
        codec=str(stream.get("codec_name") or ""),
    )
    logger.debug("probed %s: %dx%d @ %s", path, info.width, info.height, info.r_frame_rate)
    return info


def count_frames(path: Path, bins: Binaries | None = None) -> int:
    """Count frames exactly by decoding the file.

    Slow, and only needed when a stage requires an exact count that the
    container did not provide.
    """
    bins = bins or binaries()
    try:
        completed = run(
            [
                bins.ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(path),
            ]
        )
    except ToolError:
        logger.warning("exact frame count failed for %s", path)
        raise
    return int((completed.stdout or "0").strip() or 0)
