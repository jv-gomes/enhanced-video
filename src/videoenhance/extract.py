"""Turn a video into a directory of numbered frames.

Two things matter here. The frames must come out in a single, predictable
numbering so the NCNN binaries and FFmpeg agree on their order, and the
timeline must be constant frame rate. A VFR source — which is what most phone
recordings are — is normalised to CFR first; extracting its frames directly and
re-encoding them at a fixed rate is the usual cause of audio drifting out of
sync by the end of a long clip.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from . import config
from .config import Binaries, binaries
from .probe import VideoInfo, probe
from .process import run
from .workdir import WorkDir, count_frames_on_disk, frame_pattern, stage_is_done

logger = logging.getLogger(__name__)

#: Quality for intermediate JPEG frames: 2 is FFmpeg's best, 31 its worst.
JPEG_QUALITY = 2


class ExtractError(RuntimeError):
    """Extraction produced no frames, or fewer than the source should yield."""


@dataclass(frozen=True)
class ExtractResult:
    """What the extract stage produced and what the later stages need to know."""

    frames_dir: Path
    frame_count: int
    fps: Fraction
    #: The file the final encode should take its audio from. For a VFR source
    #: this is the CFR copy, whose audio matches the extracted timeline.
    audio_source: Path
    has_audio: bool
    normalised: bool = False
    #: True when the frames were already on disk and nothing was re-extracted.
    skipped: bool = False

    @property
    def duration(self) -> float:
        return float(self.frame_count / self.fps) if self.fps else 0.0


def target_cfr(info: VideoInfo) -> Fraction:
    """The constant rate to normalise a VFR source to.

    ``avg_frame_rate`` is what the file actually achieved, so it keeps the
    duration — and therefore the audio — closest to the original.
    """
    return info.avg_frame_rate or info.r_frame_rate


def to_cfr(
    info: VideoInfo,
    destination: Path,
    *,
    fps: Fraction | None = None,
    bins: Binaries | None = None,
) -> Path:
    """Rewrite a VFR source at a constant frame rate, keeping the audio.

    The video is re-encoded losslessly (``libx264 -qp 0``) because the frames
    extracted from this copy are the pixels every later stage works on.
    """
    bins = bins or binaries()
    rate = fps or target_cfr(info)
    if destination.exists():
        logger.info("reusing CFR copy %s", destination)
        return destination

    logger.info("normalising %s to constant %s fps", info.path, rate)
    # The extension has to stay last: FFmpeg picks the container from it.
    partial = destination.with_suffix(f".part{destination.suffix}")
    run(
        [
            bins.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            info.path,
            "-fps_mode",
            "cfr",
            "-r",
            str(rate),
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-preset",
            "veryfast",
            "-c:a",
            "copy",
            partial,
        ],
        capture=False,
    )
    partial.replace(destination)
    return destination


def _extract_cmd(
    source: Path,
    out_dir: Path,
    frame_format: str,
    bins: Binaries,
) -> list[object]:
    cmd: list[object] = [
        bins.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source,
        # The source is already CFR by this point, so passthrough keeps exactly
        # the frames it contains: no duplicates, none dropped.
        "-fps_mode",
        "passthrough",
    ]
    if frame_format == "jpg":
        cmd += ["-q:v", str(JPEG_QUALITY)]
    cmd.append(out_dir / frame_pattern(frame_format))
    return cmd


def extract(
    info: VideoInfo,
    work: WorkDir,
    *,
    frame_format: str = config.DEFAULT_FRAME_FORMAT,
    bins: Binaries | None = None,
    force: bool = False,
) -> ExtractResult:
    """Extract every frame of ``info`` into ``work.frames_in``.

    Normalises a VFR source to CFR first and reports that copy as the audio
    source. Skips the work entirely when the frames are already on disk.

    Raises:
        ExtractError: FFmpeg wrote no frames.
        ToolError: FFmpeg failed.
    """
    bins = bins or binaries()
    work.create()
    out_dir = work.frames_in
    out_dir.mkdir(parents=True, exist_ok=True)

    source = info.path
    fps = info.r_frame_rate
    normalised = False
    if info.is_vfr:
        source = to_cfr(info, work.cfr_video, bins=bins)
        fps = target_cfr(info)
        normalised = True

    def result(count: int, skipped: bool = False) -> ExtractResult:
        return ExtractResult(
            frames_dir=out_dir,
            frame_count=count,
            fps=fps,
            audio_source=source,
            has_audio=info.has_audio,
            normalised=normalised,
            skipped=skipped,
        )

    if not force and stage_is_done("extract", out_dir, info.nb_frames, frame_format):
        return result(count_frames_on_disk(out_dir, frame_format), skipped=True)

    logger.info("extracting frames from %s to %s", source, out_dir)
    run(_extract_cmd(source, out_dir, frame_format, bins), capture=False)

    count = count_frames_on_disk(out_dir, frame_format)
    if count == 0:
        raise ExtractError(f"no frames were extracted from {source}")
    if info.nb_frames and count < info.nb_frames:
        # The container's count can be an estimate, so this is worth saying but
        # not worth failing over.
        logger.warning(
            "extracted %d frames but %s reported %d", count, info.path.name, info.nb_frames
        )
    logger.info("extracted %d frames", count)
    return result(count)


def extract_file(
    path: Path,
    work: WorkDir | None = None,
    **kwargs: object,
) -> ExtractResult:
    """Convenience wrapper that probes ``path`` first."""
    info = probe(path)
    return extract(info, work or WorkDir.for_input(info.path), **kwargs)  # type: ignore[arg-type]
