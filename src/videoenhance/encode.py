"""Choosing a video encoder, and falling back when it does not work.

Hardware encoding is worth a lot here — a two-hour 4K re-encode on the CPU is
the slowest part of the whole pipeline — but hardware encoders fail in ways
that only show up at run time: a driver that advertises VAAPI but cannot open
the render node, an AMF build without the runtime behind it. So the encoders
are ranked, and :func:`encode` walks down the list until one of them actually
produces a file.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from .config import DEFAULT_FRAME_FORMAT, Binaries, binaries
from .process import ToolError, run
from .workdir import frame_pattern

logger = logging.getLogger(__name__)

#: Linux VAAPI render node. Overridden per call where a system differs.
DEFAULT_VAAPI_DEVICE = Path("/dev/dri/renderD128")

#: Hardware encoders worth trying, best first, per platform.
HW_ENCODERS: dict[str, tuple[str, ...]] = {
    "linux": ("hevc_vaapi", "h264_vaapi"),
    "win32": ("hevc_amf", "h264_amf"),
    "darwin": ("hevc_videotoolbox", "h264_videotoolbox"),
}
#: CPU encoders, best first. The last one is the universal fallback.
CPU_ENCODERS: tuple[str, ...] = ("libx265", "libx264")

#: Quality settings per encoder family, from CLAUDE.md.
_QUALITY_ARGS: dict[str, list[str]] = {
    "vaapi": ["-qp", "20"],
    "amf": ["-quality", "quality", "-rc", "cqp", "-qp_i", "18", "-qp_p", "20"],
    "videotoolbox": ["-q:v", "55"],
    "libx264": ["-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"],
    "libx265": ["-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p"],
}


def _family(encoder: str) -> str:
    for family in ("vaapi", "amf", "videotoolbox"):
        if encoder.endswith(family):
            return family
    return encoder


@dataclass(frozen=True)
class Encoder:
    """One encoding attempt: the codec plus whatever it needs around it."""

    name: str
    #: Options that must precede the inputs, e.g. ``-vaapi_device``.
    init_args: list[str] = field(default_factory=list)
    #: Video filter chain required by the encoder, e.g. the VAAPI hwupload.
    filters: list[str] = field(default_factory=list)
    quality_args: list[str] = field(default_factory=list)

    @property
    def is_hardware(self) -> bool:
        return _family(self.name) != self.name


def _build(name: str, vaapi_device: Path = DEFAULT_VAAPI_DEVICE) -> Encoder:
    family = _family(name)
    init: list[str] = []
    filters: list[str] = []
    if family == "vaapi":
        init = ["-vaapi_device", str(vaapi_device)]
        # VAAPI encodes from GPU memory, so the frames have to be converted and
        # uploaded first.
        filters = ["format=nv12", "hwupload"]
    return Encoder(
        name=name,
        init_args=init,
        filters=filters,
        quality_args=list(_QUALITY_ARGS.get(family, _QUALITY_ARGS.get(name, []))),
    )


def available_encoders(ffmpeg: Path | None = None) -> set[str]:
    """Encoder names reported by ``ffmpeg -encoders``.

    Returns an empty set when FFmpeg cannot be run at all, which the callers
    treat as "nothing is available" rather than crashing.
    """
    ffmpeg = ffmpeg or binaries().ffmpeg
    try:
        out = run([ffmpeg, "-hide_banner", "-encoders"]).stdout
    except ToolError:
        return set()
    # Lines look like: " V....D hevc_vaapi   H.265/HEVC (VAAPI)"
    return set(re.findall(r"^\s*[A-Z.]{6}\s+(\S+)", out, re.MULTILINE))


def encoder_chain(
    found: set[str] | None = None,
    *,
    prefer: str | None = None,
    platform: str | None = None,
    vaapi_device: Path = DEFAULT_VAAPI_DEVICE,
    allow_hardware: bool = True,
) -> list[Encoder]:
    """Rank the usable encoders, best first.

    Args:
        found: Encoder names FFmpeg reports; queried if omitted.
        prefer: An encoder to put first, if FFmpeg has it.
        platform: ``sys.platform`` override, for tests.
        vaapi_device: Render node passed to VAAPI.
        allow_hardware: Set false to encode on the CPU only.
    """
    found = available_encoders() if found is None else found
    platform = platform or sys.platform

    order: list[str] = []
    if prefer:
        order.append(prefer)
    if allow_hardware:
        order += list(HW_ENCODERS.get(platform, ()))
    order += list(CPU_ENCODERS)

    if platform == "linux" and allow_hardware and not vaapi_device.exists():
        # No render node means VAAPI cannot work; do not waste an attempt.
        logger.debug("%s missing, skipping VAAPI encoders", vaapi_device)
        order = [name for name in order if not name.endswith("vaapi") or name == prefer]

    chain: list[Encoder] = []
    seen: set[str] = set()
    for name in order:
        if name in seen or name not in found:
            continue
        seen.add(name)
        chain.append(_build(name, vaapi_device))
    return chain


def best_encoder(**kwargs: object) -> Encoder | None:
    """The encoder that would be tried first, or ``None`` if there is none."""
    chain = encoder_chain(**kwargs)  # type: ignore[arg-type]
    return chain[0] if chain else None


def describe_chain(chain: list[Encoder]) -> str:
    """One-line summary of the fallback order, for logs and the dry run."""
    if not chain:
        return "none"
    return " -> ".join(f"{e.name}{' (hw)' if e.is_hardware else ''}" for e in chain)


class EncodeError(RuntimeError):
    """No encoder in the chain managed to produce the output file."""


#: Name of the concat demuxer's playlist, written next to the parts.
CONCAT_LIST = "list.txt"


def write_concat_list(parts: list[Path], destination: Path) -> Path:
    """Write the concat demuxer playlist for ``parts``.

    Paths are absolute and single-quoted, with any quote in a name escaped the
    way the demuxer expects, so a directory with a space or an apostrophe in it
    does not break the list.
    """
    lines = []
    for part in parts:
        quoted = str(part.resolve()).replace("'", r"'\''")
        lines.append(f"file '{quoted}'")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    return destination


def concat(
    parts: list[Path],
    output: Path,
    *,
    audio_source: Path | None = None,
    list_path: Path | None = None,
    bins: Binaries | None = None,
) -> Path:
    """Join already-encoded ``parts`` into ``output``, adding audio.

    The parts are copied, not re-encoded: they came out of the same encoder
    with the same settings, so there is nothing to gain from decoding them
    again. The audio is muxed here, once, from the original file rather than
    per part — a ``-shortest`` on every part would trim each one to its own
    shorter stream, and that sub-frame error accumulates into audible drift
    over a long video.

    Raises:
        EncodeError: ``parts`` is empty or one of the files is missing.
        ValueError: the output would overwrite a part or the audio source.
    """
    bins = bins or binaries()
    if not parts:
        raise EncodeError("nothing to concatenate: no parts were produced")
    missing = [part for part in parts if not part.exists()]
    if missing:
        raise EncodeError(
            "cannot concatenate, these parts are missing:\n  "
            + "\n  ".join(str(part) for part in missing)
        )

    resolved = output.resolve()
    if audio_source and resolved == audio_source.resolve():
        raise ValueError(f"refusing to overwrite the input video: {output}")
    if any(resolved == part.resolve() for part in parts):
        raise ValueError(f"refusing to overwrite a part with the output: {output}")

    list_path = write_concat_list(parts, list_path or parts[0].parent / CONCAT_LIST)

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(f".part{output.suffix}")
    cmd: list[object] = [
        bins.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        # The playlist names absolute paths, which the demuxer refuses to
        # follow unless unsafe filenames are allowed.
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
    ]
    if audio_source:
        cmd += ["-i", audio_source]
    cmd += ["-map", "0:v"]
    if audio_source:
        cmd += ["-map", "1:a?", "-c:a", "copy", "-shortest"]
    cmd += ["-c:v", "copy", partial]

    logger.info("concatenating %d parts into %s", len(parts), output)
    run(cmd, capture=False)
    partial.replace(output)
    return output


def build_encode_cmd(
    frames_dir: Path,
    output: Path,
    encoder: Encoder,
    *,
    fps: Fraction | float,
    audio_source: Path | None = None,
    frame_format: str = DEFAULT_FRAME_FORMAT,
    bins: Binaries | None = None,
) -> list[object]:
    """Assemble the FFmpeg call that turns a frame directory into a video.

    The frames are input 0 and the audio file, when there is one, is input 1,
    so the stream mapping stays the same whether or not audio exists.
    """
    bins = bins or binaries()
    cmd: list[object] = [bins.ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    cmd += encoder.init_args
    cmd += ["-framerate", str(fps), "-i", frames_dir / frame_pattern(frame_format)]
    if audio_source:
        cmd += ["-i", audio_source]

    cmd += ["-map", "0:v"]
    if audio_source:
        # The '?' keeps this optional: a source without audio is not an error.
        cmd += ["-map", "1:a?", "-c:a", "copy", "-shortest"]

    if encoder.filters:
        cmd += ["-filter:v", ",".join(encoder.filters)]
    cmd += ["-c:v", encoder.name, *encoder.quality_args]
    # Declare the frame rate on the output too, so the container timebase
    # matches the frames and the audio stays aligned.
    cmd += ["-r", str(fps), output]
    return cmd


def encode(
    frames_dir: Path,
    output: Path,
    *,
    fps: Fraction | float,
    audio_source: Path | None = None,
    frame_format: str = DEFAULT_FRAME_FORMAT,
    chain: list[Encoder] | None = None,
    prefer: str | None = None,
    allow_hardware: bool = True,
    bins: Binaries | None = None,
) -> Encoder:
    """Encode ``frames_dir`` into ``output`` and return the encoder that won.

    Walks the fallback chain until one encoder succeeds. Writes to a temporary
    file and moves it into place, so a failed attempt never leaves a truncated
    video behind and never touches an existing output until the new one is
    complete.

    Raises:
        EncodeError: the frame directory is empty, or every encoder failed.
        ValueError: the output would overwrite the audio source.
    """
    bins = bins or binaries()
    if not any(frames_dir.glob(f"*.{frame_format}")):
        raise EncodeError(f"no {frame_format} frames to encode in {frames_dir}")
    if audio_source and output.resolve() == audio_source.resolve():
        raise ValueError(f"refusing to overwrite the input video: {output}")

    chain = chain if chain is not None else encoder_chain(
        prefer=prefer, allow_hardware=allow_hardware
    )
    if not chain:
        raise EncodeError(
            "no usable video encoder; install an FFmpeg build that includes libx264"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(f".part{output.suffix}")
    failures: list[str] = []

    for encoder in chain:
        logger.info("encoding %s with %s", output.name, encoder.name)
        try:
            run(
                build_encode_cmd(
                    frames_dir,
                    partial,
                    encoder,
                    fps=fps,
                    audio_source=audio_source,
                    frame_format=frame_format,
                    bins=bins,
                ),
                capture=False,
            )
        except ToolError as exc:
            partial.unlink(missing_ok=True)
            failures.append(f"{encoder.name}: exit {exc.returncode}")
            logger.warning("%s failed, trying the next encoder", encoder.name)
            continue

        partial.replace(output)
        logger.info("wrote %s with %s", output, encoder.name)
        return encoder

    raise EncodeError(
        "every encoder failed:\n  " + "\n  ".join(failures)
    )
