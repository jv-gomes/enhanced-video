"""Frame interpolation via ``rife-ncnn-vulkan``.

RIFE differs from Real-ESRGAN in one way that shapes this module: it is told
how many frames to produce, not what factor to apply. So the frame count is
computed here and passed as ``-n``, and the output filenames come from a
pattern rather than from the input names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from . import config
from .config import Binaries, binaries
from .pipeline import count_frames_on_disk, frame_pattern, stage_is_done
from .process import ToolError, run
from .upscale import looks_like_vram_exhaustion

logger = logging.getLogger(__name__)


class InterpolateError(RuntimeError):
    """Interpolation produced no frames, or fewer than it was asked for."""


def needs_uhd(width: int, height: int) -> bool:
    """Whether RIFE should run in UHD mode for this resolution.

    UHD mode changes how RIFE handles large motion, which matters once the
    frames are 4K or larger. It also costs memory, so it is not enabled below
    that.
    """
    return width * height >= config.UHD_PIXEL_THRESHOLD


def target_frame_count(
    source_frames: int, source_fps: Fraction | float, target_fps: Fraction | float
) -> int:
    """How many frames the output needs to hold ``source_frames`` at ``target_fps``.

    ``source_frames * target_fps / source_fps``, rounded to the nearest whole
    frame. Keeping the ratio exact matters: a count computed from a rounded
    frame rate drifts against the audio, and 29.97 fps sources are the common
    case, not the exception.
    """
    if source_frames <= 0:
        raise InterpolateError("cannot compute a target from zero source frames")
    source = Fraction(source_fps).limit_denominator(100000)
    target = Fraction(target_fps).limit_denominator(100000)
    if source <= 0 or target <= 0:
        raise InterpolateError(f"frame rates must be positive, got {source_fps} -> {target_fps}")
    if target < source:
        # Dropping frames is not interpolation; the caller should skip the stage.
        return source_frames
    exact = Fraction(source_frames) * target / source
    return max(source_frames, round(exact))


def interpolation_factor(
    source_fps: Fraction | float, target_fps: Fraction | float
) -> Fraction:
    """The ratio between the two frame rates, for logs and estimates."""
    source = Fraction(source_fps).limit_denominator(100000)
    if source <= 0:
        raise InterpolateError(f"source frame rate must be positive, got {source_fps}")
    return Fraction(target_fps).limit_denominator(100000) / source


def is_needed(source_fps: Fraction | float, target_fps: Fraction | float) -> bool:
    """Whether interpolating would change anything at all."""
    return interpolation_factor(source_fps, target_fps) > 1


@dataclass(frozen=True)
class InterpolateResult:
    """What the interpolation stage produced."""

    frames_dir: Path
    frame_count: int
    model: str
    #: Whether UHD mode was used for the run that succeeded.
    uhd: bool = False
    skipped: bool = False


def _build_cmd(
    frames_in: Path,
    frames_out: Path,
    *,
    target_frames: int,
    model_dir: Path,
    frame_format: str,
    gpu: int,
    threads: str,
    uhd: bool,
    bins: Binaries,
) -> list[object]:
    cmd: list[object] = [
        bins.rife,
        "-i",
        frames_in,
        "-o",
        frames_out,
        "-m",
        model_dir,
        # RIFE is told the total number of output frames, not a multiplier.
        "-n",
        str(target_frames),
        # Here -f is the output filename pattern, not the image format.
        "-f",
        frame_pattern(frame_format),
        "-g",
        str(gpu),
        "-j",
        threads,
    ]
    if uhd:
        cmd.append("-u")
    return cmd


def interpolate(
    frames_in: Path,
    frames_out: Path,
    *,
    target_frames: int | None = None,
    source_fps: Fraction | float | None = None,
    target_fps: Fraction | float | None = None,
    model: str = config.DEFAULT_RIFE_MODEL,
    frame_format: str = config.DEFAULT_FRAME_FORMAT,
    gpu: int = config.DEFAULT_GPU_ID,
    threads: str = config.DEFAULT_THREADS,
    uhd: bool | None = None,
    width: int = 0,
    height: int = 0,
    bins: Binaries | None = None,
    force: bool = False,
) -> InterpolateResult:
    """Interpolate ``frames_in`` up to ``target_frames`` frames in ``frames_out``.

    Give either ``target_frames`` directly, or ``source_fps`` and
    ``target_fps`` to have it computed from the frames on disk.

    UHD mode is enabled automatically when ``width`` and ``height`` describe a
    4K-or-larger frame, and can be forced either way with ``uhd``. Because UHD
    mode is also the first thing to drop when the GPU runs out of memory, a
    Vulkan allocation failure is retried once with it off.

    Raises:
        InterpolateError: there is nothing to interpolate, the target is not a
            sensible frame count, or the binary returned too few frames.
        MissingToolError: the binary is not installed.
        ToolError: the binary failed.
    """
    bins = bins or binaries()
    source_count = count_frames_on_disk(frames_in, frame_format)
    if source_count == 0:
        raise InterpolateError(f"no {frame_format} frames to interpolate in {frames_in}")

    if target_frames is None:
        if source_fps is None or target_fps is None:
            raise InterpolateError(
                "interpolate() needs either target_frames, or both source_fps and target_fps"
            )
        target_frames = target_frame_count(source_count, source_fps, target_fps)
        logger.debug(
            "target of %d frames for %s -> %s fps", target_frames, source_fps, target_fps
        )
    if target_frames < source_count:
        raise InterpolateError(
            f"target of {target_frames} frames is below the {source_count} frames "
            "going in; interpolation can only add frames"
        )

    if uhd is None:
        uhd = needs_uhd(width, height) if width and height else False

    model_dir = config.rife_model_dir(model, bins)
    frames_out.mkdir(parents=True, exist_ok=True)

    if not force and stage_is_done("interpolate", frames_out, target_frames, frame_format):
        return InterpolateResult(
            frames_dir=frames_out,
            frame_count=count_frames_on_disk(frames_out, frame_format),
            model=model,
            uhd=bool(uhd),
            skipped=True,
        )

    logger.info(
        "interpolating %d frames to %d with %s (gpu %d, threads %s, uhd %s)",
        source_count,
        target_frames,
        model,
        gpu,
        threads,
        "on" if uhd else "off",
    )

    def attempt(use_uhd: bool) -> None:
        run(
            _build_cmd(
                frames_in,
                frames_out,
                target_frames=target_frames,
                model_dir=model_dir,
                frame_format=frame_format,
                gpu=gpu,
                threads=threads,
                uhd=use_uhd,
                bins=bins,
            ),
            capture=True,
        )

    try:
        attempt(uhd)
    except ToolError as exc:
        if not (uhd and looks_like_vram_exhaustion(exc)):
            raise
        logger.warning("UHD mode ran out of memory, retrying without it")
        attempt(False)
        uhd = False

    produced = count_frames_on_disk(frames_out, frame_format)
    if produced < target_frames:
        raise InterpolateError(
            f"interpolation produced {produced} of {target_frames} frames in "
            f"{frames_out}"
        )
    logger.info("interpolated to %d frames in %s", produced, frames_out)
    return InterpolateResult(
        frames_dir=frames_out, frame_count=produced, model=model, uhd=uhd
    )
