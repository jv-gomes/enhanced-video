"""Frame interpolation via ``rife-ncnn-vulkan``.

RIFE differs from Real-ESRGAN in one way that shapes this module: it is told
how many frames to produce, not what factor to apply. So the frame count is
computed here and passed as ``-n``, and the output filenames come from a
pattern rather than from the input names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from . import config
from .config import Binaries, binaries
from .pipeline import count_frames_on_disk, frame_pattern, stage_is_done
from .process import run

logger = logging.getLogger(__name__)


class InterpolateError(RuntimeError):
    """Interpolation produced no frames, or fewer than it was asked for."""


@dataclass(frozen=True)
class InterpolateResult:
    """What the interpolation stage produced."""

    frames_dir: Path
    frame_count: int
    model: str
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
    bins: Binaries,
) -> list[object]:
    return [
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


def interpolate(
    frames_in: Path,
    frames_out: Path,
    *,
    target_frames: int,
    model: str = config.DEFAULT_RIFE_MODEL,
    frame_format: str = config.DEFAULT_FRAME_FORMAT,
    gpu: int = config.DEFAULT_GPU_ID,
    threads: str = config.DEFAULT_THREADS,
    bins: Binaries | None = None,
    force: bool = False,
) -> InterpolateResult:
    """Interpolate ``frames_in`` up to ``target_frames`` frames in ``frames_out``.

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
    if target_frames < source_count:
        raise InterpolateError(
            f"target of {target_frames} frames is below the {source_count} frames "
            "going in; interpolation can only add frames"
        )

    model_dir = config.rife_model_dir(model, bins)
    frames_out.mkdir(parents=True, exist_ok=True)

    if not force and stage_is_done("interpolate", frames_out, target_frames, frame_format):
        return InterpolateResult(
            frames_dir=frames_out,
            frame_count=count_frames_on_disk(frames_out, frame_format),
            model=model,
            skipped=True,
        )

    logger.info(
        "interpolating %d frames to %d with %s (gpu %d, threads %s)",
        source_count,
        target_frames,
        model,
        gpu,
        threads,
    )
    run(
        _build_cmd(
            frames_in,
            frames_out,
            target_frames=target_frames,
            model_dir=model_dir,
            frame_format=frame_format,
            gpu=gpu,
            threads=threads,
            bins=bins,
        ),
        capture=True,
    )

    produced = count_frames_on_disk(frames_out, frame_format)
    if produced < target_frames:
        raise InterpolateError(
            f"interpolation produced {produced} of {target_frames} frames in "
            f"{frames_out}"
        )
    logger.info("interpolated to %d frames in %s", produced, frames_out)
    return InterpolateResult(frames_dir=frames_out, frame_count=produced, model=model)
