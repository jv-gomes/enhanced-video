"""Super-resolution via ``realesrgan-ncnn-vulkan``.

The binary does the work: it takes an input directory of frames and writes an
output directory of larger ones, on any Vulkan GPU. This module's job is to
call it correctly, to notice when it has already been run, and to check that
what came back is actually a complete set of frames — the binary can exit zero
having skipped frames it could not fit in VRAM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from . import config
from .config import Binaries, binaries
from .pipeline import count_frames_on_disk, stage_is_done
from .process import run

logger = logging.getLogger(__name__)


class UpscaleError(RuntimeError):
    """Upscaling produced no frames, or fewer than it was given."""


@dataclass(frozen=True)
class UpscaleResult:
    """What the upscale stage produced."""

    frames_dir: Path
    frame_count: int
    scale: int
    model: str
    #: True when the frames were already on disk from an earlier run.
    skipped: bool = False


def _build_cmd(
    frames_in: Path,
    frames_out: Path,
    *,
    scale: int,
    model: str,
    frame_format: str,
    gpu: int,
    tile: int,
    threads: str,
    bins: Binaries,
) -> list[object]:
    cmd: list[object] = [
        bins.realesrgan,
        "-i",
        frames_in,
        "-o",
        frames_out,
        "-n",
        model,
        "-s",
        str(scale),
        "-f",
        frame_format,
        "-g",
        str(gpu),
        "-t",
        str(tile),
        "-j",
        threads,
    ]
    # The binary looks for ./models next to itself; pass the path explicitly so
    # it also works when called from another working directory.
    if bins.realesrgan_models.is_dir():
        cmd += ["-m", bins.realesrgan_models]
    return cmd


def upscale(
    frames_in: Path,
    frames_out: Path,
    *,
    expected_frames: int = 0,
    scale: int = config.DEFAULT_SCALE,
    model: str = config.DEFAULT_UPSCALE_MODEL,
    frame_format: str = config.DEFAULT_FRAME_FORMAT,
    gpu: int = config.DEFAULT_GPU_ID,
    tile: int = config.DEFAULT_TILE_SIZE,
    threads: str = config.DEFAULT_THREADS,
    bins: Binaries | None = None,
    force: bool = False,
) -> UpscaleResult:
    """Upscale every frame in ``frames_in`` into ``frames_out``.

    Args:
        expected_frames: How many frames should come out. Defaults to the
            number going in, which is what the stage should produce.
        force: Re-run even when the output already looks complete.

    Raises:
        UpscaleError: no frames to upscale, or the binary returned fewer
            frames than it was given.
        MissingToolError: the binary is not installed.
        ToolError: the binary failed.
    """
    bins = bins or binaries()
    source_count = count_frames_on_disk(frames_in, frame_format)
    if source_count == 0:
        raise UpscaleError(f"no {frame_format} frames to upscale in {frames_in}")

    expected = expected_frames or source_count
    frames_out.mkdir(parents=True, exist_ok=True)

    if not force and stage_is_done("upscale", frames_out, expected, frame_format):
        return UpscaleResult(
            frames_dir=frames_out,
            frame_count=count_frames_on_disk(frames_out, frame_format),
            scale=scale,
            model=model,
            skipped=True,
        )

    logger.info(
        "upscaling %d frames by %dx with %s", source_count, scale, model
    )
    run(
        _build_cmd(
            frames_in,
            frames_out,
            scale=scale,
            model=model,
            frame_format=frame_format,
            gpu=gpu,
            tile=tile,
            threads=threads,
            bins=bins,
        ),
        capture=False,
    )

    produced = count_frames_on_disk(frames_out, frame_format)
    if produced < expected:
        raise UpscaleError(
            f"upscaling produced {produced} of {expected} frames in {frames_out}; "
            "the GPU may have run out of memory, try a smaller --tile"
        )
    logger.info("upscaled %d frames into %s", produced, frames_out)
    return UpscaleResult(
        frames_dir=frames_out, frame_count=produced, scale=scale, model=model
    )
