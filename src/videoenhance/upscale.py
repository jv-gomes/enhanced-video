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
from .process import ToolError, run
from .workdir import count_frames_on_disk, stage_is_done

logger = logging.getLogger(__name__)


class UpscaleError(RuntimeError):
    """Upscaling produced no frames, or fewer than it was given."""


class ModelError(ValueError):
    """The requested model cannot do the requested scale."""


class VramError(UpscaleError):
    """The GPU ran out of memory, at every tile size that was tried."""


#: Fragments that mean "out of VRAM" rather than a real bug. Vulkan surfaces
#: this in several different ways depending on the driver.
VRAM_MARKERS = (
    "vkallocatememory failed",
    "out of device memory",
    "out of memory",
    "vkmapmemory failed",
    "vkqueuesubmit failed",
    "failed to allocate",
    "device lost",
)


def looks_like_vram_exhaustion(error: ToolError) -> bool:
    """Whether a failure is the GPU running out of memory.

    Only the tile size can fix that, so it is worth separating from the
    failures that retrying will not help.
    """
    haystack = f"{error.stderr}".lower()
    return any(marker in haystack for marker in VRAM_MARKERS)


def resolve_model(model: str, scale: int) -> tuple[str, int]:
    """Check a model/scale pair, and say what to do when it does not work.

    Real-ESRGAN models are trained per factor: ``realesrgan-x4plus`` only
    exists at 4x, while ``realesr-animevideov3`` covers 2x, 3x and 4x. Asking
    for a combination that does not exist fails inside the binary with an
    unhelpful error, so it is caught here.
    """
    supported = config.UPSCALE_MODELS.get(model)
    if supported is None:
        known = ", ".join(sorted(config.UPSCALE_MODELS))
        raise ModelError(f"unknown model {model!r}; available models are {known}")
    if scale not in supported:
        factors = ", ".join(str(s) for s in supported)
        alternatives = sorted(
            name for name, scales in config.UPSCALE_MODELS.items() if scale in scales
        )
        hint = f"; {', '.join(alternatives)} can do {scale}x" if alternatives else ""
        raise ModelError(f"{model} only supports scale {factors}, not {scale}{hint}")
    return model, scale


def models_for_scale(scale: int) -> list[str]:
    """Every model that can upscale by ``scale``."""
    return sorted(
        name for name, scales in config.UPSCALE_MODELS.items() if scale in scales
    )


@dataclass(frozen=True)
class UpscaleResult:
    """What the upscale stage produced."""

    frames_dir: Path
    frame_count: int
    scale: int
    model: str
    #: Tile size that actually worked, which may be smaller than requested.
    tile: int = config.DEFAULT_TILE_SIZE
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


def tile_attempts(tile: int) -> list[int]:
    """Tile sizes to try, largest first.

    Starting from the requested size, this walks down the configured
    fallbacks. An explicit tile size the user asked for is still retried
    smaller, because the alternative is a crash hours into a run.
    """
    attempts = [tile]
    attempts += [size for size in config.TILE_FALLBACKS if tile == 0 or size < tile]
    return attempts


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
    model, scale = resolve_model(model, scale)
    if tile < 0:
        raise ValueError("tile size must be 0 (automatic) or greater")
    if frame_format not in config.FRAME_FORMATS:
        raise ValueError(f"unsupported frame format {frame_format!r}")

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
            tile=tile,
            skipped=True,
        )

    logger.info(
        "upscaling %d frames by %dx with %s (gpu %d, tile %s, threads %s)",
        source_count,
        scale,
        model,
        gpu,
        tile or "auto",
        threads,
    )

    attempts = tile_attempts(tile)
    for index, attempt in enumerate(attempts):
        if index:
            logger.warning("retrying at tile size %d", attempt)
        try:
            run(
                _build_cmd(
                    frames_in,
                    frames_out,
                    scale=scale,
                    model=model,
                    frame_format=frame_format,
                    gpu=gpu,
                    tile=attempt,
                    threads=threads,
                    bins=bins,
                ),
                capture=True,
            )
        except ToolError as exc:
            if not looks_like_vram_exhaustion(exc) or index == len(attempts) - 1:
                if looks_like_vram_exhaustion(exc):
                    raise VramError(
                        f"the GPU ran out of memory at every tile size tried "
                        f"({', '.join(str(a) for a in attempts)}); try "
                        f"--frame-format jpg, a smaller --scale, or a lower -j"
                    ) from exc
                raise
            continue

        produced = count_frames_on_disk(frames_out, frame_format)
        if produced < expected:
            # Exit code zero with frames missing is the other face of running
            # out of memory, so it feeds the same retry.
            if index < len(attempts) - 1:
                logger.warning(
                    "only %d of %d frames came back; the GPU may be short on memory",
                    produced,
                    expected,
                )
                continue
            raise UpscaleError(
                f"upscaling produced {produced} of {expected} frames in {frames_out}; "
                f"tile sizes {', '.join(str(a) for a in attempts)} were tried"
            )

        logger.info("upscaled %d frames into %s", produced, frames_out)
        return UpscaleResult(
            frames_dir=frames_out,
            frame_count=produced,
            scale=scale,
            model=model,
            tile=attempt,
        )

    raise UpscaleError(f"upscaling {frames_in} did not complete")
