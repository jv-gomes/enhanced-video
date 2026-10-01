"""Driving the stages, in order, resumably.

The pipeline owns the sequence and the decisions around it: which stages are
worth running at all, what each one should produce, and when the scratch space
can be thrown away. The stages themselves live in their own modules, and the
directories they share come from :mod:`videoenhance.workdir`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from . import config
from .config import Binaries
from .encode import encode
from .extract import extract
from .interpolate import interpolate, is_needed, target_frame_count
from .probe import VideoInfo, probe
from .upscale import upscale
from .workdir import (
    FRAMES_IN,
    FRAMES_OUT,
    FRAMES_UP,
    SpaceEstimate,
    WorkDir,
    count_frames_on_disk,
    estimate_disk_usage,
    estimate_frame_bytes,
    frame_glob,
    frame_pattern,
    frames_complete,
    human_bytes,
    stage_is_done,
)

logger = logging.getLogger(__name__)

# Re-exported so callers can treat the pipeline as the single entry point.
__all__ = [
    "FRAMES_IN",
    "FRAMES_OUT",
    "FRAMES_UP",
    "Options",
    "Result",
    "SpaceEstimate",
    "Stage",
    "WorkDir",
    "count_frames_on_disk",
    "estimate_disk_usage",
    "estimate_frame_bytes",
    "frame_glob",
    "frame_pattern",
    "frames_complete",
    "human_bytes",
    "run_pipeline",
    "stage_is_done",
]


@dataclass(frozen=True)
class Options:
    """Everything the pipeline needs to know about how to process a file."""

    scale: int = config.DEFAULT_SCALE
    target_fps: float = config.DEFAULT_TARGET_FPS
    model: str = config.DEFAULT_UPSCALE_MODEL
    rife_model: str = config.DEFAULT_RIFE_MODEL
    frame_format: str = config.DEFAULT_FRAME_FORMAT
    gpu: int = config.DEFAULT_GPU_ID
    tile: int = config.DEFAULT_TILE_SIZE
    threads: str = config.DEFAULT_THREADS
    keep_temp: bool = False
    allow_hardware: bool = True
    prefer_encoder: str | None = None


@dataclass(frozen=True)
class Stage:
    """One completed stage, for the run summary."""

    name: str
    frames: int
    detail: str = ""
    skipped: bool = False

    def render(self) -> str:
        suffix = " (already done)" if self.skipped else ""
        detail = f", {self.detail}" if self.detail else ""
        return f"{self.name}: {self.frames} frames{detail}{suffix}"


@dataclass(frozen=True)
class Result:
    """What a pipeline run produced."""

    output: Path
    encoder: str
    fps: Fraction
    frames: int
    stages: list[Stage]

    def render(self) -> str:
        lines = [stage.render() for stage in self.stages]
        lines.append(f"encoded {self.frames} frames at {float(self.fps):g} fps with {self.encoder}")
        lines.append(f"wrote {self.output}")
        return "\n".join(lines)


def run_pipeline(
    input_path: Path,
    output: Path,
    options: Options | None = None,
    *,
    info: VideoInfo | None = None,
    work: WorkDir | None = None,
    bins: Binaries | None = None,
) -> Result:
    """Run every stage and return what was produced.

    The stages write into fixed directories named after the tool that fills
    them, so a crashed run resumes from wherever it stopped, and the work
    directory is removed only after the output file exists.

    Raises:
        FileNotFoundError: the input does not exist.
        ValueError: the output would overwrite the input.
    """
    options = options or Options()
    info = info or probe(input_path)
    output = Path(output)
    if output.resolve() == Path(info.path).resolve():
        raise ValueError(f"refusing to overwrite the input video: {output}")

    work = (work or WorkDir.for_input(info.path)).create()
    stages: list[Stage] = []

    extracted = extract(info, work, frame_format=options.frame_format, bins=bins)
    stages.append(
        Stage(
            "extract",
            extracted.frame_count,
            "normalised to CFR" if extracted.normalised else "",
            extracted.skipped,
        )
    )

    frames = extracted.frames_dir
    frame_count = extracted.frame_count
    fps: Fraction = extracted.fps

    if options.scale > 1:
        upscaled = upscale(
            frames,
            work.frames_up,
            expected_frames=frame_count,
            scale=options.scale,
            model=options.model,
            frame_format=options.frame_format,
            gpu=options.gpu,
            tile=options.tile,
            threads=options.threads,
            bins=bins,
        )
        stages.append(
            Stage(
                "upscale",
                upscaled.frame_count,
                f"{options.scale}x with {options.model}",
                upscaled.skipped,
            )
        )
        frames = upscaled.frames_dir
        frame_count = upscaled.frame_count
    else:
        logger.info("skipping upscale: scale is 1")

    if is_needed(fps, options.target_fps):
        target = target_frame_count(frame_count, fps, options.target_fps)
        interpolated = interpolate(
            frames,
            work.frames_out,
            target_frames=target,
            model=options.rife_model,
            frame_format=options.frame_format,
            gpu=options.gpu,
            threads=options.threads,
            width=info.width * options.scale,
            height=info.height * options.scale,
            bins=bins,
        )
        stages.append(
            Stage(
                "interpolate",
                interpolated.frame_count,
                f"to {float(options.target_fps):g} fps"
                + (", UHD mode" if interpolated.uhd else ""),
                interpolated.skipped,
            )
        )
        frames = interpolated.frames_dir
        frame_count = interpolated.frame_count
        fps = Fraction(options.target_fps).limit_denominator(100000)
    else:
        logger.info("skipping interpolation: %s fps is not above %s", options.target_fps, fps)

    encoder = encode(
        frames,
        output,
        fps=fps,
        audio_source=extracted.audio_source if extracted.has_audio else None,
        frame_format=options.frame_format,
        prefer=options.prefer_encoder,
        allow_hardware=options.allow_hardware,
        bins=bins,
    )

    # Only now that the output exists is the scratch space safe to remove.
    work.cleanup(keep=options.keep_temp)

    return Result(
        output=output,
        encoder=encoder.name,
        fps=fps,
        frames=frame_count,
        stages=stages,
    )
