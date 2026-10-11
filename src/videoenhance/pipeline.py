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
from .dedup import FrameCache, upscale_deduplicated
from .encode import encode
from .extract import extract
from .interpolate import interpolate, is_needed, target_frame_count
from .probe import VideoInfo, probe
from .progress import track
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

# The order names live in config, with the other defaults, so that the disk
# estimate in workdir can honour them too.
ORDER_UPSCALE_FIRST = config.ORDER_UPSCALE_FIRST
ORDER_INTERPOLATE_FIRST = config.ORDER_INTERPOLATE_FIRST
ORDERS = config.ORDERS

# Re-exported so callers can treat the pipeline as the single entry point.
__all__ = [
    "FRAMES_IN",
    "FRAMES_OUT",
    "FRAMES_UP",
    "Generated",
    "ORDERS",
    "ORDER_INTERPOLATE_FIRST",
    "ORDER_UPSCALE_FIRST",
    "Options",
    "Result",
    "SpaceEstimate",
    "Stage",
    "WorkDir",
    "count_frames_on_disk",
    "estimate_disk_usage",
    "estimate_frame_bytes",
    "finish",
    "frame_glob",
    "frame_pattern",
    "frames_complete",
    "generate",
    "human_bytes",
    "run_pipeline",
    "stage_is_done",
    "stage_order",
]


@dataclass(frozen=True)
class Options:
    """Everything the pipeline needs to know about how to process a file."""

    scale: int = config.DEFAULT_SCALE
    target_fps: float = config.DEFAULT_TARGET_FPS
    model: str = config.DEFAULT_UPSCALE_MODEL
    rife_model: str = config.DEFAULT_RIFE_MODEL
    frame_format: str = config.DEFAULT_FRAME_FORMAT
    order: str = config.DEFAULT_ORDER
    #: Seconds per chunk, 0 meaning "process the file in a single pass".
    #: Honoured by :func:`videoenhance.chunk.run_chunked`, which drives this
    #: pipeline once per chunk.
    chunk_seconds: float = config.DEFAULT_CHUNK_SECONDS
    #: Carry the source's audio into the output. Chunked mode turns this off
    #: for the individual parts and muxes the audio once, at the end.
    mux_audio: bool = True
    #: Generate each distinct frame once, reusing the result for every repeat.
    dedup: bool = config.DEFAULT_DEDUP
    #: Ceiling in bytes for the cache of upscaled frames; 0 means no ceiling.
    cache_size: int = config.DEFAULT_CACHE_SIZE
    #: Where that cache lives. Chunked mode points it at the job's directory so
    #: a frame generated for one chunk is still there for the next; left unset,
    #: it goes in the work directory of the run itself.
    cache_dir: Path | None = None
    #: Draw a progress bar per stage. The CLI turns this on for a terminal.
    progress: bool = False
    gpu: int = config.DEFAULT_GPU_ID
    tile: int = config.DEFAULT_TILE_SIZE
    threads: str = config.DEFAULT_THREADS
    #: Blocks per side for block dedup, 1 meaning whole frames only.
    blocks: int = config.DEFAULT_BLOCKS
    #: Pixels of neighbours upscaled with each block. Only matters above 1.
    block_margin: int = config.DEFAULT_BLOCK_MARGIN
    #: Percent of the time the GPU tools may run; below 100 they are paused
    #: in between so the card is not saturated.
    load: int = config.DEFAULT_GPU_LOAD
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
    #: What ``frames`` counts. Every stage but the chunked run's summary
    #: counts frames.
    unit: str = "frames"

    def render(self) -> str:
        suffix = " (already done)" if self.skipped else ""
        detail = f", {self.detail}" if self.detail else ""
        return f"{self.name}: {self.frames} {self.unit}{detail}{suffix}"


@dataclass(frozen=True)
class Result:
    """What a pipeline run produced."""

    output: Path
    encoder: str
    fps: Fraction
    frames: int
    stages: list[Stage]
    #: The scratch directory the run used.
    work: Path
    #: Whether its frames are still there, i.e. ``--keep-temp`` was given.
    kept: bool = False

    def render(self) -> str:
        lines = [stage.render() for stage in self.stages]
        lines.append(f"encoded {self.frames} frames at {float(self.fps):g} fps with {self.encoder}")
        lines.append(f"wrote {self.output}")
        if self.kept:
            lines.append(f"kept the frames in {self.work}")
        return "\n".join(lines)


@dataclass(frozen=True)
class _Frames:
    """What the next stage needs to know about the frames it will read.

    Carried from stage to stage so that either order works without a stage
    having to guess what ran before it: interpolating first means RIFE sees the
    source resolution, and upscaling first means it sees the upscaled one, and
    only this record knows which.
    """

    directory: Path
    count: int
    fps: Fraction
    width: int
    height: int


def _upscale_stage(
    frames: _Frames,
    work: WorkDir,
    options: Options,
    stages: list[Stage],
    bins: Binaries | None,
) -> _Frames:
    """Run Real-ESRGAN, unless the requested scale is 1."""
    if options.scale <= 1:
        logger.info("skipping upscale: scale is 1")
        return frames

    shared = dict(
        expected_frames=frames.count,
        scale=options.scale,
        model=options.model,
        frame_format=options.frame_format,
        gpu=options.gpu,
        tile=options.tile,
        threads=options.threads,
        load=options.load,
        bins=bins,
    )
    stats = None
    with track(
        "upscale",
        work.frames_up,
        frames.count,
        frame_format=options.frame_format,
        enabled=options.progress,
    ):
        if options.dedup:
            cache = FrameCache(
                options.cache_dir or work.cache,
                model=options.model,
                scale=options.scale,
                tile=options.tile,
                frame_format=options.frame_format,
                budget=options.cache_size,
                blocks=options.blocks,
                margin=options.block_margin if options.blocks > 1 else 0,
            )
            result, stats = upscale_deduplicated(
                frames.directory,
                work.frames_up,
                scratch=work.dedup,
                cache=cache,
                blocks=options.blocks,
                margin=options.block_margin if options.blocks > 1 else 0,
                **shared,
            )
        else:
            result = upscale(frames.directory, work.frames_up, **shared)
    detail = f"{options.scale}x with {options.model}"
    if result.tile:
        detail += f", tile {result.tile}"
    if stats is not None and stats.saved:
        detail += f"; {stats.render()}"
    stages.append(Stage("upscale", result.frame_count, detail, result.skipped))
    return _Frames(
        directory=result.frames_dir,
        count=result.frame_count,
        fps=frames.fps,
        width=frames.width * options.scale,
        height=frames.height * options.scale,
    )


def _interpolate_stage(
    frames: _Frames,
    work: WorkDir,
    options: Options,
    stages: list[Stage],
    bins: Binaries | None,
) -> _Frames:
    """Run RIFE, unless the target frame rate is not above the source's."""
    if not is_needed(frames.fps, options.target_fps):
        logger.info(
            "skipping interpolation: %g fps is not above the source's %s",
            options.target_fps,
            frames.fps,
        )
        return frames
    if frames.count < 2:
        # RIFE interpolates *between* frames, so one frame has nothing to pair
        # with. A single-frame chunk is rare but reachable: it is whatever is
        # left over at the end of a chunked run.
        logger.info("skipping interpolation: %d frame is not enough", frames.count)
        return frames

    target = target_frame_count(frames.count, frames.fps, options.target_fps)
    with track(
        "interpolate",
        work.frames_out,
        target,
        frame_format=options.frame_format,
        enabled=options.progress,
    ):
        result = interpolate(
            frames.directory,
            work.frames_out,
            target_frames=target,
            model=options.rife_model,
            frame_format=options.frame_format,
            gpu=options.gpu,
            threads=options.threads,
            load=options.load,
            width=frames.width,
            height=frames.height,
            bins=bins,
        )
    detail = f"to {float(options.target_fps):g} fps" + (", UHD mode" if result.uhd else "")
    stages.append(Stage("interpolate", result.frame_count, detail, result.skipped))
    return _Frames(
        directory=result.frames_dir,
        count=result.frame_count,
        fps=Fraction(options.target_fps).limit_denominator(100000),
        width=frames.width,
        height=frames.height,
    )


#: The two stages the order switches between, by name.
_MODEL_STAGES = {"upscale": _upscale_stage, "interpolate": _interpolate_stage}


def stage_order(order: str = config.DEFAULT_ORDER) -> tuple[str, ...]:
    """The two GPU stages, in the order ``order`` asks for.

    Raises:
        ValueError: the order is not one of :data:`ORDERS`.
    """
    if order == ORDER_UPSCALE_FIRST:
        return ("upscale", "interpolate")
    if order == ORDER_INTERPOLATE_FIRST:
        return ("interpolate", "upscale")
    raise ValueError(f"unknown stage order: {order!r} (expected one of {', '.join(ORDERS)})")


@dataclass(frozen=True)
class Generated:
    """The frames a run has made, waiting to be encoded.

    The boundary between the half of a run that needs the GPU and the half
    that does not, so that a chunked run can encode one chunk while the GPU is
    already generating the next.
    """

    frames: _Frames
    stages: list[Stage]
    #: Where the final encode takes its audio from, or ``None`` for none.
    audio_source: Path | None
    work: WorkDir


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

    Each stage writes into a directory named after the tool that fills it, so
    the two orders share their scratch space and a crashed run resumes from
    wherever it stopped. The work directory is removed only after the output
    file exists.

    Raises:
        FileNotFoundError: the input does not exist.
        ValueError: the output would overwrite the input, or the order is not
            one of :data:`ORDERS`.
    """
    options = options or Options()
    info = info or probe(input_path)
    output = Path(output)
    if output.resolve() == Path(info.path).resolve():
        raise ValueError(f"refusing to overwrite the input video: {output}")
    generated = generate(input_path, options, info=info, work=work, bins=bins)
    return finish(generated, output, options, bins=bins)


def generate(
    input_path: Path,
    options: Options | None = None,
    *,
    info: VideoInfo | None = None,
    work: WorkDir | None = None,
    bins: Binaries | None = None,
) -> Generated:
    """Extract the frames and run the GPU stages, stopping short of the encode.

    Raises:
        FileNotFoundError: the input does not exist.
        ValueError: the order is not one of :data:`ORDERS`.
    """
    options = options or Options()
    order = stage_order(options.order)
    info = info or probe(input_path)
    work = (work or WorkDir.for_input(info.path)).create()
    stages: list[Stage] = []

    with track(
        "extract",
        work.frames_in,
        info.nb_frames,
        frame_format=options.frame_format,
        enabled=options.progress,
    ):
        extracted = extract(info, work, frame_format=options.frame_format, bins=bins)
    stages.append(
        Stage(
            "extract",
            extracted.frame_count,
            "normalised to CFR" if extracted.normalised else "",
            extracted.skipped,
        )
    )

    frames = _Frames(
        directory=extracted.frames_dir,
        count=extracted.frame_count,
        fps=extracted.fps,
        width=info.width,
        height=info.height,
    )
    for name in order:
        frames = _MODEL_STAGES[name](frames, work, options, stages, bins)

    return Generated(
        frames=frames,
        stages=stages,
        audio_source=(
            extracted.audio_source if extracted.has_audio and options.mux_audio else None
        ),
        work=work,
    )


def finish(
    generated: Generated,
    output: Path,
    options: Options | None = None,
    *,
    bins: Binaries | None = None,
) -> Result:
    """Encode what :func:`generate` made and clear its scratch space.

    Uses no GPU compute, only FFmpeg (and the video engine of the card, when a
    hardware encoder is picked), which is why it can run beside the next
    :func:`generate`.
    """
    options = options or Options()
    output = Path(output)
    frames, work = generated.frames, generated.work
    encoder = encode(
        frames.directory,
        output,
        fps=frames.fps,
        audio_source=generated.audio_source,
        frame_format=options.frame_format,
        prefer=options.prefer_encoder,
        allow_hardware=options.allow_hardware,
        bins=bins,
    )

    # Only once the output really exists is the scratch space safe to remove:
    # the frames are the only thing that makes the next run cheap, and hours of
    # GPU time are not worth a tidy directory.
    kept = options.keep_temp or not output.exists()
    work.cleanup(keep=kept)

    return Result(
        output=output,
        encoder=encoder.name,
        fps=frames.fps,
        frames=frames.count,
        stages=generated.stages,
        work=work.root,
        kept=kept,
    )
