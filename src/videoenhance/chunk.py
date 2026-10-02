"""Processing a long video in chunks, so the frames never pile up.

A single-pass run keeps every intermediate frame of the whole video on disk at
once: :func:`videoenhance.workdir.estimate_disk_usage` counts all three stage
directories because a resumable stage cannot delete the input of a stage it may
have to run again. That is correct, and for a feature film it is hundreds of
gigabytes.

So this module turns one long run into many short ones. The source is cut into
chunks of a few seconds, :func:`videoenhance.pipeline.run_pipeline` processes
one chunk at a time into an encoded part, and the frames of that chunk are gone
before the next one starts. The parts are concatenated at the end. Peak disk
becomes "one chunk of frames, plus the parts already encoded", which does not
grow with the length of the video.

Two details carry most of the correctness:

* **The audio is muxed once, at the end**, from the original file. Giving every
  part its own audio would put a ``-shortest`` at every chunk boundary, and
  that sub-frame trim accumulates into the drift CLAUDE.md warns about.
* **The encoder is pinned** to whatever the first part used. The parts are
  concatenated by stream copy, which requires them to share a codec, and the
  encoder chain is allowed to fall back at run time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

from . import config
from .config import Binaries, binaries
from .encode import concat
from .extract import to_cfr
from .pipeline import Options, Result, Stage, run_pipeline
from .probe import ProbeError, VideoInfo, probe
from .process import ToolError, run
from .state import Job, for_input, options_of, record_cut, record_encoder, save
from .workdir import WorkDir, human_bytes

logger = logging.getLogger(__name__)

#: Chunks and parts are Matroska: it takes any codec the source might use, so
#: cutting with ``-c copy`` never has to re-container, and it holds the output
#: of any encoder in the chain.
CHUNK_SUFFIX = ".mkv"
PART_SUFFIX = ".mkv"
#: A chunk longer than this multiple of the requested length means the cut
#: could not land where it was asked to, because ``-c copy`` can only cut on
#: keyframes. Such a chunk is re-cut with a lossless re-encode.
OVERSIZE_FACTOR = 2.0


class ChunkError(RuntimeError):
    """The source could not be cut, or the parts cannot be joined."""


@dataclass(frozen=True)
class ChunkPlan:
    """The cut of one source file into chunks."""

    #: Chunk files, in playback order.
    chunks: list[Path]
    #: Seconds requested per chunk.
    seconds: float
    #: The file the audio should come from: the source, or its CFR copy.
    audio_source: Path
    has_audio: bool
    #: True when the chunks were already on disk and nothing was re-cut.
    skipped: bool = False
    #: Chunks that had to be re-cut with a re-encode because the source's
    #: keyframes were too far apart.
    recut: int = 0
    #: The job record as it stands after the cut was written to it. ``Job`` is
    #: frozen, so a caller holding the one it passed in would be holding a
    #: stale copy and its next save would wipe the cut back out.
    job: Job | None = None

    def __len__(self) -> int:
        return len(self.chunks)


def _segment_cmd(
    source: Path,
    pattern: Path,
    seconds: float,
    bins: Binaries,
    *,
    lossless: bool = False,
) -> list[object]:
    """FFmpeg call that cuts ``source`` into ``pattern``-named chunks.

    The chunks are video only: the audio is muxed back in once, at the end, so
    carrying it through every chunk would only be a chance to lose sync.
    """
    cmd: list[object] = [
        bins.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source,
        "-map",
        "0:v:0",
        "-an",
    ]
    if lossless:
        # Forcing a keyframe at every boundary is what makes the cut land
        # exactly where asked; -qp 0 keeps it mathematically lossless, so the
        # pixels the models see are still the source's.
        cmd += [
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-preset",
            "veryfast",
            "-force_key_frames",
            f"expr:gte(t,n_forced*{seconds:g})",
        ]
    else:
        cmd += ["-c:v", "copy"]
    cmd += [
        "-f",
        "segment",
        "-segment_time",
        f"{seconds:g}",
        # Each chunk starts its own timeline at zero, so the frame extraction
        # that follows does not inherit the offset of the original.
        "-reset_timestamps",
        "1",
        pattern,
    ]
    return cmd


def _chunk_duration(path: Path, bins: Binaries | None = None) -> float:
    """Duration of a chunk, or 0 when ffprobe cannot say."""
    try:
        return probe(path, bins).duration
    except (ProbeError, ToolError, FileNotFoundError):
        # A chunk whose duration cannot be read is simply not treated as
        # oversized; the pipeline will fail on it later with a better message.
        logger.debug("could not probe %s for its duration", path)
        return 0.0


def _recut(chunk: Path, seconds: float, bins: Binaries) -> list[Path]:
    """Re-cut one oversized chunk, losslessly, and replace it.

    The pieces are named after the chunk they came from with a numeric suffix,
    which keeps them sorting between their neighbours: ``chunk_0003-00`` falls
    after ``chunk_0002`` and before ``chunk_0004``.
    """
    logger.info("re-cutting %s with a lossless re-encode: its keyframes are too sparse", chunk.name)
    pattern = chunk.with_name(f"{chunk.stem}-%02d{CHUNK_SUFFIX}")
    run(_segment_cmd(chunk, pattern, seconds, bins, lossless=True), capture=False)
    pieces = sorted(chunk.parent.glob(f"{chunk.stem}-*{CHUNK_SUFFIX}"))
    if not pieces:
        raise ChunkError(f"re-cutting {chunk.name} produced nothing")
    chunk.unlink(missing_ok=True)
    return pieces


def _recorded_cut(job: Job | None, directory: Path, seconds: float) -> list[Path] | None:
    """The cut this job already made, if it was made for the same length.

    The cut is part of the job record rather than a file of its own, so that a
    resumed run has one place to look and one place to disagree with.
    """
    if job is None or not job.chunks:
        return None
    if float(job.seconds or 0) != float(seconds):
        logger.info("the recorded cut was made for a different --chunk, cutting again")
        return None
    return [directory / name for name in job.chunks]


def split(
    info: VideoInfo,
    work: WorkDir,
    seconds: float = config.DEFAULT_CHUNK_SECONDS,
    *,
    job: Job | None = None,
    bins: Binaries | None = None,
) -> ChunkPlan:
    """Cut ``info`` into chunks of roughly ``seconds`` each.

    A VFR source is normalised to CFR once, before the cut, so that every chunk
    shares one timeline; normalising per chunk would let each one settle on its
    own average frame rate.

    The resulting cut is recorded on ``job``, when one is given, and a job that
    already carries a cut of the same length is reused instead of re-cutting.

    The cut itself is a stream copy, which costs nothing and loses nothing but
    can only land on a keyframe. A chunk that comes out more than
    :data:`OVERSIZE_FACTOR` times longer than asked is re-cut on its own with a
    lossless re-encode, which covers sources whose keyframes are minutes apart.

    Raises:
        ChunkError: FFmpeg produced no chunks.
        ToolError: FFmpeg failed.
    """
    bins = bins or binaries()
    if seconds <= 0:
        raise ValueError(f"chunk length must be above 0 seconds, got {seconds}")

    work.create(stages=False)
    work.chunks.mkdir(parents=True, exist_ok=True)

    source = info.path
    audio_source = info.path
    if info.is_vfr:
        # The CFR copy is both what gets cut and where the audio comes from,
        # exactly as in the single-pass path.
        source = to_cfr(info, work.cfr_video, bins=bins)
        audio_source = source

    def plan(
        chunks: list[Path],
        *,
        skipped: bool = False,
        recut: int = 0,
        job: Job | None = job,
    ) -> ChunkPlan:
        return ChunkPlan(
            chunks=chunks,
            seconds=seconds,
            audio_source=audio_source,
            has_audio=info.has_audio,
            skipped=skipped,
            recut=recut,
            job=job,
        )

    existing = _recorded_cut(job, work.chunks, seconds)
    if existing and all(chunk.exists() for chunk in existing):
        logger.info("reusing %d chunks already in %s", len(existing), work.chunks)
        return plan(existing, skipped=True)

    for stale in work.chunks.glob(f"*{CHUNK_SUFFIX}"):
        stale.unlink(missing_ok=True)

    logger.info("cutting %s into %g-second chunks", source.name, seconds)
    pattern = work.chunks / f"chunk_%04d{CHUNK_SUFFIX}"
    run(_segment_cmd(source, pattern, seconds, bins), capture=False)

    chunks = sorted(work.chunks.glob(f"chunk_*{CHUNK_SUFFIX}"))
    if not chunks:
        raise ChunkError(f"cutting {source} into chunks produced no files")

    limit = seconds * OVERSIZE_FACTOR
    final: list[Path] = []
    recut = 0
    for chunk in chunks:
        if _chunk_duration(chunk, bins) > limit:
            final += _recut(chunk, seconds, bins)
            recut += 1
        else:
            final.append(chunk)

    if job is not None:
        job = record_cut(job, final, seconds)
    logger.info("cut into %d chunks", len(final))
    return plan(final, recut=recut, job=job)


def _part_path(parts_dir: Path, index: int) -> Path:
    return parts_dir / f"part_{index:04d}{PART_SUFFIX}"


def _joined_frame_count(output: Path, bins: Binaries | None = None) -> int:
    """Frames in the finished file, or 0 when ffprobe cannot say."""
    try:
        return probe(output, bins).nb_frames
    except (ProbeError, ToolError, FileNotFoundError):
        logger.debug("could not count the frames in %s", output)
        return 0


def run_chunked(
    input_path: Path,
    output: Path,
    options: Options | None = None,
    *,
    info: VideoInfo | None = None,
    work: WorkDir | None = None,
    job: Job | None = None,
    bins: Binaries | None = None,
) -> Result:
    """Process ``input_path`` one chunk at a time and join the results.

    Each chunk goes through the whole pipeline into its own part, and the
    chunk's frames and source file are deleted as soon as that part exists.
    An interrupted run resumes at the first missing part, because a part file
    only appears once it is complete.

    ``job`` is the record that makes that resume safe days later; one is
    created if the caller did not bring its own. The cut and the encoder are
    written to it as they are decided, and the whole thing goes away with the
    work directory once the output exists.

    Raises:
        ChunkError: the cut failed, or the parts cannot be joined by copy.
        ValueError: the output would overwrite the input.
    """
    options = options or Options()
    info = info or probe(input_path)
    output = Path(output)
    if output.resolve() == Path(info.path).resolve():
        raise ValueError(f"refusing to overwrite the input video: {output}")

    work = (work or WorkDir.for_input(info.path)).create(stages=False)
    if job is None:
        job = save(for_input(work, info.path, output, options_of(options)))

    seconds = options.chunk_seconds or config.DEFAULT_CHUNK_SECONDS
    plan = split(info, work, seconds, job=job, bins=bins)
    job = plan.job or job
    work.parts.mkdir(parents=True, exist_ok=True)

    # The parts are joined by stream copy, so they all have to come out of the
    # same encoder. The first part decides, the rest are asked for it by name,
    # and the record carries that choice across a resume so a later run cannot
    # pick a different one.
    pinned: str | None = options.prefer_encoder or job.encoder
    encoders: dict[str, int] = {}
    parts: list[Path] = []
    total_frames = 0
    reused = 0
    fps = Fraction(options.target_fps).limit_denominator(100000)

    per_chunk = replace(options, keep_temp=False, mux_audio=False)
    for index, chunk in enumerate(plan.chunks):
        part = _part_path(work.parts, index)
        parts.append(part)
        label = f"part {index + 1}/{len(plan)}"

        if part.exists():
            logger.info("%s is already encoded, skipping %s", label, chunk.name)
            reused += 1
            if not options.keep_temp:
                chunk.unlink(missing_ok=True)
            continue

        print(f"{label}: {chunk.name}")
        result = run_pipeline(
            chunk,
            part,
            replace(per_chunk, prefer_encoder=pinned),
            work=work.claim_current(chunk),
            bins=bins,
        )
        print(result.render())

        pinned = pinned or result.encoder
        job = record_encoder(job, result.encoder)
        encoders[result.encoder] = encoders.get(result.encoder, 0) + 1
        total_frames += result.frames
        fps = result.fps
        if not options.keep_temp:
            # The chunk's frames are already gone: run_pipeline removed them
            # once the part existed. The chunk file itself has no reader left.
            chunk.unlink(missing_ok=True)

    if len(encoders) > 1:
        names = ", ".join(sorted(encoders))
        raise ChunkError(
            "the parts were encoded with different encoders "
            f"({names}), so they cannot be joined by stream copy.\n"
            "Pick one with --encoder and run the same command again; the parts "
            f"already in {work.parts} that used it will be reused."
        )

    encoder = next(iter(encoders), pinned or "copy")
    concat(
        parts,
        output,
        audio_source=plan.audio_source if plan.has_audio else None,
        bins=bins,
    )

    # Count the frames in the joined file rather than adding up the parts this
    # run happened to encode: on a resumed run most of them were encoded days
    # ago, and a summary saying "1800 frames" about a 5400-frame video is a lie
    # that would make a resume look like it lost something.
    total_frames = _joined_frame_count(output, bins) or total_frames

    stages = [
        Stage(
            "chunks",
            len(plan),
            f"{plan.seconds:g}s each"
            + (f", {plan.recut} re-cut" if plan.recut else "")
            + (f", {reused} already encoded" if reused else ""),
            plan.skipped,
            unit="chunks",
        )
    ]

    kept = options.keep_temp or not output.exists()
    work.cleanup(keep=kept)

    return Result(
        output=output,
        encoder=encoder,
        fps=fps,
        frames=total_frames,
        stages=stages,
        work=work.root,
        kept=kept,
    )


def chunk_frame_count(info: VideoInfo, seconds: float) -> int:
    """Frames in one chunk, for the disk estimate.

    Chunked mode only ever holds one chunk's frames, so this — not the whole
    file's frame count — is what the estimate should be built from.
    """
    if seconds <= 0 or not info.fps:
        return info.nb_frames
    per_chunk = int(seconds * info.fps)
    return min(per_chunk, info.nb_frames) if info.nb_frames else per_chunk


def describe_plan(info: VideoInfo, seconds: float) -> str:
    """One line about the cut, for the CLI's plan block."""
    if not info.duration:
        return f"{seconds:g}s each (count unknown: the source has no duration)"
    count = max(1, int(-(-info.duration // seconds)))
    return f"about {count} x {seconds:g}s, one at a time"


def peak_disk_note(estimate_total: int) -> str:
    """How the chunked estimate should be read."""
    return f"peak is one chunk, about {human_bytes(estimate_total)}"


__all__ = [
    "CHUNK_SUFFIX",
    "ChunkError",
    "ChunkPlan",
    "chunk_frame_count",
    "describe_plan",
    "peak_disk_note",
    "run_chunked",
    "split",
]
