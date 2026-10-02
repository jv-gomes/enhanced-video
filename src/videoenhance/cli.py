"""Command-line entry point.

Run as ``videoenhance`` after an editable install, or as
``python -m videoenhance.cli``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__, config, state
from .chunk import ChunkError, chunk_frame_count, run_chunked
from .chunk import describe_plan as describe_chunks
from .doctor import report as doctor_report
from .encode import EncodeError, available_encoders, describe_chain, encoder_chain
from .extract import ExtractError
from .interpolate import InterpolateError
from .pipeline import ORDERS, Options, WorkDir, estimate_disk_usage, run_pipeline
from .pipeline import human_bytes as pipeline_human
from .probe import ProbeError, probe
from .process import ToolError
from .progress import wanted as progress_wanted
from .upscale import ModelError, UpscaleError, resolve_model

logger = logging.getLogger("videoenhance")

EXIT_OK = 0
EXIT_ERROR = 1
#: Conventional exit code for SIGINT, so a resumable interruption is
#: distinguishable from a real failure in a shell loop.
EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="videoenhance",
        description=(
            "Upscale and interpolate video on AMD GPUs using the NCNN + Vulkan "
            "builds of Real-ESRGAN and RIFE."
        ),
        epilog="Run with --doctor to check that FFmpeg, Vulkan and the NCNN binaries are ready.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="source video file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="destination file (default: <input stem>.enhanced.mp4 beside the input)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=None,
        help=f"upscale factor (default: {config.DEFAULT_SCALE})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=f"target frame rate (default: {config.DEFAULT_TARGET_FPS})",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=sorted(config.UPSCALE_MODELS),
        help=f"Real-ESRGAN model (default: {config.DEFAULT_UPSCALE_MODEL})",
    )
    parser.add_argument(
        "--order",
        default=None,
        choices=ORDERS,
        help=(
            "stage order; upscale-first is cheaper because Real-ESRGAN then "
            "runs before the frame count is multiplied, while interpolate-first "
            "lets RIFE see the original pixels "
            f"(default: {config.DEFAULT_ORDER})"
        ),
    )
    parser.add_argument(
        "--frame-format",
        default=None,
        choices=config.FRAME_FORMATS,
        help=(
            "intermediate frame format; jpg is smaller but lossy "
            f"(default: {config.DEFAULT_FRAME_FORMAT})"
        ),
    )
    parser.add_argument(
        "--chunk",
        type=float,
        metavar="SECONDS",
        default=None,
        help=(
            "process the video in chunks of this many seconds, deleting each "
            "chunk's frames as soon as its part is encoded, then join the parts; "
            "0 processes the whole file in one pass, which needs disk space for "
            f"all of its frames at once (default: {config.DEFAULT_CHUNK_SECONDS:g})"
        ),
    )
    parser.add_argument(
        "--encoder",
        metavar="NAME",
        help=(
            "force a specific FFmpeg encoder (e.g. hevc_vaapi, libx265) instead "
            "of picking the best available one"
        ),
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=config.DEFAULT_TILE_SIZE,
        help="Real-ESRGAN tile size, 0 for automatic; lower it on out-of-VRAM errors",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=config.DEFAULT_GPU_ID,
        help=f"Vulkan device id (default: {config.DEFAULT_GPU_ID})",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=config.WORK_DIR,
        help=f"directory for intermediate frames (default: {config.WORK_DIR})",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="do not draw per-stage progress bars (the default outside a terminal)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="keep the work directory after a successful run",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="print what ffprobe reports about the input and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned run, its encoder and its disk cost, then exit",
    )
    parser.add_argument(
        "--jobs",
        action="store_true",
        help="list the unfinished jobs that can be resumed, and exit",
    )
    parser.add_argument(
        "--resume",
        metavar="JOB",
        help=(
            "continue a saved job by id (any unambiguous prefix works), reusing "
            "the options it was started with; see --jobs"
        ),
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "throw away the saved work for this input and start over, instead of "
            "resuming it"
        ),
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="check the environment (FFmpeg, encoders, Vulkan GPU, NCNN binaries) and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase logging (-v for info, -vv for debug)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


#: The arguments that carry no argparse default, so that "the user asked for
#: this" can be distinguished from "nobody said". They are also exactly the
#: options a resume may not change; :data:`videoenhance.state.PIXEL_OPTIONS`
#: names them as the pipeline knows them.
RESOLVED_ARGS = {
    "scale": config.DEFAULT_SCALE,
    "fps": config.DEFAULT_TARGET_FPS,
    "model": config.DEFAULT_UPSCALE_MODEL,
    "order": config.DEFAULT_ORDER,
    "frame_format": config.DEFAULT_FRAME_FORMAT,
    "chunk": config.DEFAULT_CHUNK_SECONDS,
}

#: Maps those argument names to the option names the pipeline and the job
#: record use.
ARG_TO_OPTION = {
    "scale": "scale",
    "fps": "target_fps",
    "model": "model",
    "order": "order",
    "frame_format": "frame_format",
    "chunk": "chunk_seconds",
}


def resolve_args(args: argparse.Namespace) -> set[str]:
    """Fill in the defaults and report which options the user actually gave.

    Everything downstream wants resolved values, but ``--resume`` has to know
    the difference between an option that was typed and one that was left out:
    typing ``--scale 4`` on a job recorded at 2 is a conflict, while not
    mentioning ``--scale`` is not.
    """
    given = {name for name in RESOLVED_ARGS if getattr(args, name) is not None}
    for name, default in RESOLVED_ARGS.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    return {ARG_TO_OPTION[name] for name in given}


def apply_job(args: argparse.Namespace, job: state.Job) -> None:
    """Take the input, output and options of ``job`` as the arguments to use."""
    args.input = job.input
    args.output = job.output
    reverse = {option: arg for arg, option in ARG_TO_OPTION.items()}
    for option, value in job.options.items():
        if option in reverse:
            setattr(args, reverse[option], value)


def options_from_args(args: argparse.Namespace) -> Options:
    """Translate the parsed arguments into the pipeline's own options."""
    return Options(
        scale=args.scale,
        target_fps=args.fps,
        model=args.model,
        frame_format=args.frame_format,
        order=args.order,
        progress=args.progress and progress_wanted(),
        gpu=args.gpu,
        tile=args.tile,
        keep_temp=args.keep_temp,
        chunk_seconds=args.chunk,
        prefer_encoder=args.encoder,
    )


def describe_plan(
    info,
    args: argparse.Namespace,
    output: Path,
    work: WorkDir,
    estimate,
) -> str:
    """The run the given arguments would perform, as a readable block."""
    chain = encoder_chain(prefer=args.encoder, allow_hardware=True)
    lines = [
        f"input:   {info.width}x{info.height} @ {info.fps:g} fps, "
        f"{info.nb_frames} frames" + (" (VFR)" if info.is_vfr else ""),
        f"output:  {info.width * args.scale}x{info.height * args.scale} @ {args.fps:g} fps"
        f" -> {output}",
        f"model:   {args.model} at scale {args.scale}, {args.frame_format} frames",
        f"order:   {args.order}",
        f"encoder: {describe_chain(chain)}",
    ]
    if chunking(info, args.chunk):
        lines.append(f"chunks:  {describe_chunks(info, args.chunk)}")
    lines += [
        f"work:    {work.root}",
        f"disk:    {estimate.render()}",
    ]
    return "\n".join(lines)


def chunking(info, seconds: float) -> bool:
    """Whether this run should be cut into chunks.

    A file no longer than one chunk would gain nothing from the cut, the
    concatenation or the second mux, so it takes the single-pass path. A file
    whose duration the container does not report is chunked anyway: guessing
    "short" there is what would blow up the disk.
    """
    if seconds <= 0:
        return False
    return not info.duration or info.duration > seconds


def warn_about_space(estimate, frame_format: str) -> bool:
    """Warn when the frames are unlikely to fit. Returns True if a warning was printed."""
    if estimate.fits:
        return False
    print(
        f"warning: the intermediate frames need about "
        f"{pipeline_human(estimate.total)} but only "
        f"{pipeline_human(estimate.free)} is free.",
        file=sys.stderr,
    )
    if frame_format != "jpg":
        print(
            "         --frame-format jpg cuts that to roughly a fifth, at a small "
            "quality cost.",
            file=sys.stderr,
        )
    return True


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    # basicConfig is a no-op once a handler exists, so set the level directly
    # as well; otherwise -v is silently ignored when logging was already set up.
    logging.getLogger().setLevel(level)


def default_output(input_path: Path) -> Path:
    """Destination used when ``--output`` is omitted, never the input itself."""
    return input_path.with_suffix("").with_name(f"{input_path.stem}.enhanced.mp4")


def validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Reject argument combinations the pipeline cannot honour."""
    if args.scale < 1:
        parser.error("--scale must be 1 or greater")
    try:
        resolve_model(args.model, args.scale)
    except ModelError as exc:
        parser.error(str(exc))
    if args.fps <= 0:
        parser.error("--fps must be greater than 0")
    if args.tile < 0:
        parser.error("--tile must be 0 (automatic) or greater")
    if args.chunk < 0:
        parser.error("--chunk must be 0 (no chunking) or greater")
    if args.encoder and args.encoder not in available_encoders():
        parser.error(
            f"--encoder {args.encoder} is not available in this FFmpeg build; "
            "run --doctor to see what is"
        )
    if args.output and args.output.resolve() == args.input.resolve():
        parser.error("--output must differ from the input file")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if args.doctor:
        return doctor_report()

    if args.jobs:
        print(state.describe(state.list_jobs(args.work_dir)))
        return EXIT_OK

    given = resolve_args(args)

    if args.resume and args.restart:
        parser.error("--resume and --restart ask for opposite things; pick one")

    resumed: state.Job | None = None
    if args.resume:
        try:
            resumed = state.find(args.resume, args.work_dir)
        except state.StateError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERROR
        if args.input is not None:
            parser.error("--resume takes its input from the saved job; drop the input argument")
        # Options typed alongside --resume are not a request to change the job,
        # they are a conflict: the work already on disk was made the old way.
        asked = {ARG_TO_OPTION[arg]: getattr(args, arg) for arg in RESOLVED_ARGS}
        conflicts = state.compare(resumed, {k: v for k, v in asked.items() if k in given})
        if conflicts:
            print(f"error: {state.render_mismatch(resumed, conflicts)}", file=sys.stderr)
            return EXIT_ERROR
        apply_job(args, resumed)
        print(f"resuming {resumed.id}: {state.progress(resumed)} done")

    if args.input is None:
        parser.error(
            "an input video is required (or use --doctor, --jobs or --resume)"
        )

    try:
        info = probe(args.input)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except (ProbeError, ToolError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.probe_only:
        print(info.describe())
        return EXIT_OK

    validate(args, parser)
    output = args.output or default_output(args.input)
    work = WorkDir.for_input(args.input, base=args.work_dir)
    options = options_from_args(args)

    job = resumed or state.load(work.root)
    if job is not None and args.restart:
        print(f"discarding the saved job {job.id} ({state.progress(job)})")
        work.cleanup(keep=False)
        job = None
    elif job is not None:
        # The guards inside each stage are frame counts, and a frame count does
        # not change when --scale or --model does. Without this check a job
        # started at 2x and resumed at 4x would skip the upscale and quietly
        # produce a 2x video.
        changed = state.compare(job, state.options_of(options))
        if changed:
            print(f"error: {state.render_mismatch(job, changed)}", file=sys.stderr)
            return EXIT_ERROR
        if state.input_changed(job):
            print(
                f"error: {job.input} has changed since this job was started, so "
                f"the work in {work.root} does not belong to it.\n"
                "Use --restart to throw that away and start over.",
                file=sys.stderr,
            )
            return EXIT_ERROR

    chunked = chunking(info, args.chunk)
    # Chunked mode only ever holds one chunk's frames on disk, so that is what
    # the estimate has to be built from; the whole file's count would describe
    # a run that never happens.
    estimate = estimate_disk_usage(
        info.width,
        info.height,
        chunk_frame_count(info, args.chunk) if chunked else info.nb_frames,
        scale=args.scale,
        target_fps=args.fps,
        source_fps=info.fps,
        frame_format=args.frame_format,
        order=args.order,
        free=work.free_bytes(),
    )

    print(describe_plan(info, args, output, work, estimate))
    warn_about_space(estimate, args.frame_format)

    if args.dry_run:
        return EXIT_OK

    print()
    # Recorded before any work starts, so an interrupt in the first minute still
    # leaves something --jobs can show and --resume can pick up.
    job = job or state.save(
        state.for_input(work, args.input, output, state.options_of(options))
    )

    try:
        if chunked:
            result = run_chunked(args.input, output, options, info=info, work=work, job=job)
        else:
            result = run_pipeline(args.input, output, options, info=info, work=work)
    except KeyboardInterrupt:
        print(
            f"\ninterrupted; the work in {work.root} is kept. "
            f"Continue it later with --resume {job.id}.",
            file=sys.stderr,
        )
        return EXIT_INTERRUPTED
    except (
        ChunkError,
        EncodeError,
        ExtractError,
        InterpolateError,
        ToolError,
        UpscaleError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            f"the work in {work.root} is kept, so fixing the cause and running "
            f"--resume {job.id} continues from there.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    print(result.render())
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
