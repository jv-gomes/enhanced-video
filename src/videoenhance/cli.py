"""Command-line entry point.

Run as ``videoenhance`` after an editable install, or as
``python -m videoenhance.cli``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__, config
from .doctor import report as doctor_report
from .encode import EncodeError, describe_chain, encoder_chain
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
        default=config.DEFAULT_SCALE,
        help=f"upscale factor (default: {config.DEFAULT_SCALE})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=config.DEFAULT_TARGET_FPS,
        help=f"target frame rate (default: {config.DEFAULT_TARGET_FPS})",
    )
    parser.add_argument(
        "--model",
        default=config.DEFAULT_UPSCALE_MODEL,
        choices=sorted(config.UPSCALE_MODELS),
        help=f"Real-ESRGAN model (default: {config.DEFAULT_UPSCALE_MODEL})",
    )
    parser.add_argument(
        "--order",
        default=config.DEFAULT_ORDER,
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
        default=config.DEFAULT_FRAME_FORMAT,
        choices=config.FRAME_FORMATS,
        help=(
            "intermediate frame format; jpg is smaller but lossy "
            f"(default: {config.DEFAULT_FRAME_FORMAT})"
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
    )


def describe_plan(
    info,
    args: argparse.Namespace,
    output: Path,
    work: WorkDir,
    estimate,
) -> str:
    """The run the given arguments would perform, as a readable block."""
    chain = encoder_chain(prefer=None, allow_hardware=True)
    return "\n".join(
        [
            f"input:   {info.width}x{info.height} @ {info.fps:g} fps, "
            f"{info.nb_frames} frames" + (" (VFR)" if info.is_vfr else ""),
            f"output:  {info.width * args.scale}x{info.height * args.scale} @ {args.fps:g} fps"
            f" -> {output}",
            f"model:   {args.model} at scale {args.scale}, {args.frame_format} frames",
            f"order:   {args.order}",
            f"encoder: {describe_chain(chain)}",
            f"work:    {work.root}",
            f"disk:    {estimate.render()}",
        ]
    )


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
    if args.output and args.output.resolve() == args.input.resolve():
        parser.error("--output must differ from the input file")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if args.doctor:
        return doctor_report()

    if args.input is None:
        parser.error("an input video is required (or use --doctor)")

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

    estimate = estimate_disk_usage(
        info.width,
        info.height,
        info.nb_frames,
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
    try:
        result = run_pipeline(
            args.input,
            output,
            options_from_args(args),
            info=info,
            work=work,
        )
    except KeyboardInterrupt:
        print(
            f"\ninterrupted; the frames in {work.root} are kept, "
            "so running the same command again resumes from here.",
            file=sys.stderr,
        )
        return EXIT_INTERRUPTED
    except (ExtractError, UpscaleError, InterpolateError, EncodeError, ToolError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            f"the frames in {work.root} are kept, so fixing the cause and "
            "running the same command again resumes from there.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    print(result.render())
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
