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
from .probe import ProbeError, probe
from .process import ToolError

logger = logging.getLogger("videoenhance")

ORDERS = ("upscale-first", "interpolate-first")

EXIT_OK = 0
EXIT_ERROR = 1


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
        default=ORDERS[0],
        choices=ORDERS,
        help=(
            "stage order; upscale-first is cheaper because it runs before the "
            f"frame count is multiplied (default: {ORDERS[0]})"
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


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def default_output(input_path: Path) -> Path:
    """Destination used when ``--output`` is omitted, never the input itself."""
    return input_path.with_suffix("").with_name(f"{input_path.stem}.enhanced.mp4")


def validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Reject argument combinations the pipeline cannot honour."""
    if args.scale < 1:
        parser.error("--scale must be 1 or greater")
    if not config.scale_supported(args.model, args.scale):
        supported = ", ".join(str(s) for s in config.UPSCALE_MODELS[args.model])
        parser.error(f"{args.model} only supports scale {supported}, not {args.scale}")
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

    print(
        "The processing pipeline is not implemented yet.\n"
        f"Planned run: {info.width}x{info.height} @ {info.fps:g} fps "
        f"-> {info.width * args.scale}x{info.height * args.scale} @ {args.fps:g} fps\n"
        f"  model:  {args.model} (scale {args.scale})\n"
        f"  order:  {args.order}\n"
        f"  output: {output}\n"
        "See roadmap.md, milestones M2-M5.",
        file=sys.stderr,
    )
    return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
