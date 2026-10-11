"""Measure whether reusing upscaled *blocks* would beat reusing whole frames.

The pipeline already generates each distinct frame once (``dedup.py``). The
idea measured here goes one step further: split every frame into a grid, hash
each block together with a margin of its neighbours, and send only the blocks
not seen before to the GPU. It pays off on limited animation — a static
background with only a mouth moving — and costs extra everywhere else, because
the margins are upscaled over and over. Whether it is worth building depends on
the footage, so this script answers the two questions before anyone writes it:

1. **Reuse** (FFmpeg only, no GPU): how much GPU work each grid would need
   compared with no dedup and with today's whole-frame dedup.
2. **Seams** (``--seam-test``, runs Real-ESRGAN once per setting): upscale one
   frame whole and as blocks with a margin, stitch the blocks back, and report
   how far apart the two are. That says how wide the margin has to be.

The block hash includes the margin because the upscaler's output at a pixel
depends on its neighbours: a block whose neighbour changed has to be redone,
or the stitched frame shows a seam.

Usage::

    python scripts/measure_tile_reuse.py input.mp4
    python scripts/measure_tile_reuse.py input.mp4 --grid 3 4 6 --margin 16 --duration 60
    python scripts/measure_tile_reuse.py input.mp4 --seam-test --margins 0 8 16 32
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# Runs from a checkout without installing the package as well.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoenhance import config  # noqa: E402
from videoenhance.blocks import (  # noqa: E402
    BlockError,
    Tile,
    block_hashes,
    crop_blocks,
    grid_tiles,
    overhead,
    stitch_blocks,
)
from videoenhance.config import Binaries, binaries  # noqa: E402
from videoenhance.probe import probe  # noqa: E402
from videoenhance.process import ToolError, run  # noqa: E402
from videoenhance.upscale import ModelError, resolve_model, upscale  # noqa: E402

#: Below this saving over whole-frame dedup, the extra moving parts are not
#: worth it. A rule of thumb, not a law: the report prints the numbers too.
WORTH_IT = 0.15


@dataclass(frozen=True)
class ReuseReport:
    """GPU work for one grid, in frame-equivalents of upscaled area."""

    grid: int
    margin: int
    frames: int
    distinct_frames: int
    #: Upscaled area with blocks, divided by the area of one frame.
    block_work: float
    #: What one frame costs when every block in it changed: 1.0 plus margins.
    overhead: float

    @property
    def frame_work(self) -> float:
        return float(self.distinct_frames)

    @property
    def saving_over_frames(self) -> float:
        """Fraction of today's GPU work the blocks would save; negative is worse."""
        return 1 - self.block_work / self.frame_work if self.frame_work else 0.0


def summarize(
    frame_hashes: list[str],
    tile_hashes: list[list[str]],
    tiles: list[Tile],
    grid: int,
    margin: int,
) -> ReuseReport:
    """Turn the hashes of one grid into the work each strategy would cost.

    ``tile_hashes[i]`` holds one hash per frame for ``tiles[i]``. Blocks are
    cached by content, not position, so the same padded pixels anywhere are
    generated once — which also covers whole repeated frames.
    """
    frame_area = sum(tile.area for tile in tiles)
    distinct: dict[str, int] = {}
    for tile, hashes in zip(tiles, tile_hashes, strict=True):
        for digest in hashes:
            distinct.setdefault(digest, tile.padded_area)
    return ReuseReport(
        grid=grid,
        margin=margin,
        frames=len(frame_hashes),
        distinct_frames=len(set(frame_hashes)),
        block_work=sum(distinct.values()) / frame_area,
        overhead=overhead(tiles),
    )


def psnr(a: Path, b: Path, bins: Binaries) -> float:
    """PSNR between two images in dB; ``inf`` when they are identical."""
    completed = run(
        [
            bins.ffmpeg,
            "-hide_banner",
            "-i",
            a,
            "-i",
            b,
            "-lavfi",
            "[0:v][1:v]psnr",
            "-f",
            "null",
            "-",
        ]
    )
    match = re.search(r"average:(inf|[\d.]+)", completed.stderr or "")
    if not match:
        raise SystemExit(f"could not read the PSNR of {a} against {b}")
    return float(match.group(1))


def diff_image(a: Path, b: Path, out: Path, bins: Binaries, gain: int = 8) -> None:
    """Save the difference of two images, brightened ``gain`` times to be seen."""
    graph = (
        "[0:v]format=gbrp[a];[1:v]format=gbrp[b];"
        f"[a][b]blend=all_mode=difference,lut=c0=val*{gain}:c1=val*{gain}:c2=val*{gain}"
    )
    run(
        [
            bins.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            a,
            "-i",
            b,
            "-filter_complex",
            graph,
            "-frames:v",
            "1",
            out,
        ]
    )


def _seam_verdict(value: float) -> str:
    if value == float("inf"):
        return "identical"
    if value >= 50:
        return "invisible"
    if value >= 40:
        return "look at the diff image"
    return "seams likely visible"


def seam_test(
    args: argparse.Namespace, width: int, height: int, duration: float, bins: Binaries
) -> None:
    """Upscale one frame whole and as blocks, and compare the two."""
    out = args.out_dir
    if out.exists():
        shutil.rmtree(out)
    end = min(args.start + args.duration, duration) if args.duration else duration
    at = args.at if args.at is not None else (args.start + end) / 2
    frame_dir = out / "frame"
    frame_dir.mkdir(parents=True)
    frame = frame_dir / "00000001.png"
    run(
        [
            bins.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{at:g}",
            "-i",
            args.input,
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            frame,
        ]
    )

    whole = out / "whole"
    upscale(frame_dir, whole, scale=args.scale, model=args.model, gpu=args.gpu, bins=bins)
    reference = whole / "00000001.png"

    print(f"\nseam test: frame at {at:g}s, {args.model} {args.scale}x, files in {out}\n")
    print(f"{'grid':>6}  {'margin':>6}  {'PSNR vs whole':>13}")
    for grid in args.grid:
        for margin in args.margins:
            tiles = grid_tiles(width, height, grid, margin)
            name = f"{grid}x{grid}-m{margin}"
            block_dirs = [out / name / "blocks" / f"{i + 1:03d}" for i in range(len(tiles))]
            crop_blocks(["-i", frame], tiles, block_dirs, bins)
            # The upscaler takes one directory, so the blocks are gathered into
            # it, numbered by position, and handed back to their own directories.
            gathered, upscaled = out / name / "in", out / name / "out"
            gathered.mkdir(parents=True)
            for i, directory in enumerate(block_dirs):
                shutil.copy(directory / "00000001.png", gathered / f"{i + 1:08d}.png")
            upscale(
                gathered,
                upscaled,
                scale=args.scale,
                model=args.model,
                gpu=args.gpu,
                bins=bins,
            )
            up_dirs = [out / name / "up" / f"{i + 1:03d}" for i in range(len(tiles))]
            for i, directory in enumerate(up_dirs):
                directory.mkdir(parents=True)
                shutil.copy(upscaled / f"{i + 1:08d}.png", directory / "00000001.png")
            stitched = out / f"{name}.png"
            stitch_blocks(up_dirs, tiles, args.scale, stitched, bins)
            diff_image(stitched, reference, out / f"{name}-diff.png", bins)
            value = psnr(stitched, reference, bins)
            shown = "inf" if value == float("inf") else f"{value:.2f} dB"
            print(f"{grid}x{grid:<4}  {margin:>6}  {shown:>13}  {_seam_verdict(value)}")
    print(
        "\nThe *-diff.png files show where the stitched frame differs from the whole one,"
        "\nbrightened 8x; a seam shows up as lines along the block edges."
    )


def print_reuse(reports: list[ReuseReport], width: int, height: int, window: str) -> None:
    first = reports[0]
    print(f"\n{first.frames} frames of {width}x{height}, {window}\n")
    print(f"{'strategy':<26} {'GPU work':>10} {'vs none':>8} {'vs today':>9}")
    rows = [("no dedup", float(first.frames), None)]
    rows.append(("whole frames (today)", first.frame_work, 0.0))
    for r in reports:
        rows.append(
            (f"{r.grid}x{r.grid} blocks, margin {r.margin}", r.block_work, -r.saving_over_frames)
        )
    for label, work, vs_today in rows:
        today = "" if vs_today is None else f"{vs_today:+.0%}" if vs_today else "-"
        print(f"{label:<26} {work:>8.1f} f {work / first.frames:>8.0%} {today:>9}")
    print("\nGPU work is upscaled area in frames; 'vs today' below zero means less work.")
    for r in reports:
        print(f"  {r.grid}x{r.grid}: a frame that changes everywhere costs {r.overhead - 1:+.0%}")

    best = max(reports, key=lambda r: r.saving_over_frames)
    if best.saving_over_frames >= WORTH_IT:
        print(
            f"\nWorth pursuing: {best.grid}x{best.grid} saves {best.saving_over_frames:.0%} "
            "of today's GPU work.\nRun again with --seam-test to see how wide the margin must be."
        )
    else:
        print(
            f"\nNot worth it on this footage: the best grid saves "
            f"{max(best.saving_over_frames, 0):.0%} over whole-frame dedup "
            f"(rule of thumb: {WORTH_IT:.0%})."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure whether reusing upscaled blocks would beat whole-frame dedup.",
    )
    parser.add_argument("input", type=Path, help="video to measure")
    parser.add_argument(
        "--grid",
        type=int,
        nargs="+",
        default=[3],
        help="blocks per side; several values compare them (default: 3)",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=16,
        help="pixels of neighbours hashed and upscaled with each block (default: 16)",
    )
    parser.add_argument("--start", type=float, default=0.0, help="seconds to skip (default: 0)")
    parser.add_argument(
        "--duration", type=float, default=30.0, help="seconds to measure, 0 for all (default: 30)"
    )
    seam = parser.add_argument_group("seam test (runs Real-ESRGAN)")
    seam.add_argument(
        "--seam-test",
        action="store_true",
        help="also upscale one frame whole and as blocks, and compare",
    )
    seam.add_argument(
        "--margins",
        type=int,
        nargs="+",
        default=[0, 8, 16, 32],
        help="margins to try (default: 0 8 16 32)",
    )
    seam.add_argument("--at", type=float, help="time of the frame (default: middle of the window)")
    seam.add_argument("--scale", type=int, default=config.DEFAULT_SCALE)
    seam.add_argument("--model", default=config.DEFAULT_UPSCALE_MODEL)
    seam.add_argument("--gpu", type=int, default=config.DEFAULT_GPU_ID)
    seam.add_argument(
        "--out-dir",
        type=Path,
        default=config.WORK_DIR / "tile-test",
        help="where the seam test writes its images (default: work/tile-test)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    if not args.input.is_file():
        parser.error(f"input video not found: {args.input}")
    if args.seam_test:
        try:
            resolve_model(args.model, args.scale)
        except ModelError as exc:
            parser.error(str(exc))

    bins = binaries()
    info = probe(args.input, bins)
    try:
        tile_sets = [(g, grid_tiles(info.width, info.height, g, args.margin)) for g in args.grid]
        for grid in args.grid:
            for margin in args.margins if args.seam_test else []:
                grid_tiles(info.width, info.height, grid, margin)
    except ValueError as exc:
        parser.error(str(exc))

    end = min(args.start + args.duration, info.duration) if args.duration else info.duration
    window = f"from {args.start:g}s to {end:g}s"
    try:
        reports = []
        for grid, tiles in tile_sets:
            inputs: list[object] = []
            if args.start:
                inputs += ["-ss", f"{args.start:g}"]
            if args.duration:
                inputs += ["-t", f"{args.duration:g}"]
            frame_hashes, tile_hashes = block_hashes([*inputs, "-i", args.input], tiles, bins)
            reports.append(summarize(frame_hashes, tile_hashes, tiles, grid, args.margin))
        print_reuse(reports, info.width, info.height, window)
        if args.seam_test:
            seam_test(args, info.width, info.height, info.duration, bins)
    except (BlockError, ToolError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
