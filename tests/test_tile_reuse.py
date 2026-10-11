"""Tests for scripts/measure_tile_reuse.py, the block-reuse measurement.

The script decides whether a feature is worth building, so a wrong number is
worse than none: the geometry must cover the frame exactly, the accounting must
agree with whole-frame dedup when there is nothing more to find, and cutting a
frame into blocks and stitching it back must return the same pixels.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from videoenhance.config import binaries
from videoenhance.process import run

from .conftest import HELD_DRAWINGS, HELD_FOR, requires_ffmpeg

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_tile_reuse.py"
_spec = importlib.util.spec_from_file_location("measure_tile_reuse", SCRIPT)
tile_reuse = importlib.util.module_from_spec(_spec)
sys.modules["measure_tile_reuse"] = tile_reuse
_spec.loader.exec_module(tile_reuse)


@pytest.mark.parametrize("width,height,grid", [(320, 240, 3), (1920, 1080, 4), (641, 359, 6)])
def test_blocks_cover_the_frame_exactly_once(width: int, height: int, grid: int):
    tiles = tile_reuse.grid_tiles(width, height, grid, margin=16)
    assert len(tiles) == grid * grid
    assert sum(tile.area for tile in tiles) == width * height
    covered = {(x, y) for t in tiles for x in (t.x0, t.x1 - 1) for y in (t.y0, t.y1 - 1)}
    assert (0, 0) in covered and (width - 1, height - 1) in covered


def test_margin_is_clamped_to_the_frame():
    tiles = tile_reuse.grid_tiles(300, 300, 3, margin=16)
    corner, middle = tiles[0], tiles[4]
    assert (corner.px0, corner.py0, corner.px1, corner.py1) == (0, 0, 116, 116)
    assert (middle.px0, middle.py0, middle.px1, middle.py1) == (84, 84, 216, 216)


def test_grid_and_margin_are_validated():
    with pytest.raises(ValueError):
        tile_reuse.grid_tiles(4, 4, 5, margin=0)
    with pytest.raises(ValueError):
        tile_reuse.grid_tiles(100, 100, 2, margin=-1)


def test_summary_counts_blocks_by_content():
    tiles = tile_reuse.grid_tiles(100, 100, 2, margin=0)
    frames = ["a", "b", "c", "c"]
    # Block 0 changes every frame but the last; the other three never change.
    blocks = [["0a", "0b", "0c", "0c"], ["1"] * 4, ["2"] * 4, ["3"] * 4]
    report = tile_reuse.summarize(frames, blocks, tiles, grid=2, margin=0)
    assert report.distinct_frames == 3
    assert report.block_work == pytest.approx(3 * 0.25 + 3 * 0.25)
    assert report.saving_over_frames == pytest.approx(0.5)
    assert report.overhead == 1.0


def test_margins_make_a_fully_changing_video_cost_more():
    tiles = tile_reuse.grid_tiles(90, 90, 3, margin=10)
    frames = [str(i) for i in range(5)]
    blocks = [[f"{t}-{i}" for i in range(5)] for t in range(9)]
    report = tile_reuse.summarize(frames, blocks, tiles, grid=3, margin=10)
    assert report.overhead > 1
    assert report.saving_over_frames < 0


@requires_ffmpeg
def test_held_video_finds_at_least_the_whole_frame_repeats(held_video: Path):
    info = tile_reuse.probe(held_video)
    tiles = tile_reuse.grid_tiles(info.width, info.height, 3, margin=16)
    frames, blocks = tile_reuse.hash_blocks(held_video, tiles, start=0, duration=0, bins=binaries())
    report = tile_reuse.summarize(frames, blocks, tiles, grid=3, margin=16)
    assert report.frames == HELD_DRAWINGS * HELD_FOR
    assert report.distinct_frames == HELD_DRAWINGS
    assert all(len(hashes) == report.frames for hashes in blocks)


@requires_ffmpeg
def test_cutting_and_stitching_returns_the_same_pixels(sample_video: Path, tmp_path: Path):
    bins = binaries()
    frame = tmp_path / "frame.png"
    run(
        [
            bins.ffmpeg,
            "-loglevel",
            "error",
            "-i",
            sample_video,
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            frame,
        ]
    )
    info = tile_reuse.probe(sample_video)
    tiles = tile_reuse.grid_tiles(info.width, info.height, 3, margin=8)
    tile_reuse.crop_tiles(frame, tiles, tmp_path / "blocks", bins)
    stitched = tmp_path / "stitched.png"
    # Scale 1 stands in for the upscaler: what comes back must be the original.
    tile_reuse.stitch(tmp_path / "blocks", tiles, 1, stitched, bins)
    assert tile_reuse.psnr(stitched, frame, bins) == float("inf")
