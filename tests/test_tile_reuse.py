"""Tests for scripts/measure_tile_reuse.py, the block-reuse measurement.

The script decides whether a feature is worth building, so a wrong number is
worse than none: the accounting must agree with whole-frame dedup when there is
nothing more to find. The geometry and the cut-and-stitch round trip are tested
in test_blocks.py, where that code now lives.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from .conftest import HELD_DRAWINGS, HELD_FOR, requires_ffmpeg

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_tile_reuse.py"
_spec = importlib.util.spec_from_file_location("measure_tile_reuse", SCRIPT)
tile_reuse = importlib.util.module_from_spec(_spec)
sys.modules["measure_tile_reuse"] = tile_reuse
_spec.loader.exec_module(tile_reuse)


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
    frames, blocks = tile_reuse.block_hashes(["-i", held_video], tiles)
    report = tile_reuse.summarize(frames, blocks, tiles, grid=3, margin=16)
    assert report.frames == HELD_DRAWINGS * HELD_FOR
    assert report.distinct_frames == HELD_DRAWINGS
    assert all(len(hashes) == report.frames for hashes in blocks)
