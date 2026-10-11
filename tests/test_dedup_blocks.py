"""Tests for generating each distinct block once.

The strongest claim available is equivalence: the scaling stub is purely local,
so cutting a frame into blocks, upscaling them with a margin and stitching them
back must give *exactly* the whole-frame result, on every frame. The rest is
about what reaches the GPU: nine blocks for the first frame and one for each
frame that changes only a box inside one block.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from videoenhance.blocks import grid_tiles
from videoenhance.config import binaries
from videoenhance.dedup import (
    DedupStats,
    FrameCache,
    _block_key,
    trim_caches,
    upscale_deduplicated,
)
from videoenhance.process import run

from .conftest import BLOCK_FRAMES, STUB_SCALING, requires_ffmpeg
from .test_blocks import psnr

GRID = 3
MARGIN = 8
#: Nine blocks for the first frame, then one for each of the five that follow.
EXPECTED_BLOCKS = GRID**2 + (BLOCK_FRAMES - 1)


@pytest.fixture
def scaling_stub(stub_bin):
    stub_bin("REALESRGAN_BIN", STUB_SCALING)
    stub_bin.counts = stub_bin.log.with_name(stub_bin.log.name + ".count")  # type: ignore[attr-defined]
    return stub_bin


def sent(stub) -> list[int]:
    """How many images each call to the stub was handed."""
    if not stub.counts.exists():
        return []
    return [int(line) for line in stub.counts.read_text().split()]


def cache_for(tmp_path: Path, blocks: int, margin: int = 0, **overrides) -> FrameCache:
    settings = {"model": "realesr-animevideov3", "scale": 2, "tile": 0, "frame_format": "png"}
    return FrameCache(
        tmp_path / "cache",
        blocks=blocks,
        margin=margin if blocks > 1 else 0,
        **{**settings, **overrides},
    )


def run_stage(frames: Path, tmp_path: Path, name: str, blocks: int, **kwargs):
    margin = kwargs.pop("margin", MARGIN if blocks > 1 else 0)
    frame_format = kwargs.get("frame_format", "png")
    return upscale_deduplicated(
        frames,
        tmp_path / name,
        scratch=tmp_path / "dedup",
        cache=kwargs.pop("cache", None)
        or cache_for(tmp_path, blocks, margin, frame_format=frame_format),
        blocks=blocks,
        margin=margin,
        scale=2,
        **kwargs,
    )


@requires_ffmpeg
def test_blocks_reproduce_the_whole_frame_result_exactly(
    moving_box_frames: Path, tmp_path: Path, scaling_stub
):
    run_stage(moving_box_frames, tmp_path / "whole", "up", blocks=1)
    run_stage(moving_box_frames, tmp_path / "blocks", "up", blocks=GRID)

    for number in range(1, BLOCK_FRAMES + 1):
        name = f"{number:08d}.png"
        whole = tmp_path / "whole" / "up" / name
        stitched = tmp_path / "blocks" / "up" / name
        assert psnr(stitched, whole) == float("inf"), name


@requires_ffmpeg
def test_the_first_frame_sends_every_block_and_later_ones_only_what_moved(
    moving_box_frames: Path, tmp_path: Path, scaling_stub
):
    _, stats = run_stage(moving_box_frames, tmp_path, "up", blocks=GRID)

    assert sent(scaling_stub) == [EXPECTED_BLOCKS]
    assert stats.blocks == GRID
    assert stats.blocks_total == BLOCK_FRAMES * GRID**2
    assert stats.blocks_upscaled == EXPECTED_BLOCKS
    assert 0 < stats.work < BLOCK_FRAMES
    assert len(list((tmp_path / "up").glob("*.png"))) == BLOCK_FRAMES


@requires_ffmpeg
def test_a_second_run_sends_nothing_and_gives_the_same_frames(
    moving_box_frames: Path, tmp_path: Path, scaling_stub
):
    cache = cache_for(tmp_path, GRID, MARGIN)
    run_stage(moving_box_frames, tmp_path, "first", blocks=GRID, cache=cache)
    calls = len(scaling_stub.log.read_text().splitlines())

    _, stats = run_stage(moving_box_frames, tmp_path, "second", blocks=GRID, cache=cache)

    assert len(scaling_stub.log.read_text().splitlines()) == calls
    assert stats.blocks_upscaled == 0 and stats.work == 0
    for number in range(1, BLOCK_FRAMES + 1):
        name = f"{number:08d}.png"
        assert psnr(tmp_path / "second" / name, tmp_path / "first" / name) == float("inf")


@requires_ffmpeg
def test_blocks_are_png_even_when_the_frames_are_jpeg(
    moving_box_frames: Path, tmp_path: Path, scaling_stub
):
    jpegs = tmp_path / "jpegs"
    jpegs.mkdir()
    run(
        [binaries().ffmpeg, "-loglevel", "error", "-i", moving_box_frames / "%08d.png"]
        + ["-q:v", "2", jpegs / "%08d.jpg"]
    )

    run_stage(jpegs, tmp_path, "up", blocks=GRID, frame_format="jpg")

    assert all(" -f png " in f" {line} " for line in scaling_stub.log.read_text().splitlines())
    assert len(list((tmp_path / "up").glob("*.jpg"))) == BLOCK_FRAMES
    assert list((tmp_path / "up").glob("*.png")) == []


@requires_ffmpeg
def test_one_block_per_side_is_the_whole_frame_path(
    moving_box_frames: Path, tmp_path: Path, scaling_stub
):
    _, stats = run_stage(moving_box_frames, tmp_path, "up", blocks=1)

    assert sent(scaling_stub) == [BLOCK_FRAMES]
    assert stats.blocks == 1 and stats.blocks_total == 0


@requires_ffmpeg
def test_the_scratch_space_is_left_empty(moving_box_frames: Path, tmp_path: Path, scaling_stub):
    run_stage(moving_box_frames, tmp_path, "up", blocks=GRID)

    assert list((tmp_path / "dedup").iterdir()) == []


@requires_ffmpeg
def test_blocks_and_whole_frames_do_not_share_a_cache_slot(tmp_path: Path):
    whole = cache_for(tmp_path, 1)
    stitched = cache_for(tmp_path, GRID, MARGIN)

    assert whole.slot != stitched.slot
    assert "-b" not in whole.slot.name
    assert stitched.for_blocks().slot.name.startswith("blocks-")
    assert stitched.for_blocks().frame_format == "png"


def test_a_cache_for_another_grid_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="built for"):
        upscale_deduplicated(
            tmp_path,
            tmp_path / "up",
            scratch=tmp_path / "s",
            cache=cache_for(tmp_path, 1),
            blocks=GRID,
            margin=MARGIN,
        )


def test_the_same_bytes_in_another_shape_are_another_block():
    """A flat colour hashes alike at 100x50 and at 50x100."""
    wide = grid_tiles(100, 50, 1, 0)[0]
    tall = grid_tiles(50, 100, 1, 0)[0]

    assert _block_key(wide, "abc") != _block_key(tall, "abc")
    assert _block_key(wide, "abc") == _block_key(wide, "abc")


def test_one_budget_covers_frames_and_blocks(tmp_path: Path):
    frames = cache_for(tmp_path, GRID, MARGIN)
    blocks = frames.for_blocks()
    for index, cache in enumerate((frames, blocks, frames, blocks)):
        source = tmp_path / f"s{index}.png"
        source.write_bytes(b"x" * 1000)
        entry = cache.put(f"h{index}", source)
        os.utime(entry, (index, index))

    removed = trim_caches([frames, blocks], budget=2500)

    assert removed == 2
    assert sorted(p.name for p in frames.entries() + blocks.entries()) == ["h2.png", "h3.png"]


def test_the_summary_says_what_the_blocks_cost():
    saved = DedupStats(
        300, 300, 0, 300, blocks=3, blocks_total=2700, blocks_upscaled=412, work=61.4
    )
    worse = DedupStats(10, 10, 0, 10, blocks=3, blocks_total=90, blocks_upscaled=90, work=11.5)

    assert saved.render() == (
        "3x3 blocks: 412 of 2700 generated, 61.4 frames of work for 300 (80% saved)"
    )
    assert "15% more than whole frames" in worse.render()
