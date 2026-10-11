"""Tests for the block geometry and the FFmpeg passes that cut and join blocks.

The claim everything rests on: cutting frames into blocks and stitching them
back returns the same pixels, so that a difference measured later is the
upscaler's and not the assembly's.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videoenhance.blocks import (
    BlockError,
    block_hashes,
    crop_blocks,
    grid_tiles,
    overhead,
    stitch_blocks,
)
from videoenhance.config import binaries
from videoenhance.probe import probe
from videoenhance.process import run

from .conftest import HELD_DRAWINGS, HELD_FOR, requires_ffmpeg


def psnr(a: Path, b: Path) -> float:
    """PSNR of two images; ``inf`` when they are the same."""
    done = run(
        [binaries().ffmpeg, "-hide_banner", "-i", a, "-i", b, "-lavfi", "[0:v][1:v]psnr"]
        + ["-f", "null", "-"]
    )
    return float(done.stderr.rsplit("average:", 1)[1].split()[0])


def extract_frames(video: Path, out: Path, count: int) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    run(
        [binaries().ffmpeg, "-loglevel", "error", "-i", video, "-frames:v", str(count)]
        + ["-pix_fmt", "rgb24", out / "%08d.png"]
    )
    return out


@pytest.mark.parametrize("width,height,grid", [(320, 240, 3), (1920, 1080, 4), (641, 359, 6)])
def test_blocks_cover_the_frame_exactly_once(width: int, height: int, grid: int):
    tiles = grid_tiles(width, height, grid, margin=16)
    assert len(tiles) == grid * grid
    assert sum(tile.area for tile in tiles) == width * height
    covered = {(x, y) for t in tiles for x in (t.x0, t.x1 - 1) for y in (t.y0, t.y1 - 1)}
    assert (0, 0) in covered and (width - 1, height - 1) in covered


def test_margin_is_clamped_to_the_frame():
    tiles = grid_tiles(300, 300, 3, margin=16)
    corner, middle = tiles[0], tiles[4]
    assert (corner.px0, corner.py0, corner.px1, corner.py1) == (0, 0, 116, 116)
    assert (middle.px0, middle.py0, middle.px1, middle.py1) == (84, 84, 216, 216)


def test_grid_and_margin_are_validated():
    with pytest.raises(ValueError):
        grid_tiles(4, 4, 5, margin=0)
    with pytest.raises(ValueError):
        grid_tiles(100, 100, 2, margin=-1)


def test_overhead_is_one_without_a_margin_and_grows_with_it():
    assert overhead(grid_tiles(90, 90, 3, margin=0)) == 1.0
    assert overhead(grid_tiles(90, 90, 3, margin=10)) > 1.0


@requires_ffmpeg
def test_block_hashes_of_a_video_window(held_video: Path):
    info = probe(held_video)
    tiles = grid_tiles(info.width, info.height, 3, margin=16)

    frames, per_block = block_hashes(["-i", held_video], tiles)

    assert len(frames) == HELD_DRAWINGS * HELD_FOR
    assert len(set(frames)) == HELD_DRAWINGS
    assert len(per_block) == 9 and all(len(h) == len(frames) for h in per_block)


@requires_ffmpeg
def test_block_hashes_of_a_directory_agree_with_the_video(held_video: Path, tmp_path: Path):
    info = probe(held_video)
    tiles = grid_tiles(info.width, info.height, 3, margin=16)
    directory = extract_frames(held_video, tmp_path / "frames", HELD_DRAWINGS * HELD_FOR)

    from_dir = block_hashes(["-framerate", "1", "-i", directory / "%08d.png"], tiles)

    assert from_dir == block_hashes(["-i", held_video], tiles)


@requires_ffmpeg
def test_a_block_hash_changes_when_only_its_margin_does(tmp_path: Path):
    """The point of hashing the margin: a changed neighbour means a redo."""
    ff = binaries().ffmpeg
    frames = tmp_path / "frames"
    frames.mkdir()
    # Two 60x60 frames, equal except for one pixel in the left third.
    for index, extra in enumerate(("", ",drawbox=x=20:y=30:w=1:h=1:color=white:t=fill"), start=1):
        run(
            [ff, "-loglevel", "error", "-f", "lavfi", "-i", "color=gray:s=60x60", "-vf"]
            + [f"format=rgb24{extra}", "-frames:v", "1", frames / f"{index:08d}.png"]
        )
    tiles = grid_tiles(60, 60, 3, margin=4)

    _, per_block = block_hashes(["-framerate", "1", "-i", frames / "%08d.png"], tiles)

    differ = [h[0] != h[1] for h in per_block]
    # Block 3 (x 0-20, y 20-40) contains the pixel; block 4 (x 20-40) only has
    # it in its margin; blocks two columns away cannot see it.
    assert differ[3] and differ[4]
    assert not differ[5] and not differ[2] and not differ[8]


@requires_ffmpeg
def test_cutting_and_stitching_a_sequence_returns_the_same_pixels(
    sample_video: Path, tmp_path: Path
):
    info = probe(sample_video)
    frames = extract_frames(sample_video, tmp_path / "frames", 5)
    tiles = grid_tiles(info.width, info.height, 3, margin=8)
    dirs = [tmp_path / "blocks" / f"{i}" for i in range(len(tiles))]

    crop_blocks(["-framerate", "1", "-i", frames / "%08d.png"], tiles, dirs)
    # Scale 1 stands in for the upscaler: what comes back must be the original.
    stitch_blocks(dirs, tiles, 1, tmp_path / "out" / "%08d.png")

    for number in range(1, 6):
        name = f"{number:08d}.png"
        assert (tmp_path / "out" / name).exists()
        assert psnr(tmp_path / "out" / name, frames / name) == float("inf")


@requires_ffmpeg
def test_stitching_a_single_frame_to_a_plain_file_name(sample_video: Path, tmp_path: Path):
    info = probe(sample_video)
    frames = extract_frames(sample_video, tmp_path / "frames", 1)
    tiles = grid_tiles(info.width, info.height, 2, margin=8)
    dirs = [tmp_path / f"b{i}" for i in range(len(tiles))]
    crop_blocks(["-i", frames / "00000001.png"], tiles, dirs)

    stitch_blocks(dirs, tiles, 1, tmp_path / "one.png")

    assert psnr(tmp_path / "one.png", frames / "00000001.png") == float("inf")


@requires_ffmpeg
def test_stitching_can_write_jpeg(sample_video: Path, tmp_path: Path):
    info = probe(sample_video)
    frames = extract_frames(sample_video, tmp_path / "frames", 2)
    tiles = grid_tiles(info.width, info.height, 2, margin=8)
    dirs = [tmp_path / f"b{i}" for i in range(len(tiles))]
    crop_blocks(["-framerate", "1", "-i", frames / "%08d.png"], tiles, dirs)

    stitch_blocks(dirs, tiles, 1, tmp_path / "out" / "%08d.jpg")

    assert len(list((tmp_path / "out").glob("*.jpg"))) == 2


def test_mismatched_directory_counts_are_refused(tmp_path: Path):
    tiles = grid_tiles(30, 30, 2, margin=0)
    with pytest.raises(ValueError):
        crop_blocks(["-i", "x.png"], tiles, [tmp_path])
    with pytest.raises(ValueError):
        stitch_blocks([tmp_path], tiles, 1, tmp_path / "x.png")


def test_the_error_type_is_a_runtime_error():
    assert issubclass(BlockError, RuntimeError)
