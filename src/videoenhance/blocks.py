"""Splitting frames into blocks, and putting them back together.

A block is a region of a frame cut out with a margin of its neighbours. The
margin is what makes the split safe: Real-ESRGAN's output at a pixel depends on
the pixels around it, so a block upscaled on its own differs at its edges from
the same pixels upscaled inside the whole frame. Upscale the block *with* its
margin, cut the margin off afterwards, and the seams are gone — measured on the
test fixture at over 60 dB against the whole-frame result from a margin of 8
pixels, against 47.7 dB and visible lines along every edge with none.

Everything here is FFmpeg over a whole sequence at a time, never one process per
frame: thousands of FFmpeg starts would cost more than the GPU saves, and no
frame is ever read into Python. The functions take a list of FFmpeg *input
arguments* rather than a path, so one implementation serves a window of a video
(``["-ss", "5", "-t", "30", "-i", video]``) and a directory of frames
(``["-framerate", "1", "-i", directory / "%08d.png"]``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import Binaries, binaries
from .extract import JPEG_QUALITY
from .process import run


class BlockError(RuntimeError):
    """The blocks of a frame sequence could not be cut, hashed or joined."""


@dataclass(frozen=True)
class Tile:
    """One block of the grid, and the region upscaled for it."""

    #: The block itself: the pixels it contributes to the stitched frame.
    x0: int
    y0: int
    x1: int
    y1: int
    #: The block plus its margin, clamped to the frame.
    px0: int
    py0: int
    px1: int
    py1: int

    @property
    def area(self) -> int:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @property
    def padded_area(self) -> int:
        return (self.px1 - self.px0) * (self.py1 - self.py0)

    def crop(self) -> str:
        """FFmpeg ``crop`` arguments for the padded region."""
        return f"{self.px1 - self.px0}:{self.py1 - self.py0}:{self.px0}:{self.py0}"


def grid_tiles(width: int, height: int, grid: int, margin: int) -> list[Tile]:
    """Split a ``width`` x ``height`` frame into ``grid`` x ``grid`` blocks."""
    if grid < 1 or grid > min(width, height):
        raise ValueError(f"a {grid}x{grid} grid does not fit a {width}x{height} frame")
    if margin < 0:
        raise ValueError("the margin cannot be negative")
    xs = [round(i * width / grid) for i in range(grid + 1)]
    ys = [round(i * height / grid) for i in range(grid + 1)]
    return [
        Tile(
            xs[col],
            ys[row],
            xs[col + 1],
            ys[row + 1],
            max(0, xs[col] - margin),
            max(0, ys[row] - margin),
            min(width, xs[col + 1] + margin),
            min(height, ys[row + 1] + margin),
        )
        for row in range(grid)
        for col in range(grid)
    ]


def overhead(tiles: Sequence[Tile]) -> float:
    """What a frame that changes everywhere costs in blocks, as a multiple of one.

    1.0 is no cost at all; the margins push it up, which is why a grid can lose
    to whole frames on footage with nothing static in it.
    """
    return sum(tile.padded_area for tile in tiles) / sum(tile.area for tile in tiles)


def _labels(prefix: str, count: int) -> str:
    return "".join(f"[{prefix}{i}]" for i in range(count))


def _parse_framemd5(text: str, streams: int) -> list[list[str]]:
    """Split ``framemd5`` output into one list of hashes per stream."""
    per_stream: list[list[str]] = [[] for _ in range(streams)]
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        per_stream[int(fields[0])].append(fields[-1])
    return per_stream


def block_hashes(
    inputs: Sequence[object],
    tiles: Sequence[Tile],
    bins: Binaries | None = None,
) -> tuple[list[str], list[list[str]]]:
    """Hash every frame, and every padded block of it, in one FFmpeg pass.

    Returns ``(frame_hashes, per_block)``, where ``per_block[i][j]`` is the hash
    of block ``i`` in frame ``j``. The hash is of decoded pixels, as in
    :func:`videoenhance.dedup.pixel_hashes`, and of the block *with* its margin,
    which is what makes it a safe cache key: a block whose neighbour changed has
    a different hash and is redone, instead of being served with a seam.

    The frames go through ``format=rgb24`` first, which is what the pipeline
    extracts to and which lets a block start at an odd pixel without chroma
    rounding.

    Raises:
        BlockError: no frames came back, or a block has a different number of
            hashes from the frames, which would stitch the wrong block in.
    """
    bins = bins or binaries()
    count = len(tiles)
    graph = [f"[0:v]format=rgb24,split={count + 1}[whole]{_labels('t', count)}"]
    graph += [f"[t{i}]crop={tile.crop()}[c{i}]" for i, tile in enumerate(tiles)]
    cmd: list[object] = [bins.ffmpeg, "-hide_banner", "-loglevel", "error", *inputs]
    cmd += ["-filter_complex", ";".join(graph), "-fps_mode", "passthrough", "-map", "[whole]"]
    for i in range(count):
        cmd += ["-map", f"[c{i}]"]
    cmd += ["-an", "-f", "framemd5", "-"]

    streams = _parse_framemd5(run(cmd).stdout or "", count + 1)
    frames, per_block = streams[0], streams[1:]
    if not frames:
        raise BlockError("no frames were decoded, so there is nothing to hash")
    for index, hashes in enumerate(per_block):
        if len(hashes) != len(frames):
            raise BlockError(
                f"block {index} has {len(hashes)} hashes for {len(frames)} frames; "
                "a miscount would put the wrong block into a frame"
            )
    return frames, per_block


def crop_blocks(
    inputs: Sequence[object],
    tiles: Sequence[Tile],
    out_dirs: Sequence[Path],
    bins: Binaries | None = None,
) -> None:
    """Write every padded block of every frame, one PNG sequence per block.

    ``out_dirs[i]`` receives block ``i`` as ``00000001.png`` onwards, one file
    per input frame. PNG whatever the frames were: a block is cropped, upscaled,
    cropped again and stitched, and JPEG would add loss at every one of those.
    """
    bins = bins or binaries()
    if len(out_dirs) != len(tiles):
        raise ValueError("crop_blocks needs one output directory per block")
    count = len(tiles)
    graph = [f"[0:v]format=rgb24,split={count}{_labels('t', count)}"]
    graph += [f"[t{i}]crop={tile.crop()}[c{i}]" for i, tile in enumerate(tiles)]
    cmd: list[object] = [bins.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *inputs]
    cmd += ["-filter_complex", ";".join(graph)]
    for i, directory in enumerate(out_dirs):
        directory.mkdir(parents=True, exist_ok=True)
        cmd += ["-map", f"[c{i}]", "-fps_mode", "passthrough", directory / "%08d.png"]
    run(cmd)


def stitch_blocks(
    block_dirs: Sequence[Path],
    tiles: Sequence[Tile],
    scale: int,
    out_pattern: Path,
    bins: Binaries | None = None,
) -> None:
    """Cut the margin off each upscaled block and join the blocks into frames.

    ``block_dirs[i]`` holds block ``i`` as ``00000001.png`` onwards, already
    upscaled by ``scale``. The margin is cut at ``(x0 - px0) * scale`` and the
    blocks are laid out at the *upscaled* offsets, so the frame comes out at
    ``scale`` times the size of the source. ``out_pattern`` is a file name with
    ``%08d`` for a sequence, or a plain file name for a single frame.
    """
    bins = bins or binaries()
    if len(block_dirs) != len(tiles):
        raise ValueError("stitch_blocks needs one directory per block")
    out_pattern.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[object] = [bins.ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for directory in block_dirs:
        cmd += ["-framerate", "1", "-i", directory / "%08d.png"]
    graph = []
    for i, tile in enumerate(tiles):
        w, h = (tile.x1 - tile.x0) * scale, (tile.y1 - tile.y0) * scale
        x, y = (tile.x0 - tile.px0) * scale, (tile.y0 - tile.py0) * scale
        graph.append(f"[{i}:v]crop={w}:{h}:{x}:{y}[s{i}]")
    if len(tiles) == 1:
        graph.append("[s0]null[out]")
    else:
        layout = "|".join(f"{t.x0 * scale}_{t.y0 * scale}" for t in tiles)
        graph.append(f"{_labels('s', len(tiles))}xstack=inputs={len(tiles)}:layout={layout}[out]")
    cmd += ["-filter_complex", ";".join(graph), "-map", "[out]", "-fps_mode", "passthrough"]
    if out_pattern.suffix.lower() in (".jpg", ".jpeg"):
        cmd += ["-q:v", str(JPEG_QUALITY)]
    if "%" not in out_pattern.name:
        cmd += ["-frames:v", "1", "-update", "1"]
    cmd.append(out_pattern)
    run(cmd)


__all__ = [
    "BlockError",
    "Tile",
    "block_hashes",
    "crop_blocks",
    "grid_tiles",
    "overhead",
    "stitch_blocks",
]
