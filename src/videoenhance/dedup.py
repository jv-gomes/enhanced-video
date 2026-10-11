"""Generating each distinct frame once, by hashing its pixels.

Real-ESRGAN is the expensive stage and it is deterministic: the same frame, with
the same model, scale and tile size, comes back identical every time. So every
repeated frame in a video is being generated more than once for nothing.

In cel animation that is most of the video. A drawing is held for two or three
frames — "on twos", "on threes" — so hashing the frames and sending only the
distinct ones to the GPU cuts the stage's work by half or by two thirds. The
project's default model, ``realesr-animevideov3``, exists for exactly this kind
of footage.

**Whether the saving is there depends on the content**, and the measurement is
worth stating. Held frames of flat, cel-like artwork survive a lossy encode
bit-for-bit, in x264, x265 and lossless alike: the encoder has nothing to code,
emits skip, and the decoder reproduces the previous frame exactly. Dense or
grainy footage does not, because the encoder spends its bits differently on
every frame, so the decoded frames differ in their low bits and nothing matches.
That is why the saving is reported per run rather than promised.

Per frame, not per sequence: the upscale treats each frame on its own and does
not care about order, so hashing frames catches every repeat, including those
that are not part of any repeated animation — and a repeated animation is, by
definition, made of repeated frames.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import config
from .blocks import Tile, block_hashes, crop_blocks, grid_tiles, stitch_blocks
from .config import Binaries, binaries
from .probe import frame_size
from .process import run
from .upscale import UpscaleResult, upscale
from .workdir import (
    clear_frames,
    count_frames_on_disk,
    frame_glob,
    frame_pattern,
    human_bytes,
    stage_is_done,
)

logger = logging.getLogger(__name__)


class DedupError(RuntimeError):
    """The frames could not be hashed."""


def pixel_hashes(
    frames_dir: Path,
    frame_format: str = config.DEFAULT_FRAME_FORMAT,
    bins: Binaries | None = None,
) -> list[str]:
    """One hash per frame in ``frames_dir``, in order.

    Uses FFmpeg's ``framemd5``, which hashes **decoded pixels** rather than file
    bytes, in a single pass over the whole sequence. Both halves of that matter.
    Hashing the bytes of a PNG would be hostage to the encoder: anything it ever
    chose to embed per frame would make every frame look unique and this whole
    module would quietly stop doing anything, which is the worst way for an
    optimisation to fail. And one process beats reading hundreds of megabytes of
    PNG through Python, with no new dependency — FFmpeg is already required.

    Raises:
        DedupError: FFmpeg produced no hashes, or not one per frame.
        ToolError: FFmpeg failed.
    """
    bins = bins or binaries()
    count = count_frames_on_disk(frames_dir, frame_format)
    if count == 0:
        raise DedupError(f"no {frame_format} frames to hash in {frames_dir}")

    completed = run(
        [
            bins.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            frames_dir / frame_pattern(frame_format),
            "-an",
            "-f",
            "framemd5",
            "-",
        ]
    )
    hashes = [
        line.rsplit(",", 1)[-1].strip()
        for line in (completed.stdout or "").splitlines()
        if line and not line.startswith("#")
    ]
    if len(hashes) != count:
        raise DedupError(
            f"hashed {len(hashes)} frames but {frames_dir} holds {count}; "
            "the frame numbering may have a gap"
        )
    return hashes


@dataclass(frozen=True)
class FrameIndex:
    """Which frames a directory holds, and which of them are the same."""

    #: One hash per frame, in frame order.
    hashes: list[str]
    #: Frame files, in the same order.
    names: list[Path]

    @property
    def frames(self) -> int:
        return len(self.hashes)

    @property
    def unique(self) -> list[str]:
        """Distinct hashes, in the order they first appear."""
        seen: dict[str, None] = {}
        for digest in self.hashes:
            seen.setdefault(digest, None)
        return list(seen)

    def representative(self, digest: str) -> Path:
        """The first frame carrying ``digest``; any of them would do."""
        return self.names[self.hashes.index(digest)]


def index_frames(
    frames_dir: Path,
    frame_format: str = config.DEFAULT_FRAME_FORMAT,
    bins: Binaries | None = None,
) -> FrameIndex:
    """Hash every frame in ``frames_dir`` and pair the hashes with the files."""
    names = sorted(frames_dir.glob(frame_glob(frame_format)))
    return FrameIndex(hashes=pixel_hashes(frames_dir, frame_format, bins), names=names)


def link_or_copy(source: Path, destination: Path) -> Path:
    """Hard-link ``source`` to ``destination``, copying if that cannot be done.

    Linking is what makes the duplicates nearly free in both time and disk: a
    frame that appears thirty times is stored once and pointed at thirty times.
    Nothing in the pipeline rewrites a frame in place — every binary writes new
    files into a new directory — so sharing an inode is safe. A filesystem
    without hard links, or a destination on another one, falls back to a copy.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        logger.debug("cannot hard-link %s, copying instead", destination)
        shutil.copy2(source, destination)
    return destination


class FrameCache:
    """Upscaled frames, keyed by the pixel hash of what went in.

    The key carries the settings that change the pixels, tile size included:
    tiling alters the result at the seams, and the out-of-VRAM retry can lower
    it part-way through a run. Serving a frame that was generated with a
    different tile would be a silent difference, so it counts as a miss.

    ``budget`` is a ceiling in bytes, trimmed least-recently-used. It is not
    decoration: without it the cache would hold every distinct frame of the
    whole video and undo the reason for processing in chunks at all.
    """

    def __init__(
        self,
        root: Path,
        *,
        model: str,
        scale: int,
        tile: int,
        frame_format: str,
        budget: int = config.DEFAULT_CACHE_SIZE,
        blocks: int = 1,
        margin: int = 0,
        kind: str = "frames",
    ) -> None:
        self.root = Path(root)
        self.model = model
        self.scale = scale
        self.tile = tile
        self.blocks = blocks
        self.margin = margin
        self.kind = kind
        if kind == "blocks":
            # A block is PNG whatever the frames are: it is cropped, upscaled,
            # cropped again and stitched, and JPEG would add loss at each step.
            self.frame_format = "png"
            self.slot = self.root / f"blocks-{model}-{scale}x-t{tile}-m{margin}"
        else:
            self.frame_format = frame_format
            name = f"{model}-{scale}x-t{tile}-{frame_format}"
            # A stitched frame is not the same pixels as a whole one, so it must
            # never be served to a run that did not ask for blocks. The slot
            # name of whole frames is left alone, or every cache made before
            # blocks existed would stop hitting.
            self.slot = self.root / (f"{name}-b{blocks}m{margin}" if blocks > 1 else name)
        self.budget = budget
        self.hits = 0

    def for_blocks(self) -> FrameCache:
        """The cache of upscaled blocks that sits beside this cache of frames."""
        return FrameCache(
            self.root,
            model=self.model,
            scale=self.scale,
            tile=self.tile,
            frame_format=self.frame_format,
            budget=self.budget,
            blocks=self.blocks,
            margin=self.margin,
            kind="blocks",
        )

    def with_tile(self, tile: int) -> FrameCache:
        """The cache for another tile size, which is another slot.

        The tile that finally worked can be smaller than the one asked for, when
        the GPU ran out of memory part-way through, and a frame must be filed
        under the tile that made it.
        """
        if tile == self.tile:
            return self
        return FrameCache(
            self.root,
            model=self.model,
            scale=self.scale,
            tile=tile,
            frame_format=self.frame_format,
            budget=self.budget,
            blocks=self.blocks,
            margin=self.margin,
            kind=self.kind,
        )

    def path_for(self, digest: str) -> Path:
        return self.slot / f"{digest}.{self.frame_format}"

    def get(self, digest: str) -> Path | None:
        """The cached frame for ``digest``, marked as just used."""
        path = self.path_for(digest)
        if not path.exists():
            return None
        os.utime(path, None)
        self.hits += 1
        return path

    def put(self, digest: str, source: Path) -> Path:
        """Store ``source`` as the frame for ``digest``."""
        return link_or_copy(source, self.path_for(digest))

    def entries(self) -> list[Path]:
        if not self.slot.is_dir():
            return []
        return list(self.slot.glob(frame_glob(self.frame_format)))

    def size(self) -> int:
        """Bytes the cache occupies, counting a shared inode once."""
        seen: set[int] = set()
        total = 0
        for entry in self.entries():
            stat = entry.stat()
            if stat.st_ino in seen:
                continue
            seen.add(stat.st_ino)
            total += stat.st_size
        return total

    def trim(self) -> int:
        """Drop the least recently used entries until inside the budget."""
        return trim_caches([self], self.budget)


def trim_caches(caches: Sequence[FrameCache], budget: int) -> int:
    """Drop the least recently used entries of all ``caches`` until they fit.

    One budget covers the frames and the blocks together, because they are one
    cost on one disk; two halves would let either grow past what was asked for.
    It runs only once a stage is finished: a trim in the middle could delete a
    block the stitching is about to need.
    """
    if budget <= 0:
        return 0
    entries = sorted(
        (entry for cache in caches for entry in cache.entries()),
        key=lambda p: p.stat().st_mtime,
    )
    seen: set[int] = set()
    total = 0
    for entry in entries:
        stat = entry.stat()
        if stat.st_ino not in seen:
            seen.add(stat.st_ino)
            total += stat.st_size
    removed = 0
    for entry in entries:
        if total <= budget:
            break
        size = entry.stat().st_size
        entry.unlink(missing_ok=True)
        total -= size
        removed += 1
    if removed:
        logger.info("trimmed %d cached frames to stay under %s", removed, human_bytes(budget))
    return removed


@dataclass(frozen=True)
class DedupStats:
    """What the hashing saved, for the run summary."""

    frames: int
    unique: int
    #: Distinct frames that were already in the cache from an earlier chunk.
    cached: int
    #: Distinct frames actually sent to the GPU, or built from blocks.
    upscaled: int
    #: Blocks per side; 1 means whole frames, and the fields below are unused.
    blocks: int = 1
    #: Blocks the frames of the run are made of, ``frames * blocks ** 2``.
    blocks_total: int = 0
    #: Blocks actually sent to the GPU.
    blocks_upscaled: int = 0
    #: Upscaled area sent to the GPU, in frames. Margins count, which is why it
    #: can exceed ``upscaled`` when a grid does not pay for itself.
    work: float = 0.0

    @property
    def saved(self) -> int:
        if self.blocks > 1:
            return round(self.frames - self.work)
        return self.frames - self.upscaled

    @property
    def ratio(self) -> float:
        if self.blocks > 1:
            return (self.frames - self.work) / self.frames if self.frames else 0.0
        return self.saved / self.frames if self.frames else 0.0

    def render(self) -> str:
        if self.blocks > 1:
            detail = (
                f"{self.blocks}x{self.blocks} blocks: {self.blocks_upscaled} of "
                f"{self.blocks_total} generated, {self.work:.1f} frames of work "
                f"for {self.frames}"
            )
            if self.cached:
                detail += f", {self.cached} frames reused from earlier chunks"
            if self.ratio < 0:
                return f"{detail} ({-self.ratio:.0%} more than whole frames)"
            return f"{detail} ({self.ratio:.0%} saved)"
        if not self.saved:
            return f"{self.frames} distinct frames, nothing to reuse"
        detail = f"{self.upscaled} of {self.frames} frames generated"
        if self.cached:
            detail += f", {self.cached} reused from earlier chunks"
        return f"{detail} ({self.ratio:.0%} saved)"


def _block_key(tile: Tile, digest: str) -> str:
    """The cache key of a block: its pixels and the shape they are laid out in.

    The shape is in the key because the hash is of raw bytes, so the same bytes
    at 100x50 and at 50x100 hash alike, which a flat colour makes common. Served
    one for the other, a block would come back the wrong shape.
    """
    return f"{tile.px1 - tile.px0}x{tile.py1 - tile.py0}-{digest}"


def _generate_from_blocks(
    frames_in: Path,
    index: FrameIndex,
    missing: list[str],
    *,
    scratch: Path,
    cache: FrameCache,
    blocks: int,
    margin: int,
    scale: int,
    model: str,
    gpu: int,
    tile: int,
    threads: str,
    load: int,
    frame_format: str,
    bins: Binaries,
) -> tuple[int, int, float]:
    """Build the frames ``missing`` from the cache by upscaling only new blocks.

    Every frame goes through blocks, the unchanged ones included. A block
    upscaled alone differs at its edges from the same pixels inside a whole
    frame, so a frame upscaled whole next to one stitched from blocks would
    shift slightly between the two and flicker; stitched throughout, a reused
    block is bit for bit what recomputing it would give.

    Returns ``(tile that worked, blocks sent to the GPU, work in frames)``.
    """
    block_cache = cache.for_blocks()
    names = ("frames", "crops", "todo", "todo_up", "stitch", "out")
    dirs = {name: scratch / name for name in names}
    for directory in dirs.values():
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)

    try:
        for number, digest in enumerate(missing, start=1):
            link_or_copy(
                index.representative(digest), dirs["frames"] / f"{number:08d}.{frame_format}"
            )
        width, height = frame_size(next(iter(sorted(dirs["frames"].iterdir()))), bins)
        tiles = grid_tiles(width, height, blocks, margin)
        inputs = ["-framerate", "1", "-i", dirs["frames"] / frame_pattern(frame_format)]
        _, per_block = block_hashes(inputs, tiles, bins)

        # Each distinct block once, in the order a frame first needs it.
        scheduled: dict[str, tuple[int, int]] = {}
        for frame in range(len(missing)):
            for position, block_tile in enumerate(tiles):
                key = _block_key(block_tile, per_block[position][frame])
                if key not in scheduled and block_cache.get(key) is None:
                    scheduled[key] = (position, frame)
        logger.info(
            "%d frames to build from %d blocks each, %d blocks to generate",
            len(missing),
            len(tiles),
            len(scheduled),
        )

        used_tile = tile
        work = 0.0
        if scheduled:
            crop_dirs = [dirs["crops"] / f"{i:03d}" for i in range(len(tiles))]
            crop_blocks(inputs, tiles, crop_dirs, bins)
            for rank, (position, frame) in enumerate(scheduled.values(), start=1):
                link_or_copy(
                    crop_dirs[position] / f"{frame + 1:08d}.png",
                    dirs["todo"] / f"{rank:08d}.png",
                )
                work += tiles[position].padded_area / (width * height)
            shutil.rmtree(dirs["crops"], ignore_errors=True)

            result = upscale(
                dirs["todo"],
                dirs["todo_up"],
                expected_frames=len(scheduled),
                scale=scale,
                model=model,
                frame_format="png",
                gpu=gpu,
                tile=tile,
                threads=threads,
                load=load,
                bins=bins,
                force=True,
            )
            used_tile = result.tile
            made_by = block_cache.with_tile(used_tile)
            for rank, key in enumerate(scheduled, start=1):
                made_by.put(key, dirs["todo_up"] / f"{rank:08d}.png")

        # A block is wherever the tile that made it filed it.
        block_sources = [block_cache]
        if used_tile != block_cache.tile:
            block_sources.append(block_cache.with_tile(used_tile))
        for position, block_tile in enumerate(tiles):
            farm = dirs["stitch"] / f"{position:03d}"
            farm.mkdir()
            for frame in range(len(missing)):
                key = _block_key(block_tile, per_block[position][frame])
                found = next((hit for c in block_sources if (hit := c.get(key))), None)
                if found is None:
                    raise DedupError(
                        f"block {position} of frame {frame + 1} is missing from the cache "
                        f"after upscaling; its key is {key}"
                    )
                link_or_copy(found, farm / f"{frame + 1:08d}.png")

        stitch_blocks(
            [dirs["stitch"] / f"{i:03d}" for i in range(len(tiles))],
            tiles,
            scale,
            dirs["out"] / frame_pattern(frame_format),
            bins,
        )
        made_frames = cache.with_tile(used_tile)
        for number, digest in enumerate(missing, start=1):
            made_frames.put(digest, dirs["out"] / f"{number:08d}.{frame_format}")
        return used_tile, len(scheduled), work
    finally:
        for directory in dirs.values():
            shutil.rmtree(directory, ignore_errors=True)


def upscale_deduplicated(
    frames_in: Path,
    frames_out: Path,
    *,
    scratch: Path,
    cache: FrameCache,
    expected_frames: int = 0,
    scale: int = config.DEFAULT_SCALE,
    model: str = config.DEFAULT_UPSCALE_MODEL,
    frame_format: str = config.DEFAULT_FRAME_FORMAT,
    gpu: int = config.DEFAULT_GPU_ID,
    tile: int = config.DEFAULT_TILE_SIZE,
    threads: str = config.DEFAULT_THREADS,
    load: int = config.DEFAULT_GPU_LOAD,
    blocks: int = config.DEFAULT_BLOCKS,
    margin: int = 0,
    bins: Binaries | None = None,
    force: bool = False,
) -> tuple[UpscaleResult, DedupStats]:
    """Upscale ``frames_in`` into ``frames_out``, generating each frame once.

    Wraps :func:`videoenhance.upscale.upscale` rather than replacing it, so the
    out-of-VRAM tile retry, the output count check and the error messages all
    still apply — the binary is simply handed a smaller directory.

    ``frames_out`` ends up holding exactly as many frames as ``frames_in``, so
    the resume rule does not change: a stage whose output directory holds the
    expected count is still a stage that has run.

    With ``blocks`` above 1 a frame that misses the cache is cut into a grid of
    that many blocks per side, each with a ``margin`` of its neighbours, and only
    the blocks not seen before reach the GPU; ``cache`` must have been built for
    the same grid, since a stitched frame is not the same pixels as a whole one.
    """
    bins = bins or binaries()
    if blocks < 1 or margin < 0:
        raise ValueError("blocks must be 1 or more and the margin 0 or more")
    if (cache.blocks, cache.margin) != (blocks, margin if blocks > 1 else 0):
        raise ValueError(
            f"the cache was built for {cache.blocks} blocks with margin {cache.margin}, "
            f"not {blocks} with {margin}"
        )
    count = count_frames_on_disk(frames_in, frame_format)
    if count == 0:
        raise DedupError(f"no {frame_format} frames to upscale in {frames_in}")
    expected = expected_frames or count
    frames_out.mkdir(parents=True, exist_ok=True)

    if not force and stage_is_done("upscale", frames_out, expected, frame_format, exact=True):
        return (
            UpscaleResult(
                frames_dir=frames_out,
                frame_count=count_frames_on_disk(frames_out, frame_format),
                scale=scale,
                model=model,
                tile=tile,
                skipped=True,
            ),
            DedupStats(frames=expected, unique=0, cached=0, upscaled=0),
        )

    index = index_frames(frames_in, frame_format, bins)
    unique = index.unique
    missing = [digest for digest in unique if cache.get(digest) is None]
    logger.info(
        "%d frames, %d distinct, %d to generate", index.frames, len(unique), len(missing)
    )

    result = UpscaleResult(
        frames_dir=frames_out, frame_count=index.frames, scale=scale, model=model, tile=tile
    )
    blocks_upscaled, work = 0, 0.0
    if missing and blocks > 1:
        used_tile, blocks_upscaled, work = _generate_from_blocks(
            frames_in,
            index,
            missing,
            scratch=scratch,
            cache=cache,
            blocks=blocks,
            margin=margin,
            scale=scale,
            model=model,
            gpu=gpu,
            tile=tile,
            threads=threads,
            load=load,
            frame_format=frame_format,
            bins=bins,
        )
        result = UpscaleResult(
            frames_dir=frames_out,
            frame_count=index.frames,
            scale=scale,
            model=model,
            tile=used_tile,
        )
    elif missing:
        todo, todo_up = scratch / "in", scratch / "out"
        for directory in (todo, todo_up):
            directory.mkdir(parents=True, exist_ok=True)
            clear_frames(directory, frame_format)
        # Renumbered from one: the binary names its output after its input, so
        # the rank is what ties a result back to the hash that asked for it.
        for rank, digest in enumerate(missing, start=1):
            link_or_copy(index.representative(digest), todo / f"{rank:08d}.{frame_format}")

        result = upscale(
            todo,
            todo_up,
            expected_frames=len(missing),
            scale=scale,
            model=model,
            frame_format=frame_format,
            gpu=gpu,
            tile=tile,
            threads=threads,
            load=load,
            bins=bins,
            force=True,
        )
        made_by = cache.with_tile(result.tile)
        for rank, digest in enumerate(missing, start=1):
            made_by.put(digest, todo_up / f"{rank:08d}.{frame_format}")
        clear_frames(todo, frame_format)
        clear_frames(todo_up, frame_format)

    # A frame is where the tile that made it filed it: the requested slot for the
    # hits, the slot of the tile that worked for what was just generated.
    sources = [cache] if result.tile == cache.tile else [cache, cache.with_tile(result.tile)]
    clear_frames(frames_out, frame_format)
    for position, digest in enumerate(index.hashes, start=1):
        cached = next((found for c in sources if (found := c.get(digest))), None)
        if cached is None:
            raise DedupError(
                f"frame {position} is missing from the cache after upscaling; "
                f"its hash is {digest}"
            )
        link_or_copy(cached, frames_out / f"{position:08d}.{frame_format}")

    # Frames and blocks are trimmed together, and only now that the frames are
    # linked: a trim earlier could delete a block the stitching still needs.
    caches = list(sources)
    if blocks > 1:
        caches += [c.for_blocks() for c in sources]
    trim_caches(caches, cache.budget)
    stats = DedupStats(
        frames=index.frames,
        unique=len(unique),
        cached=len(unique) - len(missing),
        upscaled=len(missing),
        blocks=blocks,
        blocks_total=index.frames * blocks**2 if blocks > 1 else 0,
        blocks_upscaled=blocks_upscaled,
        work=work,
    )
    logger.info("upscale %s", stats.render())
    return (
        UpscaleResult(
            frames_dir=frames_out,
            frame_count=count_frames_on_disk(frames_out, frame_format),
            scale=scale,
            model=model,
            tile=result.tile,
        ),
        stats,
    )


__all__ = [
    "DedupError",
    "DedupStats",
    "FrameCache",
    "FrameIndex",
    "index_frames",
    "link_or_copy",
    "pixel_hashes",
    "trim_caches",
    "upscale_deduplicated",
]
