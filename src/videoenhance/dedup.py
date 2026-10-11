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
from dataclasses import dataclass
from pathlib import Path

from . import config
from .config import Binaries, binaries
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
    ) -> None:
        self.slot = Path(root) / f"{model}-{scale}x-t{tile}-{frame_format}"
        self.frame_format = frame_format
        self.budget = budget
        self.hits = 0

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
        if self.budget <= 0:
            return 0
        entries = sorted(self.entries(), key=lambda p: p.stat().st_mtime)
        total = self.size()
        removed = 0
        for entry in entries:
            if total <= self.budget:
                break
            size = entry.stat().st_size
            entry.unlink(missing_ok=True)
            total -= size
            removed += 1
        if removed:
            logger.info(
                "trimmed %d cached frames to stay under %s", removed, human_bytes(self.budget)
            )
        return removed


@dataclass(frozen=True)
class DedupStats:
    """What the hashing saved, for the run summary."""

    frames: int
    unique: int
    #: Distinct frames that were already in the cache from an earlier chunk.
    cached: int
    #: Distinct frames actually sent to the GPU.
    upscaled: int

    @property
    def saved(self) -> int:
        return self.frames - self.upscaled

    @property
    def ratio(self) -> float:
        return self.saved / self.frames if self.frames else 0.0

    def render(self) -> str:
        if not self.saved:
            return f"{self.frames} distinct frames, nothing to reuse"
        detail = f"{self.upscaled} of {self.frames} frames generated"
        if self.cached:
            detail += f", {self.cached} reused from earlier chunks"
        return f"{detail} ({self.ratio:.0%} saved)"


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
    """
    bins = bins or binaries()
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
    if missing:
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
            bins=bins,
            force=True,
        )
        for rank, digest in enumerate(missing, start=1):
            cache.put(digest, todo_up / f"{rank:08d}.{frame_format}")
        clear_frames(todo, frame_format)
        clear_frames(todo_up, frame_format)

    clear_frames(frames_out, frame_format)
    for position, digest in enumerate(index.hashes, start=1):
        cached = cache.get(digest)
        if cached is None:
            raise DedupError(
                f"frame {position} is missing from the cache after upscaling; "
                f"its hash is {digest}"
            )
        link_or_copy(cached, frames_out / f"{position:08d}.{frame_format}")

    cache.trim()
    stats = DedupStats(
        frames=index.frames,
        unique=len(unique),
        cached=len(unique) - len(missing),
        upscaled=len(missing),
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
    "upscale_deduplicated",
]
