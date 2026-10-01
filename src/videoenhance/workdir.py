"""The work directory, the resume rule and the disk estimate.

These are the primitives every stage shares: where frames live, how to tell a
finished stage from an interrupted one, and what the frames will cost on disk.
:mod:`videoenhance.pipeline` drives the stages on top of them.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

#: Stage directory names inside a work directory.
FRAMES_IN = "frames_in"
FRAMES_UP = "frames_up"
FRAMES_OUT = "frames_out"
#: CFR-normalised copy of a VFR input; also the audio source in that case.
CFR_NAME = "cfr.mkv"


def frame_glob(frame_format: str = config.DEFAULT_FRAME_FORMAT) -> str:
    """Glob matching the frames a stage writes, e.g. ``*.png``."""
    return f"*.{frame_format}"


def frame_pattern(frame_format: str = config.DEFAULT_FRAME_FORMAT) -> str:
    """FFmpeg/NCNN filename pattern, e.g. ``%08d.png``."""
    return f"{config.FRAME_PATTERN}.{frame_format}"


def count_frames_on_disk(
    directory: Path, frame_format: str = config.DEFAULT_FRAME_FORMAT
) -> int:
    """How many frames ``directory`` currently holds."""
    if not directory.is_dir():
        return 0
    return sum(1 for _ in directory.glob(frame_glob(frame_format)))


def frames_complete(
    directory: Path,
    expected: int,
    frame_format: str = config.DEFAULT_FRAME_FORMAT,
) -> bool:
    """Whether ``directory`` already holds every expected frame.

    An ``expected`` count of zero means "unknown", which can never count as
    complete: guessing wrong here would silently skip real work.
    """
    if expected <= 0:
        return False
    return count_frames_on_disk(directory, frame_format) >= expected


def stage_is_done(
    name: str,
    directory: Path,
    expected: int,
    frame_format: str = config.DEFAULT_FRAME_FORMAT,
) -> bool:
    """Like :func:`frames_complete`, but says out loud that it is skipping."""
    if frames_complete(directory, expected, frame_format):
        logger.info("skipping %s: %s already holds %d frames", name, directory, expected)
        return True
    return False


@dataclass(frozen=True)
class WorkDir:
    """Scratch space for one input file.

    Attributes:
        root: The per-input directory, e.g. ``work/clip-1a2b3c4d/``.
    """

    root: Path

    @classmethod
    def for_input(cls, input_path: Path, base: Path | None = None) -> WorkDir:
        """Derive the work directory for ``input_path``.

        The name carries the file stem for readability plus a short digest of
        the resolved path, so two different files with the same name do not
        share a directory and the same file always maps back to its own
        frames, which is what makes resuming work across runs.
        """
        base = base or config.WORK_DIR
        digest = hashlib.sha256(str(input_path.resolve()).encode()).hexdigest()[:8]
        stem = "".join(c if c.isalnum() or c in "-_" else "-" for c in input_path.stem)[:40]
        return cls(root=base / f"{stem or 'video'}-{digest}")

    @property
    def frames_in(self) -> Path:
        """Frames extracted from the source."""
        return self.root / FRAMES_IN

    @property
    def frames_up(self) -> Path:
        """Frames after Real-ESRGAN."""
        return self.root / FRAMES_UP

    @property
    def frames_out(self) -> Path:
        """Frames after RIFE, i.e. what gets encoded."""
        return self.root / FRAMES_OUT

    @property
    def cfr_video(self) -> Path:
        """CFR-normalised copy of a VFR source."""
        return self.root / CFR_NAME

    @property
    def stage_dirs(self) -> tuple[Path, Path, Path]:
        return (self.frames_in, self.frames_up, self.frames_out)

    def create(self) -> WorkDir:
        """Create the directory tree, reusing whatever is already there."""
        for directory in (self.root, *self.stage_dirs):
            directory.mkdir(parents=True, exist_ok=True)
        logger.debug("work directory ready: %s", self.root)
        return self

    def cleanup(self, keep: bool = False) -> bool:
        """Remove the work directory, unless ``keep`` is set.

        Returns whether the directory is gone afterwards. Failing to delete it
        is never fatal — the output video is already written — but it is worth
        saying, because the frames can be tens of gigabytes.
        """
        if keep:
            logger.info("keeping work directory %s", self.root)
            return False
        if not self.root.exists():
            return True
        shutil.rmtree(self.root, ignore_errors=True)
        if self.root.exists():
            logger.warning(
                "could not fully remove %s; delete it by hand to reclaim the space",
                self.root,
            )
            return False
        logger.debug("removed work directory %s", self.root)
        return True

    def free_bytes(self) -> int:
        """Free space on the filesystem that will hold the frames."""
        probe_dir = self.root
        while not probe_dir.exists() and probe_dir != probe_dir.parent:
            probe_dir = probe_dir.parent
        return shutil.disk_usage(probe_dir).free


#: Rough compressed size of one frame, in bytes per pixel. PNG of real video
#: content lands around 1.5 B/px for 8-bit RGB; JPEG at quality 2 is far
#: smaller. Deliberately generous: a warning that comes too early is cheap,
#: running out of disk three hours into a job is not.
BYTES_PER_PIXEL = {"png": 1.5, "jpg": 0.3}


def estimate_frame_bytes(
    width: int, height: int, frame_format: str = config.DEFAULT_FRAME_FORMAT
) -> int:
    """Estimated size of a single frame file."""
    return int(width * height * BYTES_PER_PIXEL.get(frame_format, 1.5))


@dataclass(frozen=True)
class SpaceEstimate:
    """What the intermediate frames are expected to cost on disk."""

    stages: dict[str, int]
    free: int

    @property
    def total(self) -> int:
        return sum(self.stages.values())

    @property
    def fits(self) -> bool:
        return self.free >= self.total

    def render(self) -> str:
        parts = ", ".join(f"{name} {human_bytes(size)}" for name, size in self.stages.items())
        return f"{human_bytes(self.total)} of frames ({parts}); {human_bytes(self.free)} free"


def human_bytes(size: float) -> str:
    """Format a byte count the way a person reads it."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def estimate_disk_usage(
    width: int,
    height: int,
    frame_count: int,
    *,
    scale: int,
    target_fps: float,
    source_fps: float,
    frame_format: str = config.DEFAULT_FRAME_FORMAT,
    order: str = config.DEFAULT_ORDER,
    free: int = 0,
) -> SpaceEstimate:
    """Estimate the peak disk cost of the intermediate frame directories.

    All three stage directories are counted, because a resumable pipeline
    cannot delete the input frames of a stage it may have to run again.

    The order matters to the total. Upscaling first stores two full-size
    directories, the upscaled source frames and the interpolated ones;
    interpolating first keeps its intermediate at the source resolution and so
    costs less disk, while paying for it in GPU time, because Real-ESRGAN then
    has the multiplied frame count to work through.
    """
    per_source = estimate_frame_bytes(width, height, frame_format)
    per_upscaled = estimate_frame_bytes(width * scale, height * scale, frame_format)
    interpolated = frame_count
    if source_fps > 0 and target_fps > source_fps:
        interpolated = int(frame_count * (target_fps / source_fps))

    if order == config.ORDER_INTERPOLATE_FIRST:
        # frames_in -> frames_out (RIFE, source resolution) -> frames_up.
        stages = {
            FRAMES_IN: per_source * frame_count,
            FRAMES_OUT: per_source * interpolated,
            FRAMES_UP: per_upscaled * interpolated,
        }
    else:
        # frames_in -> frames_up (Real-ESRGAN, source frame count) -> frames_out.
        stages = {
            FRAMES_IN: per_source * frame_count,
            FRAMES_UP: per_upscaled * frame_count,
            FRAMES_OUT: per_upscaled * interpolated,
        }

    return SpaceEstimate(stages=stages, free=free)
