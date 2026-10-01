"""The work directory and the rules that make every stage resumable.

A long video takes hours, so a crash must not throw the work away. Each stage
writes its frames into its own directory under ``work/``, and a stage whose
directory already holds every frame it would produce is skipped on the next
run. The layout is keyed by the input file, so processing two videos in
sequence never mixes their frames.
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

    def cleanup(self, keep: bool = False) -> None:
        """Remove the work directory, unless ``keep`` is set."""
        if keep:
            logger.info("keeping work directory %s", self.root)
            return
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
            logger.debug("removed work directory %s", self.root)

    def free_bytes(self) -> int:
        """Free space on the filesystem that will hold the frames."""
        probe_dir = self.root
        while not probe_dir.exists() and probe_dir != probe_dir.parent:
            probe_dir = probe_dir.parent
        return shutil.disk_usage(probe_dir).free
