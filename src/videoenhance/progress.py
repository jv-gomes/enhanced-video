"""Per-stage progress bars, driven by the frames on disk.

The NCNN binaries and FFmpeg each report progress in their own format, and
neither is worth parsing: every stage writes numbered files into a directory, so
counting those files is an accurate, tool-independent measure of how far along
it is. A background thread polls the directory while the stage runs and feeds a
:mod:`tqdm` bar.

The bar is a convenience, never a requirement: if tqdm is missing or the output
is not a terminal, :func:`track` quietly does nothing, and nothing else in the
pipeline has to care.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from . import config
from .workdir import count_frames_on_disk

logger = logging.getLogger(__name__)

#: How often the watcher thread recounts the output directory. Frequent enough
#: to look alive, rare enough that the directory listing costs nothing next to
#: the GPU work it is measuring.
POLL_SECONDS = 0.5


def available() -> bool:
    """Whether a progress bar can be drawn at all."""
    try:
        import tqdm  # noqa: F401
    except ImportError:
        return False
    return True


def wanted(stream: object | None = None) -> bool:
    """Whether a progress bar should be drawn by default.

    True for an interactive terminal with tqdm installed. A redirected stream
    gets no bar, so logs and CI output stay readable.
    """
    stream = stream or sys.stderr
    isatty = getattr(stream, "isatty", None)
    return available() and bool(isatty and isatty())


@contextmanager
def track(
    label: str,
    directory: Path,
    total: int,
    *,
    frame_format: str = config.DEFAULT_FRAME_FORMAT,
    enabled: bool = True,
    stream: object | None = None,
) -> Iterator[None]:
    """Show a bar for ``label`` while ``directory`` fills up to ``total`` frames.

    Args:
        label: Stage name, shown as the bar's description.
        directory: The directory the stage writes its frames into.
        total: How many frames the stage is expected to produce. A total of
            zero means the size is unknown, and no bar is drawn.
        frame_format: Extension of the frames to count.
        enabled: Set to False to skip the bar entirely.
        stream: Where to draw the bar. Defaults to standard error, so the bar
            never lands in a redirected stdout.
    """
    if not enabled or total <= 0 or not available():
        yield
        return

    from tqdm import tqdm

    # Frames already on disk belong to an earlier, interrupted run: count them
    # as done so a resumed stage does not start its bar from zero.
    start = count_frames_on_disk(directory, frame_format)
    done = threading.Event()

    with tqdm(
        total=total,
        initial=min(start, total),
        desc=label,
        unit="frame",
        leave=False,
        file=stream or sys.stderr,
        # The watcher only reports every POLL_SECONDS, so every update it does
        # make is worth drawing; tqdm's default would swallow some of them.
        mininterval=0,
    ) as bar:

        def watch() -> None:
            while not done.wait(POLL_SECONDS):
                _advance(bar, directory, total, frame_format)

        watcher = threading.Thread(target=watch, name=f"progress-{label}", daemon=True)
        watcher.start()
        try:
            yield
        finally:
            done.set()
            watcher.join(timeout=POLL_SECONDS * 2)
            # One last count, so a stage that finished between two polls does
            # not leave its bar short of the total it actually reached.
            _advance(bar, directory, total, frame_format)


def _advance(bar: object, directory: Path, total: int, frame_format: str) -> None:
    """Move ``bar`` to the number of frames currently on disk."""
    try:
        current = min(count_frames_on_disk(directory, frame_format), total)
        bar.update(current - bar.n)  # type: ignore[attr-defined]
    except OSError as exc:  # pragma: no cover - a vanished directory is not fatal
        logger.debug("progress for %s stopped: %s", directory, exc)
