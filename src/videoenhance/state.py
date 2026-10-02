"""The record of an unfinished job, so it can be picked up days later.

Resuming works by looking at the files: a part that exists is a part that is
done, a stage directory with the right number of frames is a stage that ran.
That is the right way round — the files are the work — but it cannot answer two
questions. *With which settings* was this started, and *what* did I leave
unfinished?

Without the first, resuming silently corrupts the result. Every stage guard is a
frame count (see :func:`videoenhance.workdir.stage_is_done`), and the frame
count does not change when ``--scale`` or ``--model`` does, so a job
interrupted at 2x and resumed at 4x skips the upscale and produces a 2x video
while the CLI says 4x. This module writes down the intent so that case can be
refused instead.

So: **the state file holds what cannot be derived, and nothing else.** The
settings, the input, the output, the cut. Progress is read back off the disk
every time, because a recorded "18 parts done" is a number that can disagree
with reality, and a resume that lies is worse than one that is slow.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .workdir import WorkDir, count_frames_on_disk

logger = logging.getLogger(__name__)

#: Name of the record, inside the job's work directory.
STATE_NAME = "state.json"
#: Bumped when the fields change shape. An unknown version is treated as an
#: unreadable file: the job is restarted rather than misread.
STATE_VERSION = 1

#: Options that change the pixels of the output, and therefore may not change
#: between a run and its resume. ``gpu``, ``tile``, ``threads``, ``progress``
#: and ``keep_temp`` are deliberately absent: they change how the work is done,
#: not what comes out, so resuming with different ones is legitimate.
PIXEL_OPTIONS = (
    "scale",
    "target_fps",
    "model",
    "rife_model",
    "frame_format",
    "order",
    "chunk_seconds",
)

#: How each option is spelled on the command line, for the mismatch message.
OPTION_FLAGS = {
    "scale": "--scale",
    "target_fps": "--fps",
    "model": "--model",
    "rife_model": "--rife-model",
    "frame_format": "--frame-format",
    "order": "--order",
    "chunk_seconds": "--chunk",
}


class StateError(RuntimeError):
    """A job could not be found, or the name given matches more than one."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class Job:
    """What one unfinished job remembers about itself."""

    id: str
    root: Path
    input: Path
    output: Path
    #: The subset of the options in :data:`PIXEL_OPTIONS`.
    options: dict[str, object]
    #: Size and modification time of the input when the job started, so the
    #: same path holding a different file is not mistaken for the same work.
    input_size: int = 0
    input_mtime_ns: int = 0
    #: The chunked cut: names in playback order, and the length asked for.
    chunks: list[str] = field(default_factory=list)
    seconds: float = 0.0
    #: The encoder the first part settled on. Every later part is asked for it
    #: by name, so the parts can be joined by stream copy.
    encoder: str | None = None
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)

    @property
    def path(self) -> Path:
        return self.root / STATE_NAME

    @property
    def chunked(self) -> bool:
        return bool(self.chunks)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": STATE_VERSION,
            "id": self.id,
            "input": str(self.input),
            "output": str(self.output),
            "input_size": self.input_size,
            "input_mtime_ns": self.input_mtime_ns,
            "options": dict(self.options),
            "chunks": list(self.chunks),
            "seconds": self.seconds,
            "encoder": self.encoder,
            "created": self.created,
            "updated": self.updated,
        }


def for_input(
    work: WorkDir,
    input_path: Path,
    output: Path,
    options: dict[str, object],
) -> Job:
    """A fresh record for a job about to start."""
    stat = input_path.stat() if input_path.exists() else None
    return Job(
        id=work.root.name,
        root=work.root,
        input=input_path.resolve(),
        output=Path(output).resolve(),
        options={name: options[name] for name in PIXEL_OPTIONS if name in options},
        input_size=stat.st_size if stat else 0,
        input_mtime_ns=stat.st_mtime_ns if stat else 0,
    )


def options_of(options: object) -> dict[str, object]:
    """Pull the pixel-affecting settings out of an ``Options`` instance."""
    return {name: getattr(options, name) for name in PIXEL_OPTIONS if hasattr(options, name)}


def load(root: Path) -> Job | None:
    """Read the job recorded in ``root``, or ``None`` if there is none.

    An unreadable or future-version file is reported as absent rather than
    raising: the worst case is redoing work, and refusing to run because a
    scratch file got truncated would be worse than that.
    """
    path = Path(root) / STATE_NAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("ignoring unreadable job record %s", path)
        return None
    if data.get("version") != STATE_VERSION:
        logger.warning(
            "ignoring job record %s: it was written by another version", path
        )
        return None
    try:
        return Job(
            id=str(data["id"]),
            root=Path(root),
            input=Path(data["input"]),
            output=Path(data["output"]),
            options=dict(data.get("options") or {}),
            input_size=int(data.get("input_size") or 0),
            input_mtime_ns=int(data.get("input_mtime_ns") or 0),
            chunks=[str(name) for name in data.get("chunks") or []],
            seconds=float(data.get("seconds") or 0.0),
            encoder=data.get("encoder") or None,
            created=str(data.get("created") or ""),
            updated=str(data.get("updated") or ""),
        )
    except (KeyError, TypeError, ValueError):
        logger.warning("ignoring job record %s: unexpected contents", path)
        return None


def save(job: Job) -> Job:
    """Write ``job`` out, stamping it as updated now.

    Written to a temporary name and renamed, so an interrupt during the write
    cannot leave a half-written record where a valid one used to be.
    """
    job = replace(job, updated=_now())
    job.root.mkdir(parents=True, exist_ok=True)
    partial = job.path.with_suffix(".json.tmp")
    partial.write_text(json.dumps(job.to_dict(), indent=2) + "\n")
    partial.replace(job.path)
    logger.debug("recorded job %s in %s", job.id, job.path)
    return job


def record_cut(job: Job, chunks: list[Path], seconds: float) -> Job:
    """Store the chunked cut on the job and save it."""
    return save(replace(job, chunks=[chunk.name for chunk in chunks], seconds=seconds))


def record_encoder(job: Job, encoder: str) -> Job:
    """Pin the encoder the first part used, so later parts match it."""
    if job.encoder == encoder:
        return save(job)
    return save(replace(job, encoder=encoder))


def list_jobs(base: Path | None = None) -> list[Job]:
    """Every recorded job under ``base``, most recently touched first."""
    base = Path(base or config.WORK_DIR)
    if not base.is_dir():
        return []
    jobs = [job for child in sorted(base.iterdir()) if child.is_dir() if (job := load(child))]
    return sorted(jobs, key=lambda job: job.updated, reverse=True)


def find(name: str, base: Path | None = None) -> Job:
    """Look up a job by id, or by any prefix of one that is unambiguous.

    Raises:
        StateError: nothing matches, or more than one thing does.
    """
    jobs = list_jobs(base)
    if not jobs:
        raise StateError("there are no saved jobs to resume")

    exact = [job for job in jobs if job.id == name]
    if exact:
        return exact[0]

    matches = [job for job in jobs if job.id.startswith(name)]
    if not matches:
        known = ", ".join(job.id for job in jobs)
        raise StateError(f"no saved job matches {name!r}; there is {known}")
    if len(matches) > 1:
        candidates = ", ".join(job.id for job in matches)
        raise StateError(f"{name!r} matches more than one job: {candidates}")
    return matches[0]


def compare(job: Job, options: dict[str, object]) -> dict[str, tuple[object, object]]:
    """Which pixel-affecting options differ, as ``{name: (recorded, asked)}``.

    Only names present in ``options`` are compared, so a caller that knows the
    user spelled out two of them can ask about just those.
    """
    changed: dict[str, tuple[object, object]] = {}
    for name, asked in options.items():
        if name not in PIXEL_OPTIONS or name not in job.options:
            continue
        recorded = job.options[name]
        if _differs(recorded, asked):
            changed[name] = (recorded, asked)
    return changed


def _differs(recorded: object, asked: object) -> bool:
    """Compare two option values, tolerating ``2`` versus ``2.0``."""
    if isinstance(recorded, (int, float)) and isinstance(asked, (int, float)):
        return float(recorded) != float(asked)
    return recorded != asked


def input_changed(job: Job) -> bool:
    """Whether the input file is not the one the job was started on."""
    if not job.input.exists():
        return False
    if not job.input_size and not job.input_mtime_ns:
        return False
    stat = job.input.stat()
    return (stat.st_size, stat.st_mtime_ns) != (job.input_size, job.input_mtime_ns)


def parts_done(job: Job) -> int:
    """How many encoded parts are on disk, which is the real progress."""
    parts = WorkDir(job.root).parts
    if not parts.is_dir():
        return 0
    return sum(1 for _ in parts.glob("part_*.mkv"))


def progress(job: Job) -> str:
    """A short, disk-derived summary of how far the job got."""
    if job.chunked:
        return f"{parts_done(job)}/{len(job.chunks)} parts"

    work = WorkDir(job.root)
    frame_format = str(job.options.get("frame_format") or config.DEFAULT_FRAME_FORMAT)
    # Report the furthest stage that has anything in it; that is as much as can
    # be said without re-deriving every stage's expected count.
    for name, directory in (
        ("interpolated", work.frames_out),
        ("upscaled", work.frames_up),
        ("extracted", work.frames_in),
    ):
        count = count_frames_on_disk(directory, frame_format)
        if count:
            return f"{count} frames {name}"
    return "not started"


def human_age(stamp: str, now: datetime | None = None) -> str:
    """How long ago ``stamp`` was, in words."""
    if not stamp:
        return "unknown"
    try:
        when = datetime.fromisoformat(stamp)
    except ValueError:
        return "unknown"
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    seconds = ((now or datetime.now(timezone.utc)) - when).total_seconds()
    if seconds < 90:
        return "just now"
    for limit, size, unit in (
        (5400, 60, "minute"),       # up to 90 minutes
        (86400, 3600, "hour"),      # up to a day
        (1209600, 86400, "day"),    # up to a fortnight
        (float("inf"), 604800, "week"),
    ):
        if seconds < limit:
            value = int(seconds // size)
            return f"{value} {unit}{'s' if value != 1 else ''} ago"
    return "a long time ago"


def describe(jobs: list[Job]) -> str:
    """The ``--jobs`` table."""
    if not jobs:
        return "No unfinished jobs."

    rows = [
        (job.id, job.input.name, progress(job), human_age(job.updated), _settings(job))
        for job in jobs
    ]
    headers = ("JOB", "INPUT", "PROGRESS", "STOPPED", "SETTINGS")
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def line(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells)).rstrip()

    out = [line(headers), *(line(row) for row in rows)]
    out.append("")
    out.append(f"Resume one with --resume {jobs[0].id}, or discard it with --restart.")
    return "\n".join(out)


def _settings(job: Job) -> str:
    """The settings worth showing in the listing, shortest form that is useful."""
    parts = []
    if "scale" in job.options:
        parts.append(f"{job.options['scale']}x")
    if "target_fps" in job.options:
        parts.append(f"{float(job.options['target_fps'] or 0):g}fps")
    if job.options.get("chunk_seconds"):
        parts.append(f"chunk {float(job.options['chunk_seconds']):g}s")
    return ", ".join(parts)


def render_mismatch(job: Job, changed: dict[str, tuple[object, object]]) -> str:
    """The message shown when a resume asks for different settings.

    It names every difference, because the whole point is that the user cannot
    see from the outside that the parts on disk were made another way.
    """
    lines = ["this job was started with different options:"]
    width = max(len(OPTION_FLAGS.get(name, name)) for name in changed)
    for name, (recorded, asked) in sorted(changed.items()):
        flag = OPTION_FLAGS.get(name, name)
        lines.append(f"  {flag.ljust(width)}  {_show(recorded)} -> {_show(asked)}")

    if job.chunked:
        lines.append(
            f"{parts_done(job)} of {len(job.chunks)} parts already encoded use the old ones."
        )
    else:
        lines.append(f"the frames already in {job.root} use the old ones.")
    lines.append(
        "Use --restart to throw that away and start over, or run with the original options."
    )
    return "\n".join(lines)


def _show(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


__all__ = [
    "OPTION_FLAGS",
    "PIXEL_OPTIONS",
    "STATE_NAME",
    "Job",
    "StateError",
    "compare",
    "describe",
    "find",
    "for_input",
    "human_age",
    "input_changed",
    "list_jobs",
    "load",
    "options_of",
    "parts_done",
    "progress",
    "record_cut",
    "record_encoder",
    "render_mismatch",
    "save",
]
