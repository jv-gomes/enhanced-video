"""The single entry point for every external command this package runs.

Nothing else in the project should call :mod:`subprocess` directly: routing all
calls through :func:`run` means every command is logged the same way, every
failure carries the tool's own error output, and no call can accidentally be
made with ``shell=True``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

#: How many trailing lines of stderr to quote in a ToolError message.
STDERR_TAIL_LINES = 15


class ToolError(RuntimeError):
    """An external command exited with a non-zero status."""

    def __init__(self, cmd: Sequence[str], returncode: int, stderr: str = "") -> None:
        self.cmd = [str(part) for part in cmd]
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(self._message())

    def _message(self) -> str:
        lines = [
            f"{Path(self.cmd[0]).name} failed with exit code {self.returncode}",
            f"  command: {format_cmd(self.cmd)}",
        ]
        tail = _tail(self.stderr, STDERR_TAIL_LINES)
        if tail:
            lines.append("  output:")
            lines.extend(f"    {line}" for line in tail)
        return "\n".join(lines)


class MissingToolError(ToolError):
    """An external command could not be found at all."""

    def __init__(self, cmd: Sequence[str]) -> None:
        super().__init__(cmd, returncode=127)

    def _message(self) -> str:
        return (
            f"{Path(self.cmd[0]).name} was not found. "
            "Install it or point the matching environment variable at it "
            "(see README, 'Setup')."
        )


def format_cmd(cmd: Iterable[object]) -> str:
    """Render a command as a copy-pasteable string, for logs and errors only."""
    parts = []
    for part in cmd:
        text = str(part)
        parts.append(f'"{text}"' if " " in text else text)
    return " ".join(parts)


def _tail(text: str, limit: int) -> list[str]:
    lines = [line.rstrip() for line in (text or "").splitlines() if line.strip()]
    return lines[-limit:]


def run(
    cmd: Sequence[object],
    *,
    capture: bool = True,
    check: bool = True,
    cwd: Path | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` and return the completed process.

    Args:
        cmd: The command as an argument list. Every element is coerced to
            ``str``, so :class:`~pathlib.Path` arguments can be passed as-is.
        capture: Capture stdout and stderr instead of letting the child write
            to the terminal. Pass ``False`` for long-running tools whose
            progress output the user should see live.
        check: Raise :class:`ToolError` on a non-zero exit code.
        cwd: Working directory for the child process.
        timeout: Seconds to wait before killing the child.

    Raises:
        MissingToolError: The executable does not exist.
        ToolError: The command exited non-zero and ``check`` is true, or it
            exceeded ``timeout``.
    """
    argv = [str(part) for part in cmd]
    if not argv:
        raise ValueError("run() needs at least a program name")

    logger.debug("running: %s", format_cmd(argv))

    try:
        completed = subprocess.run(  # noqa: S603 - argv list, never shell=True
            argv,
            capture_output=capture,
            text=True,
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MissingToolError(argv) from exc
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        detail = stderr or f"timed out after {timeout}s"
        raise ToolError(argv, returncode=-1, stderr=detail) from exc

    if check and completed.returncode != 0:
        raise ToolError(argv, completed.returncode, completed.stderr or "")

    return completed


def which(program: str) -> Path | None:
    """Locate ``program`` on ``PATH``, or return ``None``."""
    found = shutil.which(program)
    return Path(found) if found else None
