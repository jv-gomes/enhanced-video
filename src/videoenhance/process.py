"""The single entry point for every external command this package runs.

Nothing else in the project should call :mod:`subprocess` directly: routing all
calls through :func:`run` means every command is logged the same way, every
failure carries the tool's own error output, and no call can accidentally be
made with ``shell=True``.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
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


def _set_suspended(proc: subprocess.Popen[str], suspended: bool) -> None:
    """Pause or resume ``proc``, ignoring a process that has already exited."""
    try:
        if sys.platform == "win32":
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x0800, False, proc.pid)  # SUSPEND_RESUME
            if handle:
                ntdll = ctypes.windll.ntdll
                (ntdll.NtSuspendProcess if suspended else ntdll.NtResumeProcess)(handle)
                ctypes.windll.kernel32.CloseHandle(handle)
        else:
            os.kill(proc.pid, signal.SIGSTOP if suspended else signal.SIGCONT)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _throttle(proc: subprocess.Popen[str], load: int, stop: threading.Event) -> None:
    """Let ``proc`` run ``load`` percent of the time by pausing it in between."""
    from .config import THROTTLE_CYCLE

    run_for = THROTTLE_CYCLE * load / 100
    pause_for = THROTTLE_CYCLE - run_for
    while not stop.wait(run_for):
        _set_suspended(proc, True)
        stop.wait(pause_for)
        _set_suspended(proc, False)
    _set_suspended(proc, False)


def run(
    cmd: Sequence[object],
    *,
    capture: bool = True,
    check: bool = True,
    cwd: Path | None = None,
    timeout: float | None = None,
    load: int = 100,
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
        load: Percent of the time the child may run, 1-100. Below 100 it is
            suspended for the rest of each short cycle, which is how the GPU
            tools are kept from saturating the card.

    Raises:
        MissingToolError: The executable does not exist.
        ToolError: The command exited non-zero and ``check`` is true, or it
            exceeded ``timeout``.
    """
    argv = [str(part) for part in cmd]
    if not argv:
        raise ValueError("run() needs at least a program name")

    if not 1 <= load <= 100:
        raise ValueError("load must be between 1 and 100")

    logger.debug("running: %s", format_cmd(argv))

    try:
        if load < 100:
            completed = _run_throttled(argv, load, capture, cwd, timeout)
        else:
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


def _run_throttled(
    argv: list[str], load: int, capture: bool, cwd: Path | None, timeout: float | None
) -> subprocess.CompletedProcess[str]:
    pipe = subprocess.PIPE if capture else None
    proc = subprocess.Popen(  # noqa: S603 - argv list, never shell=True
        argv, stdout=pipe, stderr=pipe, text=True, cwd=str(cwd) if cwd else None
    )
    stop = threading.Event()
    thread = threading.Thread(target=_throttle, args=(proc, load, stop), daemon=True)
    thread.start()
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop.set()
        thread.join()
        proc.kill()
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(argv, timeout, output=stdout, stderr=stderr) from None
    except BaseException:
        stop.set()
        thread.join()
        proc.kill()
        raise
    finally:
        stop.set()
        thread.join()
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)


def which(program: str) -> Path | None:
    """Locate ``program`` on ``PATH``, or return ``None``."""
    found = shutil.which(program)
    return Path(found) if found else None
