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


class Cancelled(RuntimeError):
    """The run is being stopped, so this command was refused or cut short.

    Deliberately *not* a :class:`ToolError`. Code that walks a fallback chain
    catches ``ToolError`` and tries the next tool, which is right for a tool that
    failed and wrong for one that was stopped: the encoder chain would otherwise
    answer a Ctrl-C by starting a whole new encode with the next encoder.
    """

    def __init__(self, cmd: Sequence[str]) -> None:
        self.cmd = [str(part) for part in cmd]
        super().__init__(f"{Path(self.cmd[0]).name} was stopped: the run is being cancelled")


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


#: Set while the run is being stopped; see :func:`cancel_all`.
_CANCELLED = threading.Event()
#: Children started by :func:`run` that have not exited yet.
_LIVE: set[subprocess.Popen[str]] = set()
_LIVE_LOCK = threading.Lock()


def cancel_all() -> None:
    """Stop every command :func:`run` started, and refuse to start new ones.

    A Ctrl-C reaches the main thread as KeyboardInterrupt, but a command running
    in another thread only sees its child die, which looks like any other
    failure. This is how the main thread tells those threads the difference:
    after it, :func:`run` raises :class:`Cancelled` instead of :class:`ToolError`
    and starts nothing. :func:`reset_cancel` turns it off again.
    """
    _CANCELLED.set()
    with _LIVE_LOCK:
        live = list(_LIVE)
    for proc in live:
        try:
            proc.terminate()
        except OSError:
            pass


def reset_cancel() -> None:
    """Allow :func:`run` to start commands again after :func:`cancel_all`."""
    _CANCELLED.clear()


def cancelling() -> bool:
    """Whether :func:`cancel_all` is in effect."""
    return _CANCELLED.is_set()


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
        Cancelled: :func:`cancel_all` was called before or while it ran.
    """
    argv = [str(part) for part in cmd]
    if not argv:
        raise ValueError("run() needs at least a program name")

    if not 1 <= load <= 100:
        raise ValueError("load must be between 1 and 100")

    logger.debug("running: %s", format_cmd(argv))
    if _CANCELLED.is_set():
        raise Cancelled(argv)

    try:
        completed = _spawn(argv, load, capture, cwd, timeout)
    except FileNotFoundError as exc:
        raise MissingToolError(argv) from exc
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        detail = stderr or f"timed out after {timeout}s"
        raise ToolError(argv, returncode=-1, stderr=detail) from exc

    if _CANCELLED.is_set():
        # Whatever the exit code, a command that ended while the run was being
        # stopped did not fail on its own, and must not be retried as if it had.
        raise Cancelled(argv)

    if check and completed.returncode != 0:
        raise ToolError(argv, completed.returncode, completed.stderr or "")

    return completed


def _spawn(
    argv: list[str], load: int, capture: bool, cwd: Path | None, timeout: float | None
) -> subprocess.CompletedProcess[str]:
    """Start ``argv``, keep it registered while it runs, and wait for it.

    Registered so that :func:`cancel_all` can reach it from another thread, and
    throttled below ``load`` 100 by a helper thread that pauses it.
    """
    pipe = subprocess.PIPE if capture else None
    proc = subprocess.Popen(  # noqa: S603 - argv list, never shell=True
        argv, stdout=pipe, stderr=pipe, text=True, cwd=str(cwd) if cwd else None
    )
    with _LIVE_LOCK:
        _LIVE.add(proc)
    stop = threading.Event()
    thread = None
    if load < 100:
        thread = threading.Thread(target=_throttle, args=(proc, load, stop), daemon=True)
        thread.start()
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop.set()
        if thread:
            thread.join()
        proc.kill()
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(argv, timeout, output=stdout, stderr=stderr) from None
    except BaseException:
        stop.set()
        if thread:
            thread.join()
        proc.kill()
        proc.wait()
        raise
    finally:
        stop.set()
        if thread:
            thread.join()
        with _LIVE_LOCK:
            _LIVE.discard(proc)
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)


def which(program: str) -> Path | None:
    """Locate ``program`` on ``PATH``, or return ``None``."""
    found = shutil.which(program)
    return Path(found) if found else None
