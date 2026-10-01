"""Tests for the central subprocess helper."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from videoenhance.process import (
    MissingToolError,
    ToolError,
    format_cmd,
    run,
    which,
)


def test_run_returns_stdout():
    result = run([sys.executable, "-c", "print('hello')"])
    assert result.stdout.strip() == "hello"
    assert result.returncode == 0


def test_run_accepts_path_arguments(tmp_path: Path):
    target = tmp_path / "note.txt"
    target.write_text("content")
    result = run([sys.executable, "-c", "import sys; print(open(sys.argv[1]).read())", target])
    assert "content" in result.stdout


def test_run_raises_with_stderr_tail():
    with pytest.raises(ToolError) as excinfo:
        run([sys.executable, "-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(3)"])
    error = excinfo.value
    assert error.returncode == 3
    assert "boom" in str(error)
    assert "exit code 3" in str(error)


def test_run_check_false_does_not_raise():
    result = run([sys.executable, "-c", "import sys; sys.exit(4)"], check=False)
    assert result.returncode == 4


def test_run_missing_binary():
    with pytest.raises(MissingToolError) as excinfo:
        run(["videoenhance-does-not-exist"])
    assert "was not found" in str(excinfo.value)
    assert excinfo.value.returncode == 127


def test_run_timeout_raises_tool_error():
    with pytest.raises(ToolError) as excinfo:
        run([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.3)
    assert "timed out" in str(excinfo.value)


def test_run_rejects_empty_command():
    with pytest.raises(ValueError):
        run([])


def test_format_cmd_quotes_spaces():
    assert format_cmd(["ffmpeg", "-i", Path("/tmp/my file.mp4")]) == 'ffmpeg -i "/tmp/my file.mp4"'


def test_which_finds_the_interpreter():
    assert which(Path(sys.executable).name) is not None
    assert which("videoenhance-does-not-exist") is None
