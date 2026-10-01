"""Tests for the frame-counting progress bars."""

from __future__ import annotations

import io
from pathlib import Path

from videoenhance import progress


class NotATerminal(io.StringIO):
    def isatty(self) -> bool:
        return False


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_a_redirected_stream_gets_no_bar():
    assert progress.wanted(NotATerminal()) is False


def test_a_terminal_gets_a_bar_when_tqdm_is_installed():
    assert progress.wanted(Terminal()) is progress.available()


def test_disabled_tracking_draws_nothing(tmp_path: Path):
    stream = Terminal()
    with progress.track("upscale", tmp_path, 10, enabled=False, stream=stream):
        pass
    assert stream.getvalue() == ""


def test_an_unknown_total_draws_nothing(tmp_path: Path):
    """A total of zero means nobody knows the size, which is not a bar."""
    stream = Terminal()
    with progress.track("upscale", tmp_path, 0, stream=stream):
        pass
    assert stream.getvalue() == ""


def test_the_bar_follows_the_frames_on_disk(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(progress, "POLL_SECONDS", 0.01)
    stream = Terminal()
    with progress.track("upscale", tmp_path, 4, stream=stream):
        for index in range(1, 5):
            (tmp_path / f"{index:08d}.png").write_bytes(b"frame")
    rendered = stream.getvalue()
    assert "upscale" in rendered
    assert "4/4" in rendered


def test_frames_from_an_earlier_run_start_the_bar_part_way(tmp_path: Path):
    """A resumed stage should not pretend it is starting from nothing."""
    for index in range(1, 4):
        (tmp_path / f"{index:08d}.png").write_bytes(b"frame")
    stream = Terminal()
    with progress.track("interpolate", tmp_path, 6, stream=stream):
        pass
    assert "3/6" in stream.getvalue()


def test_only_the_right_frame_format_is_counted(tmp_path: Path):
    for index in range(1, 3):
        (tmp_path / f"{index:08d}.jpg").write_bytes(b"frame")
    (tmp_path / "00000003.png").write_bytes(b"frame")
    stream = Terminal()
    with progress.track("upscale", tmp_path, 4, frame_format="jpg", stream=stream):
        pass
    assert "2/4" in stream.getvalue()
