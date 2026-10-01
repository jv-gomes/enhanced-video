"""Tests for the ffprobe metadata reader."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from videoenhance.probe import ProbeError, VideoInfo, count_frames, probe

from .conftest import requires_ffmpeg


def _info(**overrides) -> VideoInfo:
    defaults = dict(
        path=Path("sample.mp4"),
        width=1920,
        height=1080,
        r_frame_rate=Fraction(30),
        avg_frame_rate=Fraction(30),
        nb_frames=90,
        duration=3.0,
        has_audio=True,
    )
    return VideoInfo(**{**defaults, **overrides})


@requires_ffmpeg
def test_probe_reads_sample(sample_video: Path):
    info = probe(sample_video)
    assert (info.width, info.height) == (320, 240)
    assert info.r_frame_rate == Fraction(30)
    assert info.nb_frames == 90
    assert info.duration == pytest.approx(3.0, abs=0.05)
    assert info.has_audio is True
    assert info.is_vfr is False
    assert info.codec == "h264"


@requires_ffmpeg
def test_probe_detects_missing_audio(silent_video: Path):
    info = probe(silent_video)
    assert info.has_audio is False
    assert info.r_frame_rate == Fraction(30000, 1001)
    assert info.fps == pytest.approx(29.97, abs=0.01)


@requires_ffmpeg
def test_probe_describe_mentions_resolution(sample_video: Path):
    assert "320x240" in probe(sample_video).describe()


@requires_ffmpeg
def test_count_frames_matches_container(sample_video: Path):
    assert count_frames(sample_video) == 90


def test_probe_missing_file():
    with pytest.raises(FileNotFoundError):
        probe("definitely-not-a-video.mp4")


@requires_ffmpeg
def test_probe_rejects_non_video(tmp_path: Path):
    text = tmp_path / "notes.txt"
    text.write_text("not a video")
    with pytest.raises((ProbeError, Exception)):
        probe(text)


def test_is_vfr_compares_rates_as_fractions():
    assert _info(r_frame_rate=Fraction(30), avg_frame_rate=Fraction(30)).is_vfr is False
    # 30000/1001 and 30 are close as floats but are not the same rate.
    assert _info(avg_frame_rate=Fraction(30000, 1001)).is_vfr is True
    # A missing average rate is not evidence of VFR.
    assert _info(avg_frame_rate=Fraction(0)).is_vfr is False


def test_pixels_and_fps_helpers():
    info = _info()
    assert info.pixels == 1920 * 1080
    assert info.fps == 30.0


def test_describe_flags_vfr():
    assert "VFR" in _info(avg_frame_rate=Fraction(30000, 1001)).describe()
