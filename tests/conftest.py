"""Shared fixtures.

The test suite needs a real video file but should not carry one in git, so the
fixture is generated with FFmpeg's ``lavfi`` sources on first use and cached in
``tests/fixtures/`` (which is gitignored). Everything that needs FFmpeg is
skipped, not failed, when it is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


def _generate(path: Path, args: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", *args, str(path)],
            check=True,
        )
    return path


@pytest.fixture(scope="session")
def sample_video() -> Path:
    """3 seconds of 320x240 CFR 30 fps test pattern with a 440 Hz tone."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    return _generate(
        FIXTURE_DIR / "sample.mp4",
        [
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440",
            "-t", "3",
            "-c:v", "libx264", "-c:a", "aac",
        ],
    )


@pytest.fixture(scope="session")
def silent_video() -> Path:
    """1 second of 64x48 video at 30000/1001 fps and no audio stream."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    return _generate(
        FIXTURE_DIR / "silent.mp4",
        [
            "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30000/1001",
            "-t", "1",
            "-c:v", "libx264",
        ],
    )
