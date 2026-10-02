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


@pytest.fixture(scope="session")
def vfr_video() -> Path:
    """2 seconds of 160x120 with variable frame timing and audio.

    Built by dropping two of every three frames and rewriting the
    presentation timestamps, which is what makes r_frame_rate and
    avg_frame_rate disagree the way a phone recording does.
    """
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    return _generate(
        FIXTURE_DIR / "vfr.mp4",
        [
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440",
            "-t", "2",
            "-vf", "select='not(mod(n,3))',setpts=N/(10*TB)",
            "-fps_mode", "vfr",
            "-c:v", "libx264", "-c:a", "aac",
        ],
    )


@pytest.fixture(scope="session")
def keyframed_video() -> Path:
    """4 seconds of 160x120 at 30 fps with a keyframe every half second.

    Frequent keyframes are what let the chunk cut happen by stream copy, so
    this is the fixture for the ordinary path through ``chunk.split``.
    """
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    return _generate(
        FIXTURE_DIR / "keyframed.mp4",
        [
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440",
            "-t", "4",
            "-c:v", "libx264", "-g", "15", "-c:a", "aac",
        ],
    )


@pytest.fixture(scope="session")
def longgop_video() -> Path:
    """4 seconds of 160x120 whose only keyframe is the first one.

    A stream-copy cut cannot land anywhere inside this file, so it is what
    forces ``chunk.split`` down its lossless re-cut path.
    """
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    return _generate(
        FIXTURE_DIR / "longgop.mp4",
        [
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30",
            "-t", "4",
            "-c:v", "libx264", "-g", "1000", "-sc_threshold", "0",
        ],
    )


def write_stub(path: Path, body: str) -> Path:
    """Write an executable stand-in for an NCNN binary.

    The real executables need a GPU and a model download, so the wrappers are
    exercised against scripts that accept the same arguments. That still tests
    what the wrappers are responsible for: the argument list, the retry
    behaviour and how the output directory is judged.
    """
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)
    return path


#: Records every argument list it is called with, one per line, then copies the
#: input frames to the output directory. A numeric ``-n`` (RIFE's output frame
#: count; Real-ESRGAN's ``-n`` is a model name) is honoured by duplicating the
#: last frame, so the stub returns as many frames as the real binary would.
STUB_RECORDING = """
set -euo pipefail
echo "$@" >> "$ARGV_LOG"
IN=""; OUT=""; FMT="png"; WANT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i) IN="$2"; shift 2;;
    -o) OUT="$2"; shift 2;;
    -f) FMT="${2##*.}"; shift 2;;
    -n) [[ "$2" =~ ^[0-9]+$ ]] && WANT="$2"; shift 2;;
    *) shift;;
  esac
done
mkdir -p "$OUT"
count=0
for f in "$IN"/*."$FMT"; do
  [[ -e "$f" ]] || continue
  count=$((count + 1))
  cp "$f" "$OUT/$(printf '%08d' $count).$FMT"
done
[[ $count -gt 0 ]] || exit 0
last="$OUT/$(printf '%08d' $count).$FMT"
while [[ $count -lt $WANT ]]; do
  count=$((count + 1))
  cp "$last" "$OUT/$(printf '%08d' $count).$FMT"
done
"""


@pytest.fixture
def stub_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Factory for stub NCNN binaries, with a log of how they were called."""
    argv_log = tmp_path / "argv.log"
    argv_log.touch()
    monkeypatch.setenv("ARGV_LOG", str(argv_log))

    def make(env_var: str, body: str = STUB_RECORDING, name: str = "stub-ncnn") -> Path:
        path = write_stub(tmp_path / name, body)
        monkeypatch.setenv(env_var, str(path))
        return path

    make.log = argv_log  # type: ignore[attr-defined]
    return make


@pytest.fixture
def frames(tmp_path: Path):
    """Factory for a directory of placeholder frame files."""

    def make(count: int, name: str = "frames_in", ext: str = "png") -> Path:
        directory = tmp_path / name
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(1, count + 1):
            (directory / f"{index:08d}.{ext}").write_bytes(b"frame")
        return directory

    return make
