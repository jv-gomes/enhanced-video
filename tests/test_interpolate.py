"""Tests for the RIFE wrapper and the frame-count arithmetic.

The frame count is the part worth testing hard: it is computed from two frame
rates, and getting it wrong does not crash, it just drifts the video against
the audio over the length of the clip.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from videoenhance import config
from videoenhance.interpolate import (
    InterpolateError,
    interpolate,
    interpolation_factor,
    is_needed,
    needs_uhd,
    target_frame_count,
)
from videoenhance.process import ToolError

# Writes exactly the number of frames it is told to, using the -f pattern.
STUB_RIFE = """
set -euo pipefail
echo "$@" >> "$ARGV_LOG"
IN=""; OUT=""; N="0"; FMT="%08d.png"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i) IN="$2"; shift 2;;
    -o) OUT="$2"; shift 2;;
    -n) N="$2"; shift 2;;
    -f) FMT="$2"; shift 2;;
    *) shift;;
  esac
done
mkdir -p "$OUT"
first=$(ls "$IN" | head -1)
for i in $(seq 1 "$N"); do cp "$IN/$first" "$OUT/$(printf "$FMT" "$i")"; done
"""

# Fails with a Vulkan allocation error whenever -u is passed.
STUB_UHD_OOM = """
ARGS="$*"
echo "$@" >> "$ARGV_LOG"
IN=""; OUT=""; N="0"; FMT="%08d.png"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i) IN="$2"; shift 2;;
    -o) OUT="$2"; shift 2;;
    -n) N="$2"; shift 2;;
    -f) FMT="$2"; shift 2;;
    *) shift;;
  esac
done
if [[ "$ARGS" == *" -u"* ]]; then
  echo "vkAllocateMemory failed: out of device memory" >&2
  exit 1
fi
mkdir -p "$OUT"
first=$(ls "$IN" | head -1)
for i in $(seq 1 "$N"); do cp "$IN/$first" "$OUT/$(printf "$FMT" "$i")"; done
"""

STUB_BROKEN = """
echo "$@" >> "$ARGV_LOG"
echo "invalid model dir" >&2
exit 1
"""

STUB_SHORT = """
OUT=""
while [[ $# -gt 0 ]]; do case "$1" in -o) OUT="$2"; shift 2;; *) shift;; esac; done
mkdir -p "$OUT"
touch "$OUT/00000001.png"
"""


@pytest.mark.parametrize(
    ("frames", "source", "target", "expected"),
    [
        (90, Fraction(30), 60, 180),
        (90, Fraction(30), 120, 360),
        (240, Fraction(24), 60, 600),
        (100, Fraction(25), 50, 200),
        # 29.97 to 59.94 is exactly double; computing from rounded rates is not.
        (1800, Fraction(30000, 1001), Fraction(60000, 1001), 3600),
        # 29.97 to a true 60 needs more than double.
        (1800, Fraction(30000, 1001), 60, 3604),
    ],
)
def test_target_frame_count(frames: int, source: Fraction, target, expected: int):
    assert target_frame_count(frames, source, target) == expected


def test_target_is_never_below_the_source():
    assert target_frame_count(100, 60, 30) == 100
    assert target_frame_count(100, 30, 30) == 100


def test_target_rejects_impossible_inputs():
    with pytest.raises(InterpolateError, match="zero source frames"):
        target_frame_count(0, 30, 60)
    with pytest.raises(InterpolateError, match="must be positive"):
        target_frame_count(10, 0, 60)
    with pytest.raises(InterpolateError, match="must be positive"):
        target_frame_count(10, 30, 0)


def test_factor_is_exact():
    assert interpolation_factor(Fraction(30000, 1001), Fraction(60000, 1001)) == 2
    assert interpolation_factor(24, 60) == Fraction(5, 2)


def test_is_needed_only_when_the_rate_rises():
    assert is_needed(30, 60) is True
    assert is_needed(30, 30) is False
    assert is_needed(60, 30) is False


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (1920, 1080, False),
        (2560, 1440, False),
        (3840, 2160, True),
        (4096, 2160, True),
        (7680, 4320, True),
    ],
)
def test_uhd_threshold(width: int, height: int, expected: bool):
    assert needs_uhd(width, height) is expected
    assert (width * height >= config.UHD_PIXEL_THRESHOLD) is expected


def test_argument_list_matches_the_binary(stub_bin, frames, tmp_path: Path):
    stub_bin("RIFE_BIN", STUB_RIFE)
    interpolate(frames(10), tmp_path / "out", target_frames=20, gpu=1)
    argv = stub_bin.log.read_text().split()
    assert argv[argv.index("-n") + 1] == "20"
    # For RIFE, -f is the output filename pattern, not the image format.
    assert argv[argv.index("-f") + 1] == "%08d.png"
    assert argv[argv.index("-g") + 1] == "1"
    assert argv[argv.index("-j") + 1] == config.DEFAULT_THREADS
    assert config.DEFAULT_RIFE_MODEL in argv[argv.index("-m") + 1]


def test_interpolate_produces_the_target_count(stub_bin, frames, tmp_path: Path):
    stub_bin("RIFE_BIN", STUB_RIFE)
    result = interpolate(frames(10), tmp_path / "out", target_frames=25)
    assert result.frame_count == 25
    assert result.skipped is False
    assert (tmp_path / "out" / "00000025.png").exists()


def test_count_is_derived_from_the_frame_rates(stub_bin, frames, tmp_path: Path):
    stub_bin("RIFE_BIN", STUB_RIFE)
    result = interpolate(
        frames(90), tmp_path / "out", source_fps=Fraction(30), target_fps=60
    )
    assert result.frame_count == 180


def test_rates_or_target_are_required(stub_bin, frames, tmp_path: Path):
    stub_bin("RIFE_BIN", STUB_RIFE)
    with pytest.raises(InterpolateError, match="needs either target_frames"):
        interpolate(frames(10), tmp_path / "out")
    with pytest.raises(InterpolateError, match="needs either target_frames"):
        interpolate(frames(10), tmp_path / "out", source_fps=30)


def test_a_target_below_the_source_is_rejected(stub_bin, frames, tmp_path: Path):
    stub_bin("RIFE_BIN", STUB_RIFE)
    with pytest.raises(InterpolateError, match="can only add frames"):
        interpolate(frames(10), tmp_path / "out", target_frames=5)


def test_empty_input_is_rejected(stub_bin, tmp_path: Path):
    stub_bin("RIFE_BIN", STUB_RIFE)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(InterpolateError, match="no png frames"):
        interpolate(empty, tmp_path / "out", target_frames=10)


def test_second_run_skips_completed_work(stub_bin, frames, tmp_path: Path):
    stub_bin("RIFE_BIN", STUB_RIFE)
    source = frames(10)
    interpolate(source, tmp_path / "out", target_frames=20)
    calls = len(stub_bin.log.read_text().splitlines())

    again = interpolate(source, tmp_path / "out", target_frames=20)
    assert again.skipped is True
    assert len(stub_bin.log.read_text().splitlines()) == calls


def test_a_higher_target_re_runs_the_stage(stub_bin, frames, tmp_path: Path):
    stub_bin("RIFE_BIN", STUB_RIFE)
    source = frames(10)
    interpolate(source, tmp_path / "out", target_frames=20)
    # 30 frames are not on disk yet, so this must not be skipped.
    again = interpolate(source, tmp_path / "out", target_frames=30)
    assert again.skipped is False
    assert again.frame_count == 30


def test_force_re_runs_a_completed_stage(stub_bin, frames, tmp_path: Path):
    stub_bin("RIFE_BIN", STUB_RIFE)
    source = frames(10)
    interpolate(source, tmp_path / "out", target_frames=20)
    again = interpolate(source, tmp_path / "out", target_frames=20, force=True)
    assert again.skipped is False


def test_uhd_is_enabled_from_the_resolution(stub_bin, frames, tmp_path: Path):
    stub_bin("RIFE_BIN", STUB_RIFE)
    result = interpolate(
        frames(10), tmp_path / "out", target_frames=20, width=3840, height=2160
    )
    assert result.uhd is True
    assert "-u" in stub_bin.log.read_text().split()


def test_uhd_stays_off_below_4k(stub_bin, frames, tmp_path: Path):
    stub_bin("RIFE_BIN", STUB_RIFE)
    result = interpolate(
        frames(10), tmp_path / "out", target_frames=20, width=1920, height=1080
    )
    assert result.uhd is False
    assert "-u" not in stub_bin.log.read_text().split()


def test_uhd_can_be_forced_on_at_any_resolution(stub_bin, frames, tmp_path: Path):
    stub_bin("RIFE_BIN", STUB_RIFE)
    result = interpolate(
        frames(10), tmp_path / "out", target_frames=20, uhd=True, width=640, height=480
    )
    assert result.uhd is True


def test_uhd_can_be_forced_off_above_4k(stub_bin, frames, tmp_path: Path):
    stub_bin("RIFE_BIN", STUB_RIFE)
    result = interpolate(
        frames(10),
        tmp_path / "out",
        target_frames=20,
        uhd=False,
        width=3840,
        height=2160,
    )
    assert result.uhd is False
    assert "-u" not in stub_bin.log.read_text().split()


def test_uhd_out_of_memory_retries_without_it(stub_bin, frames, tmp_path: Path):
    stub_bin("RIFE_BIN", STUB_UHD_OOM)
    result = interpolate(
        frames(10), tmp_path / "out", target_frames=20, width=3840, height=2160
    )
    assert result.uhd is False
    assert result.frame_count == 20
    attempts = stub_bin.log.read_text().splitlines()
    assert len(attempts) == 2
    assert "-u" in attempts[0]
    assert "-u" not in attempts[1]


def test_a_failure_without_uhd_is_not_retried(stub_bin, frames, tmp_path: Path):
    stub_bin("RIFE_BIN", STUB_BROKEN)
    with pytest.raises(ToolError):
        interpolate(frames(10), tmp_path / "out", target_frames=20)
    assert len(stub_bin.log.read_text().splitlines()) == 1


def test_a_short_result_is_reported(stub_bin, frames, tmp_path: Path):
    stub_bin("RIFE_BIN", STUB_SHORT)
    with pytest.raises(InterpolateError, match="produced 1 of 20"):
        interpolate(frames(10), tmp_path / "out", target_frames=20)


def test_missing_binary_is_reported_clearly(frames, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RIFE_BIN", str(tmp_path / "not-installed"))
    with pytest.raises(ToolError, match="was not found"):
        interpolate(frames(10), tmp_path / "out", target_frames=20)
