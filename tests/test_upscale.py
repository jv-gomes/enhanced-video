"""Tests for the Real-ESRGAN wrapper, driven by stub binaries.

The real executable needs a GPU and a model download, so what is tested here is
what the wrapper owns: the argument list it builds, the model/scale rules, the
tile retry and how it judges the output directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videoenhance import config
from videoenhance.process import ToolError
from videoenhance.upscale import (
    ModelError,
    UpscaleError,
    VramError,
    looks_like_vram_exhaustion,
    models_for_scale,
    resolve_model,
    tile_attempts,
    upscale,
)

# Fails with a Vulkan allocation error until the tile size drops to 128.
STUB_VRAM = """
IN=""; OUT=""; TILE="0"; FMT="png"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i) IN="$2"; shift 2;;
    -o) OUT="$2"; shift 2;;
    -t) TILE="$2"; shift 2;;
    -f) FMT="$2"; shift 2;;
    *) shift;;
  esac
done
if [[ "$TILE" == "0" || "$TILE" -gt 128 ]]; then
  echo "vkAllocateMemory failed: out of device memory (tile=$TILE)" >&2
  exit 1
fi
mkdir -p "$OUT"
for f in "$IN"/*."$FMT"; do [[ -e "$f" ]] || continue; cp "$f" "$OUT/$(basename "$f")"; done
"""

STUB_ALWAYS_OOM = """
echo "vkAllocateMemory failed: out of device memory" >&2
exit 1
"""

STUB_BROKEN_MODEL = """
echo "find_blob_index_by_name data failed" >&2
exit 1
"""

# Exits successfully but writes only one frame.
STUB_SHORT_OUTPUT = """
OUT=""
while [[ $# -gt 0 ]]; do case "$1" in -o) OUT="$2"; shift 2;; *) shift;; esac; done
mkdir -p "$OUT"
touch "$OUT/00000001.png"
"""


def test_resolve_model_accepts_supported_pairs():
    assert resolve_model("realesr-animevideov3", 2) == ("realesr-animevideov3", 2)
    assert resolve_model("realesrgan-x4plus", 4) == ("realesrgan-x4plus", 4)


def test_resolve_model_rejects_an_unsupported_scale():
    with pytest.raises(ModelError, match="only supports scale 4"):
        resolve_model("realesrgan-x4plus", 2)


def test_rejection_names_a_model_that_can_do_it():
    with pytest.raises(ModelError, match="realesr-animevideov3 can do 2x"):
        resolve_model("realesrgan-x4plus", 2)


def test_resolve_model_rejects_an_unknown_model():
    with pytest.raises(ModelError, match="unknown model"):
        resolve_model("not-a-model", 4)


def test_models_for_scale_matches_the_config_table():
    assert models_for_scale(2) == ["realesr-animevideov3"]
    assert len(models_for_scale(4)) == len(config.UPSCALE_MODELS)
    assert models_for_scale(8) == []


def test_tile_attempts_walk_down_from_the_request():
    assert tile_attempts(0) == [0, *config.TILE_FALLBACKS]
    assert tile_attempts(512) == [512, 256, 128]
    assert tile_attempts(256) == [256, 128]
    # Nothing smaller to try.
    assert tile_attempts(128) == [128]
    assert tile_attempts(64) == [64]


@pytest.mark.parametrize(
    "message",
    [
        "vkAllocateMemory failed",
        "ERROR: out of device memory",
        "vkQueueSubmit failed",
        "failed to allocate 2097152 bytes",
        "device lost",
    ],
)
def test_vram_messages_are_recognised(message: str):
    assert looks_like_vram_exhaustion(ToolError(["bin"], 1, message)) is True


@pytest.mark.parametrize(
    "message",
    ["find_blob_index_by_name failed", "invalid model path", ""],
)
def test_other_messages_are_not_mistaken_for_vram(message: str):
    assert looks_like_vram_exhaustion(ToolError(["bin"], 1, message)) is False


def test_argument_list_matches_the_binary(stub_bin, frames, tmp_path: Path):
    stub_bin("REALESRGAN_BIN")
    upscale(frames(3), tmp_path / "out", scale=2, model="realesr-animevideov3", gpu=1)
    argv = stub_bin.log.read_text().split()
    for flag, value in [
        ("-n", "realesr-animevideov3"),
        ("-s", "2"),
        ("-f", "png"),
        ("-g", "1"),
        ("-j", config.DEFAULT_THREADS),
    ]:
        assert argv[argv.index(flag) + 1] == value
    assert "-i" in argv and "-o" in argv


def test_upscale_copies_every_frame_through(stub_bin, frames, tmp_path: Path):
    stub_bin("REALESRGAN_BIN")
    result = upscale(frames(5), tmp_path / "out", scale=2)
    assert result.frame_count == 5
    assert result.scale == 2
    assert result.skipped is False
    assert (tmp_path / "out" / "00000005.png").exists()


def test_jpg_frames_are_passed_through(stub_bin, frames, tmp_path: Path):
    stub_bin("REALESRGAN_BIN")
    source = frames(3, ext="jpg")
    result = upscale(source, tmp_path / "out", scale=2, frame_format="jpg")
    assert result.frame_count == 3
    argv = stub_bin.log.read_text().split()
    assert argv[argv.index("-f") + 1] == "jpg"


def test_second_run_skips_completed_work(stub_bin, frames, tmp_path: Path):
    stub_bin("REALESRGAN_BIN")
    source = frames(4)
    upscale(source, tmp_path / "out", scale=2)
    calls = len(stub_bin.log.read_text().splitlines())

    again = upscale(source, tmp_path / "out", scale=2)
    assert again.skipped is True
    assert len(stub_bin.log.read_text().splitlines()) == calls


def test_force_re_runs_a_completed_stage(stub_bin, frames, tmp_path: Path):
    stub_bin("REALESRGAN_BIN")
    source = frames(4)
    upscale(source, tmp_path / "out", scale=2)
    again = upscale(source, tmp_path / "out", scale=2, force=True)
    assert again.skipped is False


def test_an_incomplete_output_is_not_skipped(stub_bin, frames, tmp_path: Path):
    stub_bin("REALESRGAN_BIN")
    source = frames(4)
    upscale(source, tmp_path / "out", scale=2)
    (tmp_path / "out" / "00000001.png").unlink()
    again = upscale(source, tmp_path / "out", scale=2)
    assert again.skipped is False
    assert again.frame_count == 4


def test_empty_input_is_rejected(stub_bin, tmp_path: Path):
    stub_bin("REALESRGAN_BIN")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(UpscaleError, match="no png frames"):
        upscale(empty, tmp_path / "out", scale=2)


def test_invalid_options_are_rejected_before_running(stub_bin, frames, tmp_path: Path):
    stub_bin("REALESRGAN_BIN")
    source = frames(2)
    with pytest.raises(ValueError, match="tile size"):
        upscale(source, tmp_path / "out", scale=2, tile=-1)
    with pytest.raises(ValueError, match="frame format"):
        upscale(source, tmp_path / "out", scale=2, frame_format="tiff")
    with pytest.raises(ModelError):
        upscale(source, tmp_path / "out", scale=3, model="realesrgan-x4plus")
    assert stub_bin.log.read_text() == ""


def test_retry_drops_the_tile_until_it_fits(stub_bin, frames, tmp_path: Path):
    stub_bin("REALESRGAN_BIN", STUB_VRAM)
    result = upscale(frames(3), tmp_path / "out", scale=2)
    assert result.tile == 128
    assert result.frame_count == 3


def test_retry_gives_up_with_a_vram_error(stub_bin, frames, tmp_path: Path):
    stub_bin("REALESRGAN_BIN", STUB_ALWAYS_OOM)
    with pytest.raises(VramError, match="every tile size tried"):
        upscale(frames(3), tmp_path / "out", scale=2)


def test_a_non_vram_failure_is_not_retried(stub_bin, frames, tmp_path: Path):
    stub_bin("REALESRGAN_BIN", STUB_BROKEN_MODEL)
    with pytest.raises(ToolError) as excinfo:
        upscale(frames(3), tmp_path / "out", scale=2)
    assert not isinstance(excinfo.value, VramError)
    # One attempt only: retrying a broken model cannot help.
    assert len(stub_bin.log.read_text().splitlines()) <= 1


def test_a_short_result_is_reported_after_the_retries(stub_bin, frames, tmp_path: Path):
    stub_bin("REALESRGAN_BIN", STUB_SHORT_OUTPUT)
    with pytest.raises(UpscaleError, match="produced 1 of 3"):
        upscale(frames(3), tmp_path / "out", scale=2)


def test_missing_binary_is_reported_clearly(frames, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REALESRGAN_BIN", str(tmp_path / "not-installed"))
    with pytest.raises(ToolError, match="was not found"):
        upscale(frames(2), tmp_path / "out", scale=2)
