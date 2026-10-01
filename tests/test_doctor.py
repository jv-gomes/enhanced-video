"""Tests for the environment preflight report."""

from __future__ import annotations

from pathlib import Path

import pytest

from videoenhance import doctor
from videoenhance.doctor import FAIL, OK, WARN, Check

from .conftest import requires_ffmpeg


def test_check_render_includes_hint_only_on_failure():
    assert "->" not in Check("vulkan", OK, "a GPU", hint="ignored").render()
    assert "-> fix it" in Check("rife", FAIL, "missing", hint="fix it").render()


def test_report_exit_code_reflects_worst_status(capsys: pytest.CaptureFixture[str]):
    assert doctor.report([Check("a", OK)]) == 0
    assert doctor.report([Check("a", OK), Check("b", WARN)]) == 0
    assert doctor.report([Check("a", OK), Check("b", FAIL)]) == 1
    assert "failed" in capsys.readouterr().out


@requires_ffmpeg
def test_available_encoders_includes_libx264():
    assert "libx264" in doctor.available_encoders(Path("ffmpeg"))


def test_available_encoders_empty_when_ffmpeg_missing():
    assert doctor.available_encoders(Path("videoenhance-no-ffmpeg")) == set()


def test_missing_ncnn_binary_names_the_env_var(tmp_path: Path):
    checks = doctor._check_ncnn_binary(
        "rife", tmp_path / "rife-ncnn-vulkan", "https://example.invalid", tmp_path / "rife-v4.6"
    )
    assert len(checks) == 1
    assert checks[0].status == FAIL
    assert "RIFE_BIN" in checks[0].hint


def test_present_binary_without_models_fails_on_models(tmp_path: Path):
    binary = tmp_path / "rife-ncnn-vulkan"
    binary.touch()
    checks = doctor._check_ncnn_binary(
        "rife", binary, "https://example.invalid", tmp_path / "rife-v4.6"
    )
    assert [c.status for c in checks] == [OK, FAIL]
    assert "models" in checks[1].name


def test_collect_checks_covers_the_whole_environment():
    names = [check.name for check in doctor.collect_checks()]
    for expected in ("platform", "ffmpeg", "ffprobe", "encoders", "vulkan", "realesrgan", "rife"):
        assert expected in names
