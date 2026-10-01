"""Tests for the command-line surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from videoenhance import cli

from .conftest import requires_ffmpeg


def test_default_output_never_equals_input():
    output = cli.default_output(Path("/videos/clip.mp4"))
    assert output == Path("/videos/clip.enhanced.mp4")
    assert output != Path("/videos/clip.mp4")


@requires_ffmpeg
def test_probe_only_prints_summary(sample_video: Path, capsys: pytest.CaptureFixture[str]):
    assert cli.main([str(sample_video), "--probe-only"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "320x240" in out
    assert "audio:      yes" in out


def test_missing_input_file_is_reported(capsys: pytest.CaptureFixture[str]):
    assert cli.main(["no-such-clip.mp4", "--probe-only"]) == cli.EXIT_ERROR
    assert "not found" in capsys.readouterr().err


def test_input_is_required_without_doctor():
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2


@requires_ffmpeg
def test_unsupported_scale_for_model_is_rejected(sample_video: Path):
    with pytest.raises(SystemExit) as excinfo:
        cli.main([str(sample_video), "--model", "realesrgan-x4plus", "--scale", "2"])
    assert excinfo.value.code == 2


@requires_ffmpeg
def test_output_equal_to_input_is_rejected(sample_video: Path):
    with pytest.raises(SystemExit) as excinfo:
        cli.main([str(sample_video), "-o", str(sample_video)])
    assert excinfo.value.code == 2


@requires_ffmpeg
def test_pipeline_not_implemented_yet_exits_non_zero(
    sample_video: Path, capsys: pytest.CaptureFixture[str]
):
    assert cli.main([str(sample_video), "--scale", "2", "--fps", "60"]) == cli.EXIT_ERROR
    assert "not implemented yet" in capsys.readouterr().err


def test_doctor_runs_without_an_input(capsys: pytest.CaptureFixture[str]):
    code = cli.main(["--doctor"])
    assert code in (0, 1)
    assert "ffmpeg" in capsys.readouterr().out


def test_verbosity_maps_to_log_levels():
    import logging

    cli.configure_logging(0)
    assert logging.getLogger().level == logging.WARNING
    cli.configure_logging(2)
    assert logging.getLogger().level == logging.DEBUG
