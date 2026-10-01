"""Tests for the stage sequence.

The GPU binaries are replaced by stubs (see ``conftest.write_stub``), so these
exercise what the pipeline is responsible for: which stages run, in which
order, what each one is told to produce and what the summary then says.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videoenhance import pipeline
from videoenhance.pipeline import Options, run_pipeline

from .conftest import requires_ffmpeg


@pytest.fixture
def ncnn_stubs(stub_bin):
    """Stand-ins for both NCNN binaries, sharing one argument log."""
    stub_bin("REALESRGAN_BIN", name="realesrgan-stub")
    stub_bin("RIFE_BIN", name="rife-stub")
    return stub_bin


def stage_names(result) -> list[str]:
    return [stage.name for stage in result.stages]


def test_default_order_upscales_before_interpolating():
    assert pipeline.stage_order() == ("upscale", "interpolate")


def test_interpolate_first_reverses_the_two_model_stages():
    assert pipeline.stage_order("interpolate-first") == ("interpolate", "upscale")


def test_an_unknown_order_is_rejected():
    with pytest.raises(ValueError, match="unknown stage order"):
        pipeline.stage_order("sideways")


@requires_ffmpeg
def test_full_run_reports_every_stage_in_order(
    sample_video: Path, tmp_path: Path, ncnn_stubs
):
    result = run_pipeline(
        sample_video,
        tmp_path / "out.mp4",
        Options(scale=2, target_fps=60, keep_temp=True),
        work=pipeline.WorkDir.for_input(sample_video, base=tmp_path / "work"),
    )
    assert stage_names(result) == ["extract", "upscale", "interpolate"]
    assert result.output.exists()
    # 3 seconds of 30 fps doubled to 60.
    assert result.frames == 180
    assert float(result.fps) == 60


@requires_ffmpeg
def test_interpolate_first_runs_rife_on_the_source_frames(
    sample_video: Path, tmp_path: Path, ncnn_stubs
):
    """RIFE must see the extracted frames, and Real-ESRGAN the interpolated ones."""
    work = pipeline.WorkDir.for_input(sample_video, base=tmp_path / "work")
    result = run_pipeline(
        sample_video,
        tmp_path / "out.mp4",
        Options(scale=2, target_fps=60, order="interpolate-first", keep_temp=True),
        work=work,
    )
    assert stage_names(result) == ["extract", "interpolate", "upscale"]
    assert result.frames == 180

    calls = ncnn_stubs.log.read_text().splitlines()
    rife = next(line for line in calls if "rife" in line)
    esrgan = next(line for line in calls if "realesr" in line)
    assert f"-i {work.frames_in} " in f"{rife} "
    assert f"-i {work.frames_out} " in f"{esrgan} "


@requires_ffmpeg
def test_scale_one_skips_the_upscale_stage(sample_video: Path, tmp_path: Path, ncnn_stubs):
    result = run_pipeline(
        sample_video,
        tmp_path / "out.mp4",
        Options(scale=1, target_fps=60),
        work=pipeline.WorkDir.for_input(sample_video, base=tmp_path / "work"),
    )
    assert stage_names(result) == ["extract", "interpolate"]


@requires_ffmpeg
def test_a_target_fps_at_or_below_the_source_skips_interpolation(
    sample_video: Path, tmp_path: Path, ncnn_stubs
):
    result = run_pipeline(
        sample_video,
        tmp_path / "out.mp4",
        Options(scale=2, target_fps=30),
        work=pipeline.WorkDir.for_input(sample_video, base=tmp_path / "work"),
    )
    assert stage_names(result) == ["extract", "upscale"]
    assert float(result.fps) == 30


@requires_ffmpeg
def test_refuses_to_overwrite_the_input(sample_video: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="overwrite the input"):
        run_pipeline(sample_video, sample_video, work=pipeline.WorkDir.for_input(
            sample_video, base=tmp_path / "work"
        ))
