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


def test_options_carry_every_processing_argument():
    args = cli.build_parser().parse_args(
        ["clip.mp4", "--scale", "3", "--fps", "48", "--tile", "128", "--keep-temp"]
    )
    options = cli.options_from_args(args)
    assert (options.scale, options.target_fps, options.tile) == (3, 48, 128)
    assert options.keep_temp is True


@requires_ffmpeg
def test_a_missing_binary_is_reported_and_the_frames_are_kept(
    sample_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """A failed stage must explain itself and leave the work behind to resume."""
    monkeypatch.setenv("REALESRGAN_BIN", str(tmp_path / "not-installed"))
    work_base = tmp_path / "work"
    code = cli.main(
        [
            str(sample_video),
            "-o", str(tmp_path / "out.mp4"),
            "--work-dir", str(work_base),
        ]
    )
    assert code == cli.EXIT_ERROR
    err = capsys.readouterr().err
    assert "was not found" in err
    # The message has to name the job, or the user cannot act on it.
    assert "--resume sample-" in err
    extracted = list(work_base.glob("*/frames_in/*.png"))
    assert extracted, "the extracted frames should survive a later stage failing"
    assert list(work_base.glob("*/state.json")), "the job record should survive too"


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


def test_options_carry_the_chunk_settings():
    args = cli.build_parser().parse_args(
        ["clip.mp4", "--chunk", "15", "--encoder", "libx264"]
    )
    options = cli.options_from_args(args)
    assert options.chunk_seconds == 15
    assert options.prefer_encoder == "libx264"


@requires_ffmpeg
def test_a_file_shorter_than_one_chunk_runs_in_a_single_pass(sample_video: Path):
    """Cutting a 3-second clip into 30-second chunks would buy nothing."""
    from videoenhance.probe import probe

    info = probe(sample_video)
    assert cli.chunking(info, 30.0) is False
    assert cli.chunking(info, 1.0) is True
    assert cli.chunking(info, 0) is False


@requires_ffmpeg
def test_a_source_without_a_reported_duration_is_still_chunked(sample_video: Path):
    """Guessing "short" here is what would fill the disk."""
    import dataclasses

    from videoenhance.probe import probe

    unknown = dataclasses.replace(probe(sample_video), duration=0.0)
    assert cli.chunking(unknown, 30.0) is True


@requires_ffmpeg
def test_a_negative_chunk_length_is_rejected(sample_video: Path):
    with pytest.raises(SystemExit) as excinfo:
        cli.main([str(sample_video), "--chunk", "-5"])
    assert excinfo.value.code == 2


@requires_ffmpeg
def test_an_encoder_ffmpeg_does_not_have_is_rejected(sample_video: Path):
    with pytest.raises(SystemExit) as excinfo:
        cli.main([str(sample_video), "--encoder", "no_such_encoder"])
    assert excinfo.value.code == 2


@requires_ffmpeg
def test_the_dry_run_shows_the_chunks_and_a_smaller_estimate(
    sample_video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """The estimate is the argument for the mode, so it has to be visible."""
    base = ["-o", str(tmp_path / "out.mp4"), "--work-dir", str(tmp_path), "--dry-run"]

    assert cli.main([str(sample_video), *base, "--chunk", "0"]) == cli.EXIT_OK
    single = capsys.readouterr().out
    assert "chunks:" not in single

    assert cli.main([str(sample_video), *base, "--chunk", "1"]) == cli.EXIT_OK
    chunked = capsys.readouterr().out
    assert "chunks:  about 3 x 1s" in chunked

    def frames_mb(block: str) -> float:
        line = next(line for line in block.splitlines() if line.startswith("disk:"))
        return float(line.split()[1])

    assert frames_mb(chunked) < frames_mb(single)


# --------------------------------------------------------------------------
# saved jobs: --jobs, --resume, --restart
# --------------------------------------------------------------------------


@pytest.fixture
def ncnn_stubs(stub_bin):
    """Stand-ins for both NCNN binaries, so a run needs no GPU."""
    stub_bin("REALESRGAN_BIN", name="realesrgan-stub")
    stub_bin("RIFE_BIN", name="rife-stub")
    return stub_bin


def run_cli(video: Path, out: Path, work: Path, *extra: str) -> int:
    return cli.main(
        [
            str(video), "-o", str(out), "--work-dir", str(work),
            "--no-progress", "--keep-temp", *extra,
        ]
    )


def test_resolve_args_fills_defaults_and_reports_what_was_given():
    args = cli.build_parser().parse_args(["clip.mp4", "--scale", "4"])
    given = cli.resolve_args(args)

    assert given == {"scale"}
    assert args.scale == 4
    # Everything else came from config, not from the user.
    assert args.fps == cli.config.DEFAULT_TARGET_FPS
    assert args.chunk == cli.config.DEFAULT_CHUNK_SECONDS
    assert args.model == cli.config.DEFAULT_UPSCALE_MODEL


def test_jobs_says_so_when_there_is_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    assert cli.main(["--jobs", "--work-dir", str(tmp_path)]) == cli.EXIT_OK
    assert "No unfinished jobs" in capsys.readouterr().out


@requires_ffmpeg
def test_jobs_lists_a_job_that_was_left_behind(
    sample_video: Path, tmp_path: Path, ncnn_stubs, capsys: pytest.CaptureFixture[str]
):
    work = tmp_path / "work"
    assert run_cli(sample_video, tmp_path / "out.mp4", work, "--chunk", "0") == cli.EXIT_OK
    capsys.readouterr()

    assert cli.main(["--jobs", "--work-dir", str(work)]) == cli.EXIT_OK
    listing = capsys.readouterr().out
    assert "sample-" in listing
    assert "sample.mp4" in listing
    assert "--resume sample-" in listing


def test_resuming_an_unknown_job_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    assert cli.main(["--resume", "nope", "--work-dir", str(tmp_path)]) == cli.EXIT_ERROR
    assert "no saved jobs" in capsys.readouterr().err


@requires_ffmpeg
def test_resume_does_not_take_an_input_as_well(
    sample_video: Path, tmp_path: Path, ncnn_stubs
):
    work = tmp_path / "work"
    run_cli(sample_video, tmp_path / "out.mp4", work, "--chunk", "0")
    with pytest.raises(SystemExit) as excinfo:
        cli.main([str(sample_video), "--resume", "sample", "--work-dir", str(work)])
    assert excinfo.value.code == 2


def test_resume_and_restart_contradict_each_other(tmp_path: Path):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--resume", "x", "--restart", "--work-dir", str(tmp_path)])
    assert excinfo.value.code == 2


@requires_ffmpeg
def test_resuming_a_finished_job_reuses_everything(
    sample_video: Path, tmp_path: Path, ncnn_stubs, capsys: pytest.CaptureFixture[str]
):
    """--resume has to find the input, the output and the options by itself."""
    work = tmp_path / "work"
    output = tmp_path / "out.mp4"
    assert run_cli(sample_video, output, work, "--chunk", "0", "--scale", "2") == cli.EXIT_OK
    capsys.readouterr()

    assert cli.main(["--resume", "sample", "--work-dir", str(work)]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "resuming sample-" in out
    # The stages were all already done, so nothing was recomputed.
    assert out.count("already done") >= 2
    assert output.exists()


@requires_ffmpeg
def test_changing_the_scale_on_a_resume_is_refused(
    sample_video: Path, tmp_path: Path, ncnn_stubs, capsys: pytest.CaptureFixture[str]
):
    """The regression test for the hole this closes.

    Every stage guard is a frame count, and the frame count does not change with
    --scale, so without the job record the upscale is skipped and the output is
    2x while the plan printed above it says 4x.
    """
    work = tmp_path / "work"
    output = tmp_path / "out.mp4"
    assert run_cli(sample_video, output, work, "--chunk", "0", "--scale", "2") == cli.EXIT_OK
    capsys.readouterr()
    before = output.stat().st_mtime_ns

    assert run_cli(sample_video, output, work, "--chunk", "0", "--scale", "4") == cli.EXIT_ERROR
    err = capsys.readouterr().err
    assert "--scale" in err
    assert "2 -> 4" in err
    assert "--restart" in err
    assert output.stat().st_mtime_ns == before, "the old output must not be touched"


@requires_ffmpeg
def test_changing_the_fps_on_a_chunked_resume_is_refused(
    sample_video: Path, tmp_path: Path, ncnn_stubs, capsys: pytest.CaptureFixture[str]
):
    """The chunked version of the same hole, and the quieter one.

    A different --fps leaves the resolution alone, so the parts still concatenate
    by stream copy and the result simply plays parts of itself at the wrong speed.
    """
    work = tmp_path / "work"
    assert run_cli(sample_video, tmp_path / "a.mkv", work, "--chunk", "1", "--fps", "60") == 0
    capsys.readouterr()

    code = run_cli(sample_video, tmp_path / "a.mkv", work, "--chunk", "1", "--fps", "45")
    assert code == cli.EXIT_ERROR
    assert "--fps" in capsys.readouterr().err


@requires_ffmpeg
def test_restart_throws_the_old_work_away(
    sample_video: Path, tmp_path: Path, ncnn_stubs, capsys: pytest.CaptureFixture[str]
):
    work = tmp_path / "work"
    output = tmp_path / "out.mp4"
    assert run_cli(sample_video, output, work, "--chunk", "0", "--scale", "2") == cli.EXIT_OK
    capsys.readouterr()

    code = run_cli(sample_video, output, work, "--chunk", "0", "--scale", "4", "--restart")
    assert code == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "discarding the saved job sample-" in out
    # Nothing was reused: the frames were thrown away with the record.
    assert "already done" not in out


@requires_ffmpeg
def test_a_changed_input_at_the_same_path_is_refused(
    sample_video: Path, silent_video: Path, tmp_path: Path, ncnn_stubs,
    capsys: pytest.CaptureFixture[str],
):
    """The work on disk belongs to the file that was there before."""
    work = tmp_path / "work"
    moving = tmp_path / "moving.mp4"
    moving.write_bytes(sample_video.read_bytes())
    assert run_cli(moving, tmp_path / "out.mp4", work, "--chunk", "0") == cli.EXIT_OK
    capsys.readouterr()

    moving.write_bytes(silent_video.read_bytes())
    assert run_cli(moving, tmp_path / "out.mp4", work, "--chunk", "0") == cli.EXIT_ERROR
    assert "has changed since this job was started" in capsys.readouterr().err


@requires_ffmpeg
def test_a_successful_run_leaves_no_record_behind(
    sample_video: Path, tmp_path: Path, ncnn_stubs
):
    """Without --keep-temp the record goes with the work directory."""
    work = tmp_path / "work"
    code = cli.main(
        [
            str(sample_video), "-o", str(tmp_path / "out.mp4"),
            "--work-dir", str(work), "--no-progress", "--chunk", "0",
        ]
    )
    assert code == cli.EXIT_OK
    assert list(work.glob("*/state.json")) == []
