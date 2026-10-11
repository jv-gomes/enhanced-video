"""Tests for --blocks, --block-margin and the config file, on the command line.

The precedence is the part to get right: a typed flag, then the recorded job,
then the file, then the default. The middle one is what keeps a resume from
being refused over a file nobody typed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videoenhance import cli, config, configfile, state
from videoenhance.workdir import WorkDir, estimate_disk_usage

from .conftest import requires_ffmpeg
from .test_state import SETTINGS


def parse(*argv: str):
    return cli.build_parser().parse_args(["clip.mp4", *argv])


def write_config(tmp_path: Path, text: str = "[dedup]\nblocks = 3\nmargin = 12\n") -> Path:
    path = tmp_path / "videoenhance.toml"
    path.write_text(text)
    return path


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "root")
    monkeypatch.delenv(configfile.ENV_VAR, raising=False)


# --------------------------------------------------------------------------
# precedence
# --------------------------------------------------------------------------


def test_the_defaults_are_whole_frames():
    args = parse()
    given = cli.resolve_args(args)

    assert (args.blocks, args.block_margin) == (1, config.DEFAULT_BLOCK_MARGIN)
    assert given == set()


def test_the_file_fills_what_was_not_typed_and_is_not_given():
    args = parse()
    given = cli.resolve_args(args, {"blocks": 3, "block_margin": 12})

    assert (args.blocks, args.block_margin) == (3, 12)
    assert given == set()


def test_a_typed_flag_beats_the_file():
    args = parse("--blocks", "4")
    given = cli.resolve_args(args, {"blocks": 3, "block_margin": 12})

    assert (args.blocks, args.block_margin) == (4, 12)
    assert given == {"blocks"}


def test_a_recorded_job_beats_the_file(tmp_path: Path):
    """What apply_job sets over the file's value, legacy records included."""
    args = parse()
    cli.resolve_args(args, {"blocks": 3, "block_margin": 12})
    work = WorkDir(tmp_path / "j")
    job = state.for_input(work, tmp_path / "in.mp4", tmp_path / "o.mp4", SETTINGS)

    cli.apply_job(args, job)

    # SETTINGS has no block keys, so the record is read as whole frames.
    assert args.blocks == 1


# --------------------------------------------------------------------------
# the plan and the validation
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_the_plan_shows_the_grid_and_the_file(
    sample_video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    cfg = write_config(tmp_path)

    code = cli.main(
        [str(sample_video), "--dry-run", "--config", str(cfg), "--work-dir", str(tmp_path / "w")]
    )

    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "blocks:  3x3 grid, 12 px margin" in out
    assert f"config:  {cfg}" in out


@requires_ffmpeg
def test_a_flag_beats_the_file_in_the_plan(
    sample_video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    cli.main(
        [str(sample_video), "--dry-run", "--blocks", "1"]
        + ["--config", str(write_config(tmp_path)), "--work-dir", str(tmp_path / "w")]
    )

    assert "blocks:" not in capsys.readouterr().out


@requires_ffmpeg
@pytest.mark.parametrize(
    "flags,words",
    [
        (["--blocks", "0"], "--blocks must be 1"),
        (["--block-margin", "-1"], "--block-margin must be 0"),
        (["--blocks", "20"], "under 32"),
        (["--blocks", "500"], "does not fit"),
    ],
)
def test_a_grid_that_cannot_work_is_rejected(
    sample_video: Path, tmp_path: Path, flags: list[str], words: str, capsys
):
    with pytest.raises(SystemExit) as excinfo:
        cli.main([str(sample_video), "--dry-run", "--work-dir", str(tmp_path / "w"), *flags])

    assert excinfo.value.code == 2
    assert words in capsys.readouterr().err


@requires_ffmpeg
def test_a_thin_margin_is_a_warning_not_an_error(sample_video: Path, tmp_path: Path, capsys):
    code = cli.main(
        [str(sample_video), "--dry-run", "--blocks", "2", "--block-margin", "0"]
        + ["--work-dir", str(tmp_path / "w")]
    )

    assert code == cli.EXIT_OK
    assert "seams" in capsys.readouterr().err


@requires_ffmpeg
def test_a_grid_that_costs_a_lot_more_is_a_warning(sample_video: Path, tmp_path: Path, capsys):
    cli.main(
        [str(sample_video), "--dry-run", "--blocks", "4", "--block-margin", "30"]
        + ["--work-dir", str(tmp_path / "w")]
    )

    assert "only pays off if most of the picture holds still" in capsys.readouterr().err


@requires_ffmpeg
def test_blocks_without_dedup_run_as_whole_frames(sample_video: Path, tmp_path: Path, capsys):
    cli.main(
        [str(sample_video), "--dry-run", "--blocks", "3", "--no-dedup"]
        + ["--work-dir", str(tmp_path / "w")]
    )

    captured = capsys.readouterr()
    assert "--blocks needs dedup" in captured.err
    assert "blocks:" not in captured.out


def test_a_missing_config_file_is_an_error(sample_video: Path, tmp_path: Path, capsys):
    code = cli.main([str(sample_video), "--dry-run", "--config", str(tmp_path / "nope.toml")])

    assert code == cli.EXIT_ERROR
    assert "config file not found" in capsys.readouterr().err


# --------------------------------------------------------------------------
# a job that already exists
# --------------------------------------------------------------------------


@pytest.fixture
def old_job(sample_video: Path, tmp_path: Path) -> state.Job:
    """A job started before blocks existed: no block keys in its record."""
    work = WorkDir.for_input(sample_video, base=tmp_path / "w")
    return state.save(state.for_input(work, sample_video, tmp_path / "out.mp4", SETTINGS))


@requires_ffmpeg
def test_resuming_after_the_file_changed_keeps_the_recorded_setting(
    old_job: state.Job, tmp_path: Path, capsys
):
    """The file now says 3 blocks; the job was whole frames, and nobody typed 3."""
    code = cli.main(
        ["--resume", old_job.id, "--dry-run", "--work-dir", str(tmp_path / "w")]
        + ["--config", str(write_config(tmp_path))]
    )

    assert code == cli.EXIT_OK
    assert "blocks:" not in capsys.readouterr().out


@requires_ffmpeg
def test_typing_blocks_on_a_resume_is_a_conflict(old_job: state.Job, tmp_path: Path, capsys):
    code = cli.main(
        ["--resume", old_job.id, "--blocks", "3", "--dry-run", "--work-dir", str(tmp_path / "w")]
    )

    assert code == cli.EXIT_ERROR
    assert "--blocks" in capsys.readouterr().err


@requires_ffmpeg
def test_a_rerun_names_the_file_as_the_reason_for_the_refusal(
    old_job: state.Job, sample_video: Path, tmp_path: Path, capsys
):
    cfg = write_config(tmp_path)

    code = cli.main(
        [str(sample_video), "--dry-run", "--work-dir", str(tmp_path / "w"), "--config", str(cfg)]
    )

    err = capsys.readouterr().err
    assert code == cli.EXIT_ERROR
    assert "--blocks" in err and f"(from {cfg})" in err


def test_the_mismatch_names_a_source_only_for_the_options_that_have_one(tmp_path: Path):
    work = WorkDir(tmp_path / "j")
    job = state.for_input(work, tmp_path / "i.mp4", tmp_path / "o.mp4", SETTINGS)
    changed = {"blocks": (1, 3), "scale": (2, 4)}

    text = state.render_mismatch(job, changed, {"blocks": "videoenhance.toml"})

    assert "1 -> 3  (from videoenhance.toml)" in text
    assert "2 -> 4\n" in text + "\n"


# --------------------------------------------------------------------------
# disk
# --------------------------------------------------------------------------


def test_blocks_add_their_scratch_to_the_estimate():
    plain = estimate_disk_usage(1920, 1080, 900, scale=2, target_fps=60, source_fps=30)
    grid = estimate_disk_usage(
        1920, 1080, 900, scale=2, target_fps=60, source_fps=30, blocks=3, block_margin=16
    )

    assert "blocks" not in plain.stages
    assert grid.stages["blocks"] > 0
    assert grid.total > plain.total
    # The rest of the estimate is untouched.
    assert {k: v for k, v in grid.stages.items() if k != "blocks"} == plain.stages


def test_a_margin_makes_the_scratch_bigger():
    kwargs = {"scale": 2, "target_fps": 60, "source_fps": 30, "blocks": 3}
    thin = estimate_disk_usage(1920, 1080, 900, block_margin=0, **kwargs)
    wide = estimate_disk_usage(1920, 1080, 900, block_margin=32, **kwargs)

    assert wide.stages["blocks"] > thin.stages["blocks"]


def test_no_scratch_when_the_upscale_is_skipped():
    skipped = estimate_disk_usage(
        1920, 1080, 900, scale=1, target_fps=60, source_fps=30, blocks=3, block_margin=16
    )

    assert "blocks" not in skipped.stages
