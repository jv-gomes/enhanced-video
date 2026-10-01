"""Tests for the work directory, the resume rule and the disk estimate."""

from __future__ import annotations

from pathlib import Path

from videoenhance.workdir import (
    SpaceEstimate,
    WorkDir,
    clear_frames,
    count_frames_on_disk,
    estimate_disk_usage,
    estimate_frame_bytes,
    frame_pattern,
    frames_complete,
    human_bytes,
    stage_is_done,
)


def _fill(directory: Path, count: int, ext: str = "png") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        (directory / f"{index:08d}.{ext}").touch()


def test_work_dir_is_stable_per_input(tmp_path: Path):
    first = WorkDir.for_input(Path("clip.mp4"), base=tmp_path)
    again = WorkDir.for_input(Path("clip.mp4"), base=tmp_path)
    assert first.root == again.root


def test_work_dir_differs_for_same_name_in_another_folder(tmp_path: Path):
    a = WorkDir.for_input(tmp_path / "a" / "clip.mp4", base=tmp_path)
    b = WorkDir.for_input(tmp_path / "b" / "clip.mp4", base=tmp_path)
    assert a.root != b.root
    assert a.root.name.startswith("clip-")


def test_work_dir_name_survives_awkward_characters(tmp_path: Path):
    work = WorkDir.for_input(Path("my holiday #1 (4k).mp4"), base=tmp_path)
    assert " " not in work.root.name
    assert "#" not in work.root.name


def test_create_is_idempotent_and_cleanup_removes(tmp_path: Path):
    work = WorkDir.for_input(Path("clip.mp4"), base=tmp_path).create()
    _fill(work.frames_in, 2)
    work.create()  # must not wipe what is already there
    assert count_frames_on_disk(work.frames_in) == 2
    work.cleanup()
    assert not work.root.exists()


def test_cleanup_keep_leaves_the_directory(tmp_path: Path):
    work = WorkDir.for_input(Path("clip.mp4"), base=tmp_path).create()
    work.cleanup(keep=True)
    assert work.root.exists()


def test_cleanup_is_safe_when_nothing_exists(tmp_path: Path):
    WorkDir.for_input(Path("clip.mp4"), base=tmp_path).cleanup()


def test_free_bytes_walks_up_to_an_existing_parent(tmp_path: Path):
    work = WorkDir.for_input(Path("clip.mp4"), base=tmp_path / "not" / "created" / "yet")
    assert work.free_bytes() > 0


def test_frames_complete_counts_only_the_right_format(tmp_path: Path):
    _fill(tmp_path, 3, "png")
    assert frames_complete(tmp_path, 3) is True
    assert frames_complete(tmp_path, 3, "jpg") is False


def test_frames_complete_needs_every_frame(tmp_path: Path):
    _fill(tmp_path, 2)
    assert frames_complete(tmp_path, 3) is False
    assert frames_complete(tmp_path, 2) is True


def test_unknown_expected_count_is_never_complete(tmp_path: Path):
    _fill(tmp_path, 5)
    # Zero means "we do not know", which must not skip the stage.
    assert frames_complete(tmp_path, 0) is False
    assert stage_is_done("extract", tmp_path, 0) is False


def test_missing_directory_counts_as_empty(tmp_path: Path):
    assert count_frames_on_disk(tmp_path / "nope") == 0
    assert frames_complete(tmp_path / "nope", 1) is False


def test_frame_pattern_matches_the_padding():
    assert frame_pattern("png") == "%08d.png"
    assert frame_pattern("jpg") == "%08d.jpg"


def test_jpg_frames_are_estimated_smaller_than_png():
    assert estimate_frame_bytes(1920, 1080, "jpg") < estimate_frame_bytes(1920, 1080, "png")


def test_estimate_scales_with_resolution_and_frame_count():
    small = estimate_disk_usage(640, 480, 100, scale=2, target_fps=30, source_fps=30)
    large = estimate_disk_usage(1920, 1080, 100, scale=2, target_fps=30, source_fps=30)
    assert large.total > small.total * 5


def test_estimate_counts_interpolated_frames():
    same = estimate_disk_usage(640, 480, 100, scale=1, target_fps=30, source_fps=30)
    doubled = estimate_disk_usage(640, 480, 100, scale=1, target_fps=60, source_fps=30)
    assert doubled.stages["frames_out"] == same.stages["frames_out"] * 2


def test_estimate_ignores_a_lower_target_fps():
    estimate = estimate_disk_usage(640, 480, 100, scale=1, target_fps=15, source_fps=30)
    assert estimate.stages["frames_out"] == estimate.stages["frames_in"]


def test_estimate_fits_compares_against_free_space():
    assert SpaceEstimate(stages={"a": 10}, free=100).fits is True
    assert SpaceEstimate(stages={"a": 1000}, free=100).fits is False


def test_estimate_render_names_each_stage():
    rendered = estimate_disk_usage(
        640, 480, 10, scale=2, target_fps=60, source_fps=30, free=10**9
    ).render()
    for stage in ("frames_in", "frames_up", "frames_out"):
        assert stage in rendered
    assert "free" in rendered


def test_human_bytes_picks_sensible_units():
    assert human_bytes(512) == "512 B"
    assert human_bytes(2 * 1024) == "2 KB"
    assert human_bytes(5 * 1024**2) == "5.0 MB"
    assert human_bytes(3 * 1024**3) == "3.0 GB"


def test_the_order_changes_which_stage_pays_for_the_frames(tmp_path: Path):
    """Interpolating first keeps its intermediate small, so it costs less disk."""
    common = dict(scale=2, target_fps=60, source_fps=30, frame_count=100)
    upscale_first = estimate_disk_usage(1920, 1080, order="upscale-first", **common)
    interpolate_first = estimate_disk_usage(1920, 1080, order="interpolate-first", **common)
    assert interpolate_first.total < upscale_first.total
    # Upscaling last is what makes every interpolated frame a full-size file.
    assert interpolate_first.stages["frames_up"] > upscale_first.stages["frames_up"]
    assert interpolate_first.stages["frames_out"] < upscale_first.stages["frames_out"]


def test_an_exact_stage_rejects_a_directory_holding_too_many(tmp_path: Path):
    """More frames than expected are somebody else's, not finished work."""
    _fill(tmp_path / "frames_up", 12)
    assert frames_complete(tmp_path / "frames_up", 10) is True
    assert frames_complete(tmp_path / "frames_up", 10, exact=True) is False
    assert frames_complete(tmp_path / "frames_up", 12, exact=True) is True


def test_clearing_removes_only_the_frames(tmp_path: Path):
    _fill(tmp_path, 3)
    (tmp_path / "00000001.jpg").touch()
    (tmp_path / "notes.txt").touch()
    assert clear_frames(tmp_path) == 3
    assert sorted(path.name for path in tmp_path.iterdir()) == ["00000001.jpg", "notes.txt"]


def test_clearing_a_missing_directory_is_harmless(tmp_path: Path):
    assert clear_frames(tmp_path / "nowhere") == 0
