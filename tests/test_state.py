"""Tests for the saved job record.

The record exists to answer two questions the files cannot: with which settings
was this started, and what did I leave unfinished. So these tests are mostly
about the first one — the comparison that turns "resume with different options"
from silent corruption into a refusal — and about never trusting the file more
than the disk.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from videoenhance import state
from videoenhance.workdir import WorkDir

SETTINGS = {
    "scale": 2,
    "target_fps": 60.0,
    "model": "realesr-animevideov3",
    "rife_model": "rife-v4.6",
    "frame_format": "png",
    "order": "upscale-first",
    "chunk_seconds": 30.0,
}


def make_job(tmp_path: Path, name: str = "clip-1234abcd", **overrides) -> state.Job:
    work = WorkDir(tmp_path / name)
    settings = {**SETTINGS, **overrides}
    return state.save(
        state.for_input(work, tmp_path / "clip.mp4", tmp_path / "out.mp4", settings)
    )


# --------------------------------------------------------------------------
# reading and writing
# --------------------------------------------------------------------------


def test_a_job_survives_a_round_trip(tmp_path: Path):
    saved = make_job(tmp_path)
    loaded = state.load(saved.root)

    assert loaded is not None
    assert loaded.id == "clip-1234abcd"
    assert loaded.options == SETTINGS
    assert loaded.output == (tmp_path / "out.mp4").resolve()


def test_saving_leaves_no_temporary_file(tmp_path: Path):
    """The write is atomic, so an interrupt cannot replace a good record."""
    saved = make_job(tmp_path)
    assert saved.path.exists()
    assert list(saved.root.glob("*.tmp")) == []


def test_an_unreadable_record_is_treated_as_absent(tmp_path: Path):
    """Redoing work beats refusing to run because a scratch file got truncated."""
    saved = make_job(tmp_path)
    saved.path.write_text("{ this is not json")
    assert state.load(saved.root) is None


def test_a_record_from_another_version_is_ignored(tmp_path: Path):
    saved = make_job(tmp_path)
    data = json.loads(saved.path.read_text())
    data["version"] = state.STATE_VERSION + 1
    saved.path.write_text(json.dumps(data))
    assert state.load(saved.root) is None


def test_a_record_missing_its_fields_is_ignored(tmp_path: Path):
    saved = make_job(tmp_path)
    saved.path.write_text(json.dumps({"version": state.STATE_VERSION}))
    assert state.load(saved.root) is None


def test_no_record_means_no_job(tmp_path: Path):
    assert state.load(tmp_path / "nothing-here") is None


def test_the_cut_is_stored_on_the_job(tmp_path: Path):
    saved = make_job(tmp_path)
    updated = state.record_cut(saved, [Path("chunk_0000.mkv"), Path("chunk_0001.mkv")], 15.0)

    assert updated.chunks == ["chunk_0000.mkv", "chunk_0001.mkv"]
    assert updated.chunked is True
    reloaded = state.load(saved.root)
    assert reloaded is not None
    assert reloaded.seconds == 15.0
    assert reloaded.chunks == updated.chunks


def test_recording_the_encoder_does_not_lose_the_cut(tmp_path: Path):
    """Both are written to the same file, and Job is frozen.

    A caller that kept the record it passed in would hold a stale copy, and its
    next save would wipe the cut back out — which would make a resumed run
    re-cut the source and risk lining the parts up against different chunks.
    """
    saved = make_job(tmp_path)
    with_cut = state.record_cut(saved, [Path("chunk_0000.mkv")], 30.0)
    state.record_encoder(with_cut, "libx265")

    reloaded = state.load(saved.root)
    assert reloaded is not None
    assert reloaded.chunks == ["chunk_0000.mkv"]
    assert reloaded.encoder == "libx265"


# --------------------------------------------------------------------------
# comparing options
# --------------------------------------------------------------------------


def test_comparing_names_every_changed_option(tmp_path: Path):
    job = make_job(tmp_path)
    changed = state.compare(job, {**SETTINGS, "scale": 4, "target_fps": 30.0})
    assert changed == {"scale": (2, 4), "target_fps": (60.0, 30.0)}


def test_the_same_options_are_no_change(tmp_path: Path):
    job = make_job(tmp_path)
    assert state.compare(job, dict(SETTINGS)) == {}


def test_an_int_and_the_same_float_are_not_a_change(tmp_path: Path):
    """JSON brings numbers back as int or float depending on how they went in."""
    job = make_job(tmp_path)
    assert state.compare(job, {"scale": 2.0, "target_fps": 60}) == {}


def test_options_that_do_not_change_pixels_are_not_compared(tmp_path: Path):
    """Resuming on another GPU, or with a smaller tile, is legitimate."""
    job = make_job(tmp_path)
    assert state.compare(job, {"gpu": 1, "tile": 128, "keep_temp": True}) == {}
    assert "gpu" not in state.PIXEL_OPTIONS
    assert "tile" not in state.PIXEL_OPTIONS


def test_the_mismatch_message_names_the_flags_and_the_way_out(tmp_path: Path):
    job = state.record_cut(make_job(tmp_path), [Path("chunk_0000.mkv")], 30.0)
    message = state.render_mismatch(job, state.compare(job, {"scale": 4}))

    assert "--scale" in message
    assert "2 -> 4" in message
    assert "0 of 1 parts" in message
    assert "--restart" in message


def test_options_of_reads_them_off_an_options_object():
    from videoenhance.pipeline import Options

    taken = state.options_of(Options(scale=3, target_fps=48.0))
    assert taken["scale"] == 3
    assert taken["target_fps"] == 48.0
    assert set(taken) == set(state.PIXEL_OPTIONS)


# --------------------------------------------------------------------------
# the input file itself
# --------------------------------------------------------------------------


def test_the_same_path_holding_a_different_file_is_detected(tmp_path: Path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"first")
    job = make_job(tmp_path)
    assert state.input_changed(job) is False

    source.write_bytes(b"a different video entirely")
    assert state.input_changed(job) is True


def test_a_missing_input_is_not_reported_as_changed(tmp_path: Path):
    """It will fail with a better message when it is probed."""
    job = make_job(tmp_path)
    assert state.input_changed(job) is False


# --------------------------------------------------------------------------
# listing and looking up
# --------------------------------------------------------------------------


def test_listing_finds_every_job_newest_first(tmp_path: Path):
    """Most recently touched first, so the one you just stopped is on top."""
    older = make_job(tmp_path, "old-aaaaaaaa")
    data = json.loads(older.path.read_text())
    data["updated"] = "2020-01-01T00:00:00+00:00"
    older.path.write_text(json.dumps(data))

    newer = make_job(tmp_path, "new-bbbbbbbb")

    assert [job.id for job in state.list_jobs(tmp_path)] == [newer.id, older.id]


def test_listing_an_empty_directory_finds_nothing(tmp_path: Path):
    assert state.list_jobs(tmp_path) == []
    assert state.list_jobs(tmp_path / "not-created") == []


def test_a_directory_without_a_record_is_not_a_job(tmp_path: Path):
    (tmp_path / "leftovers").mkdir()
    assert state.list_jobs(tmp_path) == []


def test_finding_a_job_by_its_full_id(tmp_path: Path):
    job = make_job(tmp_path)
    assert state.find("clip-1234abcd", tmp_path).id == job.id


def test_finding_a_job_by_an_unambiguous_prefix(tmp_path: Path):
    make_job(tmp_path)
    assert state.find("clip", tmp_path).id == "clip-1234abcd"


def test_an_ambiguous_prefix_lists_the_candidates(tmp_path: Path):
    make_job(tmp_path, "clip-11111111")
    make_job(tmp_path, "clip-22222222")
    with pytest.raises(state.StateError, match="more than one job"):
        state.find("clip", tmp_path)


def test_an_unknown_id_says_what_there_is(tmp_path: Path):
    make_job(tmp_path)
    with pytest.raises(state.StateError, match="clip-1234abcd"):
        state.find("nope", tmp_path)


def test_resuming_with_no_jobs_at_all_says_so(tmp_path: Path):
    with pytest.raises(state.StateError, match="no saved jobs"):
        state.find("anything", tmp_path)


# --------------------------------------------------------------------------
# progress, which is read off the disk and never stored
# --------------------------------------------------------------------------


def test_chunked_progress_counts_the_parts_on_disk(tmp_path: Path):
    job = state.record_cut(
        make_job(tmp_path), [Path(f"chunk_{i:04d}.mkv") for i in range(4)], 30.0
    )
    assert state.progress(job) == "0/4 parts"

    parts = WorkDir(job.root).parts
    parts.mkdir(parents=True)
    (parts / "part_0000.mkv").write_bytes(b"done")
    (parts / "part_0001.mkv").write_bytes(b"done")
    # Nothing was re-saved: the files are the progress.
    assert state.progress(job) == "2/4 parts"


def test_single_pass_progress_reports_the_furthest_stage(tmp_path: Path):
    job = make_job(tmp_path)
    work = WorkDir(job.root)
    assert state.progress(job) == "not started"

    work.frames_in.mkdir(parents=True)
    (work.frames_in / "00000001.png").write_bytes(b"frame")
    assert state.progress(job) == "1 frames extracted"

    work.frames_up.mkdir(parents=True)
    (work.frames_up / "00000001.png").write_bytes(b"frame")
    assert state.progress(job) == "1 frames upscaled"


# --------------------------------------------------------------------------
# presentation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (timedelta(seconds=5), "just now"),
        (timedelta(minutes=20), "20 minutes ago"),
        (timedelta(hours=3), "3 hours ago"),
        (timedelta(days=1), "1 day ago"),
        (timedelta(days=2), "2 days ago"),
        (timedelta(days=30), "4 weeks ago"),
    ],
)
def test_ages_are_readable(delta: timedelta, expected: str):
    now = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    stamp = (now - delta).isoformat()
    assert state.human_age(stamp, now=now) == expected


def test_an_unparseable_age_does_not_crash_the_listing():
    assert state.human_age("") == "unknown"
    assert state.human_age("whenever") == "unknown"


def test_the_listing_says_when_there_is_nothing():
    assert state.describe([]) == "No unfinished jobs."


def test_the_listing_shows_the_job_and_how_to_continue(tmp_path: Path):
    make_job(tmp_path)
    table = state.describe(state.list_jobs(tmp_path))

    assert "clip-1234abcd" in table
    assert "clip.mp4" in table
    assert "2x, 60fps, chunk 30s" in table
    assert "--resume clip-1234abcd" in table
