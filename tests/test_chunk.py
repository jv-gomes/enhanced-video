"""Tests for chunked processing.

Two things are being checked here, and they fail in different ways. The cut has
to cover the whole source exactly once — a gap or an overlap would silently
shorten or stutter the output — and the run has to *not* keep frames around,
which is the entire reason the mode exists. The GPU binaries are stubs (see
``conftest.write_stub``), so what these exercise is the orchestration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videoenhance import chunk, state
from videoenhance.chunk import ChunkError, chunk_frame_count, run_chunked, split
from videoenhance.pipeline import Options
from videoenhance.probe import probe
from videoenhance.workdir import WorkDir

from .conftest import requires_ffmpeg

# The parts are encoded on the CPU so the result does not depend on whichever
# GPU the suite happens to run on, as in tests/test_e2e.py.
CPU_ONLY = {"allow_hardware": False}


@pytest.fixture
def ncnn_stubs(stub_bin):
    """Stand-ins for both NCNN binaries, sharing one argument log."""
    stub_bin("REALESRGAN_BIN", name="realesrgan-stub")
    stub_bin("RIFE_BIN", name="rife-stub")
    return stub_bin


def work_for(video: Path, tmp_path: Path) -> WorkDir:
    return WorkDir.for_input(video, base=tmp_path / "work")


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_split_cuts_a_keyframed_source_by_copy(keyframed_video: Path, tmp_path: Path):
    """The ordinary path: frequent keyframes, so no re-encode is needed."""
    info = probe(keyframed_video)
    plan = split(info, work_for(keyframed_video, tmp_path), 1.0)

    assert len(plan) == 4
    assert plan.recut == 0
    assert plan.has_audio is True
    assert plan.audio_source == keyframed_video
    for piece in plan.chunks:
        assert piece.exists()


@requires_ffmpeg
def test_the_chunks_together_cover_the_whole_source(keyframed_video: Path, tmp_path: Path):
    """No gap and no overlap: the durations have to add back up to the source."""
    info = probe(keyframed_video)
    plan = split(info, work_for(keyframed_video, tmp_path), 1.0)

    total = sum(probe(piece).duration for piece in plan.chunks)
    assert total == pytest.approx(info.duration, abs=0.1)


@requires_ffmpeg
def test_chunks_carry_no_audio(keyframed_video: Path, tmp_path: Path):
    """Audio is muxed once at the end, so no chunk should carry a copy of it.

    Per-part audio would mean a ``-shortest`` at every boundary, and that trim
    accumulates into drift over a long video.
    """
    info = probe(keyframed_video)
    plan = split(info, work_for(keyframed_video, tmp_path), 1.0)

    assert all(probe(piece).has_audio is False for piece in plan.chunks)


def job_for(video: Path, work: WorkDir, **options) -> state.Job:
    """A saved job record, the way the CLI and run_chunked make one."""
    settings = {"scale": 2, "target_fps": 60.0, "chunk_seconds": 1.0, **options}
    return state.save(state.for_input(work, video, work.root / "out.mkv", settings))


@requires_ffmpeg
def test_splitting_twice_reuses_the_recorded_cut(keyframed_video: Path, tmp_path: Path):
    """The cut is written to the job, so a resumed run does not re-cut."""
    info = probe(keyframed_video)
    work = work_for(keyframed_video, tmp_path)
    job = job_for(keyframed_video, work)

    first = split(info, work, 1.0, job=job)
    stamps = {piece: piece.stat().st_mtime_ns for piece in first.chunks}

    second = split(info, work, 1.0, job=state.load(work.root))
    assert second.skipped is True
    assert second.chunks == first.chunks
    assert {piece: piece.stat().st_mtime_ns for piece in second.chunks} == stamps


@requires_ffmpeg
def test_the_cut_is_recorded_on_the_job(keyframed_video: Path, tmp_path: Path):
    info = probe(keyframed_video)
    work = work_for(keyframed_video, tmp_path)
    split(info, work, 1.0, job=job_for(keyframed_video, work))

    saved = state.load(work.root)
    assert saved is not None
    assert saved.seconds == 1.0
    assert saved.chunks == [f"chunk_{i:04d}.mkv" for i in range(4)]


@requires_ffmpeg
def test_a_different_chunk_length_recuts(keyframed_video: Path, tmp_path: Path):
    """The job records the length it was cut for, so changing it re-cuts."""
    info = probe(keyframed_video)
    work = work_for(keyframed_video, tmp_path)
    job = job_for(keyframed_video, work)

    assert len(split(info, work, 1.0, job=job)) == 4
    assert len(split(info, work, 2.0, job=state.load(work.root))) == 2


@requires_ffmpeg
def test_split_recuts_a_chunk_whose_keyframes_are_too_sparse(
    longgop_video: Path, tmp_path: Path
):
    """A single-keyframe source cannot be cut by copy, so it is re-encoded.

    Without this fallback the cut would silently return one chunk the length of
    the whole video, and the mode would do nothing at all.
    """
    info = probe(longgop_video)
    plan = split(info, work_for(longgop_video, tmp_path), 1.0)

    assert plan.recut == 1
    assert len(plan) == 4
    assert all(probe(piece).duration <= 2.0 for piece in plan.chunks)


@requires_ffmpeg
def test_recut_pieces_stay_in_playback_order(longgop_video: Path, tmp_path: Path):
    """The re-cut names must sort between their neighbours, not after them."""
    info = probe(longgop_video)
    plan = split(info, work_for(longgop_video, tmp_path), 1.0)

    assert plan.chunks == sorted(plan.chunks)
    assert [piece.name for piece in plan.chunks] == [
        "chunk_0000-00.mkv",
        "chunk_0000-01.mkv",
        "chunk_0000-02.mkv",
        "chunk_0000-03.mkv",
    ]


@requires_ffmpeg
def test_split_normalises_a_vfr_source_before_cutting(vfr_video: Path, tmp_path: Path):
    """One CFR copy for the whole file, and the audio comes from it.

    Normalising per chunk would let each one settle on its own average rate.
    """
    info = probe(vfr_video)
    assert info.is_vfr is True
    work = work_for(vfr_video, tmp_path)

    plan = split(info, work, 1.0)
    assert work.cfr_video.exists()
    assert plan.audio_source == work.cfr_video
    assert all(probe(piece).is_vfr is False for piece in plan.chunks)


def test_split_rejects_a_non_positive_length(tmp_path: Path):
    """Checked before anything is probed, so the source does not matter."""
    with pytest.raises(ValueError, match="above 0 seconds"):
        split(None, WorkDir(tmp_path), 0)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# run_chunked
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_a_chunked_run_matches_the_source_duration(
    keyframed_video: Path, tmp_path: Path, ncnn_stubs
):
    source = probe(keyframed_video)
    output = tmp_path / "out.mkv"

    result = run_chunked(
        keyframed_video,
        output,
        Options(scale=2, target_fps=60, chunk_seconds=1.0, **CPU_ONLY),
        work=work_for(keyframed_video, tmp_path),
    )

    assert output.exists()
    joined = probe(output)
    assert joined.has_audio is True
    assert joined.duration == pytest.approx(source.duration, abs=0.2)
    assert float(result.fps) == 60
    # 4 seconds of 30 fps doubled to 60, give or take a frame per boundary.
    assert result.frames == pytest.approx(240, abs=4)


@requires_ffmpeg
def test_a_chunked_run_leaves_nothing_on_disk(
    keyframed_video: Path, tmp_path: Path, ncnn_stubs
):
    """The whole point of the mode: no frames, no chunks, no parts afterwards."""
    work = work_for(keyframed_video, tmp_path)
    run_chunked(
        keyframed_video,
        tmp_path / "out.mkv",
        Options(target_fps=60, chunk_seconds=1.0, **CPU_ONLY),
        work=work,
    )
    assert work.root.exists() is False


@requires_ffmpeg
def test_frames_of_a_finished_chunk_are_gone_before_the_next_one(
    keyframed_video: Path, tmp_path: Path, ncnn_stubs
):
    """Peak disk is one chunk, so ``current/`` must not accumulate.

    With --keep-temp the parts and chunks survive, but the frame directories of
    a finished chunk still have to be cleared, otherwise a long run grows
    exactly the way the single-pass mode does.
    """
    work = work_for(keyframed_video, tmp_path)
    run_chunked(
        keyframed_video,
        tmp_path / "out.mkv",
        Options(target_fps=60, chunk_seconds=1.0, keep_temp=True, **CPU_ONLY),
        work=work,
    )

    frames = list(work.root.rglob("*.png"))
    assert frames == []
    assert len(list(work.parts.glob("part_*.mkv"))) == 4


@requires_ffmpeg
def test_a_chunked_run_resumes_from_the_parts_it_already_has(
    keyframed_video: Path, tmp_path: Path, ncnn_stubs
):
    """A rerun must re-encode only the missing part."""
    work = work_for(keyframed_video, tmp_path)
    options = Options(target_fps=60, chunk_seconds=1.0, keep_temp=True, **CPU_ONLY)

    run_chunked(keyframed_video, tmp_path / "first.mkv", options, work=work)
    parts = sorted(work.parts.glob("part_*.mkv"))
    assert len(parts) == 4

    parts[2].unlink()
    kept = {part: part.stat().st_mtime_ns for part in parts if part.exists()}

    result = run_chunked(keyframed_video, tmp_path / "second.mkv", options, work=work)

    assert (tmp_path / "second.mkv").exists()
    assert parts[2].exists()
    assert {part: part.stat().st_mtime_ns for part in kept} == kept
    assert "3 already encoded" in result.stages[0].detail


@requires_ffmpeg
def test_a_chunked_run_of_a_silent_video_stays_silent(
    silent_video: Path, tmp_path: Path, ncnn_stubs
):
    output = tmp_path / "silent.mkv"
    run_chunked(
        silent_video,
        output,
        Options(target_fps=60, chunk_seconds=0.5, **CPU_ONLY),
        work=work_for(silent_video, tmp_path),
    )
    assert probe(output).has_audio is False


@requires_ffmpeg
def test_a_chunked_vfr_run_keeps_the_audio_aligned(
    vfr_video: Path, tmp_path: Path, ncnn_stubs
):
    """The VFR case, which is where audio sync actually goes wrong."""
    work = work_for(vfr_video, tmp_path)
    output = tmp_path / "vfr.mkv"
    run_chunked(
        vfr_video,
        output,
        Options(target_fps=60, chunk_seconds=0.5, keep_temp=True, **CPU_ONLY),
        work=work,
    )

    joined = probe(output)
    normalised = probe(work.cfr_video)
    assert joined.has_audio is True
    assert joined.duration == pytest.approx(normalised.duration, abs=0.2)


@requires_ffmpeg
def test_a_chunked_run_refuses_to_overwrite_the_input(sample_video: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="refusing to overwrite"):
        run_chunked(sample_video, sample_video, Options(chunk_seconds=1.0))


def fake_pipeline(encoders: list[str], seen: list):
    """A stand-in for run_pipeline that hands out encoders from a list.

    The encoder a part ends up with is decided at run time by the fallback
    chain, which is exactly what cannot be provoked on demand, so the two tests
    below drive it from here instead.
    """
    from fractions import Fraction

    from videoenhance.pipeline import Result

    def fake(chunk_path, part, options, **kwargs):
        seen.append((chunk_path, options))
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"part")
        return Result(
            output=part,
            encoder=encoders[len(seen) - 1],
            fps=Fraction(60),
            frames=10,
            stages=[],
            work=part.parent,
        )

    return fake


@requires_ffmpeg
def test_every_part_after_the_first_is_asked_for_the_same_encoder(
    keyframed_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The parts are joined by copy, so they have to share a codec."""
    seen: list = []
    monkeypatch.setattr(chunk, "run_pipeline", fake_pipeline(["libx265"] * 4, seen))
    monkeypatch.setattr(chunk, "concat", lambda *a, **k: None)

    run_chunked(
        keyframed_video,
        tmp_path / "out.mkv",
        Options(chunk_seconds=1.0),
        work=work_for(keyframed_video, tmp_path),
    )

    assert len(seen) == 4
    assert seen[0][1].prefer_encoder is None
    assert [options.prefer_encoder for _, options in seen[1:]] == ["libx265"] * 3
    # No part carries audio; it is muxed once by concat.
    assert all(options.mux_audio is False for _, options in seen)


@requires_ffmpeg
def test_parts_with_different_encoders_are_not_joined_by_copy(
    keyframed_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Copying mismatched codecs into one file would produce a broken video."""
    seen: list = []
    monkeypatch.setattr(
        chunk, "run_pipeline", fake_pipeline(["hevc_vaapi", "libx265", "libx265", "libx265"], seen)
    )

    with pytest.raises(ChunkError, match="different encoders"):
        run_chunked(
            keyframed_video,
            tmp_path / "out.mkv",
            Options(chunk_seconds=1.0),
            work=work_for(keyframed_video, tmp_path),
        )


# --------------------------------------------------------------------------
# the disk estimate helpers
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_the_disk_estimate_counts_one_chunk(sample_video: Path):
    """The estimate has to describe the run that happens, not the whole file."""
    info = probe(sample_video)
    assert info.nb_frames == 90
    # 1 second of 30 fps.
    assert chunk_frame_count(info, 1.0) == 30
    # A chunk longer than the file is still only the file.
    assert chunk_frame_count(info, 600.0) == 90
    assert chunk_frame_count(info, 0) == 90


@requires_ffmpeg
def test_the_plan_describes_how_many_chunks(sample_video: Path):
    info = probe(sample_video)
    assert chunk.describe_plan(info, 1.0).startswith("about 3 x 1s")


@requires_ffmpeg
def test_a_chunked_run_does_not_create_the_single_pass_directories(
    keyframed_video: Path, tmp_path: Path, ncnn_stubs
):
    """Its frames live in ``current/``, so the top-level ones would be empty."""
    work = work_for(keyframed_video, tmp_path)
    run_chunked(
        keyframed_video,
        tmp_path / "out.mkv",
        Options(target_fps=60, chunk_seconds=1.0, keep_temp=True, **CPU_ONLY),
        work=work,
    )
    assert work.frames_in.exists() is False
    assert work.frames_up.exists() is False
    assert work.frames_out.exists() is False
