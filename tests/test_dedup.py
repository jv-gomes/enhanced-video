"""Tests for generating each distinct frame once.

The central claim is that the GPU is handed only the distinct frames and that
the output is nevertheless identical to what it would have been. The stub
binaries record their arguments, so the first half can be asserted directly;
the second half is checked by hashing both outputs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from videoenhance import dedup
from videoenhance.dedup import (
    DedupError,
    DedupStats,
    FrameCache,
    index_frames,
    link_or_copy,
    pixel_hashes,
    upscale_deduplicated,
)
from videoenhance.extract import extract
from videoenhance.pipeline import Options, WorkDir, run_pipeline
from videoenhance.probe import probe

from .conftest import HELD_DRAWINGS, HELD_FOR, requires_ffmpeg

CPU_ONLY = {"allow_hardware": False}


@pytest.fixture
def ncnn_stubs(stub_bin):
    """Stand-ins for both NCNN binaries, sharing one argument log."""
    stub_bin("REALESRGAN_BIN", name="realesrgan-stub")
    stub_bin("RIFE_BIN", name="rife-stub")
    return stub_bin


def frames_of(video: Path, tmp_path: Path, name: str = "work") -> Path:
    """Extract ``video`` and return the directory of frames."""
    work = WorkDir.for_input(video, base=tmp_path / name)
    return extract(probe(video), work).frames_dir


def cache_in(tmp_path: Path, **overrides) -> FrameCache:
    settings = {"model": "realesr-animevideov3", "scale": 2, "tile": 0, "frame_format": "png"}
    return FrameCache(tmp_path / "cache", **{**settings, **overrides})


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_held_frames_hash_the_same(held_video: Path, tmp_path: Path):
    """The premise: a drawing held for three frames is three equal hashes."""
    index = index_frames(frames_of(held_video, tmp_path))

    assert index.frames == HELD_DRAWINGS * HELD_FOR
    assert len(index.unique) == HELD_DRAWINGS
    assert [index.hashes.count(h) for h in index.unique] == [HELD_FOR] * HELD_DRAWINGS


@requires_ffmpeg
def test_distinct_frames_hash_differently(sample_video: Path, tmp_path: Path):
    """A moving test pattern has nothing to reuse, and must not pretend it does."""
    index = index_frames(frames_of(sample_video, tmp_path))
    assert len(index.unique) == index.frames


@requires_ffmpeg
def test_the_representative_is_a_frame_that_carries_the_hash(
    held_video: Path, tmp_path: Path
):
    frames = frames_of(held_video, tmp_path)
    index = index_frames(frames)
    for digest in index.unique:
        representative = index.representative(digest)
        assert representative.exists()
        assert index.hashes[index.names.index(representative)] == digest


@requires_ffmpeg
def test_hashing_an_empty_directory_is_an_error(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(DedupError, match="no png frames"):
        pixel_hashes(tmp_path / "empty")


@requires_ffmpeg
def test_a_gap_in_the_numbering_is_caught(held_video: Path, tmp_path: Path):
    """FFmpeg stops at a gap, so the hash count would silently disagree."""
    frames = frames_of(held_video, tmp_path)
    sorted(frames.glob("*.png"))[5].unlink()
    with pytest.raises(DedupError, match="gap"):
        pixel_hashes(frames)


# --------------------------------------------------------------------------
# linking
# --------------------------------------------------------------------------


def test_linking_shares_one_inode(tmp_path: Path):
    source = tmp_path / "a.png"
    source.write_bytes(b"frame")
    linked = link_or_copy(source, tmp_path / "b.png")

    assert linked.stat().st_ino == source.stat().st_ino
    assert source.stat().st_nlink == 2


def test_linking_falls_back_to_copying(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A filesystem without hard links must still work, just not for free."""
    source = tmp_path / "a.png"
    source.write_bytes(b"frame")
    monkeypatch.setattr(os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))

    copied = link_or_copy(source, tmp_path / "b.png")
    assert copied.read_bytes() == b"frame"
    assert copied.stat().st_ino != source.stat().st_ino


def test_linking_over_an_existing_file_replaces_it(tmp_path: Path):
    source = tmp_path / "a.png"
    source.write_bytes(b"new")
    destination = tmp_path / "b.png"
    destination.write_bytes(b"old")

    link_or_copy(source, destination)
    assert destination.read_bytes() == b"new"


# --------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------


def test_a_miss_then_a_hit(tmp_path: Path):
    cache = cache_in(tmp_path)
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"upscaled")

    assert cache.get("abc") is None
    cache.put("abc", frame)
    assert cache.get("abc") is not None
    assert cache.get("abc").read_bytes() == b"upscaled"


def test_the_key_separates_different_settings(tmp_path: Path):
    """Tile size changes the result at the seams, so it cannot be ignored."""
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"upscaled")
    cache_in(tmp_path, tile=0).put("abc", frame)

    assert cache_in(tmp_path, tile=128).get("abc") is None
    assert cache_in(tmp_path, scale=4).get("abc") is None
    assert cache_in(tmp_path, model="realesrgan-x4plus").get("abc") is None
    assert cache_in(tmp_path, tile=0).get("abc") is not None


def test_the_cache_trims_to_its_budget(tmp_path: Path):
    """Without a ceiling a long video's frames would accumulate all run."""
    cache = cache_in(tmp_path, budget=300)
    for index in range(10):
        frame = tmp_path / f"f{index}.png"
        frame.write_bytes(b"x" * 100)
        cache.put(f"hash{index}", frame)
        os.utime(cache.path_for(f"hash{index}"), (index, index))

    assert cache.size() == 1000
    assert cache.trim() == 7
    assert cache.size() <= 300
    # The oldest went first, so the most recent three survive.
    assert cache.get("hash9") is not None
    assert cache.get("hash0") is None


def test_no_budget_means_no_trimming(tmp_path: Path):
    cache = cache_in(tmp_path, budget=0)
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"x" * 10_000)
    cache.put("abc", frame)

    assert cache.trim() == 0
    assert cache.get("abc") is not None


def test_a_shared_inode_is_counted_once(tmp_path: Path):
    """Entries are hard links, so counting their sizes would overstate the cache."""
    cache = cache_in(tmp_path)
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"x" * 100)
    cache.put("one", frame)
    cache.put("two", frame)

    assert len(cache.entries()) == 2
    assert cache.size() == 100


# --------------------------------------------------------------------------
# the deduplicated stage
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_only_the_distinct_frames_reach_the_binary(
    held_video: Path, tmp_path: Path, ncnn_stubs, monkeypatch: pytest.MonkeyPatch
):
    """The central assertion: the GPU is handed ten frames, not thirty.

    Read from the call itself rather than from the scratch directory, which is
    cleared afterwards on purpose.
    """
    seen: dict[str, object] = {}
    original = dedup.upscale

    def spy(frames_in: Path, frames_out: Path, **kwargs):
        seen["given"] = len(list(frames_in.glob("*.png")))
        seen["expected"] = kwargs.get("expected_frames")
        return original(frames_in, frames_out, **kwargs)

    monkeypatch.setattr(dedup, "upscale", spy)

    frames = frames_of(held_video, tmp_path)
    result, stats = upscale_deduplicated(
        frames,
        tmp_path / "up",
        scratch=tmp_path / "dedup",
        cache=cache_in(tmp_path),
    )

    assert seen == {"given": HELD_DRAWINGS, "expected": HELD_DRAWINGS}
    assert stats.frames == HELD_DRAWINGS * HELD_FOR
    assert stats.unique == HELD_DRAWINGS
    assert stats.upscaled == HELD_DRAWINGS
    assert stats.saved == HELD_DRAWINGS * (HELD_FOR - 1)
    assert stats.ratio == pytest.approx(2 / 3)
    assert result.frame_count == HELD_DRAWINGS * HELD_FOR


@requires_ffmpeg
def test_the_scratch_directories_are_left_empty(
    held_video: Path, tmp_path: Path, ncnn_stubs
):
    """Load-bearing, not tidiness.

    The scratch output holds hard links to cache entries, so a link left behind
    would keep a trimmed entry's data alive and the budget would stop freeing
    anything.
    """
    scratch = tmp_path / "dedup"
    upscale_deduplicated(
        frames_of(held_video, tmp_path),
        tmp_path / "up",
        scratch=scratch,
        cache=cache_in(tmp_path),
    )
    assert list(scratch.rglob("*.png")) == []


@requires_ffmpeg
def test_the_output_holds_every_frame_with_duplicates_linked(
    held_video: Path, tmp_path: Path, ncnn_stubs
):
    """Thirty frames out, but only ten of them occupying disk."""
    frames = frames_of(held_video, tmp_path)
    out = tmp_path / "up"
    upscale_deduplicated(
        frames, out, scratch=tmp_path / "dedup", cache=cache_in(tmp_path)
    )

    produced = sorted(out.glob("*.png"))
    assert len(produced) == HELD_DRAWINGS * HELD_FOR
    assert len({frame.stat().st_ino for frame in produced}) == HELD_DRAWINGS
    # Each distinct frame is pointed at by its three positions plus the cache.
    assert produced[0].stat().st_nlink == HELD_FOR + 1


@requires_ffmpeg
def test_a_cache_hit_skips_the_binary_entirely(
    held_video: Path, tmp_path: Path, ncnn_stubs
):
    """What makes a frame generated for one chunk free for the next."""
    frames = frames_of(held_video, tmp_path)
    cache = cache_in(tmp_path)
    upscale_deduplicated(
        frames, tmp_path / "up1", scratch=tmp_path / "dedup", cache=cache
    )
    before = len(ncnn_stubs.log.read_text().splitlines())

    _, stats = upscale_deduplicated(
        frames, tmp_path / "up2", scratch=tmp_path / "dedup", cache=cache
    )

    assert stats.cached == HELD_DRAWINGS
    assert stats.upscaled == 0
    assert len(ncnn_stubs.log.read_text().splitlines()) == before
    assert len(list((tmp_path / "up2").glob("*.png"))) == HELD_DRAWINGS * HELD_FOR


@requires_ffmpeg
def test_a_finished_stage_is_not_redone(held_video: Path, tmp_path: Path, ncnn_stubs):
    frames = frames_of(held_video, tmp_path)
    out = tmp_path / "up"
    upscale_deduplicated(
        frames, out, scratch=tmp_path / "dedup", cache=cache_in(tmp_path)
    )
    before = len(ncnn_stubs.log.read_text().splitlines())

    result, _ = upscale_deduplicated(
        frames, out, scratch=tmp_path / "dedup", cache=cache_in(tmp_path / "other")
    )
    assert result.skipped is True
    assert len(ncnn_stubs.log.read_text().splitlines()) == before


@requires_ffmpeg
def test_all_distinct_frames_cost_one_call_and_no_links(
    sample_video: Path, tmp_path: Path, ncnn_stubs
):
    """Live-action footage gains nothing, and must lose nothing either."""
    frames = frames_of(sample_video, tmp_path)
    result, stats = upscale_deduplicated(
        frames, tmp_path / "up", scratch=tmp_path / "dedup", cache=cache_in(tmp_path)
    )

    assert stats.unique == stats.frames
    assert stats.saved == 0
    assert stats.render() == f"{stats.frames} distinct frames, nothing to reuse"
    assert result.frame_count == stats.frames


def test_stats_describe_what_was_reused():
    assert DedupStats(frames=90, unique=30, cached=0, upscaled=30).render() == (
        "30 of 90 frames generated (67% saved)"
    )
    assert "reused from earlier chunks" in (
        DedupStats(frames=90, unique=30, cached=12, upscaled=18).render()
    )


# --------------------------------------------------------------------------
# the output must not change
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_dedup_does_not_change_the_output(held_video: Path, tmp_path: Path, ncnn_stubs):
    """The test that matters: reuse has to be invisible in the result.

    Both runs go through the whole pipeline and their frames are compared by
    pixel hash. A dedup that moves a single pixel is a bug, not an optimisation.
    """
    plain = run_pipeline(
        held_video,
        tmp_path / "plain.mkv",
        Options(scale=2, target_fps=60, dedup=False, **CPU_ONLY),
        work=WorkDir.for_input(held_video, base=tmp_path / "w1"),
    )
    deduped = run_pipeline(
        held_video,
        tmp_path / "deduped.mkv",
        Options(scale=2, target_fps=60, dedup=True, **CPU_ONLY),
        work=WorkDir.for_input(held_video, base=tmp_path / "w2"),
    )

    assert deduped.frames == plain.frames
    assert pixel_hashes_of(deduped.output) == pixel_hashes_of(plain.output)


def pixel_hashes_of(video: Path) -> list[str]:
    """Every frame of ``video`` as a pixel hash, for comparing two outputs."""
    import subprocess

    completed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-an", "-f", "framemd5", "-"],
        capture_output=True, text=True, check=True,
    )
    return [
        line.rsplit(",", 1)[-1].strip()
        for line in completed.stdout.splitlines()
        if line and not line.startswith("#")
    ]
