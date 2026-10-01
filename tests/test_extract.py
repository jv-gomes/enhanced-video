"""Tests for frame extraction, including the VFR normalisation path."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from videoenhance.extract import ExtractError, extract, target_cfr, to_cfr
from videoenhance.pipeline import WorkDir, count_frames_on_disk
from videoenhance.probe import VideoInfo, probe
from videoenhance.process import ToolError

from .conftest import requires_ffmpeg


@pytest.fixture
def work(tmp_path: Path) -> WorkDir:
    return WorkDir.for_input(Path("clip.mp4"), base=tmp_path)


@requires_ffmpeg
def test_extract_writes_every_frame(sample_video: Path, work: WorkDir):
    info = probe(sample_video)
    result = extract(info, work)
    assert result.frame_count == 90
    assert count_frames_on_disk(result.frames_dir) == 90
    assert (result.frames_dir / "00000001.png").exists()
    assert (result.frames_dir / "00000090.png").exists()


@requires_ffmpeg
def test_cfr_source_keeps_its_own_audio(sample_video: Path, work: WorkDir):
    result = extract(probe(sample_video), work)
    assert result.normalised is False
    assert result.audio_source == sample_video
    assert result.has_audio is True
    assert result.fps == Fraction(30)
    assert result.duration == pytest.approx(3.0, abs=0.05)


@requires_ffmpeg
def test_second_run_skips_the_work(sample_video: Path, work: WorkDir, caplog):
    info = probe(sample_video)
    extract(info, work)
    mtime = (work.frames_in / "00000001.png").stat().st_mtime_ns
    with caplog.at_level("INFO"):
        again = extract(info, work)
    assert again.frame_count == 90
    assert (work.frames_in / "00000001.png").stat().st_mtime_ns == mtime
    assert "skipping extract" in caplog.text


@requires_ffmpeg
def test_force_re_extracts(sample_video: Path, work: WorkDir):
    info = probe(sample_video)
    extract(info, work)
    (work.frames_in / "00000001.png").unlink()
    result = extract(info, work, force=True)
    assert result.frame_count == 90


@requires_ffmpeg
def test_jpg_frames_are_smaller(sample_video: Path, tmp_path: Path):
    info = probe(sample_video)
    png = extract(info, WorkDir.for_input(Path("p.mp4"), base=tmp_path))
    jpg = extract(info, WorkDir.for_input(Path("j.mp4"), base=tmp_path), frame_format="jpg")
    assert jpg.frame_count == png.frame_count
    assert count_frames_on_disk(jpg.frames_dir, "jpg") == 90
    assert count_frames_on_disk(jpg.frames_dir, "png") == 0

    def total(directory: Path, ext: str) -> int:
        return sum(p.stat().st_size for p in directory.glob(f"*.{ext}"))

    assert total(jpg.frames_dir, "jpg") < total(png.frames_dir, "png")


@requires_ffmpeg
def test_vfr_source_is_normalised_before_extraction(vfr_video: Path, work: WorkDir):
    info = probe(vfr_video)
    assert info.is_vfr is True

    result = extract(info, work)
    assert result.normalised is True
    # The audio must come from the CFR copy, not the drifting original.
    assert result.audio_source == work.cfr_video
    assert work.cfr_video.exists()

    normalised = probe(work.cfr_video)
    assert normalised.is_vfr is False
    assert normalised.has_audio is True
    # The extracted frame timeline now matches the file the audio comes from.
    assert result.duration == pytest.approx(normalised.duration, abs=0.1)


@requires_ffmpeg
def test_cfr_copy_is_reused_on_a_second_run(vfr_video: Path, work: WorkDir, caplog):
    info = probe(vfr_video)
    extract(info, work)
    stamp = work.cfr_video.stat().st_mtime_ns
    with caplog.at_level("INFO"):
        extract(info, work, force=True)
    assert work.cfr_video.stat().st_mtime_ns == stamp
    assert "reusing CFR copy" in caplog.text


@requires_ffmpeg
def test_no_partial_files_are_left_behind(vfr_video: Path, work: WorkDir):
    extract(probe(vfr_video), work)
    assert list(work.root.glob("*.part*")) == []


@requires_ffmpeg
def test_target_cfr_prefers_the_average_rate(vfr_video: Path):
    info = probe(vfr_video)
    assert target_cfr(info) == info.avg_frame_rate


@requires_ffmpeg
def test_to_cfr_names_its_output_with_a_usable_extension(vfr_video: Path, work: WorkDir):
    work.create()
    # FFmpeg picks the container from the extension, so the temporary name must
    # keep it last.
    produced = to_cfr(probe(vfr_video), work.cfr_video)
    assert produced == work.cfr_video
    assert produced.suffix == ".mkv"


@requires_ffmpeg
def test_extract_raises_when_ffmpeg_cannot_read_the_source(tmp_path: Path, work: WorkDir):
    # A VideoInfo can describe a file that is not decodable, e.g. because it was
    # truncated between the probe and the extract.
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"\x00" * 128)
    info = VideoInfo(
        path=broken,
        width=320,
        height=240,
        r_frame_rate=Fraction(30),
        avg_frame_rate=Fraction(30),
        nb_frames=90,
        duration=3.0,
        has_audio=False,
    )
    with pytest.raises((ExtractError, ToolError)):
        extract(info, work)
