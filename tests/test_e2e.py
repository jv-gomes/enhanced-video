"""End-to-end round trip: video -> frames -> video, with no model in the loop.

This is the milestone's real test. It does not check the models; it checks that
the plumbing around them preserves what matters — the frame count, the
duration, the resolution and the audio — because those are what break silently
and only become visible hours into a real run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videoenhance.encode import encode
from videoenhance.extract import extract
from videoenhance.pipeline import WorkDir
from videoenhance.probe import probe

from .conftest import requires_ffmpeg

# The round trip is encoded on the CPU so the result does not depend on
# whichever GPU the suite happens to run on.
CPU_ONLY = {"allow_hardware": False}


@requires_ffmpeg
def test_round_trip_preserves_duration_and_audio(sample_video: Path, tmp_path: Path):
    source = probe(sample_video)
    work = WorkDir.for_input(sample_video, base=tmp_path)

    extracted = extract(source, work)
    assert extracted.frame_count == source.nb_frames

    output = tmp_path / "round_trip.mp4"
    encode(
        extracted.frames_dir,
        output,
        fps=extracted.fps,
        audio_source=extracted.audio_source,
        **CPU_ONLY,
    )

    result = probe(output)
    assert (result.width, result.height) == (source.width, source.height)
    assert result.r_frame_rate == source.r_frame_rate
    assert result.has_audio is True
    assert result.duration == pytest.approx(source.duration, abs=0.1)
    assert result.nb_frames == pytest.approx(source.nb_frames, abs=1)


@requires_ffmpeg
def test_encoding_at_double_the_rate_halves_the_duration(sample_video: Path, tmp_path: Path):
    """Encoding the same frames at double the rate halves the duration.

    Proof that the frame rate reaches the container: this is exactly what the
    interpolation stage will rely on once it produces twice the frames.
    """
    source = probe(sample_video)
    work = WorkDir.for_input(sample_video, base=tmp_path)
    extracted = extract(source, work)

    output = tmp_path / "double_rate.mp4"
    encode(extracted.frames_dir, output, fps=source.fps * 2, **CPU_ONLY)

    result = probe(output)
    assert result.fps == pytest.approx(source.fps * 2)
    assert result.duration == pytest.approx(source.duration / 2, abs=0.1)


@requires_ffmpeg
def test_round_trip_of_a_silent_video_stays_silent(silent_video: Path, tmp_path: Path):
    source = probe(silent_video)
    work = WorkDir.for_input(silent_video, base=tmp_path)
    extracted = extract(source, work)

    output = tmp_path / "silent_round_trip.mp4"
    encode(
        extracted.frames_dir,
        output,
        fps=extracted.fps,
        audio_source=extracted.audio_source,
        **CPU_ONLY,
    )

    result = probe(output)
    assert result.has_audio is False
    assert result.duration == pytest.approx(source.duration, abs=0.1)


@requires_ffmpeg
def test_vfr_round_trip_keeps_audio_aligned(vfr_video: Path, tmp_path: Path):
    """The VFR case, which is where audio sync actually goes wrong.

    The frames and the audio must come from the same CFR timeline, so the
    output duration matches the normalised copy rather than drifting.
    """
    source = probe(vfr_video)
    assert source.is_vfr is True

    work = WorkDir.for_input(vfr_video, base=tmp_path)
    extracted = extract(source, work)
    assert extracted.normalised is True

    output = tmp_path / "vfr_round_trip.mp4"
    encode(
        extracted.frames_dir,
        output,
        fps=extracted.fps,
        audio_source=extracted.audio_source,
        **CPU_ONLY,
    )

    result = probe(output)
    normalised = probe(work.cfr_video)
    assert result.is_vfr is False
    assert result.has_audio is True
    # Video and audio agree to within a frame of the timeline they came from.
    assert result.duration == pytest.approx(normalised.duration, abs=0.1)


@requires_ffmpeg
def test_round_trip_through_jpg_frames(sample_video: Path, tmp_path: Path):
    source = probe(sample_video)
    work = WorkDir.for_input(sample_video, base=tmp_path)
    extracted = extract(source, work, frame_format="jpg")

    output = tmp_path / "jpg_round_trip.mp4"
    encode(
        extracted.frames_dir,
        output,
        fps=extracted.fps,
        audio_source=extracted.audio_source,
        frame_format="jpg",
        **CPU_ONLY,
    )

    result = probe(output)
    assert (result.width, result.height) == (source.width, source.height)
    assert result.duration == pytest.approx(source.duration, abs=0.1)


@requires_ffmpeg
def test_a_resumed_round_trip_produces_the_same_output(sample_video: Path, tmp_path: Path):
    """A crash after extraction must not change the result of the next run."""
    source = probe(sample_video)
    work = WorkDir.for_input(sample_video, base=tmp_path)

    first = extract(source, work)
    # Second call finds the frames already there and skips the work.
    second = extract(source, work)
    assert second.frame_count == first.frame_count

    output = tmp_path / "resumed.mp4"
    encode(
        second.frames_dir,
        output,
        fps=second.fps,
        audio_source=second.audio_source,
        **CPU_ONLY,
    )
    assert probe(output).duration == pytest.approx(source.duration, abs=0.1)
