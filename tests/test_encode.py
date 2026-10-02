"""Tests for encoder selection and for the frames-plus-audio assembly."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from videoenhance.encode import (
    CPU_ENCODERS,
    EncodeError,
    Encoder,
    available_encoders,
    best_encoder,
    build_encode_cmd,
    concat,
    describe_chain,
    encode,
    encoder_chain,
    write_concat_list,
)
from videoenhance.extract import extract
from videoenhance.pipeline import WorkDir
from videoenhance.probe import probe

from .conftest import requires_ffmpeg

ALL = {"hevc_vaapi", "h264_vaapi", "hevc_amf", "h264_amf", "libx265", "libx264"}
PRESENT_DEVICE = Path(__file__)  # any existing path stands in for a render node


def _names(chain: list[Encoder]) -> list[str]:
    return [encoder.name for encoder in chain]


def test_linux_prefers_vaapi_then_falls_back_to_cpu():
    chain = encoder_chain(ALL, platform="linux", vaapi_device=PRESENT_DEVICE)
    assert _names(chain) == ["hevc_vaapi", "h264_vaapi", "libx265", "libx264"]


def test_windows_prefers_amf():
    assert _names(encoder_chain(ALL, platform="win32"))[0] == "hevc_amf"


def test_chain_only_offers_encoders_ffmpeg_has():
    assert _names(encoder_chain({"libx264"}, platform="linux")) == ["libx264"]


def test_missing_render_node_drops_vaapi():
    chain = encoder_chain(ALL, platform="linux", vaapi_device=Path("/dev/dri/definitely-not"))
    assert not any(name.endswith("vaapi") for name in _names(chain))
    assert _names(chain) == list(CPU_ENCODERS)


def test_hardware_can_be_disabled():
    chain = encoder_chain(
        ALL, platform="linux", allow_hardware=False, vaapi_device=PRESENT_DEVICE
    )
    assert _names(chain) == list(CPU_ENCODERS)


def test_prefer_moves_an_encoder_to_the_front():
    chain = encoder_chain(ALL, prefer="libx264", platform="linux", vaapi_device=PRESENT_DEVICE)
    assert _names(chain)[0] == "libx264"
    assert "libx264" not in _names(chain)[1:]


def test_empty_when_ffmpeg_has_nothing():
    assert encoder_chain(set(), platform="linux") == []
    assert best_encoder(found=set(), platform="linux") is None
    assert describe_chain([]) == "none"


def test_vaapi_carries_its_device_and_upload_filters():
    encoder = best_encoder(found=ALL, platform="linux", vaapi_device=PRESENT_DEVICE)
    assert encoder is not None
    assert encoder.is_hardware is True
    assert "-vaapi_device" in encoder.init_args
    assert encoder.filters == ["format=nv12", "hwupload"]


def test_cpu_encoder_needs_no_init_or_filters():
    encoder = best_encoder(found={"libx264"}, platform="linux")
    assert encoder is not None
    assert encoder.is_hardware is False
    assert encoder.init_args == []
    assert encoder.filters == []
    assert "-crf" in encoder.quality_args


def test_command_maps_frames_and_audio(tmp_path: Path):
    encoder = Encoder(name="libx264", quality_args=["-crf", "18"])
    cmd = [
        str(part)
        for part in build_encode_cmd(
            tmp_path / "frames",
            tmp_path / "out.mp4",
            encoder,
            fps=Fraction(60),
            audio_source=tmp_path / "source.mp4",
        )
    ]
    assert "-framerate" in cmd and "60" in cmd
    assert cmd[cmd.index("-map") + 1] == "0:v"
    # Optional audio mapping, so a silent source is not an error.
    assert "1:a?" in cmd
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert "-shortest" in cmd
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[-1].endswith("out.mp4")


def test_command_omits_audio_when_there_is_none(tmp_path: Path):
    cmd = [
        str(part)
        for part in build_encode_cmd(
            tmp_path / "frames", tmp_path / "out.mp4", Encoder(name="libx264"), fps=30
        )
    ]
    assert "1:a?" not in cmd
    assert "-c:a" not in cmd
    assert "-shortest" not in cmd


def test_vaapi_command_places_device_before_the_input(tmp_path: Path):
    encoder = Encoder(
        name="hevc_vaapi",
        init_args=["-vaapi_device", "/dev/dri/renderD128"],
        filters=["format=nv12", "hwupload"],
    )
    cmd = [
        str(part)
        for part in build_encode_cmd(tmp_path / "f", tmp_path / "o.mp4", encoder, fps=30)
    ]
    assert cmd.index("-vaapi_device") < cmd.index("-i")
    assert cmd[cmd.index("-filter:v") + 1] == "format=nv12,hwupload"


@requires_ffmpeg
def test_available_encoders_reports_libx264():
    assert "libx264" in available_encoders()


def test_available_encoders_is_empty_without_ffmpeg():
    assert available_encoders(Path("videoenhance-no-such-ffmpeg")) == set()


def test_encode_refuses_an_empty_frame_directory(tmp_path: Path):
    (tmp_path / "frames").mkdir()
    with pytest.raises(EncodeError, match="no png frames"):
        encode(tmp_path / "frames", tmp_path / "out.mp4", fps=30)


@requires_ffmpeg
def test_encode_refuses_to_overwrite_the_source(sample_video: Path, tmp_path: Path):
    result = extract(probe(sample_video), WorkDir.for_input(Path("c.mp4"), base=tmp_path))
    with pytest.raises(ValueError, match="refusing to overwrite"):
        encode(result.frames_dir, sample_video, fps=30, audio_source=sample_video)


@requires_ffmpeg
def test_encode_falls_back_when_an_encoder_fails(sample_video: Path, tmp_path: Path):
    result = extract(probe(sample_video), WorkDir.for_input(Path("c.mp4"), base=tmp_path))
    output = tmp_path / "fallback.mp4"
    used = encode(
        result.frames_dir,
        output,
        fps=result.fps,
        chain=[
            Encoder(name="not_a_real_encoder"),
            Encoder(name="libx264", quality_args=["-crf", "28"]),
        ],
    )
    assert used.name == "libx264"
    assert output.exists()


@requires_ffmpeg
def test_encode_raises_when_every_encoder_fails(sample_video: Path, tmp_path: Path):
    result = extract(probe(sample_video), WorkDir.for_input(Path("c.mp4"), base=tmp_path))
    with pytest.raises(EncodeError, match="every encoder failed"):
        encode(result.frames_dir, tmp_path / "x.mp4", fps=30, chain=[Encoder(name="nope")])


@requires_ffmpeg
def test_a_failed_encode_leaves_no_partial_file(sample_video: Path, tmp_path: Path):
    result = extract(probe(sample_video), WorkDir.for_input(Path("c.mp4"), base=tmp_path))
    with pytest.raises(EncodeError):
        encode(result.frames_dir, tmp_path / "x.mp4", fps=30, chain=[Encoder(name="nope")])
    assert list(tmp_path.glob("*.part*")) == []
    assert not (tmp_path / "x.mp4").exists()


def test_encode_reports_when_no_encoder_exists(tmp_path: Path):
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "00000001.png").touch()
    with pytest.raises(EncodeError, match="no usable video encoder"):
        encode(frames, tmp_path / "out.mp4", fps=30, chain=[])


# --------------------------------------------------------------------------
# joining already-encoded parts
# --------------------------------------------------------------------------


def test_concat_needs_something_to_join(tmp_path: Path):
    with pytest.raises(EncodeError, match="no parts"):
        concat([], tmp_path / "out.mkv")


def test_concat_names_the_parts_that_are_missing(tmp_path: Path):
    with pytest.raises(EncodeError, match="part_0001"):
        concat([tmp_path / "part_0001.mkv"], tmp_path / "out.mkv")


def test_concat_refuses_to_write_over_a_part(tmp_path: Path):
    part = tmp_path / "part_0000.mkv"
    part.write_bytes(b"data")
    with pytest.raises(ValueError, match="overwrite a part"):
        concat([part], part)


def test_the_concat_list_quotes_awkward_names(tmp_path: Path):
    """A directory with a space or an apostrophe must not break the playlist."""
    listing = write_concat_list(
        [tmp_path / "a b.mkv", tmp_path / "it's.mkv"], tmp_path / "list.txt"
    )
    lines = listing.read_text().splitlines()
    assert lines[0] == f"file '{tmp_path}/a b.mkv'"
    assert lines[1] == f"file '{tmp_path}/it'\\''s.mkv'"
