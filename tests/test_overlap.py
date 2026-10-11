"""Tests for encoding one chunk while the next is generated.

The GPU stages and the encode use different parts of the machine, so a chunked
run encodes part N in a thread while chunk N+1 is generated. What can go wrong
is all about ordering: generating into a slot whose frames are still being
encoded, a pin that arrives too late, a failure noticed only at the end, and an
interrupt that leaves an encode running. Generation and encode are replaced by
stand-ins here, so the tests are about the orchestration and run in seconds.
"""

from __future__ import annotations

import threading
import time
from fractions import Fraction
from pathlib import Path

import pytest

from videoenhance import chunk
from videoenhance.chunk import chunk_frame_count, describe_plan, run_chunked
from videoenhance.pipeline import Options, Result
from videoenhance.probe import probe
from videoenhance.process import Cancelled, ToolError, cancelling, reset_cancel, run
from videoenhance.workdir import WorkDir

from .conftest import requires_ffmpeg

CPU_ONLY = {"allow_hardware": False}


@pytest.fixture(autouse=True)
def clean_cancel():
    reset_cancel()
    yield
    reset_cancel()


def work_for(video: Path, tmp_path: Path) -> WorkDir:
    return WorkDir.for_input(video, base=tmp_path / "work")


class Recorder:
    """Stand-ins for generate and finish that log what happens, and when."""

    def __init__(self, encode_seconds: float = 0.0) -> None:
        self.events: list[tuple[str, int]] = []
        self.slots_in_encode: dict[Path, int] = {}
        self.clashes: list[str] = []
        self.lock = threading.Lock()
        self.encode_seconds = encode_seconds
        self.generate_hook = None
        self.finish_hook = None

    def generate(self, chunk_path: Path, options, *, work, bins=None):
        index = int(chunk_path.stem.rsplit("_", 1)[-1])
        with self.lock:
            if work.root in self.slots_in_encode:
                self.clashes.append(
                    f"chunk {index} generated into {work.root.name} while part "
                    f"{self.slots_in_encode[work.root]} was still being encoded from it"
                )
            self.events.append(("generate", index))
        if self.generate_hook:
            self.generate_hook(index)
        return (index, work.root)

    def finish(self, generated, part: Path, options, *, bins=None):
        index, slot = generated
        with self.lock:
            self.slots_in_encode[slot] = index
            self.events.append(("encode-start", index))
        try:
            if self.finish_hook:
                self.finish_hook(index)
            time.sleep(self.encode_seconds)
            part.parent.mkdir(parents=True, exist_ok=True)
            part.write_bytes(b"part")
        finally:
            with self.lock:
                self.slots_in_encode.pop(slot, None)
                self.events.append(("encode-end", index))
        return Result(
            output=part, encoder="libx264", fps=Fraction(60), frames=10, stages=[], work=slot
        )

    def install(self, monkeypatch: pytest.MonkeyPatch) -> Recorder:
        monkeypatch.setattr(chunk, "generate", self.generate)
        monkeypatch.setattr(chunk, "finish", self.finish)
        monkeypatch.setattr(chunk, "concat", lambda *a, **k: None)
        return self

    def order(self, name: str) -> list[int]:
        return [index for event, index in self.events if event == name]


def run_four_chunks(video: Path, tmp_path: Path, **options) -> None:
    run_chunked(
        video,
        tmp_path / "out.mkv",
        Options(chunk_seconds=1.0, **options),
        work=work_for(video, tmp_path),
    )


# --------------------------------------------------------------------------
# the overlap itself
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_the_next_chunk_is_generated_while_the_last_is_encoded(
    keyframed_video: Path, tmp_path: Path, monkeypatch
):
    """Part 1's encode waits for chunk 2 to start generating: only overlap lets it."""
    recorder = Recorder().install(monkeypatch)
    second_started = threading.Event()
    seen: dict[str, bool] = {}
    recorder.generate_hook = lambda index: index == 1 and second_started.set()

    def first_encode_waits(index: int) -> None:
        if index == 0:
            seen["overlapped"] = second_started.wait(timeout=10)

    recorder.finish_hook = first_encode_waits

    run_four_chunks(keyframed_video, tmp_path)

    assert seen["overlapped"] is True
    assert recorder.order("encode-end") == [0, 1, 2, 3]


@requires_ffmpeg
def test_without_overlap_each_part_is_encoded_before_the_next_chunk(
    keyframed_video: Path, tmp_path: Path, monkeypatch
):
    recorder = Recorder().install(monkeypatch)

    run_four_chunks(keyframed_video, tmp_path, overlap=False)

    assert recorder.events == [
        (event, index) for index in range(4) for event in ("generate", "encode-start", "encode-end")
    ]


@requires_ffmpeg
def test_a_slot_is_never_generated_into_while_its_part_is_encoded(
    keyframed_video: Path, tmp_path: Path, monkeypatch
):
    """Slow encodes make the clash likely if the wait on the slot were missing."""
    recorder = Recorder(encode_seconds=0.3).install(monkeypatch)

    run_four_chunks(keyframed_video, tmp_path)

    assert recorder.clashes == []


@requires_ffmpeg
def test_the_chunks_alternate_between_two_slots(keyframed_video: Path, tmp_path: Path, monkeypatch):
    recorder = Recorder().install(monkeypatch)
    used: list[str] = []
    original = recorder.generate

    def spy(chunk_path, options, *, work, bins=None):
        used.append(work.root.name)
        return original(chunk_path, options, work=work, bins=bins)

    monkeypatch.setattr(chunk, "generate", spy)

    run_four_chunks(keyframed_video, tmp_path)

    assert used == ["current", "current-1", "current", "current-1"]


@requires_ffmpeg
def test_the_encoder_pin_reaches_every_part_after_the_first(
    keyframed_video: Path, tmp_path: Path, monkeypatch
):
    """Chunk 2 is queued before part 1 has picked an encoder; it must still get it."""
    recorder = Recorder(encode_seconds=0.2).install(monkeypatch)
    asked: list[str | None] = []
    original = recorder.finish

    def spy(generated, part, options, *, bins=None):
        asked.append(options.prefer_encoder)
        return original(generated, part, options, bins=bins)

    monkeypatch.setattr(chunk, "finish", spy)

    run_four_chunks(keyframed_video, tmp_path)

    assert asked == [None, "libx264", "libx264", "libx264"]


# --------------------------------------------------------------------------
# stopping
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_a_failed_encode_stops_the_run_at_the_next_chunk(
    keyframed_video: Path, tmp_path: Path, monkeypatch
):
    recorder = Recorder().install(monkeypatch)

    def fail_first(index: int) -> None:
        if index == 0:
            raise ToolError(["ffmpeg"], 1, "boom")

    recorder.finish_hook = fail_first

    with pytest.raises(ToolError, match="boom"):
        run_four_chunks(keyframed_video, tmp_path)

    # Chunk 1 may already have been generated beside the failing encode, but
    # chunk 2 needs part 0's slot, so the failure is seen before it.
    assert recorder.order("generate") in ([0], [0, 1])
    assert not cancelling()


@requires_ffmpeg
def test_an_interrupt_stops_the_encode_in_flight_at_once(
    keyframed_video: Path, tmp_path: Path, monkeypatch
):
    """Ctrl-C while part 1 is encoding and chunk 2 is generating."""
    recorder = Recorder().install(monkeypatch)
    encoding = threading.Event()
    outcome: dict[str, object] = {}

    def long_encode(index: int) -> None:
        if index == 1:
            encoding.set()
            try:
                run(["sleep", "60"])
            except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
                outcome["error"] = exc
                raise

    def interrupt_third(index: int) -> None:
        if index == 2:
            assert encoding.wait(timeout=10)
            time.sleep(0.2)
            raise KeyboardInterrupt

    recorder.finish_hook = long_encode
    recorder.generate_hook = interrupt_third

    start = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        run_four_chunks(keyframed_video, tmp_path)

    assert time.monotonic() - start < 10
    assert isinstance(outcome["error"], Cancelled)
    assert not any(t.name.startswith("encode") for t in threading.enumerate())
    assert not cancelling()
    # Part 0 was finished and stays for the resume; part 1 never appeared.
    parts = work_for(keyframed_video, tmp_path).parts
    assert (parts / "part_0000.mkv").exists()
    assert not (parts / "part_0001.mkv").exists()


# --------------------------------------------------------------------------
# a real run, and what it costs on disk
# --------------------------------------------------------------------------


@pytest.fixture
def ncnn_stubs(stub_bin):
    stub_bin("REALESRGAN_BIN", name="realesrgan-stub")
    stub_bin("RIFE_BIN", name="rife-stub")
    return stub_bin


@requires_ffmpeg
def test_overlapping_gives_the_same_video_and_leaves_no_frames(
    keyframed_video: Path, tmp_path: Path, ncnn_stubs
):
    outputs = {}
    for overlap in (True, False):
        base = tmp_path / ("on" if overlap else "off")
        work = work_for(keyframed_video, base)
        output = base / "out.mkv"
        run_chunked(
            keyframed_video,
            output,
            Options(target_fps=60, chunk_seconds=1.0, overlap=overlap, keep_temp=True, **CPU_ONLY),
            work=work,
        )
        outputs[overlap] = probe(output)
        for slot in (0, 1):
            assert list(work.slot(slot).rglob("*.png")) == []

    assert outputs[True].nb_frames == outputs[False].nb_frames
    assert outputs[True].duration == pytest.approx(outputs[False].duration, abs=0.05)


@requires_ffmpeg
def test_the_disk_estimate_counts_two_chunks_when_overlapping(sample_video: Path):
    info = probe(sample_video)

    assert chunk_frame_count(info, 1.0) == 30
    assert chunk_frame_count(info, 1.0, overlap=True) == 60
    # Never more than the file itself.
    assert chunk_frame_count(info, 2.0, overlap=True) == 90


@requires_ffmpeg
def test_the_plan_says_how_the_chunks_are_run(sample_video: Path):
    info = probe(sample_video)

    assert "encoding one while the next is generated" in describe_plan(info, 1.0, overlap=True)
    assert "one at a time" in describe_plan(info, 1.0)
    # A single chunk has nothing to overlap with.
    assert "one at a time" in describe_plan(info, 600.0, overlap=True)
