"""Binary locations and pipeline defaults.

Every path in the project originates here, so that a user with the executables
somewhere unusual can point environment variables at them instead of editing
code. Paths are always :class:`~pathlib.Path`, and executable names grow a
``.exe`` suffix on Windows.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .process import which

#: Repository root, i.e. the directory that holds ``bin/`` and ``work/``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BIN_DIR = PROJECT_ROOT / "bin"
WORK_DIR = PROJECT_ROOT / "work"

#: Real-ESRGAN models, mapped to the upscale factors each one supports.
UPSCALE_MODELS: dict[str, tuple[int, ...]] = {
    "realesr-animevideov3": (2, 3, 4),
    "realesrgan-x4plus": (4,),
    "realesrgan-x4plus-anime": (4,),
}

DEFAULT_UPSCALE_MODEL = "realesr-animevideov3"
DEFAULT_RIFE_MODEL = "rife-v4.6"

#: Upscale before interpolating. The default, and the cheaper one: Real-ESRGAN
#: is the expensive stage, so it should run before RIFE multiplies the frame
#: count.
ORDER_UPSCALE_FIRST = "upscale-first"
#: Interpolate at the source resolution first. Slower overall, but RIFE sees
#: the original pixels rather than Real-ESRGAN's reconstruction of them, which
#: some people prefer on live-action footage.
ORDER_INTERPOLATE_FIRST = "interpolate-first"
ORDERS = (ORDER_UPSCALE_FIRST, ORDER_INTERPOLATE_FIRST)
DEFAULT_ORDER = ORDER_UPSCALE_FIRST

#: Blocks per side for block dedup. 1 means whole frames only, which is what
#: dedup did before blocks existed. Off by default because every block carries
#: a margin, so on footage with nothing static in it a grid does more GPU work
#: than whole frames; the saving is measured first with
#: ``scripts/measure_tile_reuse.py``.
DEFAULT_BLOCKS = 1
#: Pixels of neighbours upscaled with each block so that stitching leaves no
#: seams: with none the seams measured 47.7 dB and were visible along every
#: edge, with 8 and above over 60 dB.
DEFAULT_BLOCK_MARGIN = 16

DEFAULT_SCALE = 2
DEFAULT_TARGET_FPS = 60
DEFAULT_FRAME_FORMAT = "png"

#: Reuse an upscaled frame whenever the same pixels come round again. Lossless
#: for the output, because Real-ESRGAN is deterministic, and worth a lot on cel
#: animation, where a drawing is held for two or three frames. Grainy live
#: action has almost no exact repeats and simply pays the hashing cost.
DEFAULT_DEDUP = True
#: Ceiling for the upscaled-frame cache that outlives each chunk, in bytes.
#: Without one, a long video's unique frames would accumulate for the whole run
#: and undo the point of processing in chunks. 0 means no ceiling.
DEFAULT_CACHE_SIZE = 2 * 1024**3

#: Seconds of video processed per chunk. The frames of one chunk are the whole
#: disk cost of a run, so this is what keeps a two-hour film from needing
#: hundreds of gigabytes of scratch space. 0 disables chunking and processes
#: the file in a single pass.
DEFAULT_CHUNK_SECONDS = 30.0
FRAME_FORMATS = ("png", "jpg")
FRAME_PATTERN = "%08d"

#: ``-g``: GPU device id. 0 is the first Vulkan device; make sure it is the
#: dedicated AMD card on machines that also have an integrated GPU.
DEFAULT_GPU_ID = 0
#: ``-t``: tile size, 0 meaning automatic. Lower it on Vulkan OOM errors.
DEFAULT_TILE_SIZE = 0
#: Tile sizes to retry with when the GPU runs out of memory.
TILE_FALLBACKS = (256, 128)
#: Share of the time the GPU binaries are allowed to run, in percent. The
#: NCNN tools have no throttle of their own, so ``process.run`` pauses the
#: process for the rest of each cycle; 100 means no limit.
DEFAULT_GPU_LOAD = 100
#: Length of one run-then-pause cycle, in seconds. Short enough that the card
#: visibly idles between bursts, long enough that suspending costs nothing.
THROTTLE_CYCLE = 0.5
#: ``-j``: load:proc:save thread counts.
DEFAULT_THREADS = "1:2:2"

#: Resolution above which RIFE should run in UHD mode (``-u``).
UHD_PIXEL_THRESHOLD = 3840 * 2160


def _exe(name: str) -> str:
    """Add the platform's executable suffix to ``name``."""
    return f"{name}.exe" if sys.platform == "win32" else name


def _resolve(env_var: str, default: Path, *, path_lookup: str | None = None) -> Path:
    """Pick a binary location: the environment override, then ``default``.

    Falls back to ``PATH`` when ``default`` does not exist and ``path_lookup``
    is given, so a system-wide install works without any configuration.
    """
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser()
    if not default.exists() and path_lookup:
        found = which(path_lookup)
        if found:
            return found
    return default


@dataclass(frozen=True)
class Binaries:
    """Resolved locations of the four executables the pipeline drives."""

    ffmpeg: Path = field(default_factory=lambda: _resolve("FFMPEG_BIN", Path(_exe("ffmpeg"))))
    ffprobe: Path = field(default_factory=lambda: _resolve("FFPROBE_BIN", Path(_exe("ffprobe"))))
    realesrgan: Path = field(
        default_factory=lambda: _resolve(
            "REALESRGAN_BIN",
            BIN_DIR / "realesrgan" / _exe("realesrgan-ncnn-vulkan"),
            path_lookup="realesrgan-ncnn-vulkan",
        )
    )
    rife: Path = field(
        default_factory=lambda: _resolve(
            "RIFE_BIN",
            BIN_DIR / "rife" / _exe("rife-ncnn-vulkan"),
            path_lookup="rife-ncnn-vulkan",
        )
    )

    @property
    def realesrgan_models(self) -> Path:
        """Directory holding the Real-ESRGAN ``.param``/``.bin`` model files."""
        return self.realesrgan.parent / "models"

    @property
    def rife_models(self) -> Path:
        """Directory holding the RIFE model folders, e.g. ``rife-v4.6/``."""
        return self.rife.parent


def binaries() -> Binaries:
    """Resolve the binary locations from the current environment.

    Called per invocation rather than cached at import time, so tests and
    callers can change the environment variables and see the effect.
    """
    return Binaries()


def rife_model_dir(name: str = DEFAULT_RIFE_MODEL, bins: Binaries | None = None) -> Path:
    """Path of a RIFE model directory, which is passed to the binary via ``-m``."""
    return (bins or binaries()).rife_models / name


def scale_supported(model: str, scale: int) -> bool:
    """Whether ``model`` can upscale by ``scale``."""
    return scale in UPSCALE_MODELS.get(model, ())
