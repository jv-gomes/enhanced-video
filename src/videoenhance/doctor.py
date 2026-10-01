"""Environment preflight check behind ``--doctor``.

A run can take hours, so it is worth a second up front to confirm that FFmpeg
is present, that Vulkan actually sees the GPU, that a usable encoder exists and
that the NCNN binaries have been downloaded. Every failure is reported with the
action that fixes it, and all checks run before anything is reported, so the
user sees the complete picture in one pass.
"""

from __future__ import annotations

import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from . import config
from .encode import CPU_ENCODERS, HW_ENCODERS, available_encoders
from .process import ToolError, run, which

OK = "ok"
WARN = "warn"
FAIL = "fail"

_MARKS = {OK: "[ ok ]", WARN: "[warn]", FAIL: "[fail]"}

REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases"
RIFE_URL = "https://github.com/nihui/rife-ncnn-vulkan/releases"


@dataclass
class Check:
    """One line of the doctor report."""

    name: str
    status: str
    detail: str = ""
    hint: str = ""

    def render(self) -> str:
        line = f"{_MARKS[self.status]} {self.name}"
        if self.detail:
            line += f": {self.detail}"
        if self.hint and self.status != OK:
            line += f"\n       -> {self.hint}"
        return line


def _check_ffmpeg_tool(name: str, path: Path) -> Check:
    located = path if path.is_absolute() else which(str(path))
    if located is None or not Path(located).exists():
        return Check(
            name,
            FAIL,
            "not found",
            f"install FFmpeg and put {name} on PATH, or set {name.upper()}_BIN",
        )
    try:
        version = run([located, "-version"]).stdout.splitlines()[0]
    except (ToolError, IndexError):
        return Check(name, FAIL, f"{located} is not runnable", "reinstall FFmpeg")
    return Check(name, OK, version.replace(f"{name} version ", "").split(" Copyright")[0])


def _check_encoders(ffmpeg: Path) -> Check:
    found = available_encoders(ffmpeg)
    if not found:
        return Check("encoders", FAIL, "could not query ffmpeg -encoders", "reinstall FFmpeg")

    preferred = HW_ENCODERS.get(sys.platform, ())
    hardware = [name for name in preferred if name in found]
    cpu = [name for name in CPU_ENCODERS if name in found]

    if hardware:
        detail = f"hardware {hardware[0]}, fallback {cpu[0] if cpu else 'none'}"
        return Check("encoders", OK, detail)
    if cpu:
        return Check(
            "encoders",
            WARN,
            f"no hardware encoder, will use {cpu[0]}",
            "CPU encoding is much slower; install the drivers that provide "
            f"{preferred[0] if preferred else 'a hardware encoder'}",
        )
    return Check("encoders", FAIL, "no usable encoder", "install an FFmpeg build with libx264")


def _check_vaapi_device() -> Check | None:
    """Only meaningful on Linux, where VAAPI needs a render node."""
    if sys.platform != "linux":
        return None
    device = Path("/dev/dri/renderD128")
    if device.exists():
        return Check("vaapi device", OK, str(device))
    return Check(
        "vaapi device",
        WARN,
        "no /dev/dri/renderD128",
        "hardware encoding will be unavailable; check your GPU drivers",
    )


def vulkan_devices() -> list[str]:
    """GPU names reported by ``vulkaninfo --summary``, if it is installed."""
    if which("vulkaninfo") is None:
        return []
    try:
        out = run(["vulkaninfo", "--summary"], check=False).stdout or ""
    except ToolError:
        return []
    return [name.strip() for name in re.findall(r"deviceName\s*=\s*(.+)", out)]


def _check_vulkan() -> Check:
    if which("vulkaninfo") is None:
        return Check(
            "vulkan",
            WARN,
            "vulkaninfo not installed, cannot verify the GPU",
            "install vulkan-tools to confirm the GPU is visible before a long run",
        )
    devices = vulkan_devices()
    if not devices:
        return Check(
            "vulkan",
            FAIL,
            "no Vulkan device found",
            "update your GPU driver (Adrenalin on Windows, Mesa/RADV on Linux)",
        )
    detail = devices[0]
    if len(devices) > 1:
        detail += f" (+{len(devices) - 1} more; select with --gpu)"
    return Check("vulkan", OK, detail)


def _check_ncnn_binary(name: str, path: Path, url: str, models: Path) -> list[Check]:
    env_var = "REALESRGAN_BIN" if name == "realesrgan" else "RIFE_BIN"
    if not path.exists():
        return [
            Check(
                name,
                FAIL,
                f"not found at {path}",
                f"download the release from {url}, unpack it into "
                f"{path.parent}, or set {env_var}",
            )
        ]

    checks = [Check(name, OK, str(path))]
    if not models.exists():
        checks.append(
            Check(
                f"{name} models",
                FAIL,
                f"missing {models}",
                "unpack the model folder shipped with the release next to the binary",
            )
        )
    return checks


def collect_checks() -> list[Check]:
    """Run every environment check and return the report lines."""
    bins = config.binaries()
    checks: list[Check] = [
        Check(
            "platform",
            OK,
            f"{platform.system()} {platform.machine()}, Python {sys.version.split()[0]}",
        ),
        _check_ffmpeg_tool("ffmpeg", bins.ffmpeg),
        _check_ffmpeg_tool("ffprobe", bins.ffprobe),
    ]
    checks.append(_check_encoders(bins.ffmpeg))
    vaapi = _check_vaapi_device()
    if vaapi:
        checks.append(vaapi)
    checks.append(_check_vulkan())
    checks += _check_ncnn_binary(
        "realesrgan", bins.realesrgan, REALESRGAN_URL, bins.realesrgan_models
    )
    checks += _check_ncnn_binary(
        "rife", bins.rife, RIFE_URL, config.rife_model_dir(bins=bins)
    )
    return checks


def report(checks: list[Check] | None = None) -> int:
    """Print the report and return a process exit code (0 only if nothing failed)."""
    checks = checks if checks is not None else collect_checks()
    for check in checks:
        print(check.render())

    failures = [c for c in checks if c.status == FAIL]
    warnings = [c for c in checks if c.status == WARN]
    print()
    if failures:
        print(f"{len(failures)} check(s) failed; the pipeline cannot run yet.")
        return 1
    if warnings:
        print(f"Ready, with {len(warnings)} warning(s).")
        return 0
    print("Ready.")
    return 0
