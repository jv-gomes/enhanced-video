"""Tests for scripts/fetch-bins.sh, against zips built in the test.

The real releases are 47 MB and 430 MB, so the script's download locations can
be pointed at ``file://`` URLs and the whole flow runs offline. What is under
test is the part that goes wrong: the two archives are laid out differently, so
each has to be flattened its own way, and a layout that has changed must be
refused without touching an install that works.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch-bins.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("curl") is None or os.name == "nt",
    reason="needs bash and curl",
)


def build_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


@pytest.fixture
def releases(tmp_path: Path) -> Path:
    """A directory of fake releases, laid out the way the real ones are."""
    directory = tmp_path / "releases"
    directory.mkdir()
    build_zip(
        directory / "realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
        {
            "realesrgan-ncnn-vulkan": "#!/bin/sh\n",
            "models/realesr-animevideov3-x2.bin": "m",
            "models/realesr-animevideov3-x2.param": "p",
            "input.jpg": "demo",
            "onepiece_demo.mp4": "demo",
        },
    )
    build_zip(
        directory / "rife-ncnn-vulkan-20221029-ubuntu.zip",
        {
            "rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan": "#!/bin/sh\n",
            "rife-ncnn-vulkan-20221029-ubuntu/LICENSE": "MIT",
            "rife-ncnn-vulkan-20221029-ubuntu/README.md": "readme",
            "rife-ncnn-vulkan-20221029-ubuntu/rife-v4.6/flownet.bin": "b",
            "rife-ncnn-vulkan-20221029-ubuntu/rife-v4.6/flownet.param": "p",
            "rife-ncnn-vulkan-20221029-ubuntu/rife-anime/flownet.bin": "b",
        },
    )
    build_zip(
        directory / "realesrgan-ncnn-vulkan-20220424-windows.zip",
        {"realesrgan-ncnn-vulkan.exe": "MZ", "models/realesrgan-x4plus.bin": "m"},
    )
    build_zip(
        directory / "rife-ncnn-vulkan-20221029-windows.zip",
        {
            "rife-ncnn-vulkan-20221029-windows/rife-ncnn-vulkan.exe": "MZ",
            "rife-ncnn-vulkan-20221029-windows/rife-v4.6/flownet.bin": "b",
        },
    )
    return directory


def fetch(tmp_path: Path, releases: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "REALESRGAN_URL_BASE": releases.as_uri(),
        "RIFE_URL_BASE": releases.as_uri(),
    }
    return subprocess.run(
        ["bash", str(SCRIPT), "--os", "ubuntu", "--dest", str(tmp_path / "bin"), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_each_archive_is_flattened_its_own_way(tmp_path: Path, releases: Path):
    done = fetch(tmp_path, releases)

    assert done.returncode == 0, done.stderr
    bin_dir = tmp_path / "bin"
    assert (bin_dir / "realesrgan" / "realesrgan-ncnn-vulkan").is_file()
    assert (bin_dir / "realesrgan" / "models" / "realesr-animevideov3-x2.bin").is_file()
    assert (bin_dir / "rife" / "rife-ncnn-vulkan").is_file()
    assert (bin_dir / "rife" / "rife-v4.6" / "flownet.bin").is_file()
    # The wrapper folder is gone: its contents are what moved.
    assert not list(bin_dir.rglob("rife-ncnn-vulkan-20221029-ubuntu"))


def test_the_binaries_are_made_executable(tmp_path: Path, releases: Path):
    fetch(tmp_path, releases)

    for binary in ("realesrgan/realesrgan-ncnn-vulkan", "rife/rife-ncnn-vulkan"):
        assert os.access(tmp_path / "bin" / binary, os.X_OK), binary


def test_the_demo_files_and_spare_models_are_dropped(tmp_path: Path, releases: Path):
    fetch(tmp_path, releases)

    bin_dir = tmp_path / "bin"
    assert not (bin_dir / "realesrgan" / "input.jpg").exists()
    assert not (bin_dir / "realesrgan" / "onepiece_demo.mp4").exists()
    assert not (bin_dir / "rife" / "rife-anime").exists()
    assert (bin_dir / "rife" / "LICENSE").is_file()


def test_every_rife_model_can_be_kept(tmp_path: Path, releases: Path):
    fetch(tmp_path, releases, "--all-rife-models")

    assert (tmp_path / "bin" / "rife" / "rife-anime" / "flownet.bin").is_file()


def test_an_existing_install_is_left_alone_unless_forced(tmp_path: Path, releases: Path):
    fetch(tmp_path, releases)
    marker = tmp_path / "bin" / "realesrgan" / "marker"
    marker.write_text("mine")

    again = fetch(tmp_path, releases)
    assert "already in" in again.stdout
    assert marker.exists()

    forced = fetch(tmp_path, releases, "--force")
    assert forced.returncode == 0
    assert not marker.exists()


def test_only_fetches_one_of_the_two(tmp_path: Path, releases: Path):
    fetch(tmp_path, releases, "--only", "rife")

    assert (tmp_path / "bin" / "rife").is_dir()
    assert not (tmp_path / "bin" / "realesrgan").exists()


def test_windows_names_get_an_exe(tmp_path: Path, releases: Path):
    done = fetch(tmp_path, releases, "--os", "windows")

    assert done.returncode == 0, done.stderr
    assert (tmp_path / "bin" / "realesrgan" / "realesrgan-ncnn-vulkan.exe").is_file()
    assert (tmp_path / "bin" / "rife" / "rife-ncnn-vulkan.exe").is_file()


def test_a_dry_run_names_the_urls_and_changes_nothing(tmp_path: Path, releases: Path):
    done = fetch(tmp_path, releases, "--dry-run")

    assert done.returncode == 0
    assert "realesrgan-ncnn-vulkan-20220424-ubuntu.zip" in done.stdout
    assert "rife-ncnn-vulkan-20221029-ubuntu.zip" in done.stdout
    assert not (tmp_path / "bin").exists()


def test_a_changed_layout_is_refused_and_a_working_install_survives(tmp_path: Path, releases: Path):
    fetch(tmp_path, releases)
    kept = tmp_path / "bin" / "realesrgan" / "models" / "realesr-animevideov3-x2.bin"
    # The Real-ESRGAN-ncnn-vulkan release: the binary and nothing else.
    build_zip(
        releases / "realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
        {"realesrgan-ncnn-vulkan-v0.2.0-ubuntu/realesrgan-ncnn-vulkan": "#!/bin/sh\n"},
    )

    done = fetch(tmp_path, releases, "--only", "realesrgan", "--force")

    assert done.returncode != 0
    assert "layout changed" in done.stderr
    assert kept.is_file()


def test_a_download_that_fails_is_an_error(tmp_path: Path, releases: Path):
    (releases / "rife-ncnn-vulkan-20221029-ubuntu.zip").unlink()

    done = fetch(tmp_path, releases, "--only", "rife")

    assert done.returncode != 0
    assert "could not download" in done.stderr


@pytest.mark.parametrize("flags", [["--os", "beos"], ["--only", "both"], ["--nope"]])
def test_bad_arguments_are_rejected(tmp_path: Path, releases: Path, flags: list[str]):
    done = fetch(tmp_path, releases, *flags)

    assert done.returncode != 0
    assert not (tmp_path / "bin" / "rife").exists()
