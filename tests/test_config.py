"""Tests for binary resolution and pipeline defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from videoenhance import config


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REALESRGAN_BIN", "/opt/esrgan/realesrgan-ncnn-vulkan")
    monkeypatch.setenv("RIFE_BIN", "/opt/rife/rife-ncnn-vulkan")
    bins = config.binaries()
    assert bins.realesrgan == Path("/opt/esrgan/realesrgan-ncnn-vulkan")
    assert bins.rife == Path("/opt/rife/rife-ncnn-vulkan")


def test_env_override_expands_user(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RIFE_BIN", "~/tools/rife-ncnn-vulkan")
    assert "~" not in str(config.binaries().rife)


def test_defaults_point_into_bin_dir(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("REALESRGAN_BIN", raising=False)
    monkeypatch.delenv("RIFE_BIN", raising=False)
    bins = config.binaries()
    # Either the bundled bin/ location or a PATH hit, never an empty path.
    assert bins.realesrgan.name.startswith("realesrgan-ncnn-vulkan")
    assert bins.rife.name.startswith("rife-ncnn-vulkan")


def test_model_directories_sit_next_to_binaries(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REALESRGAN_BIN", "/opt/esrgan/realesrgan-ncnn-vulkan")
    monkeypatch.setenv("RIFE_BIN", "/opt/rife/rife-ncnn-vulkan")
    bins = config.binaries()
    assert bins.realesrgan_models == Path("/opt/esrgan/models")
    assert config.rife_model_dir("rife-v4.6", bins) == Path("/opt/rife/rife-v4.6")


def test_exe_suffix_only_on_windows(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config.sys, "platform", "win32")
    assert config._exe("rife-ncnn-vulkan") == "rife-ncnn-vulkan.exe"
    monkeypatch.setattr(config.sys, "platform", "linux")
    assert config._exe("rife-ncnn-vulkan") == "rife-ncnn-vulkan"


def test_scale_support_matches_model_table():
    assert config.scale_supported("realesr-animevideov3", 2)
    assert config.scale_supported("realesrgan-x4plus", 4)
    assert not config.scale_supported("realesrgan-x4plus", 2)
    assert not config.scale_supported("unknown-model", 4)


def test_defaults_are_sane():
    assert config.DEFAULT_UPSCALE_MODEL in config.UPSCALE_MODELS
    assert config.DEFAULT_FRAME_FORMAT in config.FRAME_FORMATS
    assert config.scale_supported(config.DEFAULT_UPSCALE_MODEL, config.DEFAULT_SCALE)
    assert config.TILE_FALLBACKS == tuple(sorted(config.TILE_FALLBACKS, reverse=True))
