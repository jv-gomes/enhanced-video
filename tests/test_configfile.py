"""Tests for reading videoenhance.toml."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from videoenhance import config, configfile
from videoenhance.configfile import ConfigError, load, locate


def write(tmp_path: Path, text: str, name: str = "videoenhance.toml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """No stray file in the real project root, no inherited variable."""
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "root")
    (tmp_path / "root").mkdir()
    monkeypatch.delenv(configfile.ENV_VAR, raising=False)


# --------------------------------------------------------------------------
# where the file is looked for
# --------------------------------------------------------------------------


def test_no_file_is_not_an_error():
    assert locate() is None
    assert load(None) == {}


def test_the_project_root_file_is_found(tmp_path: Path):
    path = write(tmp_path / "root", "[dedup]\nblocks = 2\n")
    assert locate() == path


def test_the_variable_beats_the_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    write(tmp_path / "root", "[dedup]\nblocks = 2\n")
    elsewhere = write(tmp_path, "[dedup]\nblocks = 4\n", "other.toml")
    monkeypatch.setenv(configfile.ENV_VAR, str(elsewhere))
    assert locate() == elsewhere


def test_the_flag_beats_the_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    chosen = write(tmp_path, "[dedup]\nblocks = 3\n", "chosen.toml")
    monkeypatch.setenv(configfile.ENV_VAR, str(write(tmp_path, "", "env.toml")))
    assert locate(chosen) == chosen


def test_a_named_file_that_is_missing_is_an_error(tmp_path: Path, monkeypatch):
    with pytest.raises(ConfigError, match="not found"):
        locate(tmp_path / "nope.toml")
    monkeypatch.setenv(configfile.ENV_VAR, str(tmp_path / "nope.toml"))
    with pytest.raises(ConfigError, match=configfile.ENV_VAR):
        locate()


# --------------------------------------------------------------------------
# what it may say
# --------------------------------------------------------------------------


def test_the_keys_come_back_under_their_argument_names(tmp_path: Path):
    path = write(tmp_path, "[dedup]\nblocks = 3\nmargin = 16\n")
    assert load(path) == {"blocks": 3, "block_margin": 16}


def test_a_file_may_set_only_one_of_them(tmp_path: Path):
    assert load(write(tmp_path, "[dedup]\nblocks = 4\n")) == {"blocks": 4}
    assert load(write(tmp_path, "")) == {}


@pytest.mark.parametrize(
    "text,words",
    [
        ("[dedupe]\nblocks = 3\n", "unknown table [dedupe]"),
        ("[dedup]\nblock = 3\n", "unknown key 'block'"),
        ("dedup = 3\n", "must be a table"),
        ("[dedup]\nblocks = 2.5\n", "whole number"),
        ("[dedup]\nblocks = true\n", "whole number"),
        ('[dedup]\nmargin = "16"\n', "whole number"),
        ("[dedup]\nblocks = 0\n", "1 or greater"),
        ("[dedup]\nmargin = -1\n", "negative"),
        ("[dedup\nblocks = 3\n", "not valid TOML"),
    ],
)
def test_a_bad_file_is_rejected_with_its_path_and_the_reason(tmp_path: Path, text: str, words: str):
    path = write(tmp_path, text)
    with pytest.raises(ConfigError, match=re.escape(words)) as excinfo:
        load(path)
    assert str(path) in str(excinfo.value)


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="tomllib is in the standard library")
def test_old_python_without_tomli_says_what_to_install(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tomli", None)
    with pytest.raises(ConfigError, match="tomli"):
        load(write(tmp_path, "[dedup]\nblocks = 3\n"))


def test_the_example_file_is_valid_and_documents_every_key():
    example = Path(__file__).resolve().parents[1] / "videoenhance.example.toml"
    values = load(example)
    assert set(values) == {arg for keys in configfile.SCHEMA.values() for arg in keys.values()}
