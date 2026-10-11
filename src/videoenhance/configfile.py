"""Options read from ``videoenhance.toml``.

The file holds what a person sets once and forgets: how to split frames into
blocks, and how wide a margin each block carries. It sits below the command
line and above the built-in defaults, so a flag always wins and the file only
changes what happens when nobody said anything.

Where it is looked for, first match wins:

1. ``--config PATH``, which is an error if the file is not there;
2. ``$VIDEOENHANCE_CONFIG``, the same;
3. ``videoenhance.toml`` in the project root, if it exists.

Having no file is normal and is not an error. Having a file that says something
the program does not understand is: an unknown key is rejected by name, because
a typo in ``blocks`` that silently did nothing would leave the user waiting for
a saving that never comes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from . import config

ENV_VAR = "VIDEOENHANCE_CONFIG"
FILE_NAME = "videoenhance.toml"

#: The tables the file may hold, with the keys of each and the command-line
#: argument each key stands for.
SCHEMA: dict[str, dict[str, str]] = {
    "dedup": {"blocks": "blocks", "margin": "block_margin"},
}


class ConfigError(ValueError):
    """The config file could not be found, read or understood."""


def _toml_loader():
    """The TOML parser: the standard library's, or ``tomli`` before Python 3.11."""
    if sys.version_info >= (3, 11):
        import tomllib

        return tomllib
    try:
        import tomli
    except ImportError:
        raise ConfigError(
            "reading videoenhance.toml needs Python 3.11 or newer, or the tomli "
            "package on this version (pip install tomli)"
        ) from None
    return tomli


def locate(explicit: Path | None = None) -> Path | None:
    """The config file to use, or ``None`` when there is none.

    Raises:
        ConfigError: a file was named, by flag or by variable, and is missing.
    """
    if explicit is not None:
        if not explicit.is_file():
            raise ConfigError(f"config file not found: {explicit}")
        return explicit
    named = os.environ.get(ENV_VAR)
    if named:
        path = Path(named).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file not found: {path} (from ${ENV_VAR})")
        return path
    default = config.PROJECT_ROOT / FILE_NAME
    return default if default.is_file() else None


def _integer(path: Path, where: str, value: object) -> int:
    # bool is an int in Python, and `blocks = true` is not a grid size.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}: {where} must be a whole number, not {value!r}")
    return value


def load(path: Path | None) -> dict[str, int]:
    """Read ``path`` into ``{argument name: value}``; empty when there is no file.

    Raises:
        ConfigError: the file is not valid TOML, names a table or key that does
            not exist, or holds a value out of range.
    """
    if path is None:
        return {}
    parser = _toml_loader()
    try:
        with path.open("rb") as handle:
            document = parser.load(handle)
    except parser.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: not valid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc

    values: dict[str, int] = {}
    for table, body in document.items():
        if table not in SCHEMA:
            known = ", ".join(f"[{name}]" for name in SCHEMA)
            raise ConfigError(f"{path}: unknown table [{table}]; the file may hold {known}")
        if not isinstance(body, dict):
            raise ConfigError(f"{path}: {table} must be a table, written [{table}]")
        for key, value in body.items():
            if key not in SCHEMA[table]:
                known = ", ".join(SCHEMA[table])
                raise ConfigError(f"{path}: unknown key {key!r} in [{table}]; the keys are {known}")
            values[SCHEMA[table][key]] = _integer(path, f"{key} in [{table}]", value)

    if values.get("blocks", 1) < 1:
        raise ConfigError(f"{path}: blocks in [dedup] must be 1 or greater (1 means off)")
    if values.get("block_margin", 0) < 0:
        raise ConfigError(f"{path}: margin in [dedup] cannot be negative")
    return values


__all__ = ["ConfigError", "ENV_VAR", "FILE_NAME", "SCHEMA", "load", "locate"]
