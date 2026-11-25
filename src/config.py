"""Configuration loader with ${VAR:-default} environment expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR:-default} placeholders in loaded YAML values."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(
            lambda m: os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else ""),
            value,
        )
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def load_hardware_config(path: str | Path = "config/hardware.yaml") -> dict[str, Any]:
    """Load hardware configuration YAML after loading .env variables."""
    load_dotenv()
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _expand_env(raw)


def update_yaml_numeric_keys(
    config_path: str | Path,
    updates: dict[str, float],
    decimals: int = 1,
    section_header: str | None = None,
) -> bool:
    """Update numeric `key: value` lines in-place while preserving comments and formatting."""
    path = Path(config_path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    in_section = section_header is None
    section_indent = 0
    updated = 0
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if section_header is not None and not stripped.startswith("#"):
            if not in_section and stripped == f"{section_header}:":
                in_section = True
                section_indent = len(line) - len(line.lstrip())
            elif in_section and stripped and (len(line) - len(line.lstrip())) <= section_indent:
                in_section = False

        if in_section and not line.lstrip().startswith("#"):
            m = re.match(r"^(\s+)([A-Za-z_][A-Za-z0-9_]*):\s*[-+0-9.eE]+(.*)$", line.rstrip("\r\n"))
            if m and m.group(2) in updates:
                indent, key, suffix = m.group(1), m.group(2), m.group(3)
                eol = "\r\n" if line.endswith("\r\n") else "\n"
                line = f"{indent}{key}: {updates[key]:.{decimals}f}{suffix}{eol}"
                updated += 1
        new_lines.append(line)

    if updated != len(updates):
        return False
    path.write_text("".join(new_lines), encoding="utf-8")
    return True
