"""YAML config loading with recursive import support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path, visited: set[Path] | None = None) -> dict[str, Any]:
    """Load config file and recursively merge imported fragments.

    Args:
        config_path: Path to root YAML config.
        visited: Internal recursion guard for cycle detection.

    Returns:
        Merged configuration dictionary.

    Raises:
        ValueError: If config shape is invalid or cyclic imports are detected.
    """
    resolved_path = config_path.resolve()
    visited = visited or set()
    if resolved_path in visited:
        raise ValueError(f"Cyclic config import detected: {resolved_path}")
    visited.add(resolved_path)

    with resolved_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")

    imports = config.pop("imports", [])
    if imports is None:
        imports = []
    if not isinstance(imports, list):
        raise ValueError("Config key 'imports' must be a list")

    merged: dict[str, Any] = {}
    for import_path in imports:
        if not isinstance(import_path, str):
            raise ValueError("Each import path must be a string")
        child_path = (resolved_path.parent / import_path).resolve()
        child_cfg = load_config(child_path, visited=visited)
        merged = _deep_merge(merged, child_cfg)
    return _deep_merge(merged, config)

