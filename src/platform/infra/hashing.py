"""Fingerprint helpers for cache invalidation decisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def compute_inputs_fingerprint(
    inputs: list[Path], config_snapshot: dict[str, Any]
) -> str:
    """Build stable hash from input file metadata and config snapshot.

    Args:
        inputs: Files that determine whether stage output is reusable.
        config_snapshot: Stage-specific config subset affecting output semantics.

    Returns:
        SHA-256 hex digest used as cache fingerprint.
    """
    payload: dict[str, Any] = {"inputs": [], "config": config_snapshot}
    for path in sorted(inputs):
        if path.exists():
            stat = path.stat()
            payload["inputs"].append(
                {
                    "path": str(path.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        else:
            payload["inputs"].append({"path": str(path.resolve()), "missing": True})

    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
