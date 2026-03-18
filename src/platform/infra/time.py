"""Time helper primitives used by platform components."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Provide UTC timestamp used in logs and metadata snapshots.

    Returns:
        ISO-8601 timestamp with explicit UTC timezone offset.
    """
    return datetime.now(tz=timezone.utc).isoformat()
