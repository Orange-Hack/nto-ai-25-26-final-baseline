"""Artifact persistence and cache-status tracking primitives."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.platform.infra.time import utc_now_iso


class ArtifactsManager:
    """Track stage status metadata and cache fingerprints on disk.

    The manager centralizes atomic writes for run metadata and step statuses so
    interrupted runs never leave partially written control files.
    """

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.meta_dir = artifacts_dir / "_meta"
        self.step_status_path = self.meta_dir / "step_status.json"
        self.run_meta_path = self.meta_dir / "run.json"
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def should_run(self, step_name: str, fingerprint: str, output_path: Path) -> bool:
        """Decide whether stage output can be reused from cache.

        Args:
            step_name: Canonical stage identifier.
            fingerprint: Current stage fingerprint for inputs/config.
            output_path: Expected artifact path for stage output.

        Returns:
            True if stage must be executed, False if cache-hit is valid.
        """
        if not output_path.exists():
            return True
        status = self._read_json(self.step_status_path)
        step_info = status.get(step_name)
        if not step_info:
            return True
        return (
            step_info.get("status") != "done"
            or step_info.get("fingerprint") != fingerprint
        )

    def mark_started(self, step_name: str, fingerprint: str) -> None:
        """Persist stage start marker for run diagnostics."""
        status = self._read_json(self.step_status_path)
        status.setdefault(step_name, {})
        status[step_name].update(
            {
                "status": "running",
                "fingerprint": fingerprint,
                "started_at": utc_now_iso(),
            }
        )
        atomic_write_json(self.step_status_path, status)

    def _mark_done_internal(
        self,
        step_name: str,
        fingerprint: str,
        payload: dict[str, Any],
        duration_sec: float | None,
    ) -> None:
        status = self._read_json(self.step_status_path)
        status.setdefault(step_name, {})
        step_stats = dict(payload)
        if duration_sec is not None:
            step_stats["duration_sec"] = float(round(duration_sec, 3))
            rows = payload.get("rows")
            if isinstance(rows, int) and rows > 0 and duration_sec > 0:
                step_stats["rows_per_sec"] = float(round(rows / duration_sec, 3))
        status[step_name].update(
            {
                "status": "done",
                "fingerprint": fingerprint,
                "finished_at": utc_now_iso(),
                "duration_sec": float(round(duration_sec, 3))
                if duration_sec is not None
                else None,
                "stats": step_stats,
            }
        )
        atomic_write_json(self.step_status_path, status)

    def mark_done(
        self,
        step_name: str,
        fingerprint: str,
        payload: dict[str, Any],
        duration_sec: float | None = None,
    ) -> None:
        """Persist successful stage completion metadata."""
        self._mark_done_internal(
            step_name=step_name,
            fingerprint=fingerprint,
            payload=payload,
            duration_sec=duration_sec,
        )

    def write_run_meta(self, run_meta: dict[str, Any]) -> None:
        """Write run metadata snapshot atomically."""
        atomic_write_json(self.run_meta_path, run_meta)

    def get_step_durations(self, step_names: list[str]) -> dict[str, float]:
        """Fetch historical stage durations for ETA estimation.

        Args:
            step_names: Ordered stage names expected in the current run.

        Returns:
            Mapping of stage name to previously measured duration in seconds.
        """
        status = self._read_json(self.step_status_path)
        durations: dict[str, float] = {}
        for step_name in step_names:
            duration = status.get(step_name, {}).get("duration_sec")
            if isinstance(duration, (int, float)) and duration > 0:
                durations[step_name] = float(duration)
        return durations


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON payload using same-directory temporary file.

    Args:
        path: Final destination path.
        payload: JSON-serializable dictionary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def atomic_write_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Atomically write dataframe artifact in parquet or csv format.

    Args:
        df: DataFrame payload to persist.
        path: Target artifact path with `.parquet` or `.csv` suffix.

    Raises:
        ValueError: If file suffix is not supported.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        if suffix == ".parquet":
            df.to_parquet(tmp_path, index=False)
        elif suffix == ".csv":
            df.to_csv(tmp_path, index=False)
        else:
            raise ValueError(f"Unsupported artifact format: {suffix}")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
