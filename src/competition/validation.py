"""Submission validation helpers for participant solution output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.platform.core.submission_contract import validate_submission_frame


def validate_submission(submission: pd.DataFrame, data_dir: Path, k: int) -> None:
    """Validate submission against target users and rank constraints.

    Args:
        submission: Final submission frame with three required columns.
        data_dir: Directory that contains `targets.csv`.
        k: Required number of ranked items per target user.

    Raises:
        ValueError: If submission schema or per-user ranking contract is invalid.
    """
    target_users = set(
        pd.read_csv(data_dir / "targets.csv")["user_id"].astype("int64").tolist()
    )
    validate_submission_frame(
        submission_df=submission, target_users=target_users, k=int(k)
    )
