from __future__ import annotations

import pandas as pd
import pytest

from src.platform.core.submission_contract import validate_submission_frame


def _build_valid_submission() -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for rank in range(1, 21):
        rows.append({"user_id": 1, "edition_id": 1000 + rank, "rank": rank})
        rows.append({"user_id": 2, "edition_id": 2000 + rank, "rank": rank})
    return pd.DataFrame(rows)


def test_validate_submission_accepts_correct_format() -> None:
    submission = _build_valid_submission()
    validate_submission_frame(submission_df=submission, target_users={1, 2}, k=20)


def test_validate_submission_rejects_duplicate_rank() -> None:
    submission = _build_valid_submission()
    submission.loc[(submission["user_id"] == 1) & (submission["rank"] == 2), "rank"] = 1
    with pytest.raises(ValueError):
        validate_submission_frame(submission_df=submission, target_users={1, 2}, k=20)


def test_validate_submission_rejects_missing_user() -> None:
    submission = _build_valid_submission()
    submission = submission[submission["user_id"] == 1]
    with pytest.raises(ValueError):
        validate_submission_frame(submission_df=submission, target_users={1, 2}, k=20)
