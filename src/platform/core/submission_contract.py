"""Strict submission contract checks for final output integrity."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd


def validate_submission_frame(
    submission_df: pd.DataFrame, target_users: set[int], k: int = 20
) -> None:
    """Enforce submission schema and per-user ranking rules.

    Args:
        submission_df: Candidate submission frame to validate.
        target_users: Set of required users from targets file.
        k: Required recommendation list length per user.

    Raises:
        ValueError: If schema, rank domain, completeness, or uniqueness fail.
    """
    required_columns = {"user_id", "edition_id", "rank"}
    if not required_columns.issubset(submission_df.columns):
        raise ValueError("submission must contain columns: user_id, edition_id, rank")

    by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
    errors: list[str] = []
    for row in submission_df.to_dict(orient="records"):
        user_id = int(row["user_id"])
        edition_id = int(row["edition_id"])
        rank = int(row["rank"])
        if rank < 1 or rank > k:
            errors.append(f"user {user_id}: rank must be in 1..{k}, got {rank}")
        by_user[user_id].append((rank, edition_id))

    missing_users = target_users - set(by_user.keys())
    extra_users = set(by_user.keys()) - target_users
    if missing_users:
        errors.append(f"missing users from targets: {len(missing_users)}")
    if extra_users:
        errors.append(f"extra users in submission: {len(extra_users)}")

    expected_ranks = set(range(1, k + 1))
    for user_id, rows in by_user.items():
        if len(rows) != k:
            errors.append(f"user {user_id}: expected {k} rows, got {len(rows)}")
            continue
        ranks = [rank for rank, _ in rows]
        edition_ids = [edition_id for _, edition_id in rows]
        if set(ranks) != expected_ranks:
            errors.append(f"user {user_id}: ranks must be unique 1..{k}")
        if len(set(edition_ids)) != k:
            errors.append(f"user {user_id}: edition_id must be unique")

    if errors:
        raise ValueError("; ".join(errors))
