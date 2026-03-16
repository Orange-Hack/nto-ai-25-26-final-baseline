"""Submission validation and NDCG@20 scoring for final task."""

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import click
import pandas as pd


TOP_K = 20


@dataclass(frozen=True)
class ScoreResult:
    """Structured scoring result used by API and CLI."""

    public_score: float
    private_score: float
    overall_score: float
    per_user: pd.DataFrame


def validate_submission_rows(
    submission_rows: list[dict], target_users: set[str]
) -> tuple[bool, list[str]]:
    """Validate submission rows against contract."""
    errors: list[str] = []
    by_user: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in submission_rows:
        user_id = str(row.get("user_id", "")).strip()
        edition_id = str(row.get("edition_id", "")).strip()
        rank_raw = str(row.get("rank", "")).strip()
        if not user_id or not edition_id or not rank_raw:
            errors.append("Rows must contain non-empty user_id, edition_id, rank")
            continue
        if not rank_raw.isdigit():
            errors.append(f"Rank must be integer for user {user_id}")
            continue
        rank = int(rank_raw)
        if rank < 1 or rank > TOP_K:
            errors.append(f"Rank out of range 1..20 for user {user_id}: {rank}")
        by_user[user_id].append((rank, edition_id))

    missing = target_users - set(by_user.keys())
    extra = set(by_user.keys()) - target_users
    if missing:
        errors.append(f"Missing target users: {len(missing)}")
    if extra:
        errors.append(f"Unexpected users in submission: {len(extra)}")

    expected_ranks = set(range(1, TOP_K + 1))
    for user_id in sorted(by_user):
        rows = by_user[user_id]
        if len(rows) != TOP_K:
            errors.append(f"User {user_id}: expected 20 rows, got {len(rows)}")
            continue
        ranks = [rank for rank, _ in rows]
        editions = [edition_id for _, edition_id in rows]
        if len(set(ranks)) != TOP_K or set(ranks) != expected_ranks:
            errors.append(f"User {user_id}: ranks must be unique 1..20")
        if len(set(editions)) != TOP_K:
            errors.append(f"User {user_id}: edition_id must be unique")

    return len(errors) == 0, errors


def validate_submission_file(submission_csv: Path, platform_dir: Path) -> None:
    """Validate submission file and raise ValueError on failure."""
    submission = pd.read_csv(submission_csv)
    solution = pd.read_csv(platform_dir / "solution.csv")
    target_users = {
        str(user_id) for user_id in solution["user_id"].astype(str).unique()
    }
    ok, errors = validate_submission_rows(
        submission.to_dict(orient="records"), target_users
    )
    if not ok:
        raise ValueError("; ".join(errors))


def _ndcg_at_20(predicted: list[str], relevant: set[str]) -> float:
    dcg = 0.0
    for idx, edition_id in enumerate(predicted[:TOP_K], start=1):
        rel = 1.0 if edition_id in relevant else 0.0
        dcg += rel / math.log2(idx + 1)
    idcg = sum(
        1.0 / math.log2(idx + 1) for idx in range(1, min(len(relevant), TOP_K) + 1)
    )
    return dcg / idcg if idcg > 0.0 else 0.0


def score_submission_frames(
    submission: pd.DataFrame, solution: pd.DataFrame
) -> ScoreResult:
    """Score already loaded dataframes and return structured result."""
    required_submission_cols = {"user_id", "edition_id", "rank"}
    required_solution_cols = {"user_id", "edition_id", "stage"}
    if not required_submission_cols.issubset(submission.columns):
        raise ValueError("submission must contain columns: user_id, edition_id, rank")
    if not required_solution_cols.issubset(solution.columns):
        raise ValueError("solution must contain columns: user_id, edition_id, stage")

    target_users = {
        str(user_id) for user_id in solution["user_id"].astype(str).unique()
    }
    submission_rows = submission.to_dict(orient="records")
    ok, errors = validate_submission_rows(submission_rows, target_users)
    if not ok:
        raise ValueError("; ".join(errors))

    predicted_by_user: dict[str, dict[int, str]] = defaultdict(dict)
    for row in submission_rows:
        predicted_by_user[str(row["user_id"])][int(row["rank"])] = str(
            row["edition_id"]
        )

    relevant_by_user_stage: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"public": set(), "private": set()}
    )
    stage_by_user: dict[str, str] = {}
    for row in solution.to_dict(orient="records"):
        user_id = str(row["user_id"])
        stage = str(row["stage"])
        edition_id = str(row["edition_id"])
        relevant_by_user_stage[user_id][stage].add(edition_id)
        stage_by_user[user_id] = stage

    per_user_rows = []
    for user_id in sorted(predicted_by_user):
        predicted = [predicted_by_user[user_id][rank] for rank in range(1, TOP_K + 1)]
        stage = stage_by_user.get(user_id, "public")
        relevant = relevant_by_user_stage[user_id][stage]
        ndcg = _ndcg_at_20(predicted, relevant)
        per_user_rows.append({"user_id": user_id, "stage": stage, "ndcg@20": ndcg})

    per_user_df = pd.DataFrame(per_user_rows)
    public_df = per_user_df[per_user_df["stage"] == "public"]
    private_df = per_user_df[per_user_df["stage"] == "private"]

    public_score = float(public_df["ndcg@20"].mean()) if not public_df.empty else 0.0
    private_score = float(private_df["ndcg@20"].mean()) if not private_df.empty else 0.0
    overall_score = (
        float(per_user_df["ndcg@20"].mean()) if not per_user_df.empty else 0.0
    )
    return ScoreResult(
        public_score=public_score,
        private_score=private_score,
        overall_score=overall_score,
        per_user=per_user_df,
    )


def score_submission(submission_csv: Path, platform_dir: Path) -> dict:
    """Backward-compatible path-based scoring API used in pipeline CLI."""
    submission = pd.read_csv(submission_csv)
    solution = pd.read_csv(platform_dir / "solution.csv")
    result = score_submission_frames(submission, solution)
    return {
        "public_ndcg@20": result.public_score,
        "private_ndcg@20": result.private_score,
        "overall_ndcg@20": result.overall_score,
        "users_count": int(len(result.per_user)),
    }


@click.command()
@click.option(
    "--submission",
    "submission_path",
    default="submission.csv",
    type=click.Path(exists=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--solution",
    "solution_path",
    default="solution.csv",
    type=click.Path(exists=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--per-user-out",
    "per_user_path",
    required=False,
    type=click.Path(path_type=Path),
)
def main(
    submission_path: Path, solution_path: Path, per_user_path: Path | None
) -> None:
    """CLI entrypoint for platform scoring."""
    submission = pd.read_csv(submission_path)
    solution = pd.read_csv(solution_path)
    try:
        result = score_submission_frames(submission, solution)
    except ValueError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(2) from exc

    output = {
        "public_score": result.public_score,
        "private_score": result.private_score,
        "overall_score": result.overall_score,
    }
    print(json.dumps(output, ensure_ascii=False))

    if per_user_path:
        per_user_path.parent.mkdir(parents=True, exist_ok=True)
        result.per_user.to_csv(per_user_path, index=False)


if __name__ == "__main__":
    main()
