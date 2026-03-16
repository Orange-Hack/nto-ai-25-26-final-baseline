"""Metric helpers used by local validation workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MetricsSummary:
    """Bundle local-validation aggregate statistics for reporting."""

    mean_ndcg: float
    quantiles: dict[str, float]
    per_user: pd.DataFrame


def ndcg_at_k(predicted: list[int], relevant: set[int], k: int) -> float:
    """Compute binary NDCG@k for a single user ranking.

    Args:
        predicted: Ordered edition IDs predicted for a user.
        relevant: Set of relevant edition IDs for that user.
        k: Cutoff depth for DCG/IDCG computation.

    Returns:
        NDCG score in the `[0, 1]` range.
    """
    dcg = 0.0
    for rank, edition_id in enumerate(predicted[:k], start=1):
        rel = 1.0 if edition_id in relevant else 0.0
        dcg += rel / math.log2(rank + 1)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def summarize_ndcg(
    per_user_df: pd.DataFrame, score_column: str = "ndcg@20"
) -> MetricsSummary:
    """Aggregate local-validation per-user scores into summary statistics.

    Args:
        per_user_df: DataFrame with per-user metric values.
        score_column: Column name containing per-user NDCG values.

    Returns:
        `MetricsSummary` with mean and key quantiles for quick monitoring.
    """
    if per_user_df.empty:
        return MetricsSummary(
            mean_ndcg=0.0,
            quantiles={"q25": 0.0, "q50": 0.0, "q75": 0.0},
            per_user=per_user_df,
        )
    if score_column not in per_user_df.columns:
        raise ValueError(f"Missing score column: {score_column}")
    scores = per_user_df[score_column]
    quantiles = {
        "q25": float(scores.quantile(0.25)),
        "q50": float(scores.quantile(0.50)),
        "q75": float(scores.quantile(0.75)),
    }
    return MetricsSummary(
        mean_ndcg=float(scores.mean()),
        quantiles=quantiles,
        per_user=per_user_df,
    )

