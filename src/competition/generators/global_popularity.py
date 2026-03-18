"""Global popularity generator for participant solution."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.platform.core.dataset import Dataset


class GlobalPopularityGenerator:
    """Recommend globally popular editions for each target user.

    The generator serves as a robust baseline source and is used as fallback
    recall in blended ranking. It scores items by unique-user popularity from
    precomputed feature tables and broadcasts top-k popular editions to all
    requested users.
    """

    name = "global_popularity"

    def __init__(self, show_progress: bool = False) -> None:
        """Initialize progress behavior for user-level iteration.

        Args:
            show_progress: Whether to render tqdm bars in interactive sessions.
        """
        self.show_progress = show_progress

    def generate(
        self,
        dataset: Dataset,
        user_ids: np.ndarray,
        features: pd.DataFrame,
        k: int,
        seed: int,
    ) -> pd.DataFrame:
        """Generate candidate rows from global popularity statistics.

        Args:
            dataset: Runtime dataset (unused directly but part of generator contract).
            user_ids: Users for whom candidates must be emitted.
            features: Long feature table containing `edition_popularity_all`.
            k: Maximum candidate count per user.
            seed: Pipeline seed (unused by this deterministic generator).

        Returns:
            Candidate DataFrame with `user_id`, `edition_id`, `score`, `source`.
        """
        del dataset, seed
        popularity = features[features["feature_type"] == "edition_popularity_all"][
            ["edition_id", "value"]
        ].copy()
        popularity = popularity.sort_values(
            ["value", "edition_id"], ascending=[False, True]
        ).head(k)

        if popularity.empty:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        rows: list[dict[str, float | int | str]] = []
        for user_id in tqdm(
            user_ids.tolist(),
            total=len(user_ids),
            desc=f"{self.name}_users",
            leave=False,
            dynamic_ncols=True,
            disable=not (self.show_progress and sys.stdout.isatty()),
            file=sys.stdout,
        ):
            for _, row in popularity.iterrows():
                rows.append(
                    {
                        "user_id": int(user_id),
                        "edition_id": int(row["edition_id"]),
                        "score": float(row["value"]),
                        "source": self.name,
                    }
                )
        return pd.DataFrame(rows)
