"""User-author affinity generator for participant solution."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.platform.core.dataset import Dataset


class UserAuthorGenerator:
    """Generate candidates from user author preference profiles.

    The generator expands author-level affinity into concrete editions by
    combining normalized user-author weights with edition popularity prior.
    """

    name = "user_author"

    def __init__(
        self, author_smoothing: float = 1.0, show_progress: bool = False
    ) -> None:
        """Store hyperparameters used by author-based scoring.

        Args:
            author_smoothing: Additive prior for each author-edition edge.
            show_progress: Whether to display per-user tqdm updates.
        """
        self.author_smoothing = author_smoothing
        self.show_progress = show_progress

    def generate(
        self,
        dataset: Dataset,
        user_ids: np.ndarray,
        features: pd.DataFrame,
        k: int,
        seed: int,
    ) -> pd.DataFrame:
        """Emit top-k author-driven candidates for every target user.

        Args:
            dataset: Runtime dataset containing edition-to-author mapping.
            user_ids: Target users for candidate generation.
            features: Long feature table with `user_author_profile`.
            k: Maximum number of candidates generated per user.
            seed: Pipeline seed (unused by deterministic score aggregation).

        Returns:
            Candidate DataFrame with required schema and source name.
        """
        del seed
        user_profile = features[features["feature_type"] == "user_author_profile"][
            ["user_id", "author_id", "value"]
        ].copy()
        if user_profile.empty:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        edition_pop = features[features["feature_type"] == "edition_popularity_all"][
            ["edition_id", "value"]
        ].copy()
        pop_map = {
            int(row["edition_id"]): float(row["value"])
            for row in edition_pop.to_dict(orient="records")
        }

        author_to_edition = dataset.catalog_df[["edition_id", "author_id"]].copy()
        author_to_edition["pop"] = (
            author_to_edition["edition_id"].map(pop_map).fillna(0.0)
        )
        author_to_edition = author_to_edition.sort_values(
            ["author_id", "pop", "edition_id"], ascending=[True, False, True]
        )
        top_per_author = max(k * 5, 200)
        author_to_editions: dict[int, list[tuple[int, float]]] = {}
        for author_id, group in author_to_edition.groupby("author_id"):
            rows = group.head(top_per_author)
            author_to_editions[int(author_id)] = [
                (int(edition_id), float(pop))
                for edition_id, pop in zip(
                    rows["edition_id"].tolist(), rows["pop"].tolist()
                )
            ]

        user_profile = user_profile[user_profile["user_id"].isin(user_ids.tolist())]
        rows: list[dict[str, int | float | str]] = []
        grouped = user_profile.groupby("user_id")
        for user_id, group in tqdm(
            grouped,
            total=user_profile["user_id"].nunique(),
            desc=f"{self.name}_users",
            leave=False,
            dynamic_ncols=True,
            disable=not (self.show_progress and sys.stdout.isatty()),
            file=sys.stdout,
        ):
            score_by_edition: dict[int, float] = {}
            for _, profile_row in group.iterrows():
                author_id = int(profile_row["author_id"])
                weight = float(profile_row["value"])
                for edition_id, pop in author_to_editions.get(author_id, []):
                    score_by_edition[edition_id] = score_by_edition.get(
                        edition_id, 0.0
                    ) + (weight * (pop + self.author_smoothing))
            top_items = sorted(score_by_edition.items(), key=lambda x: (-x[1], x[0]))[
                :k
            ]
            for edition_id, score in top_items:
                rows.append(
                    {
                        "user_id": int(user_id),
                        "edition_id": int(edition_id),
                        "score": float(score),
                        "source": self.name,
                    }
                )
        return pd.DataFrame(rows, columns=["user_id", "edition_id", "score", "source"])
