"""Feature construction entrypoint for participant solution."""

from __future__ import annotations

import pandas as pd

from src.platform.core.dataset import Dataset


def build_features_frame(dataset: Dataset, recent_days: int) -> pd.DataFrame:
    """Build baseline feature matrix consumed by candidate generators.

    The function creates a compact long-form feature table that keeps the
    baseline extensible while staying model-agnostic. The generated feature
    blocks encode global popularity and user preference profiles over genres
    and authors.

    Args:
        dataset: Runtime dataset with interactions, catalog, and taxonomy tables.
        recent_days: Time window in days for recency popularity signal.

    Returns:
        Long-form feature DataFrame with columns:
        `feature_type`, `user_id`, `edition_id`, `genre_id`, `author_id`, `value`.
    """
    positives = dataset.interactions_df[
        dataset.interactions_df["event_type"].isin([1, 2])
    ]

    popularity_all = (
        positives.groupby("edition_id", as_index=False)["user_id"]
        .nunique()
        .rename(columns={"user_id": "value"})
    )
    popularity_all["feature_type"] = "edition_popularity_all"
    popularity_all["user_id"] = pd.NA
    popularity_all["genre_id"] = pd.NA
    popularity_all["author_id"] = pd.NA

    max_ts = positives["event_ts"].max()
    cutoff = max_ts - pd.Timedelta(days=recent_days)
    recent = positives[positives["event_ts"] >= cutoff]
    popularity_recent = (
        recent.groupby("edition_id", as_index=False)["user_id"]
        .nunique()
        .rename(columns={"user_id": "value"})
    )
    popularity_recent["feature_type"] = "edition_popularity_recent"
    popularity_recent["user_id"] = pd.NA
    popularity_recent["genre_id"] = pd.NA
    popularity_recent["author_id"] = pd.NA

    user_genres = positives[["user_id", "edition_id"]].merge(
        dataset.catalog_df[["edition_id", "book_id"]],
        on="edition_id",
        how="inner",
    )
    user_genres = user_genres.merge(dataset.book_genres_df, on="book_id", how="inner")
    user_genre_profile = (
        user_genres.groupby(["user_id", "genre_id"], as_index=False)["edition_id"]
        .count()
        .rename(columns={"edition_id": "value"})
    )
    user_genre_profile["value"] = user_genre_profile[
        "value"
    ] / user_genre_profile.groupby("user_id")["value"].transform("sum")
    user_genre_profile["feature_type"] = "user_genre_profile"
    user_genre_profile["edition_id"] = pd.NA
    user_genre_profile["author_id"] = pd.NA

    user_authors = positives[["user_id", "edition_id"]].merge(
        dataset.catalog_df[["edition_id", "author_id"]],
        on="edition_id",
        how="inner",
    )
    user_author_profile = (
        user_authors.groupby(["user_id", "author_id"], as_index=False)["edition_id"]
        .count()
        .rename(columns={"edition_id": "value"})
    )
    user_author_profile["value"] = user_author_profile[
        "value"
    ] / user_author_profile.groupby("user_id")["value"].transform("sum")
    user_author_profile["feature_type"] = "user_author_profile"
    user_author_profile["edition_id"] = pd.NA
    user_author_profile["genre_id"] = pd.NA

    return pd.concat(
        [
            popularity_all[
                [
                    "feature_type",
                    "user_id",
                    "edition_id",
                    "genre_id",
                    "author_id",
                    "value",
                ]
            ],
            popularity_recent[
                [
                    "feature_type",
                    "user_id",
                    "edition_id",
                    "genre_id",
                    "author_id",
                    "value",
                ]
            ],
            user_genre_profile[
                [
                    "feature_type",
                    "user_id",
                    "edition_id",
                    "genre_id",
                    "author_id",
                    "value",
                ]
            ],
            user_author_profile[
                [
                    "feature_type",
                    "user_id",
                    "edition_id",
                    "genre_id",
                    "author_id",
                    "value",
                ]
            ],
        ],
        ignore_index=True,
    )
