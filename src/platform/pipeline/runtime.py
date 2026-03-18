"""Runtime dataset assembly helpers for pipeline stages."""

from __future__ import annotations

import pandas as pd

from src.platform.core.dataset import Dataset
from src.platform.pipeline.models import PipelinePaths


def load_base_dataset(paths: PipelinePaths) -> Dataset:
    """Load immutable source dataset directly from raw data directory."""
    return Dataset.load(paths.data_dir)


def load_runtime_dataset(paths: PipelinePaths) -> Dataset:
    """Load runtime dataset using cached interactions and seen-positive pairs.

    Args:
        paths: Resolved paths for current run artifacts and source data.

    Returns:
        Dataset with cached interactions/seen positives and fresh static tables.
    """
    base_dataset = Dataset.load(paths.data_dir)
    cache = pd.read_csv(paths.data_cache_path)
    interactions = cache[cache["_record_type"] == "interaction"][
        ["user_id", "edition_id", "event_type", "rating", "event_ts"]
    ].copy()
    interactions["user_id"] = interactions["user_id"].astype("int64")
    interactions["edition_id"] = interactions["edition_id"].astype("int64")
    interactions["event_type"] = interactions["event_type"].astype("int32")
    interactions["event_ts"] = pd.to_datetime(interactions["event_ts"])

    seen_positive = cache[cache["_record_type"] == "seen_positive"][
        ["user_id", "edition_id"]
    ].copy()
    seen_positive["user_id"] = seen_positive["user_id"].astype("int64")
    seen_positive["edition_id"] = seen_positive["edition_id"].astype("int64")

    return Dataset(
        interactions_df=interactions,
        targets_df=base_dataset.targets_df,
        catalog_df=base_dataset.catalog_df,
        authors_df=base_dataset.authors_df,
        book_genres_df=base_dataset.book_genres_df,
        genres_df=base_dataset.genres_df,
        users_df=base_dataset.users_df,
        seen_positive_df=seen_positive,
    )


def pack_data_cache(dataset: Dataset) -> pd.DataFrame:
    """Serialize interactions and seen-positive pairs into single cache table."""
    interactions = dataset.interactions_df.copy()
    interactions["_record_type"] = "interaction"
    seen = dataset.seen_positive_df.copy()
    seen["_record_type"] = "seen_positive"
    seen["event_type"] = pd.NA
    seen["rating"] = pd.NA
    seen["event_ts"] = pd.NaT
    combined = pd.concat([interactions, seen], ignore_index=True, sort=False)
    return combined[
        ["_record_type", "user_id", "edition_id", "event_type", "rating", "event_ts"]
    ]
