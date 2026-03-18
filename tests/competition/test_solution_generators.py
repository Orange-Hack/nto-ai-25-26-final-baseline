from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.competition.generators.global_popularity import GlobalPopularityGenerator
from src.competition.generators.user_author import UserAuthorGenerator
from src.competition.generators.user_genre import UserGenrePopularityGenerator
from src.platform.core.dataset import Dataset


@pytest.fixture
def sample_dataset() -> Dataset:
    interactions = pd.DataFrame(
        [
            {
                "user_id": 1,
                "edition_id": 10,
                "event_type": 1,
                "rating": None,
                "event_ts": "2026-01-01",
            },
            {
                "user_id": 1,
                "edition_id": 11,
                "event_type": 2,
                "rating": 5.0,
                "event_ts": "2026-01-02",
            },
            {
                "user_id": 2,
                "edition_id": 12,
                "event_type": 1,
                "rating": None,
                "event_ts": "2026-01-03",
            },
        ]
    )
    interactions["event_ts"] = pd.to_datetime(interactions["event_ts"])
    targets = pd.DataFrame({"user_id": [1, 2]})
    catalog = pd.DataFrame(
        [
            {
                "edition_id": 10,
                "book_id": 100,
                "author_id": 1000,
                "publication_year": 2020,
                "age_restriction": 12,
                "language_id": 1,
                "publisher_id": 1,
            },
            {
                "edition_id": 11,
                "book_id": 101,
                "author_id": 1001,
                "publication_year": 2021,
                "age_restriction": 12,
                "language_id": 1,
                "publisher_id": 1,
            },
            {
                "edition_id": 12,
                "book_id": 102,
                "author_id": 1000,
                "publication_year": 2022,
                "age_restriction": 12,
                "language_id": 1,
                "publisher_id": 2,
            },
        ]
    )
    book_genres = pd.DataFrame(
        [
            {"book_id": 100, "genre_id": 500},
            {"book_id": 101, "genre_id": 501},
            {"book_id": 102, "genre_id": 500},
        ]
    )
    authors = pd.DataFrame(
        [
            {"author_id": 1000, "author_name": "Author A"},
            {"author_id": 1001, "author_name": "Author B"},
        ]
    )
    genres = pd.DataFrame({"genre_id": [500, 501], "genre_name": ["A", "B"]})
    users = pd.DataFrame({"user_id": [1, 2], "gender": [1, 2], "age": [20, 30]})
    seen = interactions[interactions["event_type"].isin([1, 2])][
        ["user_id", "edition_id"]
    ].drop_duplicates()
    return Dataset(
        interactions_df=interactions,
        targets_df=targets,
        catalog_df=catalog,
        authors_df=authors,
        book_genres_df=book_genres,
        genres_df=genres,
        users_df=users,
        seen_positive_df=seen,
    )


@pytest.fixture
def sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_type": "edition_popularity_all",
                "user_id": pd.NA,
                "edition_id": 10,
                "genre_id": pd.NA,
                "author_id": pd.NA,
                "value": 5.0,
            },
            {
                "feature_type": "edition_popularity_all",
                "user_id": pd.NA,
                "edition_id": 11,
                "genre_id": pd.NA,
                "author_id": pd.NA,
                "value": 3.0,
            },
            {
                "feature_type": "user_genre_profile",
                "user_id": 1,
                "edition_id": pd.NA,
                "genre_id": 500,
                "author_id": pd.NA,
                "value": 1.0,
            },
            {
                "feature_type": "user_author_profile",
                "user_id": 1,
                "edition_id": pd.NA,
                "genre_id": pd.NA,
                "author_id": 1000,
                "value": 1.0,
            },
        ]
    )


@pytest.mark.parametrize(
    "generator",
    [
        GlobalPopularityGenerator(),
        UserGenrePopularityGenerator(),
        UserAuthorGenerator(),
    ],
)
def test_generators_contract(
    sample_dataset: Dataset, sample_features: pd.DataFrame, generator: object
) -> None:
    user_ids = np.array([1, 2], dtype=np.int64)
    frame = generator.generate(
        dataset=sample_dataset,
        user_ids=user_ids,
        features=sample_features,
        k=5,
        seed=42,
    )
    required = {"user_id", "edition_id", "score", "source"}
    assert required.issubset(frame.columns)
    assert frame["source"].eq(generator.name).all()
