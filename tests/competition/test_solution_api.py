from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.competition.features import build_features_frame
from src.competition.generators import build_generator, run_generators
from src.competition.ranking import rank_predictions
from src.platform.cli.config_loader import load_config
from src.platform.core.dataset import Dataset


def _dataset() -> Dataset:
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
            {
                "edition_id": 13,
                "book_id": 103,
                "author_id": 1002,
                "publication_year": 2023,
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
            {"book_id": 103, "genre_id": 501},
        ]
    )
    authors = pd.DataFrame(
        [
            {"author_id": 1000, "author_name": "Author A"},
            {"author_id": 1001, "author_name": "Author B"},
            {"author_id": 1002, "author_name": "Author C"},
        ]
    )
    genres = pd.DataFrame({"genre_id": [500, 501], "genre_name": ["A", "B"]})
    users = pd.DataFrame({"user_id": [1, 2], "gender": [1, 2], "age": [20, 30]})
    seen = (
        interactions[interactions["event_type"].isin([1, 2])][["user_id", "edition_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
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


def test_experiment_config_imports_system_layer() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "experiments" / "baseline.yaml")
    assert config["paths"]["data_dir"] == "./data"
    assert config["logs"]["tqdm_enabled"] is True
    assert "candidates" in config


def test_generator_registry_and_pipeline_contract() -> None:
    dataset = _dataset()
    features = build_features_frame(dataset=dataset, recent_days=30)
    user_ids = dataset.targets_df["user_id"].astype("int64")
    generators_cfg = [
        {"name": "global_popularity", "params": {}},
        {"name": "user_author", "params": {}},
    ]
    candidates = run_generators(
        dataset=dataset,
        features=features,
        user_ids=user_ids,
        generators_cfg=generators_cfg,
        per_generator_k=5,
        seed=42,
        tqdm_enabled=False,
    )
    assert {"user_id", "edition_id", "score", "source"}.issubset(candidates.columns)

    predictions = rank_predictions(
        dataset=dataset,
        candidates=candidates,
        source_weights={"global_popularity": 1.0, "user_author": 1.0},
        k=2,
    )
    assert {"user_id", "edition_id", "rank"}.issubset(predictions.columns)
    assert predictions["rank"].between(1, 2).all()


def test_build_generator_unknown_name_has_registry_hint() -> None:
    try:
        build_generator("unknown_name", {}, tqdm_enabled=False)
    except ValueError as exc:
        message = str(exc)
        assert "Unknown generator name" in message
        assert "global_popularity" in message
    else:
        raise AssertionError("Expected ValueError for unknown generator name")
