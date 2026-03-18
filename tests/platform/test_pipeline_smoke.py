from __future__ import annotations

import json
import logging
from io import StringIO
from pathlib import Path

import pandas as pd

from src.platform.core.artifacts import ArtifactsManager
from src.platform.pipeline import PipelineRunner


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_minimal_dataset(data_dir: Path) -> None:
    _write_csv(
        data_dir / "interactions.csv",
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
            {
                "user_id": 2,
                "edition_id": 13,
                "event_type": 2,
                "rating": 4.0,
                "event_ts": "2026-01-04",
            },
        ],
    )
    _write_csv(data_dir / "targets.csv", [{"user_id": 1}, {"user_id": 2}])
    _write_csv(
        data_dir / "editions.csv",
        [
            {
                "edition_id": 10,
                "book_id": 100,
                "author_id": 1000,
                "publication_year": 2020,
                "age_restriction": 12,
                "language_id": 1,
                "publisher_id": 1,
                "title": "a",
                "description": "a",
            },
            {
                "edition_id": 11,
                "book_id": 101,
                "author_id": 1001,
                "publication_year": 2021,
                "age_restriction": 12,
                "language_id": 1,
                "publisher_id": 1,
                "title": "b",
                "description": "b",
            },
            {
                "edition_id": 12,
                "book_id": 102,
                "author_id": 1000,
                "publication_year": 2022,
                "age_restriction": 12,
                "language_id": 1,
                "publisher_id": 2,
                "title": "c",
                "description": "c",
            },
            {
                "edition_id": 13,
                "book_id": 103,
                "author_id": 1002,
                "publication_year": 2023,
                "age_restriction": 12,
                "language_id": 1,
                "publisher_id": 3,
                "title": "d",
                "description": "d",
            },
            {
                "edition_id": 14,
                "book_id": 104,
                "author_id": 1002,
                "publication_year": 2023,
                "age_restriction": 12,
                "language_id": 1,
                "publisher_id": 3,
                "title": "e",
                "description": "e",
            },
        ],
    )
    _write_csv(
        data_dir / "authors.csv",
        [
            {"author_id": 1000, "author_name": "author-a"},
            {"author_id": 1001, "author_name": "author-b"},
            {"author_id": 1002, "author_name": "author-c"},
        ],
    )
    _write_csv(
        data_dir / "book_genres.csv",
        [
            {"book_id": 100, "genre_id": 500},
            {"book_id": 101, "genre_id": 501},
            {"book_id": 102, "genre_id": 500},
            {"book_id": 103, "genre_id": 501},
            {"book_id": 104, "genre_id": 500},
        ],
    )
    _write_csv(
        data_dir / "genres.csv",
        [{"genre_id": 500, "genre_name": "x"}, {"genre_id": 501, "genre_name": "y"}],
    )
    _write_csv(
        data_dir / "users.csv",
        [
            {"user_id": 1, "gender": 1, "age": 22},
            {"user_id": 2, "gender": 2, "age": 25},
        ],
    )


def test_pipeline_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()

    _write_csv(
        data_dir / "interactions.csv",
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
            {
                "user_id": 2,
                "edition_id": 13,
                "event_type": 2,
                "rating": 4.0,
                "event_ts": "2026-01-04",
            },
        ],
    )
    _write_csv(data_dir / "targets.csv", [{"user_id": 1}, {"user_id": 2}])
    _write_csv(
        data_dir / "editions.csv",
        [
            {
                "edition_id": 10,
                "book_id": 100,
                "author_id": 1000,
                "publication_year": 2020,
                "age_restriction": 12,
                "language_id": 1,
                "publisher_id": 1,
                "title": "a",
                "description": "a",
            },
            {
                "edition_id": 11,
                "book_id": 101,
                "author_id": 1001,
                "publication_year": 2021,
                "age_restriction": 12,
                "language_id": 1,
                "publisher_id": 1,
                "title": "b",
                "description": "b",
            },
            {
                "edition_id": 12,
                "book_id": 102,
                "author_id": 1000,
                "publication_year": 2022,
                "age_restriction": 12,
                "language_id": 1,
                "publisher_id": 2,
                "title": "c",
                "description": "c",
            },
            {
                "edition_id": 13,
                "book_id": 103,
                "author_id": 1002,
                "publication_year": 2023,
                "age_restriction": 12,
                "language_id": 1,
                "publisher_id": 3,
                "title": "d",
                "description": "d",
            },
            {
                "edition_id": 14,
                "book_id": 104,
                "author_id": 1002,
                "publication_year": 2023,
                "age_restriction": 12,
                "language_id": 1,
                "publisher_id": 3,
                "title": "e",
                "description": "e",
            },
        ],
    )
    _write_csv(
        data_dir / "authors.csv",
        [
            {"author_id": 1000, "author_name": "author-a"},
            {"author_id": 1001, "author_name": "author-b"},
            {"author_id": 1002, "author_name": "author-c"},
        ],
    )
    _write_csv(
        data_dir / "book_genres.csv",
        [
            {"book_id": 100, "genre_id": 500},
            {"book_id": 101, "genre_id": 501},
            {"book_id": 102, "genre_id": 500},
            {"book_id": 103, "genre_id": 501},
            {"book_id": 104, "genre_id": 500},
        ],
    )
    _write_csv(
        data_dir / "genres.csv",
        [{"genre_id": 500, "genre_name": "x"}, {"genre_id": 501, "genre_name": "y"}],
    )
    _write_csv(
        data_dir / "users.csv",
        [
            {"user_id": 1, "gender": 1, "age": 22},
            {"user_id": 2, "gender": 2, "age": 25},
        ],
    )

    config = {
        "paths": {"data_dir": str(data_dir), "artifacts_dir": str(artifacts_dir)},
        "pipeline": {"k": 2, "seed": 42, "recent_days": 30},
        "candidates": {
            "per_generator_k": 5,
            "generators": [
                {"name": "global_popularity", "params": {}},
                {"name": "user_genre", "params": {}},
                {"name": "user_author", "params": {}},
            ],
        },
        "ranking": {"source_weights": {}},
        "validation": {"pseudo_incident_days": 2},
        "logs": {"dir": str(logs_dir), "tqdm_enabled": False},
    }
    logger = logging.getLogger("test-smoke")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    runner = PipelineRunner(config=config, logger=logger)
    runner.run()

    assert (artifacts_dir / "data_cache.csv").exists()
    assert (artifacts_dir / "features.csv").exists()
    assert (artifacts_dir / "candidates.csv").exists()
    assert (artifacts_dir / "predictions.csv").exists()
    assert (artifacts_dir / "submission.csv").exists()

    step_status = pd.read_json(artifacts_dir / "_meta" / "step_status.json")
    assert "prepare_data" in step_status.columns


def test_progress_metadata_and_skip_logs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    _write_csv(
        data_dir / "interactions.csv",
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
            {
                "user_id": 2,
                "edition_id": 13,
                "event_type": 2,
                "rating": 4.0,
                "event_ts": "2026-01-04",
            },
        ],
    )
    _write_csv(data_dir / "targets.csv", [{"user_id": 1}, {"user_id": 2}])
    _write_csv(
        data_dir / "editions.csv",
        [
            {
                "edition_id": 10,
                "book_id": 100,
                "author_id": 1000,
                "publication_year": 2020,
                "age_restriction": 12,
                "language_id": 1,
                "publisher_id": 1,
                "title": "a",
                "description": "a",
            },
            {
                "edition_id": 11,
                "book_id": 101,
                "author_id": 1001,
                "publication_year": 2021,
                "age_restriction": 12,
                "language_id": 1,
                "publisher_id": 1,
                "title": "b",
                "description": "b",
            },
            {
                "edition_id": 12,
                "book_id": 102,
                "author_id": 1000,
                "publication_year": 2022,
                "age_restriction": 12,
                "language_id": 1,
                "publisher_id": 2,
                "title": "c",
                "description": "c",
            },
            {
                "edition_id": 13,
                "book_id": 103,
                "author_id": 1002,
                "publication_year": 2023,
                "age_restriction": 12,
                "language_id": 1,
                "publisher_id": 3,
                "title": "d",
                "description": "d",
            },
            {
                "edition_id": 14,
                "book_id": 104,
                "author_id": 1002,
                "publication_year": 2023,
                "age_restriction": 12,
                "language_id": 1,
                "publisher_id": 3,
                "title": "e",
                "description": "e",
            },
        ],
    )
    _write_csv(
        data_dir / "authors.csv",
        [
            {"author_id": 1000, "author_name": "author-a"},
            {"author_id": 1001, "author_name": "author-b"},
            {"author_id": 1002, "author_name": "author-c"},
        ],
    )
    _write_csv(
        data_dir / "book_genres.csv",
        [
            {"book_id": 100, "genre_id": 500},
            {"book_id": 101, "genre_id": 501},
            {"book_id": 102, "genre_id": 500},
            {"book_id": 103, "genre_id": 501},
            {"book_id": 104, "genre_id": 500},
        ],
    )
    _write_csv(
        data_dir / "genres.csv",
        [{"genre_id": 500, "genre_name": "x"}, {"genre_id": 501, "genre_name": "y"}],
    )
    _write_csv(
        data_dir / "users.csv",
        [
            {"user_id": 1, "gender": 1, "age": 22},
            {"user_id": 2, "gender": 2, "age": 25},
        ],
    )

    config = {
        "paths": {"data_dir": str(data_dir), "artifacts_dir": str(artifacts_dir)},
        "pipeline": {"k": 2, "seed": 42, "recent_days": 30},
        "candidates": {
            "per_generator_k": 5,
            "generators": [
                {"name": "global_popularity", "params": {}},
                {"name": "user_genre", "params": {}},
                {"name": "user_author", "params": {}},
            ],
        },
        "ranking": {"source_weights": {}},
        "validation": {"pseudo_incident_days": 2},
        "logs": {"dir": str(logs_dir), "tqdm_enabled": False},
    }
    stream = StringIO()
    logger = logging.getLogger("test-progress")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(logging.StreamHandler(stream))

    runner = PipelineRunner(config=config, logger=logger)
    runner.run()
    runner.run()

    status = ArtifactsManager(artifacts_dir).get_step_durations(
        [
            "prepare_data",
            "build_features",
            "generate_candidates",
            "rank_and_select",
            "make_submission",
        ]
    )
    assert len(status) == 5
    assert all(value >= 0 for value in status.values())

    output = stream.getvalue()
    assert "Stage 1/5 prepare_data started" in output
    assert "Skip stage 1/5 prepare_data (cache hit)" in output


def test_generate_candidates_recomputes_only_changed_generator(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    _write_minimal_dataset(data_dir)

    base_config = {
        "paths": {"data_dir": str(data_dir), "artifacts_dir": str(artifacts_dir)},
        "pipeline": {"k": 2, "seed": 42, "recent_days": 30},
        "candidates": {
            "per_generator_k": 5,
            "generators": [
                {"name": "global_popularity", "params": {}},
                {"name": "user_author", "params": {"author_smoothing": 1.0}},
            ],
        },
        "ranking": {"source_weights": {}},
        "validation": {"pseudo_incident_days": 2},
        "logs": {"dir": str(logs_dir), "tqdm_enabled": False},
    }

    logger = logging.getLogger("test-generator-cache")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    runner = PipelineRunner(config=base_config, logger=logger)
    runner.run(stage="generate_candidates")

    generators_dir = artifacts_dir / "generators"
    manifest_path = artifacts_dir / "_meta" / "generator_cache_manifest.json"
    first_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_cache_files = sorted(p.name for p in generators_dir.glob("*.csv"))

    assert first_manifest["cache_hits"] == 0
    assert first_manifest["cache_misses"] == 2
    assert len(first_cache_files) == 2

    changed_config = {
        **base_config,
        "candidates": {
            "per_generator_k": 5,
            "generators": [
                {"name": "global_popularity", "params": {}},
                {"name": "user_author", "params": {"author_smoothing": 2.0}},
            ],
        },
    }
    runner_changed = PipelineRunner(config=changed_config, logger=logger)
    runner_changed.run(stage="generate_candidates")

    second_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    second_cache_files = sorted(p.name for p in generators_dir.glob("*.csv"))

    assert second_manifest["cache_hits"] == 1
    assert second_manifest["cache_misses"] == 1
    assert len(second_cache_files) == 3
    assert any(
        entry["source_name"] == "global_popularity" and entry["cache_hit"] is True
        for entry in second_manifest["entries"]
    )
    assert any(
        entry["source_name"] == "user_author" and entry["cache_hit"] is False
        for entry in second_manifest["entries"]
    )
