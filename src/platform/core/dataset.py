"""Dataset container and loader for platform pipeline runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.platform.infra.loaders import read_csv
from src.platform.infra.schema import ensure_columns


@dataclass(frozen=True)
class Dataset:
    """Store normalized data frames shared by pipeline stages.

    Attributes:
        interactions_df: Event log with `event_ts` parsed to datetime.
        targets_df: Users requiring top-k predictions.
        catalog_df: Edition metadata.
        authors_df: Author dictionary.
        book_genres_df: Book to genre links.
        genres_df: Genre dictionary.
        users_df: User profile table.
        seen_positive_df: Unique positive `(user_id, edition_id)` pairs.
    """

    interactions_df: pd.DataFrame
    targets_df: pd.DataFrame
    catalog_df: pd.DataFrame
    authors_df: pd.DataFrame
    book_genres_df: pd.DataFrame
    genres_df: pd.DataFrame
    users_df: pd.DataFrame
    seen_positive_df: pd.DataFrame

    @staticmethod
    def load(data_dir: Path) -> "Dataset":
        """Load and normalize all required CSV inputs.

        Args:
            data_dir: Directory containing task source CSV files.

        Returns:
            Fully normalized `Dataset` instance ready for pipeline stages.

        Raises:
            FileNotFoundError: If any required CSV file is missing.
            ValueError: If required columns or timestamp values are invalid.
        """
        required_files = {
            "interactions.csv": data_dir / "interactions.csv",
            "targets.csv": data_dir / "targets.csv",
            "editions.csv": data_dir / "editions.csv",
            "authors.csv": data_dir / "authors.csv",
            "book_genres.csv": data_dir / "book_genres.csv",
            "genres.csv": data_dir / "genres.csv",
            "users.csv": data_dir / "users.csv",
        }
        for file_name, path in required_files.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Required file is missing: {file_name}. Expected path: {path}"
                )

        interactions_df = read_csv(required_files["interactions.csv"])
        targets_df = read_csv(required_files["targets.csv"])
        editions_df = read_csv(required_files["editions.csv"])
        authors_df = read_csv(required_files["authors.csv"])
        book_genres_df = read_csv(required_files["book_genres.csv"])
        genres_df = read_csv(required_files["genres.csv"])
        users_df = read_csv(required_files["users.csv"])

        ensure_columns(
            interactions_df,
            ["user_id", "edition_id", "event_type", "rating", "event_ts"],
            "interactions.csv",
        )
        ensure_columns(targets_df, ["user_id"], "targets.csv")
        ensure_columns(
            editions_df,
            [
                "edition_id",
                "book_id",
                "author_id",
                "publication_year",
                "age_restriction",
                "language_id",
                "publisher_id",
            ],
            "editions.csv",
        )
        ensure_columns(authors_df, ["author_id"], "authors.csv")
        ensure_columns(book_genres_df, ["book_id", "genre_id"], "book_genres.csv")
        ensure_columns(genres_df, ["genre_id"], "genres.csv")

        interactions_df["event_ts"] = pd.to_datetime(
            interactions_df["event_ts"], errors="coerce"
        )
        if interactions_df["event_ts"].isna().any():
            raise ValueError("interactions.csv contains invalid event_ts values")

        int_columns = [
            ("user_id", targets_df),
            ("user_id", interactions_df),
            ("edition_id", interactions_df),
            ("event_type", interactions_df),
            ("user_id", users_df),
            ("edition_id", editions_df),
            ("book_id", editions_df),
            ("author_id", editions_df),
            ("author_id", authors_df),
            ("publication_year", editions_df),
            ("age_restriction", editions_df),
            ("language_id", editions_df),
            ("publisher_id", editions_df),
            ("book_id", book_genres_df),
            ("genre_id", book_genres_df),
            ("genre_id", genres_df),
        ]
        for column, frame in int_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(
                    "Int64"
                )

        interactions_df = interactions_df.astype(
            {"user_id": "int64", "edition_id": "int64", "event_type": "int32"}
        )
        targets_df = targets_df.astype({"user_id": "int64"})
        editions_df = editions_df.astype(
            {
                "edition_id": "int64",
                "book_id": "int64",
                "author_id": "int64",
                "publication_year": "int64",
                "age_restriction": "int64",
                "language_id": "int64",
                "publisher_id": "int64",
            }
        )
        authors_df = authors_df.astype({"author_id": "int64"})
        book_genres_df = book_genres_df.astype(
            {"book_id": "int64", "genre_id": "int64"}
        )
        genres_df = genres_df.astype({"genre_id": "int64"})

        seen_positive_df = (
            interactions_df[interactions_df["event_type"].isin([1, 2])][
                ["user_id", "edition_id"]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        return Dataset(
            interactions_df=interactions_df,
            targets_df=targets_df,
            catalog_df=editions_df,
            authors_df=authors_df,
            book_genres_df=book_genres_df,
            genres_df=genres_df,
            users_df=users_df,
            seen_positive_df=seen_positive_df,
        )
