"""Schema validation utilities for strict dataframe contracts."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def ensure_columns(df: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    """Assert that required columns exist before downstream processing.

    Args:
        df: DataFrame whose schema should be checked.
        columns: Required column names expected by the caller.
        table_name: Human-readable input name used in error messages.

    Raises:
        ValueError: If any required columns are missing from `df`.
    """
    required = list(columns)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {', '.join(sorted(missing))}"
        )
