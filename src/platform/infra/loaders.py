"""Low-level tabular file readers used by dataset loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    """Read CSV input for pipeline processing.

    Args:
        path: Absolute or relative path to CSV file.

    Returns:
        Parsed pandas DataFrame.
    """
    return pd.read_csv(path)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read parquet artifact for pipeline processing.

    Args:
        path: Absolute or relative path to parquet file.

    Returns:
        Parsed pandas DataFrame.
    """
    return pd.read_parquet(path)

