"""Contracts for participant candidate generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from src.platform.core.dataset import Dataset


class CandidateGenerator(Protocol):
    """Define the interface required by platform candidate runner.

    The runner invokes this method for each configured generator and expects
    a normalized candidate frame. Implementations can use any internal logic
    as long as they honor the returned schema contract.

    Attributes:
        name: Stable source identifier written to the `source` column.
    """

    name: str

    def generate(
        self,
        dataset: Dataset,
        user_ids: np.ndarray,
        features: pd.DataFrame,
        k: int,
        seed: int,
    ) -> pd.DataFrame:
        """Produce candidate rows for target users.

        Args:
            dataset: Runtime dataset with interactions, catalog, and targets.
            user_ids: Numpy array of users that require recommendations.
            features: Precomputed feature table produced by feature stage.
            k: Maximum number of rows per user for this generator.
            seed: Global deterministic seed configured for the pipeline.

        Returns:
            DataFrame with columns `user_id`, `edition_id`, `score`, `source`.
        """


@dataclass(frozen=True)
class GeneratorConfig:
    """Capture normalized generator settings from YAML configs."""

    name: str
    params: dict[str, float]
