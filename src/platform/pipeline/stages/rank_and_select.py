"""Rank-and-select stage delegating ranking logic to competition zone."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.competition.ranking import rank_predictions
from src.platform.core.artifacts import atomic_write_dataframe
from src.platform.pipeline.models import PipelineContext
from src.platform.pipeline.runtime import load_runtime_dataset


class RankAndSelectStage:
    """Rank candidates and persist top-k predictions artifact."""

    name = "rank_and_select"

    def __init__(self, context: PipelineContext) -> None:
        self.context = context

    def run(self) -> dict[str, Any]:
        """Load candidates, rank them, and persist prediction artifact."""
        dataset = load_runtime_dataset(self.context.paths)
        candidates = pd.read_csv(self.context.paths.candidates_path)
        predictions = rank_predictions(
            dataset=dataset,
            candidates=candidates,
            source_weights=self.context.config.get("ranking", {}).get(
                "source_weights", {}
            ),
            k=int(self.context.config["pipeline"]["k"]),
        )
        atomic_write_dataframe(predictions, self.context.paths.predictions_path)
        return {
            "rows": int(len(predictions)),
            "users": int(
                predictions["user_id"].nunique() if not predictions.empty else 0
            ),
        }
