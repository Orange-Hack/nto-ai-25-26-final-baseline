"""Prepare-data stage for runtime cache materialization."""

from __future__ import annotations

from typing import Any

from src.platform.core.artifacts import atomic_write_dataframe
from src.platform.pipeline.models import PipelineContext
from src.platform.pipeline.runtime import load_base_dataset, pack_data_cache


class PrepareDataStage:
    """Normalize source inputs and persist packed runtime cache artifact."""

    name = "prepare_data"

    def __init__(self, context: PipelineContext) -> None:
        self.context = context

    def run(self) -> dict[str, Any]:
        """Build `data_cache.parquet` from source interactions and positives."""
        dataset = load_base_dataset(self.context.paths)
        data_cache = pack_data_cache(dataset)
        atomic_write_dataframe(data_cache, self.context.paths.data_cache_path)
        return {
            "rows": int(len(data_cache)),
            "users": int(dataset.interactions_df["user_id"].nunique()),
            "positives": int(len(dataset.seen_positive_df)),
        }

