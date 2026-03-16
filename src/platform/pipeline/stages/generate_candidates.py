"""Candidate-generation stage using competition generator registry."""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd

from src.competition.generators.runner import run_generators_with_cache
from src.platform.core.artifacts import atomic_write_dataframe, atomic_write_json
from src.platform.infra.time import utc_now_iso
from src.platform.pipeline.models import PipelineContext
from src.platform.pipeline.runtime import load_runtime_dataset


class GenerateCandidatesStage:
    """Run configured generators and persist candidate artifact."""

    name = "generate_candidates"

    def __init__(self, context: PipelineContext) -> None:
        self.context = context

    def run(self) -> dict[str, Any]:
        """Generate candidate rows for all target users.

        Returns:
            Dictionary with row/user/source counts for metadata reporting.
        """
        dataset = load_runtime_dataset(self.context.paths)
        features = pd.read_parquet(self.context.paths.features_path)
        user_ids = dataset.targets_df["user_id"].drop_duplicates().astype("int64")
        logs_cfg = self.context.config.get("logs", {})
        tqdm_enabled = bool(logs_cfg.get("tqdm_enabled", True)) and sys.stdout.isatty()
        candidates, cache_entries = run_generators_with_cache(
            dataset=dataset,
            features=features,
            user_ids=user_ids,
            generators_cfg=list(self.context.config["candidates"]["generators"]),
            per_generator_k=int(self.context.config["candidates"]["per_generator_k"]),
            seed=int(self.context.config["pipeline"]["seed"]),
            tqdm_enabled=tqdm_enabled,
            cache_dir=self.context.paths.generators_cache_dir,
            features_input_path=self.context.paths.features_path,
            targets_input_path=self.context.paths.data_dir / "targets.csv",
        )
        atomic_write_dataframe(candidates, self.context.paths.candidates_path)
        cache_hits = sum(1 for entry in cache_entries if entry["cache_hit"])
        cache_misses = len(cache_entries) - cache_hits
        manifest_payload: dict[str, Any] = {
            "generated_at": utc_now_iso(),
            "total_generators": len(cache_entries),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "entries": cache_entries,
        }
        atomic_write_json(self.context.paths.generators_cache_manifest_path, manifest_payload)
        return {
            "rows": int(len(candidates)),
            "users": int(candidates["user_id"].nunique() if not candidates.empty else 0),
            "sources": int(candidates["source"].nunique() if not candidates.empty else 0),
            "cache_hits": int(cache_hits),
            "cache_misses": int(cache_misses),
        }

