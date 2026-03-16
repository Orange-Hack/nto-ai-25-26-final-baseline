"""Local pseudo-incident workflow for offline model quality checks."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.competition.features import build_features_frame
from src.competition.generators import run_generators
from src.competition.ranking import rank_predictions
from src.platform.core.dataset import Dataset
from src.platform.core.metrics import ndcg_at_k, summarize_ndcg
from src.platform.pipeline.models import PipelineContext


class PseudoIncidentValidationWorkflow:
    """Run local validation by masking positives in recent clean-period window."""

    def __init__(self, context: PipelineContext) -> None:
        self.context = context

    def run(self) -> dict[str, Any]:
        """Compute local proxy NDCG by simulating an incident recovery task.

        Returns:
            Dictionary with mean NDCG, quartiles, and evaluated user count.

        Raises:
            ValueError: If insufficient positive data exists for pseudo-incident.
        """
        dataset = Dataset.load(self.context.paths.data_dir)
        k = int(self.context.config["pipeline"]["k"])
        metric_label = f"ndcg@{k}"
        interactions = dataset.interactions_df.copy()
        positives = interactions[interactions["event_type"].isin([1, 2])]
        if positives.empty:
            raise ValueError("No positive interactions available for local validation")

        min_ts = positives["event_ts"].min()
        clean_end = min_ts + pd.Timedelta(days=150)
        clean_df = positives[positives["event_ts"] < clean_end].copy()
        pseudo_days = int(
            self.context.config.get("validation", {}).get("pseudo_incident_days", 14)
        )
        incident_end = clean_df["event_ts"].max()
        incident_start = incident_end - pd.Timedelta(days=pseudo_days)
        pseudo_window_df = clean_df[clean_df["event_ts"] >= incident_start].copy()
        pseudo_pairs = pseudo_window_df[["user_id", "edition_id"]].drop_duplicates()

        rng = np.random.default_rng(int(self.context.config["pipeline"]["seed"]))
        if pseudo_pairs.empty:
            raise ValueError("Pseudo-incident window does not contain positive pairs")
        mask_count = max(1, int(len(pseudo_pairs) * 0.2))
        mask_indexes = rng.choice(len(pseudo_pairs), size=mask_count, replace=False)
        masked_pairs = pseudo_pairs.iloc[mask_indexes].copy()
        masked_pairs["_masked"] = 1

        observed = clean_df.merge(masked_pairs, on=["user_id", "edition_id"], how="left")
        observed = observed[observed["_masked"].isna()].drop(columns=["_masked"])

        targets = pseudo_window_df[["user_id"]].drop_duplicates().astype({"user_id": "int64"})
        validation_dataset = Dataset(
            interactions_df=observed,
            targets_df=targets,
            catalog_df=dataset.catalog_df,
            authors_df=dataset.authors_df,
            book_genres_df=dataset.book_genres_df,
            genres_df=dataset.genres_df,
            users_df=dataset.users_df,
            seen_positive_df=observed[["user_id", "edition_id"]].drop_duplicates(),
        )

        features = build_features_frame(
            dataset=validation_dataset,
            recent_days=int(self.context.config["pipeline"]["recent_days"]),
        )
        user_ids = validation_dataset.targets_df["user_id"].drop_duplicates().astype("int64")
        logs_cfg = self.context.config.get("logs", {})
        tqdm_enabled = bool(logs_cfg.get("tqdm_enabled", True)) and sys.stdout.isatty()
        candidates = run_generators(
            dataset=validation_dataset,
            features=features,
            user_ids=user_ids,
            generators_cfg=list(self.context.config["candidates"]["generators"]),
            per_generator_k=int(self.context.config["candidates"]["per_generator_k"]),
            seed=int(self.context.config["pipeline"]["seed"]),
            tqdm_enabled=tqdm_enabled,
        )
        predictions = rank_predictions(
            dataset=validation_dataset,
            candidates=candidates,
            source_weights=self.context.config.get("ranking", {}).get("source_weights", {}),
            k=k,
        )

        relevant_by_user: dict[int, set[int]] = {}
        for row in masked_pairs.to_dict(orient="records"):
            relevant_by_user.setdefault(int(row["user_id"]), set()).add(int(row["edition_id"]))

        per_user_rows: list[dict[str, float | int]] = []
        target_user_ids = targets["user_id"].tolist()
        for user_id in tqdm(
            target_user_ids,
            total=len(target_user_ids),
            desc="validation_users",
            leave=False,
            dynamic_ncols=True,
            disable=not tqdm_enabled,
            file=sys.stdout,
        ):
            user_pred = predictions[predictions["user_id"] == int(user_id)].sort_values("rank")
            predicted = user_pred["edition_id"].astype("int64").tolist()
            relevant = relevant_by_user.get(int(user_id), set())
            ndcg = ndcg_at_k(
                predicted=predicted,
                relevant=relevant,
                k=k,
            )
            per_user_rows.append({"user_id": int(user_id), metric_label: ndcg})

        per_user_df = pd.DataFrame(per_user_rows)
        summary = summarize_ndcg(per_user_df, score_column=metric_label)
        result = {
            f"mean_{metric_label}": summary.mean_ndcg,
            "quantiles": summary.quantiles,
            "users": int(len(per_user_df)),
        }
        self.context.logger.info("Local validation result: %s", result)
        return result

