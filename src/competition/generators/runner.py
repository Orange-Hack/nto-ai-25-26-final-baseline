"""Generator runner and contract checks for participant solution."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.competition.generators.registry import build_generator
from src.platform.core.dataset import Dataset
from src.platform.infra.hashing import compute_inputs_fingerprint

GENERATOR_CACHE_SCHEMA_VERSION = 1


def _sanitize_source_name(source_name: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in source_name
    )
    return safe or "generator"


def validate_candidate_contract(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Normalize and validate generator output schema.

    Args:
        frame: Raw DataFrame emitted by a generator.
        source_name: Expected source value for all rows.

    Returns:
        Normalized DataFrame with stable column order and dtypes.

    Raises:
        ValueError: If required columns are missing or source labels mismatch.
    """
    required = {"user_id", "edition_id", "score", "source"}
    if not required.issubset(frame.columns):
        missing = required - set(frame.columns)
        raise ValueError(
            f"generator '{source_name}' returned invalid schema, missing={sorted(missing)}"
        )
    frame = frame[["user_id", "edition_id", "score", "source"]].copy()
    frame["user_id"] = frame["user_id"].astype("int64")
    frame["edition_id"] = frame["edition_id"].astype("int64")
    frame["score"] = frame["score"].astype(float)
    frame["source"] = frame["source"].astype(str)
    if (frame["source"] != source_name).any():
        raise ValueError(f"generator '{source_name}' returned rows with invalid source")
    return frame


def run_generators(
    dataset: Dataset,
    features: pd.DataFrame,
    user_ids: pd.Series,
    generators_cfg: list[dict[str, Any]],
    per_generator_k: int,
    seed: int,
    tqdm_enabled: bool,
) -> pd.DataFrame:
    """Execute configured generators and aggregate all candidate rows.

    Args:
        dataset: Runtime dataset passed to each generator.
        features: Feature matrix computed for the current run.
        user_ids: Distinct target users for candidate generation.
        generators_cfg: Ordered generator config list from YAML.
        per_generator_k: Max rows per user for each generator.
        seed: Global deterministic seed for generators.
        tqdm_enabled: Whether runner should display progress bars.

    Returns:
        Concatenated candidate DataFrame across all configured generators.
    """
    frames, _ = run_generators_with_cache(
        dataset=dataset,
        features=features,
        user_ids=user_ids,
        generators_cfg=generators_cfg,
        per_generator_k=per_generator_k,
        seed=seed,
        tqdm_enabled=tqdm_enabled,
        cache_dir=None,
        features_input_path=None,
        targets_input_path=None,
    )
    return frames


def run_generators_with_cache(
    dataset: Dataset,
    features: pd.DataFrame,
    user_ids: pd.Series,
    generators_cfg: list[dict[str, Any]],
    per_generator_k: int,
    seed: int,
    tqdm_enabled: bool,
    cache_dir: Path | None,
    features_input_path: Path | None,
    targets_input_path: Path | None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Execute generators with optional per-generator cache reuse.

    Args:
        dataset: Runtime dataset passed to each generator.
        features: Feature matrix computed for the current run.
        user_ids: Distinct target users for candidate generation.
        generators_cfg: Ordered generator config list from YAML.
        per_generator_k: Max rows per user for each generator.
        seed: Global deterministic seed for generators.
        tqdm_enabled: Whether runner should display progress bars.
        cache_dir: Directory for per-generator cache files. If None, disables cache.
        features_input_path: Optional path to features artifact used for fingerprint.
        targets_input_path: Optional path to targets file used for fingerprint.

    Returns:
        Tuple of concatenated candidates and cache metadata rows per generator.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    cache_entries: list[dict[str, Any]] = []
    for generator_cfg in tqdm(
        generators_cfg,
        total=len(generators_cfg),
        desc="generate_candidates",
        leave=False,
        dynamic_ncols=True,
        disable=not tqdm_enabled,
        file=sys.stdout,
    ):
        generator = build_generator(
            generator_cfg["name"],
            generator_cfg.get("params", {}),
            tqdm_enabled=tqdm_enabled,
        )
        generator_params = generator_cfg.get("params", {})
        fingerprint = ""
        cache_path: Path | None = None
        cache_hit = False
        if (
            cache_dir is not None
            and features_input_path is not None
            and targets_input_path is not None
        ):
            fingerprint = compute_inputs_fingerprint(
                inputs=[features_input_path, targets_input_path],
                config_snapshot={
                    "schema_version": GENERATOR_CACHE_SCHEMA_VERSION,
                    "pipeline": {
                        "seed": int(seed),
                        "per_generator_k": int(per_generator_k),
                    },
                    "generator": {
                        "config_name": generator_cfg["name"],
                        "source_name": generator.name,
                        "params": generator_params,
                    },
                },
            )
            cache_name = f"{_sanitize_source_name(generator.name)}__{fingerprint}.csv"
            cache_path = cache_dir / cache_name
            if cache_path.exists():
                generated = pd.read_csv(cache_path)
                generated = validate_candidate_contract(generated, generator.name)
                cache_hit = True
            else:
                generated = generator.generate(
                    dataset=dataset,
                    user_ids=user_ids.astype("int64").to_numpy(),
                    features=features,
                    k=int(per_generator_k),
                    seed=int(seed),
                )
                generated = validate_candidate_contract(generated, generator.name)
                generated.to_csv(cache_path, index=False)
        else:
            generated = generator.generate(
                dataset=dataset,
                user_ids=user_ids.astype("int64").to_numpy(),
                features=features,
                k=int(per_generator_k),
                seed=int(seed),
            )
            generated = validate_candidate_contract(generated, generator.name)

        cache_entries.append(
            {
                "config_name": str(generator_cfg["name"]),
                "source_name": str(generator.name),
                "fingerprint": fingerprint,
                "cache_path": str(cache_path) if cache_path is not None else None,
                "cache_hit": bool(cache_hit),
                "rows": int(len(generated)),
                "params": dict(generator_params),
            }
        )
        frames.append(generated)

    aggregated = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])
    )
    return aggregated, cache_entries
