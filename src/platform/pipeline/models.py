"""Data models and constants for stage orchestration runtime."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.platform.core.artifacts import ArtifactsManager

STAGES = [
    "prepare_data",
    "build_features",
    "generate_candidates",
    "rank_and_select",
    "make_submission",
]

DEPENDENCIES: dict[str, list[str]] = {
    "prepare_data": [],
    "build_features": ["prepare_data"],
    "generate_candidates": ["build_features"],
    "rank_and_select": ["generate_candidates"],
    "make_submission": ["rank_and_select"],
}

STAGE_SCHEMA_VERSIONS: dict[str, int] = {
    "prepare_data": 2,
    "build_features": 1,
    "generate_candidates": 2,
    "rank_and_select": 2,
    "make_submission": 2,
}


@dataclass(frozen=True)
class PipelinePaths:
    """Collect resolved file-system paths used by the pipeline run."""

    data_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    data_cache_path: Path
    features_path: Path
    candidates_path: Path
    generators_cache_dir: Path
    generators_cache_manifest_path: Path
    predictions_path: Path
    submission_path: Path


@dataclass(frozen=True)
class PipelineContext:
    """Share immutable execution context across stages and workflows."""

    config: dict[str, Any]
    paths: PipelinePaths
    logger: logging.Logger
    artifacts: ArtifactsManager
