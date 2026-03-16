"""Named stage classes used by pipeline orchestrator registry."""

from src.platform.pipeline.stages.build_features import BuildFeaturesStage
from src.platform.pipeline.stages.generate_candidates import GenerateCandidatesStage
from src.platform.pipeline.stages.make_submission import MakeSubmissionStage
from src.platform.pipeline.stages.prepare_data import PrepareDataStage
from src.platform.pipeline.stages.rank_and_select import RankAndSelectStage

__all__ = [
    "PrepareDataStage",
    "BuildFeaturesStage",
    "GenerateCandidatesStage",
    "RankAndSelectStage",
    "MakeSubmissionStage",
]

