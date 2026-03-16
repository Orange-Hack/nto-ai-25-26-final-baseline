"""Generator APIs for participant-defined candidate sources."""

from src.competition.generators.registry import GENERATOR_REGISTRY, build_generator
from src.competition.generators.runner import (
    run_generators,
    validate_candidate_contract,
)

__all__ = [
    "GENERATOR_REGISTRY",
    "build_generator",
    "run_generators",
    "validate_candidate_contract",
]

