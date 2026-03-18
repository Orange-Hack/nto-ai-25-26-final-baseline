"""Protocol contracts for platform stage implementations."""

from __future__ import annotations

from typing import Any, Protocol


class PipelineStage(Protocol):
    """Define the minimal stage interface required by orchestrator registry."""

    name: str

    def run(self) -> dict[str, Any]:
        """Execute stage side effects and return stage statistics."""
