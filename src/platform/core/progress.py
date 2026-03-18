"""Progress and ETA estimators for stage execution logs."""

from __future__ import annotations

from dataclasses import dataclass, field


def format_seconds(seconds: float) -> str:
    """Format raw seconds into compact `mm:ss` or `hh:mm:ss` string."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class StageProgressTracker:
    """Track completed stage runtimes and estimate remaining time."""

    total_stages: int
    historical_durations: dict[str, float] = field(default_factory=dict)
    completed_durations: list[float] = field(default_factory=list)

    def estimate_remaining_seconds(
        self, current_index: int, remaining_stage_names: list[str]
    ) -> float:
        """Estimate remaining runtime based on observed or historical timings.

        Args:
            current_index: One-based index of current stage (kept for API parity).
            remaining_stage_names: Stage names after the currently running stage.

        Returns:
            Estimated seconds left for completion of all remaining stages.
        """
        del current_index
        if not remaining_stage_names:
            return 0.0

        if self.completed_durations:
            avg = sum(self.completed_durations) / len(self.completed_durations)
            return avg * len(remaining_stage_names)

        known = [
            self.historical_durations[name]
            for name in remaining_stage_names
            if name in self.historical_durations
        ]
        if known:
            return float(sum(known))

        return 0.0

    def register_completed_stage(self, duration_sec: float) -> None:
        """Record one finished stage duration for future ETA estimation."""
        self.completed_durations.append(max(0.0, float(duration_sec)))
