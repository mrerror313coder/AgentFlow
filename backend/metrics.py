"""In-memory execution metrics for the AgentFlow harness.

The tracker keeps a process-local history. It is intentionally simple: one
hackathon demo does not need a database, and the export endpoint can still
hand the history to a judge as JSON.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


class MetricsTracker:
    """Record latency, token estimates, and uptime for harness calls."""

    def __init__(self) -> None:
        """Start a new tracker at the current UTC time."""
        self.total_requests: int = 0
        self.history: list[dict[str, Any]] = []
        self.start_time: datetime = datetime.now(timezone.utc)

    def record(
        self,
        skill_name: str,
        time_ms: float,
        tokens: int,
        model: str | None,
    ) -> dict[str, Any]:
        """Append one completed call to the history.

        Args:
            skill_name: Skill name, or ``direct_chat`` for a model-only turn.
            time_ms: Elapsed wall time in milliseconds.
            tokens: Estimated token count for the call.
            model: Model identifier when a model was invoked.

        Returns:
            The history entry that was stored.
        """
        entry = {
            "skill_name": skill_name,
            "execution_time_ms": round(time_ms, 3),
            "estimated_tokens": tokens,
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.history.append(entry)
        self.total_requests += 1
        return entry

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens as roughly one token per four characters.

        Args:
            text: Text to measure.

        Returns:
            A positive integer estimate.
        """
        return max(1, len(text) // 4)

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent history entries in chronological order.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            A copy of the selected history slice.
        """
        if limit <= 0:
            return []
        return [dict(item) for item in self.history[-limit:]]

    def get_status(self, ollama_ok: bool, skills: list[str]) -> dict[str, Any]:
        """Build the status dictionary consumed by the health endpoint.

        Args:
            ollama_ok: Whether the Ollama health check succeeded.
            skills: Names of skills currently loaded.

        Returns:
            Status fields matching ``SystemStatus``, with ``ollama_models``
            left empty for the orchestrator to fill in.
        """
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return {
            "ollama_connected": ollama_ok,
            "ollama_models": [],
            "loaded_skills": list(skills),
            "total_requests": self.total_requests,
            "uptime_seconds": round(elapsed, 3),
        }

    def export_json(self) -> str:
        """Serialize the full history for download.

        Returns:
            Pretty-printed JSON containing the request count and history.
        """
        payload = {
            "total_requests": self.total_requests,
            "started_at": self.start_time.isoformat(),
            "history": self.history,
        }
        return json.dumps(payload, indent=2)
