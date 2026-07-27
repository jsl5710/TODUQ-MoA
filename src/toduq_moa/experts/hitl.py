"""Human-in-the-loop expert: a safety escalation channel, not a fallback.

Enqueues the query for human review with its safety category and returns a result
that marks `needs_human=True`. The aggregator MUST NOT override this — a query
that reaches HITL is answered by a human, or held.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from toduq_moa.experts.base import BaseExpert
from toduq_moa.schema import ExpertResult, Query


class HumanInTheLoopExpert(BaseExpert):
    id = "hitl"
    route = "hitl"

    def __init__(self, enqueue: Optional[Callable[[Query, dict[str, Any]], str]] = None):
        # enqueue(query, meta) -> ticket_id; defaults to an in-memory queue.
        self._queue: list[dict[str, Any]] = []
        self.enqueue = enqueue or self._default_enqueue

    def _default_enqueue(self, query: Query, meta: dict[str, Any]) -> str:
        ticket = {"id": f"hitl-{len(self._queue)}", "text": query.text, **meta}
        self._queue.append(ticket)
        return ticket["id"]

    def run(self, query: Query) -> ExpertResult:
        category = query.metadata.get("safety_category", "unknown")
        ticket_id = self.enqueue(query, {"safety_category": category})
        return ExpertResult(
            self.id, self.route,
            content=f"Escalated to human review (ticket {ticket_id}, category={category}).",
            evidence=[{"ticket_id": ticket_id, "safety_category": category}],
            confidence=1.0, needs_human=True,
        )
