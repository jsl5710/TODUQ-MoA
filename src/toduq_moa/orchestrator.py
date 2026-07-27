"""Orchestrator: wire gate → router → experts → aggregator into one call.

    moa = MoA()                      # offline defaults (heuristic gate, stub experts)
    response = moa.handle(Query("I want to eat somewhere"))

Experts for the chosen routes run (independently — parallelizable), and the
aggregator reasons over their combined evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from toduq_moa.aggregator import aggregate
from toduq_moa.experts import Expert, default_experts
from toduq_moa.gate import HeuristicGate
from toduq_moa.router import route as route_flag
from toduq_moa.schema import ExpertResult, FinalResponse, Query, UQFlag


@dataclass
class Trace:
    """Full decision trace for auditing / evaluation."""
    flag: UQFlag
    routes: list[str]
    results: list[ExpertResult] = field(default_factory=list)
    response: Optional[FinalResponse] = None


class MoA:
    def __init__(self, *, gate: Optional[Any] = None, experts: Optional[list[Expert]] = None,
                 aggregator_llm: Optional[Any] = None, gate_llm: Optional[Any] = None):
        self.gate = gate or HeuristicGate()
        self.experts = experts or default_experts(llm=gate_llm)
        self.aggregator_llm = aggregator_llm

    def _experts_for(self, route: str) -> list[Expert]:
        return [e for e in self.experts if e.can_handle(route)]

    def handle(self, query: Query) -> Trace:
        flag = self.gate.score(query)
        decision = route_flag(flag)
        trace = Trace(flag=flag, routes=list(decision.routes))

        if decision.routes == ["answer"]:
            trace.response = FinalResponse(answer="", action_taken="answer", abstained=False)
            return trace

        # Pass the safety category through so the HITL expert can label the ticket.
        query.metadata.setdefault("safety_category", flag.safety_category)

        results: list[ExpertResult] = []
        for r in decision.routes:
            for expert in self._experts_for(r):
                results.append(expert.run(query))   # independent → safe to parallelize

        trace.results = results
        trace.response = aggregate(query, results, llm=self.aggregator_llm)
        return trace
