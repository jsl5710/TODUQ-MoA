"""Router: map a UQ flag to the expert(s) to consult.

The mapping mirrors TODUQ's taxonomy → gold-action logic, expanded to pick a
concrete RAG backend. Safety escalations short-circuit straight to HITL.
"""
from __future__ import annotations

from toduq_moa.schema import Route, RouteDecision, UQFlag

# Uncertainty type -> default route(s). Parameter (knowledge gap) fans out across
# the RAG backends; the aggregator reconciles whichever expert answered.
_TYPE_ROUTES: dict[str, list[Route]] = {
    "input": ["clarify"],
    "reasoning": ["handoff_llm"],
    "parameter": ["rag_relational", "rag_vector", "rag_web"],
    "prediction": ["handoff_llm"],
}


def route(flag: UQFlag) -> RouteDecision:
    if not flag.is_uncertain:
        return RouteDecision(routes=["answer"], reason="gate: not uncertain")

    # Safety / major severity -> human, unconditionally.
    if flag.safety_category != "none" or flag.severity == "major":
        return RouteDecision(routes=["hitl"],
                             reason=f"escalation: safety={flag.safety_category}, sev={flag.severity}")

    routes = _TYPE_ROUTES.get(flag.uncertainty_type or "", ["clarify"])
    return RouteDecision(routes=list(routes),
                         reason=f"type={flag.uncertainty_type} -> {routes}")
