"""Core datatypes for the MoA pipeline: gate → router → experts → aggregator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

UncertaintyType = Literal["input", "reasoning", "parameter", "prediction"]
Severity = Literal["none", "minor", "major"]

# Route/action vocabulary — a superset of TODUQ's gold actions, with RAG split by
# backend so the router can pick the right expert.
Route = Literal[
    "answer",            # no real uncertainty; respond directly
    "clarify",           # ask a follow-up question
    "rag_relational",    # structured query against a relational DB
    "rag_vector",        # unstructured retrieval from a vector/RAG DB
    "rag_web",           # internet search
    "hitl",              # human-in-the-loop (safety / major)
    "handoff_llm",       # escalate to a stronger reasoner
]

# Safety categories that force an HITL escalation regardless of the UQ score.
SafetyCategory = Literal["self_harm", "adversarial", "high_stakes", "none"]


@dataclass
class Query:
    """The unit flowing through the pipeline: a user turn plus dialogue context."""
    text: str
    dialogue_context: list[str] = field(default_factory=list)
    belief_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UQFlag:
    """Output of the UQ gate."""
    is_uncertain: bool
    score: float                                  # 0..1 uncertainty magnitude
    uncertainty_type: Optional[UncertaintyType] = None
    severity: Severity = "none"
    safety_category: SafetyCategory = "none"
    rationale: str = ""


@dataclass
class RouteDecision:
    """Output of the router — the experts to consult (ordered, may be several)."""
    routes: list[Route]
    reason: str = ""


@dataclass
class ExpertResult:
    """What an expert returns to the aggregator."""
    expert_id: str
    route: Route
    content: str                                  # the expert's answer/finding
    evidence: list[dict[str, Any]] = field(default_factory=list)  # citations/rows/docs
    confidence: float = 0.0
    needs_human: bool = False                     # expert escalates to HITL
    error: Optional[str] = None


@dataclass
class FinalResponse:
    """Aggregator output."""
    answer: str
    action_taken: Route
    abstained: bool
    used_experts: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    escalated_to_human: bool = False
