"""UQ gate: decide whether an input is uncertain, and classify it.

The gate composes two things:
  1. a **UQ method** from the shared `toduq_moa.uq` registry (the SAME
     implementations TODUQ uses) — loadable by name, so any method of choice can
     drive the router; and
  2. a **safety screen** that flags safety-critical inputs for HITL escalation
     independently of the UQ score, so a confident-but-harmful input still routes
     to a human.

    Gate()                                   # default: shared "lexical", offline
    Gate("semantic_entropy", client=llm)     # response-based, needs a model
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from toduq_moa.schema import Query, SafetyCategory, UQFlag
from toduq_moa.uq import Client, UQMethod as _UQMethod, load_uq

# Minimal, transparent safety lexicon. A production system swaps this for a
# trained safety classifier; the escalation semantics stay.
_SAFETY_MARKERS: dict[SafetyCategory, tuple[str, ...]] = {
    "self_harm": ("suicide", "kill myself", "self harm", "self-harm", "hurt myself"),
    "adversarial": ("ignore previous", "jailbreak", "system prompt", "disregard your rules"),
}


@runtime_checkable
class UQGate(Protocol):
    """The gate surface the orchestrator uses: Query -> UQFlag."""
    def score(self, query: Query) -> UQFlag: ...


def screen_safety(query: Query) -> SafetyCategory:
    text = query.text.lower()
    for category, markers in _SAFETY_MARKERS.items():
        if any(m in text for m in markers):
            return category
    return "none"


class Gate:
    """Composes a shared UQ method + the safety screen into a UQFlag."""

    def __init__(self, uq: str | _UQMethod = "lexical", *, client: Optional[Client] = None,
                 threshold: float = 0.5, **uq_kwargs):
        self.uq = load_uq(uq, **uq_kwargs) if isinstance(uq, str) else uq
        self.client = client
        self.threshold = threshold

    def score(self, query: Query) -> UQFlag:
        safety = screen_safety(query)
        if safety != "none":
            return UQFlag(is_uncertain=True, score=1.0, uncertainty_type="reasoning",
                          severity="major", safety_category=safety,
                          rationale=f"safety screen matched: {safety}")
        r = self.uq.score(query.text, context=query.dialogue_context, client=self.client)
        uncertain = r.score >= self.threshold
        return UQFlag(is_uncertain=uncertain, score=r.score,
                      uncertainty_type=r.uncertainty_type,
                      severity="minor" if uncertain else "none",
                      safety_category="none", rationale=f"uq={r.method}")


class HeuristicGate(Gate):
    """Offline default: the shared `lexical` UQ method + safety screen."""
    def __init__(self):
        super().__init__("lexical")


# Back-compat alias: the orchestrator's typing referenced UQMethod (Query->UQFlag).
UQMethod = UQGate
