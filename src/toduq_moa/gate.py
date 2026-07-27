"""UQ gate: decide whether an input is uncertain, and classify it.

Pluggable `UQMethod` protocol so any uncertainty quantifier fits — a TODUQ-style
detector, semantic entropy over N samples, a calibrated classifier, or an
external UQ service. Ships a keyword/heuristic gate for offline runs.

The gate ALSO runs a safety screen: safety-critical inputs are flagged for HITL
escalation independently of the uncertainty score, so a confident-but-harmful
input still routes to a human.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from toduq_moa.schema import Query, SafetyCategory, UQFlag

# Minimal, transparent safety lexicon for the offline gate. A production system
# swaps this for a trained safety classifier; the escalation semantics stay.
_SAFETY_MARKERS: dict[SafetyCategory, tuple[str, ...]] = {
    "self_harm": ("suicide", "kill myself", "self harm", "self-harm", "hurt myself"),
    "adversarial": ("ignore previous", "jailbreak", "system prompt", "disregard your rules"),
}


@runtime_checkable
class UQMethod(Protocol):
    def score(self, query: Query) -> UQFlag: ...


def screen_safety(query: Query) -> SafetyCategory:
    text = query.text.lower()
    for category, markers in _SAFETY_MARKERS.items():
        if any(m in text for m in markers):
            return category
    return "none"


class HeuristicGate:
    """Offline UQ gate. Flags underspecified / hedged / unknowable-looking turns
    and always applies the safety screen. Not a real UQ method — a stand-in that
    keeps the pipeline runnable and the interface honest."""

    threshold = 0.5

    def score(self, query: Query) -> UQFlag:
        safety = screen_safety(query)
        if safety != "none":
            return UQFlag(is_uncertain=True, score=1.0, uncertainty_type="reasoning",
                          severity="major", safety_category=safety,
                          rationale=f"safety screen matched: {safety}")

        text = query.text.lower()
        words = set(text.replace("?", " ").replace(".", " ").replace(",", " ").split())
        hedge_words = {"maybe", "somewhere", "something", "or", "anywhere", "someone"}
        hedge_phrases = ("not sure", "that one", "over there", "or something")
        unknowable = ("will it", "next week", "next friday", "tomorrow", "in the future", "guarantee")
        score = 0.0
        utype = None
        if words & hedge_words or any(p in text for p in hedge_phrases):
            score, utype = 0.6, "input"
        if any(u in text for u in unknowable):
            score, utype = max(score, 0.8), "parameter"
        return UQFlag(
            is_uncertain=score >= self.threshold, score=score,
            uncertainty_type=utype, severity="minor" if score >= self.threshold else "none",
            rationale="heuristic gate",
        )
