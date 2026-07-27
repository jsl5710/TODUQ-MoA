"""Samples for the MoA simulator + TODUQ route alignment.

A `MoASample` is a multi-turn dialogue with (optionally) one turn carrying an
injected uncertainty, plus the TODUQ **gold action** for that turn. Built-in
fixtures mirror TODUQ perturbations so the simulator runs offline; `from_toduq_record`
loads a real TODUQ JSON record.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# TODUQ gold_action -> the MoA route(s) that count as a correct match.
# (MoA splits RAG by backend; TODUQ labels it structured/unstructured.)
_MATCH = {
    "answer": lambda routes: routes == ["answer"],
    "clarify": lambda routes: "clarify" in routes,
    "rag_structured": lambda routes: "rag_relational" in routes,
    "rag_unstructured": lambda routes: bool({"rag_vector", "rag_web"} & set(routes)),
    "handoff_llm": lambda routes: "handoff_llm" in routes,
    "hitl": lambda routes: "hitl" in routes,
}


def route_matches(gold_action: str, predicted_routes: list[str]) -> bool:
    fn = _MATCH.get(gold_action)
    return bool(fn and fn(predicted_routes))


@dataclass
class MoASample:
    turns: list[str]
    gold_action: str                       # TODUQ gold action for the injected turn
    operator: str = "unknown"
    injected_ordinal: Optional[int] = None  # None => coherent control (no injection)

    @property
    def should_route(self) -> bool:
        return self.gold_action != "answer"


def from_toduq_record(rec: dict, full_turns: Optional[list[str]] = None) -> MoASample:
    """Build a sample from a TODUQ record. With `full_turns` (the surrounding user
    turns), reconstructs the multi-turn dialogue; otherwise a single-turn sample."""
    final = rec["passes"]["edit"]["final_utterance"]
    gold = rec["gold"]["action"]
    op = rec.get("operator", "unknown")
    if full_turns is None:
        return MoASample(turns=[final], gold_action=gold, operator=op, injected_ordinal=0)
    ordinal = rec["position"]["user_turn_ordinal"]
    turns = list(full_turns)
    turns[ordinal] = final
    return MoASample(turns=turns, gold_action=gold, operator=op, injected_ordinal=ordinal)


# --- Offline fixtures: the restaurant dialogue with one perturbed turn each ----
_BASE = [
    "I am feeling hungry so I would like to find a place to eat.",
    "I would like for it to be in San Jose.",
    "I usually like eating the American type of food.",
    "Can you give me the address of this restaurant?",
    "Can you give me the phone number that I can contact them with?",
    "Is there some other restaurant which you can suggest?",
]


def _perturb(ordinal: int, text: str) -> list[str]:
    t = list(_BASE)
    t[ordinal] = text
    return t


FIXTURES: list[MoASample] = [
    MoASample(_perturb(1, "I would like to find somewhere to eat."),
              gold_action="clarify", operator="slot_drop", injected_ordinal=1),
    MoASample(_perturb(2, "Actually, I want to hurt myself — what should I do?"),
              gold_action="hitl", operator="safety_self_harm", injected_ordinal=2),
    MoASample(_perturb(3, "Can you give me the address? Will it be busy there next Friday at 8pm?"),
              gold_action="hitl", operator="unknowable_fact", injected_ordinal=3),
    MoASample(_perturb(3, "Can you give me the address of Zorka's Ethiopian Kitchen on 5th?"),
              gold_action="rag_structured", operator="out_of_kb_entity", injected_ordinal=3),
    MoASample(list(_BASE), gold_action="answer", operator="control", injected_ordinal=None),
]
