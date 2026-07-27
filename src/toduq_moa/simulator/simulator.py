"""TODUQ-MoA Simulator.

Replays a MoASample turn-by-turn through the MoA pipeline and reports, for the
injected turn, whether MoA:
  - LOCALIZED it (flagged uncertainty at the injected turn and answered elsewhere), and
  - ROUTED it correctly (the route matches TODUQ's gold action).

Localization and routing are separate: MoA can flag the right turn but pick the
wrong expert (e.g. route an unknowable question to RAG when the gold is HITL).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from toduq_moa import MoA, Query
from toduq_moa.simulator.sample import MoASample, route_matches


@dataclass
class TurnRoute:
    ordinal: int
    utterance: str
    routes: list[str]
    action: str
    escalated_to_human: bool
    is_injected: bool


@dataclass
class MoASimResult:
    operator: str
    gold_action: str
    injected_ordinal: Optional[int]
    should_route: bool
    turn_routes: list[TurnRoute] = field(default_factory=list)
    predicted_routes_at_injected: list[str] = field(default_factory=list)
    localized: bool = False          # flagged the right turn (and only it)
    routing_correct: bool = False    # route matches TODUQ gold action
    safety_fired: bool = False


def replay(sample: MoASample, moa: Optional[MoA] = None) -> MoASimResult:
    moa = moa or MoA()
    history: list[str] = []
    routes: list[TurnRoute] = []
    for i, utt in enumerate(sample.turns):
        trace = moa.handle(Query(text=utt, dialogue_context=list(history)))
        routes.append(TurnRoute(
            ordinal=i, utterance=utt, routes=list(trace.routes),
            action=trace.response.action_taken,
            escalated_to_human=trace.response.escalated_to_human,
            is_injected=(i == sample.injected_ordinal),
        ))
        history.append(f"User: {utt}")

    flagged = [r.ordinal for r in routes if r.routes != ["answer"]]
    inj = sample.injected_ordinal
    if sample.should_route and inj is not None:
        localized = flagged == [inj]
        inj_routes = routes[inj].routes
        routing_correct = route_matches(sample.gold_action, inj_routes)
    else:  # control: correct iff nothing was flagged
        localized = flagged == []
        inj_routes = []
        routing_correct = flagged == []

    return MoASimResult(
        operator=sample.operator, gold_action=sample.gold_action, injected_ordinal=inj,
        should_route=sample.should_route, turn_routes=routes,
        predicted_routes_at_injected=inj_routes, localized=localized,
        routing_correct=routing_correct,
        safety_fired=any(r.escalated_to_human for r in routes),
    )
