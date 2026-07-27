"""MoA simulator: replay TODUQ samples and test localization vs routing (offline)."""
from toduq_moa import MoA
from toduq_moa.simulator import FIXTURES, MoASample, from_toduq_record, replay, route_matches


def _by_op(op):
    return next(s for s in FIXTURES if s.operator == op)


def test_route_matches_maps_toduq_actions():
    assert route_matches("clarify", ["clarify"])
    assert route_matches("rag_structured", ["rag_relational", "rag_vector", "rag_web"])
    assert route_matches("rag_unstructured", ["rag_vector"])
    assert route_matches("hitl", ["hitl"])
    assert not route_matches("hitl", ["rag_relational"])
    assert route_matches("answer", ["answer"])


def test_slot_drop_localized_and_routed():
    res = replay(_by_op("slot_drop"), MoA())
    assert res.localized is True
    assert res.routing_correct is True
    assert res.predicted_routes_at_injected == ["clarify"]


def test_safety_escalates_to_human():
    res = replay(_by_op("safety_self_harm"), MoA())
    assert res.safety_fired is True
    assert res.routing_correct is True
    assert res.predicted_routes_at_injected == ["hitl"]


def test_unknowable_localized_but_mis_routed():
    # right turn flagged, wrong expert (heuristic gate routes to RAG, gold is HITL)
    res = replay(_by_op("unknowable_fact"), MoA())
    assert res.localized is True
    assert res.routing_correct is False


def test_control_flags_nothing():
    res = replay(_by_op("control"), MoA())
    assert res.should_route is False
    assert res.localized is True
    assert all(r.routes == ["answer"] for r in res.turn_routes)


def test_from_toduq_record_single_and_multi_turn():
    rec = {"passes": {"edit": {"final_utterance": "somewhere to eat"}},
           "gold": {"action": "clarify"}, "operator": "slot_drop",
           "position": {"user_turn_ordinal": 1}}
    single = from_toduq_record(rec)
    assert single.turns == ["somewhere to eat"] and single.injected_ordinal == 0
    multi = from_toduq_record(rec, full_turns=["a", "b", "c"])
    assert multi.turns[1] == "somewhere to eat" and multi.injected_ordinal == 1
