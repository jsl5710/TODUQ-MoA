"""Minimal CLI. `toduq-moa demo` runs the offline pipeline on sample inputs."""
from __future__ import annotations

import argparse

from toduq_moa import MoA, Query

_SAMPLES = [
    "I would like to find a place to eat somewhere.",          # input -> clarify
    "Is the Ethiopian place on 5th open next Friday at 8pm?",  # parameter -> RAG
    "I want to hurt myself, what should I do?",                # safety -> hitl
    "Book a table for two at 7.",                              # confident -> answer
]


def _simulate() -> int:
    from toduq_moa import MoA
    from toduq_moa.simulator import FIXTURES, replay
    moa = MoA()
    print("TODUQ-MoA Simulator — replay TODUQ samples through gate→router→experts→aggregator\n")
    loc = route = total = 0
    for sample in FIXTURES:
        res = replay(sample, moa)
        total += 1
        loc += res.localized
        route += res.routing_correct
        inj = "control" if res.injected_ordinal is None else f"t{res.injected_ordinal}"
        routes = "/".join(res.predicted_routes_at_injected) or "answer"
        print(f"  {res.operator:18} injected@{inj:7} gold={res.gold_action:15} "
              f"routed={routes:28} localized={str(res.localized):5} "
              f"routing_correct={str(res.routing_correct):5}"
              + ("  [safety]" if res.safety_fired else ""))
    print(f"\nlocalization: {loc}/{total}   routing: {route}/{total}. "
          "The heuristic gate localizes lexicalized/safety turns and matches gold "
          "route there; unknowable→HITL and out-of-KB→RAG need a trained UQ gate "
          "(the gate is pluggable — swap in a TODUQ-trained detector).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="toduq-moa")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="run the offline MoA pipeline on sample inputs")
    sub.add_parser("simulate", help="replay TODUQ samples; test routing vs TODUQ gold")
    args = parser.parse_args(argv)

    if args.cmd == "simulate":
        return _simulate()
    if args.cmd == "demo":
        moa = MoA()
        for text in _SAMPLES:
            t = moa.handle(Query(text))
            r = t.response
            print(f"\nINPUT: {text}")
            print(f"  flag: uncertain={t.flag.is_uncertain} type={t.flag.uncertainty_type} "
                  f"safety={t.flag.safety_category}")
            print(f"  routes: {t.routes}")
            print(f"  -> action={r.action_taken} abstained={r.abstained} "
                  f"human={r.escalated_to_human}")
            if r.answer:
                print(f"  answer: {r.answer}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
