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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="toduq-moa")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="run the offline MoA pipeline on sample inputs")
    args = parser.parse_args(argv)

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
