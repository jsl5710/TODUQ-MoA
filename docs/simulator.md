# TODUQ-MoA Simulator

The simulator replays a TODUQ sample **turn-by-turn** through the full MoA
pipeline (`gate → router → experts → aggregator`) and reports, for the turn where
uncertainty was injected, whether MoA:

- **localized** it — flagged uncertainty at the injected turn and answered
  everywhere else, and
- **routed** it correctly — the route matches TODUQ's **gold action**.

These are deliberately separate measurements: MoA can flag the right turn but pick
the wrong expert.

```
TODUQ sample (perturbed dialogue + gold action)
        │ replay each turn
        ▼
   MoA.handle(turn)  →  gate flag → router route → experts → aggregator
        │
        ▼
   at the injected turn:  route == gold?  (routing_correct)
   across turns:          only the injected turn flagged?  (localized)
```

## Components (`src/toduq_moa/simulator/`)

- **`MoASample`** (`sample.py`) — a multi-turn dialogue, the injected turn ordinal
  (or `None` for a control), and the TODUQ **gold action**. `FIXTURES` mirror
  TODUQ perturbations for an offline run; `from_toduq_record(rec, full_turns=…)`
  loads a real TODUQ JSON record.
- **`route_matches`** — maps TODUQ's gold action onto MoA's route space
  (`rag_structured → rag_relational`, `rag_unstructured → rag_vector`/`rag_web`,
  etc.), so the router is scored against TODUQ labels.
- **`replay`** (`simulator.py`) — runs the pipeline per turn and returns a
  `MoASimResult` (`turn_routes`, `localized`, `routing_correct`, `safety_fired`).

## Run it

```bash
PYTHONPATH=src python -m toduq_moa.cli simulate
```

Offline output:

```
slot_drop         injected@t1  gold=clarify        routed=clarify                 localized=True  routing_correct=True
safety_self_harm  injected@t2  gold=hitl           routed=hitl                    localized=True  routing_correct=True  [safety]
unknowable_fact   injected@t3  gold=hitl           routed=rag_relational/rag_vector/rag_web  localized=True  routing_correct=False
out_of_kb_entity  injected@t3  gold=rag_structured routed=answer                  localized=False routing_correct=False
control           injected@—   gold=answer         routed=answer                  localized=True  routing_correct=True
```

## Reading the result

The default **heuristic gate** localizes lexicalized input uncertainty and safety
turns, and routes them to the right expert. It **mis-routes** `unknowable_fact`
(localizes the turn but sends it to RAG when the gold is a human escalation —
severity isn't inferred from surface form) and **misses** `out_of_kb_entity`
entirely (a fluent knowledge gap the surface gate can't see). Both gaps are the
gate's, not the router's — and the gate is pluggable: swap in a **TODUQ-trained UQ
detector** (or `SemanticEntropyMetric` / `VerbalizedConfidenceMetric` over a live
model) and re-run to measure routing accuracy against TODUQ gold with a real
uncertainty signal.

## Aggregate evaluation

Across a TODUQ seed file (`from_toduq_record` per record), `localized` and
`routing_correct` give the MoA router's **localization accuracy** and **routing
accuracy** against TODUQ gold — the headline numbers for how well the routing
policy consumes an uncertainty signal.
