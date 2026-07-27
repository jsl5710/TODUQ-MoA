# TODUQ-MoA — Mixture-of-Agents with expert routing under uncertainty

A **mixture-of-agents (MoA)** inference system for conversational / agentic AI.
When an input is **flagged as uncertain** by a UQ method, TODUQ-MoA routes it to
one or more **expert agents** whose outputs become the input to a **sophisticated
reasoning aggregator** that produces the final response — or abstains again.

This is the *inference-time* companion to
[**TODUQ**](https://github.com/jsl5710/TODUQ): TODUQ builds the labeled
uncertainty-injection dataset and defines the abstain/route decision; TODUQ-MoA
is a system that *acts* on that decision with a panel of experts.

```
                 ┌────────────┐   uncertain?   ┌──────────┐
   user input ─► │  UQ GATE   │ ───────────────►│  ROUTER  │
   + context     │ flag+type  │   no → answer   │ type/sev │
                 └────────────┘                 └────┬─────┘
                                       ┌─────────────┼───────────────┬─────────────┐
                                       ▼             ▼               ▼             ▼
                                ┌───────────┐ ┌───────────┐  ┌───────────┐  ┌──────────┐
                                │ RAG:      │ │ RAG:      │  │ HUMAN-IN- │  │ CLARIFY  │
                                │ relational│ │ vector +  │  │ THE-LOOP  │  │ ask a    │
                                │ DB (query)│ │ web       │  │ (safety)  │  │ follow-up│
                                └─────┬─────┘ └─────┬─────┘  └─────┬─────┘  └────┬─────┘
                                      └─────────────┴──── expert results ───────┘
                                                     ▼
                                            ┌──────────────────┐
                                            │  AGGREGATOR       │  sophisticated
                                            │  (reasoning LLM)  │  reasoning model
                                            └────────┬─────────┘
                                                     ▼  final answer / abstain
```

## Experts (v1)

| Expert | Handles | Backing |
| --- | --- | --- |
| `rag_relational` | facts answerable by a **structured query** | relational DB (SQL / dataset query) |
| `rag_vector` | facts needing **free-text context** | vector / RAG DB |
| `rag_web` | facts outside any local store | internet search |
| `hitl` | **safety-critical / major** issues (adversarial, self-harm, high-stakes) | human intervention queue |
| `clarify` | resolvable ambiguity | generates a follow-up question |

The routing vocabulary mirrors TODUQ's gold actions (`clarify`,
`rag_structured` → `rag_relational`, `rag_unstructured` → `rag_vector`/`rag_web`,
`hitl`, `handoff_llm`), so a TODUQ-labeled turn maps directly onto a route here —
letting you **evaluate the router against TODUQ's gold labels**.

## The reasoning aggregator

Expert outputs are collected as **structured evidence** and handed to a
higher-capability reasoning model (default a current Claude model) that
synthesizes the final answer, cites which expert supplied what, and may itself
decide to abstain (→ back to HITL). Provider-agnostic: Claude / OpenAI / open
models via the same `LLMClient` interface used in TODUQ.

## Safety posture

The `hitl` expert is a **safety escalation channel**, not a fallback of last
resort. Inputs matching safety categories (self-harm, adversarial manipulation,
other high-stakes) route to human review **regardless** of the UQ score — the
aggregator cannot override a safety escalation. See
[`docs/experts.md`](docs/experts.md).

## How the three repos fit together

- **TODUQ** — dataset: inject controlled uncertainty into task-oriented dialogue,
  label the correct abstain/route action.
- **TODUQ-MoA** (this repo) — system: route UQ-flagged inputs to experts and
  aggregate. Evaluated *with* TODUQ's labels.
- **TODDO** — measure discourse coherence of the resulting task-oriented dialogue.

## Simulator

The **TODUQ-MoA Simulator** replays a TODUQ sample turn-by-turn through the
pipeline and tests whether the flagged turn **localizes** to the injected turn and
**routes** to the correct expert (vs TODUQ's gold action):

```bash
PYTHONPATH=src python -m toduq_moa.cli simulate
```

See [`docs/simulator.md`](docs/simulator.md).

## Status

Design scaffold + simulator. Interfaces and a runnable offline pipeline (stub
experts + echo aggregator) are in place; real DB/vector/web backends, a live
aggregator, and a trained UQ gate are the next milestones. See [`docs/`](docs/).

## License

Code under MIT (see `LICENSE`). If evaluated on TODUQ-derived data, that data
remains CC BY-SA 4.0 (SGD lineage).
