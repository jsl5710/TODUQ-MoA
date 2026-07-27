# Project Overview — the TODUQ / TODUQ-MoA / TODDC program

This document explains the **three-repository research program** these projects
form, how they connect, and where **TODUQ-MoA** sits in it.

## The problem

Conversational and agentic AI systems in **task-oriented dialogue** (booking,
search, support) fail in two coupled ways:

1. They **answer when they shouldn't** — confidently hallucinating instead of
   recognizing uncertainty and deferring (asking, retrieving, escalating).
2. Even when they act correctly, the intervention can **break the discourse** —
   the dialogue stops cohering.

Measuring and *fixing* the first failure is what TODUQ-MoA is for; the other two
repos build the labeled data to drive and evaluate it, from a common base — the
**Schema-Guided Dialogue (SGD)** dataset.

## The three repositories

| Repo | Builds | Role |
| --- | --- | --- |
| **[TODUQ](https://github.com/jsl5710/TODUQ)** | uncertainty-injection dataset | supplies labeled uncertainty triggers + the gold route |
| **TODUQ-MoA** (this repo) | mixture-of-agents inference system | **acts** on the uncertainty: route to experts, aggregate |
| **[TODDC](https://github.com/jsl5710/TODDC)** | coherence-violation dataset | evaluates whether MoA's output stayed coherent |

```
   TODUQ (uncertainty + gold route) ──► TODUQ-MoA ──► TODDC (coherence of the result)
                                        gate → router → experts → aggregator
```

## TODUQ-MoA's architecture

A four-stage pipeline (`src/toduq_moa/`), each stage a swappable typed component:

```
Query ─► Gate (UQ + safety) ─► Router ─► Experts ─► Aggregator ─► FinalResponse
```

- **Gate** — a UQ method produces an uncertainty flag *and* a safety screen;
  safety-critical inputs escalate to a human regardless of the UQ score.
- **Router** — maps the flag to expert(s); the route vocabulary is a **superset
  of TODUQ's gold actions**, so the router can be scored directly against TODUQ
  labels.
- **Experts** — `rag_relational` (structured DB query), `rag_vector` (RAG DB),
  `rag_web` (internet), `hitl` (safety channel), `clarify`. Pluggable backends;
  offline stubs keep the whole system runnable.
- **Aggregator** — a higher-capability reasoning model synthesizes the experts'
  evidence into the final answer. Two invariants it cannot break: a human
  escalation forces abstain, and a clarification is returned as a question.

Provider-agnostic (Claude / OpenAI / open models) for the gate's UQ method, the
clarify expert, and the aggregator.

## The shared method (in the sibling data repos)

TODUQ and TODDC generate their datasets with a deterministic **5-pass chain**
(`analyse → document → apply → confirm → edit`) where the template owns the label
and the LLM owns the wording. TODUQ-MoA consumes those labels: TODUQ's gold route
scores the **router**, and TODDC scores the **coherence** of MoA's multi-turn
output.

## How the loop closes

1. **TODUQ** perturbs a user turn and labels the correct route.
2. **TODUQ-MoA** (this repo) flags the uncertainty and routes it to experts, then
   a reasoning model aggregates their outputs — or abstains to a human.
3. **TODDC** checks whether the resulting dialogue stayed coherent.

Each repo runs standalone and offline; together they form an end-to-end pipeline
for building, acting on, and evaluating uncertainty-aware task-oriented dialogue.

## Consolidation note

The data repos and this one each carry their own `LLMClient` runner layer. A
planned `tod-core` package would share ingest, runners, and the judge across all
three.
