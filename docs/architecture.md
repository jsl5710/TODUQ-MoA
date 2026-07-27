# Architecture

TODUQ-MoA is a four-stage pipeline. Each stage is a small, swappable component
with a typed interface (see `src/toduq_moa/schema.py`).

```
Query ─► [1] Gate ─► UQFlag ─► [2] Router ─► RouteDecision ─► [3] Experts ─► ExpertResult[] ─► [4] Aggregator ─► FinalResponse
```

## 1. Gate (`gate.py`)
Runs a UQ method (`UQMethod` protocol) to produce a `UQFlag`
(`is_uncertain`, `score`, `uncertainty_type`, `severity`) **and** a safety screen.
Safety-critical inputs are flagged for HITL escalation independently of the UQ
score, so a *confident* but harmful input still routes to a human. The offline
`HeuristicGate` is a stand-in; plug in a real UQ detector (e.g. a TODUQ-trained
classifier, semantic entropy over `LLMClient.sample(n)`, or a calibrated head).

## 2. Router (`router.py`)
Maps the flag to one or more `Route`s. The mapping mirrors TODUQ's
taxonomy→gold-action logic, expanded so `rag_*` picks a concrete backend. Safety
or `major` severity short-circuits to `hitl` — nothing else can preempt it.

## 3. Experts (`experts/`)
Each expert owns one route and returns structured `ExpertResult` (content +
evidence + confidence, or `needs_human`). Experts for the selected routes run
independently, so they parallelize cleanly. Backends (SQL, vector store, web) are
injected; offline stubs return empty evidence so the pipeline always runs.

## 4. Aggregator (`aggregator.py`)
A higher-capability reasoning LLM synthesizes the experts' evidence into the final
answer, citing sources. Two invariants it cannot break: a `needs_human` result
forces abstain+escalate; a `clarify` result is returned as a question. With no LLM
it degrades to a deterministic evidence summary.

## Orchestrator & Trace
`MoA.handle(query) -> Trace` runs all four stages and returns the full decision
trace (flag, routes, per-expert results, final response) for auditing and for
evaluation against TODUQ gold labels.

## Provider-agnostic models
The gate's UQ method, the clarify expert, and the aggregator all accept an
`LLMClient` (the same interface TODUQ uses) — Claude, OpenAI, or open models via
vLLM/Ollama, chosen in `configs/moa.yaml`.
