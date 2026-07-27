# Experts

Every expert implements `can_handle(route)` + `run(query) -> ExpertResult`
(`experts/base.py`). Results carry `content`, structured `evidence`, a
`confidence`, and a `needs_human` flag.

## RAG experts (`experts/rag.py`)

| Expert | Route | Backend interface | Maps to TODUQ |
| --- | --- | --- | --- |
| `RelationalRAGExpert` | `rag_relational` | `backend(query) -> rows` (SQL / dataset query) | `rag_structured` |
| `VectorRAGExpert` | `rag_vector` | `backend(text, k) -> docs` (vector/RAG DB) | `rag_unstructured` |
| `WebRAGExpert` | `rag_web` | `backend(text) -> results` (internet) | `rag_unstructured` |

Parameter-uncertainty inputs fan out across all three; the aggregator reconciles
whichever expert returned grounded evidence. Backends are injected — wire
SQLAlchemy / Chroma / a search API in production; the offline stubs return empty
evidence and a "no backend configured" error so the pipeline still runs.

## Human-in-the-loop (`experts/hitl.py`) — a safety channel

`HumanInTheLoopExpert` enqueues the query for human review and returns a result
with `needs_human=True`. It is reached when the gate flags a **safety category**
(self-harm, adversarial, high-stakes) or `major` severity. The aggregator
**cannot** answer over a `needs_human` result — it abstains and escalates. This
is deliberate: HITL is where the system defers to a person, not a last-resort
fallback. Swap the in-memory queue for a real ticketing/on-call integration.

## Clarify (`experts/clarify.py`)

`ClarifyExpert` produces the single follow-up question most likely to resolve the
ambiguity (via an `LLMClient`, or a template offline). The aggregator returns the
question to the user instead of answering.

## Adding an expert

1. Subclass `BaseExpert`, set `id` + `route`, implement `run`.
2. Add its route to `Route` in `schema.py` and to the router's mapping if new.
3. Register it in `experts/__init__.py:default_experts`.
