# Routing & TODUQ alignment

The router turns a `UQFlag` into a `RouteDecision`. The vocabulary is a superset
of TODUQ's gold actions so a TODUQ-labeled turn maps onto a TODUQ-MoA route,
which lets you **evaluate the router directly against TODUQ's gold labels**.

| UQ flag | Route(s) | TODUQ gold action |
| --- | --- | --- |
| not uncertain | `answer` | `answer` |
| `input` | `clarify` | `clarify` |
| `parameter` | `rag_relational` + `rag_vector` + `rag_web` | `rag_structured` / `rag_unstructured` |
| `reasoning` / `prediction` | `handoff_llm` | `handoff_llm` |
| safety category ≠ none **or** severity `major` | `hitl` (short-circuit) | `hitl` |

## Escalation precedence

Safety and `major` severity **preempt** the type-based mapping — a self-harm or
adversarial input goes to `hitl` even if its uncertainty score is low. This is
the one rule the router applies before consulting the taxonomy, and the
aggregator cannot reverse it.

## Evaluating the router with TODUQ

For each TODUQ record, feed `source.utterance` (+ context) to the gate, run the
router, and collapse the predicted routes back to TODUQ's action space
(`rag_* → rag_structured/rag_unstructured`). Then score against `gold.action`
with TODUQ's `routing_accuracy` — closing the loop between the two repos:

- **TODUQ** decides *what the right route is* (gold label).
- **TODUQ-MoA** decides *what route to take* (prediction) and *acts on it*.
