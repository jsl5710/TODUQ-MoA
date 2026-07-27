"""Aggregator: the sophisticated reasoning model that synthesizes expert outputs.

Takes the collected `ExpertResult`s plus the original query and produces the final
response. Two hard rules:
  1. If any expert escalated to a human (`needs_human`), the aggregator abstains
     and hands off — it cannot answer over a safety escalation.
  2. A `clarify` result is returned to the user as a question, not answered.
Otherwise it hands the structured evidence to an LLM to reason over; with no LLM
it falls back to a deterministic evidence summary so the pipeline still runs.
"""
from __future__ import annotations

from typing import Any, Optional

from toduq_moa.schema import ExpertResult, FinalResponse, Query

_SYSTEM = (
    "You are a careful reasoning aggregator in a mixture-of-agents system. You are "
    "given a user query and evidence gathered by expert agents. Synthesize a single "
    "well-grounded answer, citing which expert supplied each fact. If the evidence is "
    "insufficient or conflicting, say so and recommend escalation rather than guessing."
)


def aggregate(query: Query, results: list[ExpertResult], *, llm: Optional[Any] = None) -> FinalResponse:
    used = [r.expert_id for r in results]

    # Rule 1 — safety escalation wins.
    human = next((r for r in results if r.needs_human), None)
    if human is not None:
        return FinalResponse(answer=human.content, action_taken="hitl", abstained=True,
                             used_experts=used, escalated_to_human=True)

    # Rule 2 — clarification is a question back to the user.
    clar = next((r for r in results if r.route == "clarify"), None)
    if clar is not None:
        return FinalResponse(answer=clar.content, action_taken="clarify", abstained=True,
                             used_experts=used)

    evidence = [e for r in results for e in r.evidence]
    action = results[0].route if results else "answer"

    if llm is not None:
        prompt = _render(query, results)
        answer = llm.generate(prompt, system=_SYSTEM).strip()
    else:
        answer = _summarize(results)

    return FinalResponse(answer=answer, action_taken=action, abstained=False,
                         used_experts=used, citations=evidence)


def _render(query: Query, results: list[ExpertResult]) -> str:
    lines = [f"USER QUERY: {query.text}", "", "EXPERT EVIDENCE:"]
    for r in results:
        lines.append(f"- [{r.expert_id}] {r.content} (evidence: {len(r.evidence)} item(s))")
    lines.append("\nSynthesize the final answer, citing experts.")
    return "\n".join(lines)


def _summarize(results: list[ExpertResult]) -> str:
    answered = [r for r in results if r.confidence > 0 and not r.error]
    if not answered:
        return "No expert produced grounded evidence; recommend escalation."
    return " | ".join(f"{r.expert_id}: {r.content}" for r in answered)
