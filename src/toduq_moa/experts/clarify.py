"""Clarify expert: turn an ambiguous input into a single follow-up question.

Uses an LLMClient when available (a prompt that asks for the one question that
would resolve the ambiguity); otherwise a template. The aggregator returns the
question to the user instead of an answer.
"""
from __future__ import annotations

from typing import Any, Optional

from toduq_moa.experts.base import BaseExpert
from toduq_moa.schema import ExpertResult, Query

_PROMPT = (
    "A user said: {utterance!r}\n"
    "It is ambiguous or underspecified. Ask ONE concise clarifying question that "
    "would resolve the ambiguity. Return only the question."
)


class ClarifyExpert(BaseExpert):
    id = "clarify"
    route = "clarify"

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm

    def run(self, query: Query) -> ExpertResult:
        if self.llm is not None:
            question = self.llm.generate(_PROMPT.format(utterance=query.text)).strip()
        else:
            question = "Could you clarify what you mean so I can help accurately?"
        return ExpertResult(self.id, self.route, content=question, confidence=0.8)
