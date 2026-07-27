"""RAG experts: relational (structured query), vector (RAG DB), web (internet).

Each wraps a pluggable backend behind a tiny interface so real datastores drop in
without touching the pipeline. Offline stubs return empty evidence so the whole
system runs with no external services.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from toduq_moa.experts.base import BaseExpert
from toduq_moa.schema import ExpertResult, Query


class RelationalRAGExpert(BaseExpert):
    """Answers via a structured query against a relational DB (the `rag_structured`
    case in TODUQ). `backend(query) -> list[rows]`; rows are the evidence."""
    id = "rag_relational"
    route = "rag_relational"

    def __init__(self, backend: Optional[Callable[[Query], list[dict[str, Any]]]] = None):
        self.backend = backend

    def run(self, query: Query) -> ExpertResult:
        if self.backend is None:
            return ExpertResult(self.id, self.route, content="", confidence=0.0,
                                error="no relational backend configured")
        rows = self.backend(query)
        return ExpertResult(self.id, self.route,
                            content=f"{len(rows)} row(s) matched.",
                            evidence=rows, confidence=1.0 if rows else 0.0)


class VectorRAGExpert(BaseExpert):
    """Retrieves free-text context from a vector/RAG DB (the `rag_unstructured`
    case). `backend(text, k) -> list[docs]`."""
    id = "rag_vector"
    route = "rag_vector"

    def __init__(self, backend: Optional[Callable[[str, int], list[dict[str, Any]]]] = None,
                 k: int = 4):
        self.backend = backend
        self.k = k

    def run(self, query: Query) -> ExpertResult:
        if self.backend is None:
            return ExpertResult(self.id, self.route, content="", confidence=0.0,
                                error="no vector backend configured")
        docs = self.backend(query.text, self.k)
        return ExpertResult(self.id, self.route,
                            content=f"retrieved {len(docs)} passage(s).",
                            evidence=docs, confidence=1.0 if docs else 0.0)


class WebRAGExpert(BaseExpert):
    """Searches the internet. `backend(text) -> list[results]`."""
    id = "rag_web"
    route = "rag_web"

    def __init__(self, backend: Optional[Callable[[str], list[dict[str, Any]]]] = None):
        self.backend = backend

    def run(self, query: Query) -> ExpertResult:
        if self.backend is None:
            return ExpertResult(self.id, self.route, content="", confidence=0.0,
                                error="no web backend configured")
        results = self.backend(query.text)
        return ExpertResult(self.id, self.route,
                            content=f"{len(results)} web result(s).",
                            evidence=results, confidence=1.0 if results else 0.0)
