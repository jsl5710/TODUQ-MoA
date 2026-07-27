"""Expert protocol. Each expert owns one route and returns structured evidence."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from toduq_moa.schema import ExpertResult, Query, Route


@runtime_checkable
class Expert(Protocol):
    id: str
    route: Route

    def can_handle(self, route: Route) -> bool: ...
    def run(self, query: Query) -> ExpertResult: ...


class BaseExpert:
    id: str = "base"
    route: Route = "answer"

    def can_handle(self, route: Route) -> bool:
        return route == self.route

    def run(self, query: Query) -> ExpertResult:  # pragma: no cover - abstract
        raise NotImplementedError
