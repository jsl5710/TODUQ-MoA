"""Expert registry."""
from toduq_moa.experts.base import BaseExpert, Expert
from toduq_moa.experts.clarify import ClarifyExpert
from toduq_moa.experts.hitl import HumanInTheLoopExpert
from toduq_moa.experts.rag import RelationalRAGExpert, VectorRAGExpert, WebRAGExpert


def default_experts(*, llm=None) -> list[Expert]:
    """The v1 expert panel with offline stubs (no backends wired)."""
    return [
        RelationalRAGExpert(),
        VectorRAGExpert(),
        WebRAGExpert(),
        HumanInTheLoopExpert(),
        ClarifyExpert(llm=llm),
    ]


__all__ = [
    "Expert", "BaseExpert", "default_experts",
    "RelationalRAGExpert", "VectorRAGExpert", "WebRAGExpert",
    "HumanInTheLoopExpert", "ClarifyExpert",
]
