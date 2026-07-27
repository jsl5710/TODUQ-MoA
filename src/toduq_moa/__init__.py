"""TODUQ-MoA — Mixture-of-Agents with expert routing under uncertainty.

Pipeline: gate (UQ + safety) → router → expert panel → reasoning aggregator.
Companion to TODUQ (which supplies labeled uncertainty triggers).
"""
from toduq_moa.orchestrator import MoA, Trace
from toduq_moa.schema import ExpertResult, FinalResponse, Query, RouteDecision, UQFlag

__version__ = "0.1.0"

__all__ = ["MoA", "Trace", "Query", "UQFlag", "RouteDecision", "ExpertResult", "FinalResponse"]
