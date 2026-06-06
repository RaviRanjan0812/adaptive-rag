"""Typed contracts for the whole system (Pydantic — reuse your TradingAgents instinct)."""
from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HopType(str, Enum):
    single = "single_hop"
    multi = "multi_hop"
    relational = "relational"
    unanswerable = "unanswerable"


class TierName(str, Enum):
    long_context = "long_context"
    hybrid = "hybrid"
    graph_rag = "graph_rag"
    agentic = "agentic"


class QueryRequest(BaseModel):
    question: str
    force_tier: Optional[TierName] = Field(
        default=None,
        description="Bypass the router and force a tier (used by the benchmark).",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID from POST /upload. Queries run against the uploaded document.",
    )


class Citation(BaseModel):
    doc_id: str
    snippet: str = ""
    score: Optional[float] = None


class RouteDecision(BaseModel):
    tier: TierName
    estimated_hops: int
    reasoning: str
    features: Dict[str, Any]


class TierResult(BaseModel):
    tier: TierName
    answer: str
    citations: List[Citation] = []
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    num_llm_calls: int = 0
    grounded: bool = True  # every claim maps to a citation? (faithfulness gate)
    notes: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    route: RouteDecision
    result: TierResult
