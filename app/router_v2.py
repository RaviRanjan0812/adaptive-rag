"""Router v2 — LLM-judge classifier using Gemini.

Falls back to v1 heuristic on any failure.
Requires: GEMINI_API_KEY
"""
from __future__ import annotations

import json

from app.llm import llm, strip_fences
from app.schemas import RouteDecision, TierName
from app.router import AdaptiveRouter

_SYSTEM = """\
You are a query router for a financial-filing RAG system with four retrieval tiers.
Given a user question, choose the SINGLE best tier and return ONLY valid JSON.

Tier descriptions:
  long_context  - Corpus fits in context. Use for short, self-contained, single-entity questions.
                  Examples: "What was Apple's revenue in FY2024?"
  hybrid        - FAISS + BM25 + rerank. Best for single-hop factual lookups.
                  Examples: "List NVDA's reportable segments.", "What is JPM's tier-1 ratio?"
  graph_rag     - Entity-relationship graph traversal. Use when the question explicitly
                  compares or connects two or more named entities across documents.
                  Examples: "Compare AAPL supply-chain risks with the Q3 call.",
                  "How do NVDA and AAPL gross margins compare?"
  agentic       - Plan→retrieve→critique loop. Use only for open-ended multi-step questions
                  requiring synthesis across several documents.
                  Examples: "How did JPM rate assumptions relate to NII guidance AND actual Q2?"

Rules:
- Prefer cheaper tiers (long_context < hybrid < graph_rag < agentic).
- Only choose agentic if genuine multi-step reasoning is required.
- If unanswerable from filings, choose hybrid (it will say so).

Respond with EXACTLY this JSON (no prose, no markdown):
{
  "tier": "<long_context|hybrid|graph_rag|agentic>",
  "estimated_hops": <integer 1-4>,
  "reasoning": "<one sentence>",
  "confidence": <float 0.0-1.0>
}
"""


class RouterV2:
    def __init__(self):
        self._v1 = AdaptiveRouter()

    def route(self, question: str) -> RouteDecision:
        try:
            return self._llm_route(question)
        except Exception as exc:
            decision = self._v1.route(question)
            decision.features["v2_fallback"] = str(exc)[:120]
            return decision

    def _llm_route(self, question: str) -> RouteDecision:
        raw, in_tok, out_tok = llm(_SYSTEM, f"Question: {question}", max_tokens=256)
        raw = strip_fences(raw)
        obj = json.loads(raw)
        return RouteDecision(
            tier=TierName(obj["tier"]),
            estimated_hops=int(obj.get("estimated_hops", 1)),
            reasoning=obj.get("reasoning", ""),
            features={
                "v2": True,
                "confidence": float(obj.get("confidence", 1.0)),
                "in_tokens": in_tok,
                "out_tokens": out_tok,
            },
        )
