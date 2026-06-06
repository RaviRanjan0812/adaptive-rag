"""Benchmark metrics.

retrieval_precision_recall  - coverage of gold doc-ids
faithfulness                - LLM-judge (Gemini): does every claim map to a citation?
routing_accuracy            - does the router pick the gold tier?
"""
from __future__ import annotations

import json
import os
from typing import List, Tuple

from app.llm import llm, strip_fences
from app.schemas import TierResult, TierName

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

def retrieval_precision_recall(retrieved_ids: List[str], gold_ids: List[str]) -> Tuple[float, float]:
    r, g = set(retrieved_ids), set(gold_ids)
    if not g:
        return (1.0, 1.0) if not r else (0.0, 1.0)
    tp        = len(r & g)
    precision = tp / len(r) if r else 0.0
    recall    = tp / len(g)
    return precision, recall


# --------------------------------------------------------------------------- #
# Faithfulness — Gemini judge
# --------------------------------------------------------------------------- #

_FAITH_SYSTEM = """\
You are a strict faithfulness auditor for financial RAG answers.
Given an answer and the cited source excerpts, return a score from 0.0 to 1.0.
  1.0 = every factual claim is directly supported by a citation
  0.0 = claims with no citation support (hallucinations)
Partial scores for partial support.
Return ONLY a JSON object: {"score": <float>, "reason": "<one sentence>"}
"""

def faithfulness(result: TierResult, gold_answer: str = "") -> float:
    """LLM-judge faithfulness. Falls back to grounded flag if no API key."""
    if not os.getenv("GEMINI_API_KEY") or not result.citations:
        return 1.0 if result.grounded else 0.0

    snippets = "\n".join(f"[{c.doc_id}] {c.snippet}" for c in result.citations[:6])
    prompt   = (
        f"Answer:\n{result.answer[:1200]}\n\n"
        f"Citations:\n{snippets}\n\n"
        "Score the faithfulness of this answer."
    )
    try:
        raw, _, _ = llm(_FAITH_SYSTEM, prompt, max_tokens=128)
        raw = strip_fences(raw)
        obj = json.loads(raw)
        return float(obj.get("score", 0.5))
    except Exception:
        return 1.0 if result.grounded else 0.0


# --------------------------------------------------------------------------- #
# Routing accuracy
# --------------------------------------------------------------------------- #

_HOP_TYPE_TO_TIER: dict[str, TierName] = {
    "single_hop":   TierName.hybrid,
    "multi_hop":    TierName.agentic,
    "relational":   TierName.graph_rag,
    "unanswerable": TierName.hybrid,
}

GOLD_TIER_OVERRIDES: dict[str, TierName] = {}


def gold_tier(query_item: dict) -> TierName | None:
    qid = query_item.get("id", "")
    if qid in GOLD_TIER_OVERRIDES:
        return GOLD_TIER_OVERRIDES[qid]
    # Honour explicit gold_tier field in the JSONL
    if gt := query_item.get("gold_tier"):
        try:
            return TierName(gt)
        except ValueError:
            pass
    hop = query_item.get("hop_type", "")
    if hop == "single_hop" and len(query_item.get("question", "").split()) <= 8:
        return TierName.long_context
    return _HOP_TYPE_TO_TIER.get(hop)


def routing_accuracy(decisions: list[tuple[TierName, TierName | None]]) -> float:
    known   = [(c, g) for c, g in decisions if g is not None]
    if not known:
        return 0.0
    correct = sum(1 for c, g in known if c == g)
    return correct / len(known)
