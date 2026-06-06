"""Tier 4 — AGENTIC, SELF-REFLECTIVE.

LangGraph loop: plan → retrieve → critique → (re-retrieve) → generate
Uses Gemini for all LLM steps.

Requires:
  data/index/faiss.index, data/index/bm25.pkl, data/index/chunks.jsonl
  GEMINI_API_KEY
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from app.llm import llm, cost, strip_fences, MODEL, _IN_PER_M, _OUT_PER_M
from app.tiers.base import register, RetrievalTier
from app.schemas import TierResult, TierName, Citation
from app.session_store import SessionIndex
from app.tiers.hybrid import _load_indexes, _dense_retrieve, _bm25_retrieve, _rrf, _rerank

MAX_LOOPS   = 3
TOP_RETRIEVE = 6
RERANK_K    = 4


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

@dataclass
class StepLog:
    step: str
    query_used: str = ""
    chunks_retrieved: int = 0
    chunks_new: int = 0
    critique_verdict: str = ""
    critique_reason: str = ""
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class AgentProvenance:
    steps: list[StepLog] = field(default_factory=list)
    retrieval_loops: int = 0
    over_retrieval: int = 0
    bad_retrieval: int = 0
    total_in_tokens: int = 0
    total_out_tokens: int = 0

    def log(self, step: StepLog):
        self.steps.append(step)
        self.total_in_tokens  += step.tokens_in
        self.total_out_tokens += step.tokens_out

    def cost_usd(self) -> float:
        return cost(self.total_in_tokens, self.total_out_tokens)

    def summary(self) -> str:
        return (
            f"loops={self.retrieval_loops} over_retrieval={self.over_retrieval} "
            f"bad_retrieval={self.bad_retrieval} llm_calls={len(self.steps)} "
            f"in={self.total_in_tokens} out={self.total_out_tokens}"
        )


# --------------------------------------------------------------------------- #
# LangGraph state
# --------------------------------------------------------------------------- #

class AgentState(TypedDict):
    question: str
    sub_queries: list[str]
    evidence: list[dict]
    evidence_ids: set
    critique_verdict: str
    critique_reason: str
    answer: str
    loop_count: int
    provenance: AgentProvenance
    session: Any  # SessionIndex | None


# --------------------------------------------------------------------------- #
# Graph nodes
# --------------------------------------------------------------------------- #

PLAN_SYSTEM = """\
You decompose a complex financial question into 1-3 focused sub-queries.
Return a JSON array of strings (sub-query texts only).
Each sub-query should be independently searchable.
Return ONLY a JSON array, no prose.
"""

def node_plan(state: AgentState) -> AgentState:
    prov = state["provenance"]
    text, in_tok, out_tok = llm(
        PLAN_SYSTEM,
        f"Decompose this question into sub-queries:\n{state['question']}",
        max_tokens=256,
    )
    sub_queries = _parse_json_array(text) or [state["question"]]
    prov.log(StepLog(step="plan", query_used=state["question"],
                     tokens_in=in_tok, tokens_out=out_tok))
    return {**state, "sub_queries": sub_queries}


def node_retrieve(state: AgentState) -> AgentState:
    fi, bm25, chunks, embed_model, rerank_model = _load_indexes()
    session = state.get("session")
    if session is not None:
        fi, bm25, chunks = session.faiss_index, session.bm25, session.chunks
    prov         = state["provenance"]
    existing_ids = state.get("evidence_ids", set())
    new_evidence = list(state.get("evidence", []))
    loop         = state["loop_count"]
    query        = state["sub_queries"][loop % len(state["sub_queries"])]

    dense_hits = _dense_retrieve(query, fi, embed_model, chunks, TOP_RETRIEVE)
    bm25_hits  = _bm25_retrieve(query, bm25, TOP_RETRIEVE)
    merged     = _rrf(dense_hits, bm25_hits)
    candidates = [chunks[i] for i in merged[:TOP_RETRIEVE * 2] if i < len(chunks)]
    top        = _rerank(query, candidates, rerank_model, RERANK_K)

    chunks_new = 0
    for c in top:
        if c["chunk_id"] not in existing_ids:
            new_evidence.append(c)
            existing_ids.add(c["chunk_id"])
            chunks_new += 1

    if loop > 0:
        prov.retrieval_loops += 1
        if chunks_new == 0:
            prov.over_retrieval += 1

    prov.log(StepLog(step=f"retrieve_loop_{loop}", query_used=query,
                     chunks_retrieved=len(top), chunks_new=chunks_new))
    return {**state, "evidence": new_evidence, "evidence_ids": existing_ids}


CRITIQUE_SYSTEM = """\
You are a strict evidence auditor for financial research.
Given a question and retrieved filing excerpts, decide if the evidence is
SUFFICIENT to write a complete, fully-cited answer.

Respond with exactly this JSON:
{"verdict": "sufficient" | "insufficient", "reason": "one sentence"}
"""

def node_critique(state: AgentState) -> AgentState:
    prov = state["provenance"]
    evidence_text = "\n\n---\n\n".join(
        f"[{c['chunk_id']}]\n{c['text'][:400]}" for c in state["evidence"][:TOP_RETRIEVE]
    )
    prompt = (
        f"Question: {state['question']}\n\n"
        f"<evidence>\n{evidence_text}\n</evidence>\n\nIs this evidence sufficient?"
    )
    text, in_tok, out_tok = llm(CRITIQUE_SYSTEM, prompt, max_tokens=128)
    text = strip_fences(text)
    try:
        obj     = json.loads(text)
        verdict = obj.get("verdict", "insufficient")
        reason  = obj.get("reason", "")
    except json.JSONDecodeError:
        verdict, reason = "insufficient", "parse error"

    new_loop = state["loop_count"] + 1   # increment here — node functions CAN mutate state
    if new_loop >= MAX_LOOPS and verdict == "insufficient":
        prov.bad_retrieval += 1
        verdict = "sufficient"           # force exit after MAX_LOOPS

    prov.log(StepLog(step=f"critique_loop_{state['loop_count']}",
                     critique_verdict=verdict, critique_reason=reason,
                     tokens_in=in_tok, tokens_out=out_tok))
    return {**state, "critique_verdict": verdict, "critique_reason": reason,
            "loop_count": new_loop}


def edge_critique_router(state: AgentState) -> str:
    # Read-only: never mutate state here — LangGraph silently discards mutations
    if state["critique_verdict"] == "sufficient":
        return "generate"
    return "retrieve"


GEN_SYSTEM = """\
You are a senior financial analyst. Write a complete, precise answer to the
question using ONLY the provided filing excerpts.
Cite every factual claim with the chunk_id in brackets.
If evidence is incomplete, say so explicitly and cite what IS available.
"""

def node_generate(state: AgentState) -> AgentState:
    prov = state["provenance"]
    evidence_text = "\n\n---\n\n".join(
        f"[{c['chunk_id']}]\n{c['text'][:600]}"
        for c in state["evidence"][:MAX_LOOPS * RERANK_K]
    )
    prompt = f"<evidence>\n{evidence_text}\n</evidence>\n\nQuestion: {state['question']}"
    text, in_tok, out_tok = llm(GEN_SYSTEM, prompt, max_tokens=1024)
    prov.log(StepLog(step="generate", tokens_in=in_tok, tokens_out=out_tok))
    return {**state, "answer": text}


def _parse_json_array(text: str) -> list:
    text = strip_fences(text)
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


# --------------------------------------------------------------------------- #
# Build graph (lazy)
# --------------------------------------------------------------------------- #

_GRAPH = None

def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        g = StateGraph(AgentState)
        g.add_node("plan",     node_plan)
        g.add_node("retrieve", node_retrieve)
        g.add_node("critique", node_critique)
        g.add_node("generate", node_generate)
        g.set_entry_point("plan")
        g.add_edge("plan", "retrieve")
        g.add_edge("retrieve", "critique")
        g.add_conditional_edges("critique", edge_critique_router,
                                {"retrieve": "retrieve", "generate": "generate"})
        g.add_edge("generate", END)
        _GRAPH = g.compile()
    return _GRAPH


# --------------------------------------------------------------------------- #
# Tier
# --------------------------------------------------------------------------- #

@register
class AgenticTier(RetrievalTier):
    name = TierName.agentic

    def run(self, question: str, session: SessionIndex | None = None) -> TierResult:
        prov = AgentProvenance()
        initial_state: AgentState = {
            "question": question,
            "sub_queries": [],
            "evidence": [],
            "evidence_ids": set(),
            "critique_verdict": "insufficient",
            "critique_reason": "",
            "answer": "",
            "loop_count": 0,
            "provenance": prov,
            "session": session,
        }

        t0          = time.perf_counter()
        final_state = _get_graph().invoke(initial_state)
        latency_ms  = (time.perf_counter() - t0) * 1000

        evidence  = final_state["evidence"]
        citations = [Citation(doc_id=c["doc_id"], snippet=c["text"][:200]) for c in evidence]

        return TierResult(
            tier=self.name,
            answer=final_state["answer"],
            citations=citations,
            cost_usd=prov.cost_usd(),
            latency_ms=round(latency_ms, 1),
            num_llm_calls=len(prov.steps),
            notes=prov.summary(),
        )
