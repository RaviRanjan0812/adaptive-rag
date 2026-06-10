"""Tier 3 — GRAPH RAG.

Multi-hop entity-graph traversal for relational questions.
Uses Gemini for entity extraction and grounded generation.

Requires:
  data/index/graph.pkl, data/index/chunks.jsonl
  GEMINI_API_KEY
"""

import json
import pickle
import re
import time
from functools import lru_cache
from pathlib import Path

import networkx as nx

from app.llm import llm, cost, strip_fences, MODEL
from app.tiers.base import register, RetrievalTier
from app.schemas import TierResult, TierName, Citation
from app.session_store import SessionIndex

INDEX_DIR   = Path("data/index")
GRAPH_PATH  = INDEX_DIR / "graph.pkl"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"

MAX_HOPS           = 2
MAX_EVIDENCE_CHUNKS = 8
MAX_CHUNK_CHARS    = 600

ENTITY_SYSTEM = """\
You extract the key named entities from a financial question.
Return a JSON array of strings (entity names only, normalised).
Company tickers preferred (AAPL, NVDA, JPM).
Return ONLY a JSON array, no prose.
"""

GEN_SYSTEM = """\
You are a financial analyst assistant.
Answer the question using ONLY the graph-retrieved evidence below.
The evidence is a set of filing excerpts linked by the knowledge graph.
For every factual claim, cite the chunk_id in brackets.
If the evidence is insufficient, say so explicitly.
"""


@lru_cache(maxsize=1)
def _load_graph() -> nx.MultiDiGraph:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"{GRAPH_PATH} not found — run python -m ingest.build_graph first."
        )
    with GRAPH_PATH.open("rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _load_chunk_index() -> dict[str, dict]:
    idx = {}
    for line in CHUNKS_PATH.read_text().splitlines():
        if line.strip():
            c = json.loads(line)
            idx[c["chunk_id"]] = c
    return idx


def _fuzzy_match(name: str, graph: nx.MultiDiGraph) -> list[str]:
    name_l = name.lower()
    return [n for n in graph.nodes() if name_l in n.lower() or n.lower() in name_l]


def _bfs_subgraph(seeds: list[str], graph: nx.MultiDiGraph, hops: int) -> set[str]:
    visited, frontier = set(seeds), set(seeds)
    for _ in range(hops):
        nxt = set()
        for node in frontier:
            if node not in graph:
                continue
            nxt |= (set(graph.successors(node)) | set(graph.predecessors(node))) - visited
        visited |= nxt
        frontier = nxt
    return visited


def _collect_chunk_ids(nodes: set[str], graph: nx.MultiDiGraph) -> list[str]:
    chunk_ids: list[str] = []
    for n in nodes:
        if n in graph.nodes:
            chunk_ids.extend(graph.nodes[n].get("chunk_ids", []))
    for u, v, data in graph.edges(data=True):
        if u in nodes or v in nodes:
            chunk_ids.extend(data.get("chunk_ids", []))
    seen, result = set(), []
    for cid in chunk_ids:
        if cid not in seen:
            seen.add(cid)
            result.append(cid)
    return result


def _extract_entities(question: str) -> list[str]:
    raw, _, _ = llm(ENTITY_SYSTEM, question, max_tokens=256)
    raw = strip_fences(raw)
    try:
        entities = json.loads(raw) if raw.startswith("[") else []
        return [str(e) for e in entities if e]
    except json.JSONDecodeError:
        return re.findall(r"\b[A-Z]{2,5}\b", question)


def _build_evidence(chunk_ids: list[str], chunk_index: dict) -> tuple[str, list[dict]]:
    chosen = [chunk_index[cid] for cid in chunk_ids[:MAX_EVIDENCE_CHUNKS] if cid in chunk_index]
    parts = [f"[{c['chunk_id']}]\n{c['text'][:MAX_CHUNK_CHARS]}" for c in chosen]
    return "\n\n---\n\n".join(parts), chosen


@register
class GraphRagTier(RetrievalTier):
    name = TierName.graph_rag

    def run(self, question: str, session: SessionIndex | None = None) -> TierResult:
        # No graph exists for uploaded docs — fall back to hybrid retrieval
        if session is not None:
            import time as _time
            from app.tiers.hybrid import _load_indexes, _dense_retrieve, _bm25_retrieve, _rrf, _rerank, LOW_MEMORY
            _, _, _, embed_model, rerank_model = _load_indexes()
            t0 = _time.perf_counter()
            if LOW_MEMORY or embed_model is None:
                bm25_hits = _bm25_retrieve(question, session.bm25, 8)
                evidence_chunks = [session.chunks[i] for i, _ in bm25_hits if i < len(session.chunks)]
            else:
                dense_hits = _dense_retrieve(question, session.faiss_index, embed_model, session.chunks, 20)
                bm25_hits  = _bm25_retrieve(question, session.bm25, 20)
                merged     = _rrf(dense_hits, bm25_hits)
                candidates = [session.chunks[i] for i in merged[:40] if i < len(session.chunks)]
                evidence_chunks = _rerank(question, candidates, rerank_model, 8)
            context, _ = _build_evidence([c["chunk_id"] for c in evidence_chunks], session.chunk_index)
            prompt = f"<graph_evidence>\n{context}\n</graph_evidence>\n\nQuestion: {question}"
            answer, in_tok, out_tok = llm(GEN_SYSTEM, prompt, max_tokens=1024)
            return TierResult(
                tier=self.name,
                answer=answer,
                citations=[Citation(doc_id=c["doc_id"], snippet=c["text"][:200]) for c in evidence_chunks],
                cost_usd=cost(in_tok, out_tok),
                latency_ms=round((_time.perf_counter() - t0) * 1000, 1),
                num_llm_calls=1,
                notes=f"session_fallback=hybrid_retrieval model={MODEL}",
            )

        graph       = _load_graph()
        chunk_index = _load_chunk_index()

        t0 = time.perf_counter()

        # Step 1: entity extraction (1 LLM call)
        entities   = _extract_entities(question)
        seed_nodes = list(dict.fromkeys(n for e in entities for n in _fuzzy_match(e, graph)))
        notes_parts = [f"entities={entities}", f"seed_nodes={seed_nodes[:6]}"]

        # Step 2: BFS + chunk collection
        if seed_nodes:
            subgraph_nodes = _bfs_subgraph(seed_nodes, graph, MAX_HOPS)
            chunk_ids      = _collect_chunk_ids(subgraph_nodes, graph)
        else:
            tokens    = question.lower().split()
            chunk_ids = []
            for n, data in graph.nodes(data=True):
                if any(tok in n.lower() for tok in tokens):
                    chunk_ids.extend(data.get("chunk_ids", []))
            subgraph_nodes = set()

        notes_parts.append(f"subgraph_nodes={len(subgraph_nodes)} evidence_chunks={len(chunk_ids)}")
        context, evidence_chunks = _build_evidence(chunk_ids, chunk_index)

        if not evidence_chunks:
            return TierResult(
                tier=self.name,
                answer="The knowledge graph contains no evidence for this question. Consider hybrid or agentic tier.",
                citations=[],
                cost_usd=0.0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                num_llm_calls=1,
                grounded=False,
                notes=" | ".join(notes_parts),
            )

        # Step 3: grounded generation (1 LLM call)
        prompt = f"<graph_evidence>\n{context}\n</graph_evidence>\n\nQuestion: {question}"
        answer, in_tok, out_tok = llm(GEN_SYSTEM, prompt, max_tokens=1024)
        latency_ms = (time.perf_counter() - t0) * 1000

        return TierResult(
            tier=self.name,
            answer=answer,
            citations=[Citation(doc_id=c["doc_id"], snippet=c["text"][:200]) for c in evidence_chunks],
            cost_usd=cost(in_tok, out_tok),
            latency_ms=round(latency_ms, 1),
            num_llm_calls=2,
            notes=" | ".join(notes_parts) + f" | model={MODEL} in={in_tok} out={out_tok}",
        )
