"""Tier 2 — HYBRID.

FAISS (dense cosine) + BM25 (lexical), fused with Reciprocal Rank Fusion,
then cross-encoder reranked. Single grounded-generation Gemini call.

Requires:
  data/index/faiss.index, data/index/bm25.pkl, data/index/chunks.jsonl
  GEMINI_API_KEY env var
"""

import json
import pickle
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.llm import llm, cost, MODEL
from app.tiers.base import register, RetrievalTier
from app.schemas import TierResult, TierName, Citation
from app.session_store import SessionIndex

INDEX_DIR = Path("data/index")

EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_DENSE  = 20
TOP_BM25   = 20
TOP_RERANK = 5
RRF_K      = 60

# LOW_MEMORY=true → skip sentence-transformers & CrossEncoder (saves ~300MB RAM)
# Use BM25-only retrieval — no dense embeddings, no reranking
LOW_MEMORY = os.getenv("LOW_MEMORY", "false").lower() == "true"

SYSTEM = (
    "You are a financial analyst assistant. "
    "Answer the question using ONLY the retrieved filing excerpts below. "
    "For every factual claim, cite the chunk_id. "
    "If none of the excerpts contain the answer, say so explicitly."
)


# --------------------------------------------------------------------------- #
# Index loading
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _load_indexes() -> tuple[Any, Any, list[dict], SentenceTransformer | None, CrossEncoder | None]:
    faiss_path  = INDEX_DIR / "faiss.index"
    bm25_path   = INDEX_DIR / "bm25.pkl"
    chunks_path = INDEX_DIR / "chunks.jsonl"

    for p in (faiss_path, bm25_path, chunks_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — run python -m ingest.build_index first."
            )

    fi = faiss.read_index(str(faiss_path))
    with bm25_path.open("rb") as f:
        bm25 = pickle.load(f)
    chunks = [json.loads(l) for l in chunks_path.read_text().splitlines() if l.strip()]

    if LOW_MEMORY:
        # Skip heavy ML models to stay within 512MB RAM (Railway free tier)
        return fi, bm25, chunks, None, None

    embed_model  = SentenceTransformer(EMBED_MODEL)
    rerank_model = CrossEncoder(RERANK_MODEL)
    return fi, bm25, chunks, embed_model, rerank_model


# --------------------------------------------------------------------------- #
# Retrieval helpers (also imported by agentic tier)
# --------------------------------------------------------------------------- #

def _dense_retrieve(query: str, fi, embed_model, chunks, k: int) -> list[tuple[int, float]]:
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    distances, indices = fi.search(q_emb, k)
    return [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0]) if idx >= 0]


def _bm25_retrieve(query: str, bm25, k: int) -> list[tuple[int, float]]:
    # Strip punctuation so "NVIDIA's" matches corpus token "nvidia"
    import re as _re
    tokens = _re.findall(r"[a-z0-9]+", query.lower())
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_idx]


def _rrf(dense: list[tuple[int, float]], bm25: list[tuple[int, float]],
         k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _) in enumerate(bm25):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


def _rerank(query: str, candidates: list[dict], rerank_model: CrossEncoder,
            top_k: int) -> list[dict]:
    pairs  = [(query, c["text"]) for c in candidates]
    scores = rerank_model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]


# --------------------------------------------------------------------------- #
# Tier
# --------------------------------------------------------------------------- #

@register
class HybridTier(RetrievalTier):
    name = TierName.hybrid

    def run(self, question: str, session: SessionIndex | None = None) -> TierResult:
        fi, bm25, chunks, embed_model, rerank_model = _load_indexes()
        if session is not None:
            fi, bm25, chunks = session.faiss_index, session.bm25, session.chunks

        if LOW_MEMORY or embed_model is None:
            # BM25-only path: no dense embeddings, no reranking
            bm25_hits    = _bm25_retrieve(question, bm25, TOP_RERANK)
            final_chunks = [chunks[i] for i, _ in bm25_hits if i < len(chunks)]
        else:
            dense_hits = _dense_retrieve(question, fi, embed_model, chunks, TOP_DENSE)
            bm25_hits  = _bm25_retrieve(question, bm25, TOP_BM25)
            merged_idx   = _rrf(dense_hits, bm25_hits)
            candidates   = [chunks[i] for i in merged_idx[:TOP_DENSE + TOP_BM25] if i < len(chunks)]
            final_chunks = _rerank(question, candidates, rerank_model, TOP_RERANK)

        context = "\n\n---\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in final_chunks)
        prompt  = f"<excerpts>\n{context}\n</excerpts>\n\nQuestion: {question}"

        t0 = time.perf_counter()
        answer, in_tok, out_tok = llm(SYSTEM, prompt, max_tokens=1024)
        latency_ms = (time.perf_counter() - t0) * 1000

        citations = [
            Citation(doc_id=c["doc_id"], snippet=c["text"][:200])
            for c in final_chunks
        ]

        return TierResult(
            tier=self.name,
            answer=answer,
            citations=citations,
            cost_usd=cost(in_tok, out_tok),
            latency_ms=round(latency_ms, 1),
            num_llm_calls=1,
            notes=f"low_memory={LOW_MEMORY} bm25={len(bm25_hits)} rerank_k={TOP_RERANK} model={MODEL}",
        )
