# Adaptive Retrieval Router for SEC Filings

An adaptive RAG system that selects, **per query**, the optimal retrieval strategy from four
tiers — long-context, hybrid dense-lexical, GraphRAG, and self-reflective agentic — backed by
a rigorous benchmark that quantifies routing quality, inference cost, and latency.

## Overview

Equity analysts and compliance teams spend significant time cross-referencing 10-Ks, 10-Qs,
and earnings transcripts. General-purpose AI either hallucinates on high-stakes multi-document
questions or routes every query through an expensive agentic pipeline regardless of complexity.

This system resolves both failure modes: it produces accurate, **fully-cited** answers and
keeps inference cost low by matching each query to the lightest pipeline capable of answering
it correctly.

> Research acceleration tool — not financial advice. Every claim is traceable to a source document.

## Why adaptive routing

A single-hop factual lookup costs ~$0.001 and completes in under two seconds through the
hybrid tier. The same question routed through the agentic tier costs ~10× more and takes
several seconds longer. Most production systems overspend because they apply the heaviest
pipeline uniformly. The core contribution of this project is a learned routing policy that
matches query complexity to pipeline cost — and a benchmark that proves the policy works.

## Architecture

```
question ─▶ AdaptiveRouter ─┬─▶ long_context   (no retrieval; full corpus in-context)
                            ├─▶ hybrid          (FAISS + BM25 + RRF + cross-encoder rerank)
                            ├─▶ graph_rag       (entity-graph BFS, multi-hop relational Qs)
                            └─▶ agentic         (plan → retrieve → critique → re-retrieve)
                                     │
                                     ▼
                       TierResult {answer, citations, cost_usd, latency_ms, grounded}
```

Every tier returns the same typed `TierResult`, enabling head-to-head benchmarking across
quality, cost, and latency dimensions.

## Retrieval tiers

| Tier | Strategy | Best for |
|---|---|---|
| `long_context` | Full corpus in Gemini context window | Short, self-contained, single-entity questions |
| `hybrid` | FAISS dense + BM25 lexical, fused with RRF, cross-encoder reranked | Single-hop factual lookups |
| `graph_rag` | Entity extraction → NetworkX graph → BFS multi-hop traversal | Relational questions across multiple entities |
| `agentic` | LangGraph plan→retrieve→critique loop (max 3 iterations) | Open-ended multi-step synthesis |

## Quickstart

**Prerequisites:** Python 3.10+, a [Google AI Studio](https://aistudio.google.com) API key,
and an SEC EDGAR user-agent string (`"Name email"` per SEC policy).

```bash
pip install -r requirements.txt

# Environment
export GEMINI_API_KEY=AIza...
export SEC_USER_AGENT="Your Name your@email.com"

# 1. Fetch SEC filings — AAPL, JPM, NVDA (10-K + 10-Q each)
python -m ingest.fetch_filings

# 2. Build FAISS + BM25 indexes and concatenated corpus
python -m ingest.build_index

# 3. Build knowledge graph via Gemini Flash entity extraction
python -m ingest.build_graph

# 4. Start the API
uvicorn app.main:app --reload
#    POST /query  {"question": "..."}  →  routing decision + answer + cost/latency

# 5. Run the full benchmark across all tiers
python -m eval.benchmark --queries eval/queries.example.jsonl
```

## Streamlit demo

```bash
cd demo_app
streamlit run streamlit_app.py
```

Requires the API to be running on `http://localhost:8000`. Opens at `http://localhost:8501`.

## Project layout

```
app/
  main.py           FastAPI application — /query  /tiers  /healthz  /metrics
  schemas.py        Shared Pydantic contracts (QueryRequest, TierResult, RouteDecision)
  router.py         Router v1 — heuristic (query features → tier)
  router_v2.py      Router v2 — Gemini LLM-judge with confidence score; falls back to v1
  observability.py  Cost + latency tracking; Langfuse trace emission
  tiers/
    long_context.py Full-corpus Gemini generation
    hybrid.py       FAISS + BM25 + RRF + cross-encoder rerank + Gemini generation
    graph_rag.py    Entity graph BFS + grounded Gemini generation
    agentic.py      LangGraph self-reflective loop

ingest/
  fetch_filings.py  EDGAR HTTPS download — 10-K and 10-Q for each ticker
  build_index.py    Chunking, sentence-transformer embeddings, FAISS + BM25 index build
  build_graph.py    Gemini Flash entity/relation extraction → NetworkX graph

eval/
  benchmark.py      Runs every tier against every query; produces cost/quality/latency table
  metrics.py        Retrieval P/R and LLM faithfulness judge
  crossover.py      Per-query tier crossover analysis
  queries.jsonl     50 stratified gold-labeled queries (easy/medium/hard × hop type)

demo_app/
  streamlit_app.py  Interactive demo UI
scripts/
  warmup_cache.py   Pre-populates the demo-mode answer cache
```

## API reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/query` | Route and answer a question. Body: `{"question": "...", "force_tier": null}` |
| `POST` | `/query?router=v2` | Use the LLM-judge router instead of the heuristic |
| `GET` | `/tiers` | List registered retrieval tiers |
| `GET` | `/healthz` | Liveness probe |
| `GET` | `/metrics` | Prometheus-style cost and latency counters |

## Benchmark

The evaluation harness runs all four tiers against 50 stratified, gold-labeled queries
spanning three difficulty levels (easy / medium / hard) and four hop types
(single-hop / relational / multi-hop / unanswerable). Key outputs:

- Per-tier faithfulness score (LLM judge), cost, and latency
- Router v1 vs v2 accuracy comparison
- Crossover analysis: which tier wins at each difficulty × hop-type combination
- CI faithfulness gate (`FAITH_GATE=0.60`) — pipeline exits with code 1 on regression

## Deployment

See [DEPLOY.md](DEPLOY.md) for full instructions covering Docker Compose (local),
Render, Railway, and Hugging Face Spaces.

## Implementation history

| Phase | Description |
|---|---|
| 0 | Scaffold — routing, eval harness, and API architecture |
| 1 | Corpus — EDGAR fetch for AAPL, JPM, NVDA with stable `doc_id`s |
| 2 | Long-context and hybrid tiers — FAISS + BM25 + RRF + rerank + Gemini generation |
| 3 | GraphRAG tier — Gemini Flash entity extraction, NetworkX BFS traversal |
| 4 | Agentic tier — LangGraph plan→retrieve→critique loop with per-step provenance |
| 5 | Router v2 — Gemini LLM-judge with confidence score; v1 vs v2 benchmark comparison |
| 6 | Benchmark — 50 gold-labeled queries, crossover analysis, CI faithfulness gate |
| 7 | Deploy — production Dockerfile, Langfuse tracing, Streamlit demo, rate limiting |
