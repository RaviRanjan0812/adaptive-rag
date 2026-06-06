# Adaptive Retrieval Router for Financial Filings

An adaptive RAG system that, **per query**, routes among four retrieval strategies —
long-context, hybrid (dense + lexical), GraphRAG, and self-reflective agentic —
and a rigorous benchmark that **proves** the routing policy on quality, cost, and latency.

> **The real problem.** Equity analysts and compliance teams burn hours manually
> cross-referencing 10-Ks, 10-Qs, and earnings-call transcripts. Generic AI fails two
> ways: it hallucinates on high-stakes cross-document questions, or it answers correctly
> but runs everything through an expensive agentic pipeline that doesn't scale.
> This system gives accurate, **cited**, multi-hop answers *and* keeps cost low by
> invoking heavy retrieval only when a question actually needs it.
>
> *Research-acceleration tool, not financial advice — trust comes from citations.*

## Why a router (not one pipeline)

A naive RAG query is cheap; an agentic one doing the same job costs ~10x more and is
several seconds slower. Most production systems overspend by sending every query
through the heaviest pipeline. The skill — and the thing this project demonstrates — is
matching query difficulty to pipeline cost, and **measuring** that you got it right.

## Architecture

```
question ─▶ AdaptiveRouter ─┬─▶ long_context   (no retrieval; corpus fits context)
                            ├─▶ hybrid          (FAISS + BM25 + RRF + rerank)
                            ├─▶ graph_rag       (entity-graph multi-hop, relational Qs)
                            └─▶ agentic         (plan→retrieve→critique→re-retrieve)
                                     │
                                     ▼
                       TierResult {answer, citations, cost_usd, latency_ms, grounded}
```

Every tier returns the same typed `TierResult`, so the benchmark can score them head-to-head.

## Quickstart

```bash
pip install -r requirements.txt

# Set API keys
export ANTHROPIC_API_KEY=sk-ant-...
export SEC_USER_AGENT="Your Name your@email.com"

# 1) Fetch corpus (AAPL, JPM, NVDA — 10-K + 10-Q each)
python -m ingest.fetch_filings

# 2) Build FAISS + BM25 indexes + concatenated corpus
python -m ingest.build_index

# 3) Build knowledge graph (entity extraction via Claude Haiku)
python -m ingest.build_graph

# 4) Run the API (all tiers now real, not stubs)
uvicorn app.main:app --reload
#   POST /query  { "question": "..." }   ->   route decision + answer + cost/latency

# 5) Benchmark all tiers head-to-head
python -m eval.benchmark --queries eval/queries.example.jsonl
```

The scaffold runs end-to-end immediately. Each tier is a **stub** that returns a canned
result with simulated cost/latency — replace the stubs with real retrieval one at a time.

## Project layout

```
app/
  schemas.py        typed contracts (Pydantic)
  router.py         adaptive router — v1 heuristic (v2 = LLM-judge/classifier)
  observability.py  cost + latency tracking (TODO: Langfuse traces)
  main.py           FastAPI: /query /tiers /healthz
  demo.py           no-server routing demo
  tiers/            long_context | hybrid | graph_rag | agentic  (all STUBS)
eval/
  metrics.py        retrieval P/R + faithfulness (LLM-judge TODO)
  benchmark.py      every tier over every query -> table + headline finding
  queries.example.jsonl   stratified, labeled query schema
ingest/
  fetch_filings.py  EDGAR download (stub)
  build_index.py    chunk/embed/graph build (stub)
.github/workflows/eval.yml   CI that runs the benchmark on every push
Dockerfile / docker-compose.yml   router + Qdrant + Neo4j
```

## Build roadmap

- **Phase 0 — Scaffold (this repo).** Routing, observability, eval harness, API all run on stubs. ✅
- **Phase 1 — Corpus.** EDGAR fetch (10-K/10-Q, 3 tickers). Stable `doc_id`s. Manifest. ✅
- **Phase 2 — The two tiers you know cold.** `long_context` (Claude, full corpus in-context) + `hybrid` (FAISS + BM25 + RRF + cross-encoder rerank, Claude generation). ✅
- **Phase 3 — GraphRAG.** Entity/relation extraction (Claude Haiku) → NetworkX graph (`ingest/build_graph.py`). BFS multi-hop traversal + grounded generation in `graph_rag` tier. ✅
- **Phase 4 — Agentic.** LangGraph plan→retrieve→critique→generate loop (max 3 iterations). Per-step provenance; `retrieval_loops`, `over_retrieval`, `bad_retrieval` counters surfaced in benchmark. ✅
- **Phase 5 — The router.** v2 LLM-judge (`router_v2.py`) with confidence score + v1 fallback. Benchmark produces v1 vs v2 routing accuracy, faith, cost table. LLM faithfulness judge wired into `eval/metrics.py`. ✅
- **Phase 6 — Benchmark + finding.** 50 stratified gold-labeled queries (`eval/queries.jsonl`): easy/medium/hard × single_hop/relational/multi_hop/unanswerable. Crossover analysis (`eval/crossover.py`). CI faithfulness gate (`FAITH_GATE=0.60`, exit 1 on regression). Results JSON artefact uploaded per run. ✅
- **Phase 7 — Deploy.** Production Dockerfile (models pre-baked); `docker compose up` with Langfuse self-hosted; Render + Railway one-click configs; Streamlit demo (`demo_app/`) for HF Spaces; demo-mode answer cache + per-IP rate limiter; `GET /metrics`; cache warm-up script. See `DEPLOY.md`. ✅

## The deliverables that get you hired

1. A live demo link that visibly routes per query.
2. An open, labeled benchmark dataset.
3. A results dashboard (quality × cost × latency across tiers).
4. A falsifiable headline finding, e.g. *"the router matches always-agentic faithfulness within ~2% at ~70% lower cost; GraphRAG only beats hybrid above ~3 hops."*
