"""FastAPI entrypoint:  uvicorn app.main:app --host 0.0.0.0 --port 8000

POST /query                      route + answer
POST /query?router=v2            use LLM-judge router
GET  /tiers                      registered tiers
GET  /healthz                    liveness probe
GET  /metrics                    Prometheus-style text
"""
from __future__ import annotations

import hashlib
import os
import time
from collections import defaultdict, deque

from fastapi import FastAPI, File, HTTPException, Query as QParam, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from app.observability import trace_query
from app.router import AdaptiveRouter
from app.router_v2 import RouterV2
from app.schemas import QueryRequest, QueryResponse
from app.tiers.base import get_tier, all_tiers
from app import session_store
from ingest.ingest_upload import build_session_index
import app.tiers.long_context, app.tiers.hybrid, app.tiers.graph_rag, app.tiers.agentic  # noqa: F401,E401

app = FastAPI(title="Adaptive Retrieval Router", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_v1 = AdaptiveRouter()
_v2 = RouterV2()

# ── Rate limiting ────────────────────────────────────────────────────────────
_RPM = int(os.getenv("RATE_LIMIT_RPM", "30"))
_rate_windows: dict[str, deque] = defaultdict(deque)


def _check_rate(ip: str) -> None:
    now = time.time()
    window = _rate_windows[ip]
    while window and window[0] < now - 60:
        window.popleft()
    if len(window) >= _RPM:
        raise HTTPException(status_code=429, detail=f"Rate limit: {_RPM} req/min per IP.")
    window.append(now)


# ── Demo-mode answer cache ───────────────────────────────────────────────────
_DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
_answer_cache: dict[str, QueryResponse] = {}


def _cache_key(question: str, router_ver: str) -> str:
    return hashlib.sha256(f"{router_ver}:{question.strip().lower()}".encode()).hexdigest()


# ── Metrics counters ─────────────────────────────────────────────────────────
_metrics: dict[str, float] = defaultdict(float)


def _record(tier: str, cost: float, latency: float) -> None:
    _metrics[f"queries_total{{tier={tier!r}}}"] += 1
    _metrics[f"cost_usd_total{{tier={tier!r}}}"] += cost
    _metrics[f"latency_ms_sum{{tier={tier!r}}}"] += latency


# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Accept a .txt or .pdf file, build an in-memory index, return a session_id.

    Pass the session_id in subsequent POST /query requests to query the uploaded
    document instead of the pre-built SEC filings index.
    """
    allowed = {".txt", ".md", ".pdf"}
    suffix  = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed}")

    raw = await file.read()

    if suffix == ".pdf":
        try:
            import pypdf
            import io
            reader = pypdf.PdfReader(io.BytesIO(raw))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise HTTPException(status_code=422, detail="pypdf not installed — upload .txt files.")
    else:
        text = raw.decode("utf-8", errors="replace")

    try:
        session_id, idx = build_session_index(text, file.filename or "upload.txt")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    session_store.put(session_id, idx)
    return {
        "session_id": session_id,
        "chunks": len(idx.chunks),
        "message": "Index built. Pass session_id in POST /query to query this document.",
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok", "demo_mode": _DEMO_MODE, "rate_limit_rpm": _RPM}


@app.get("/tiers")
def tiers():
    return [t.name.value for t in all_tiers()]


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = [f"{k} {v}" for k, v in sorted(_metrics.items())]
    return "\n".join(lines)


@app.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    request: Request,
    router: str = QParam(default="v1", pattern="^v[12]$"),
):
    ip = request.client.host if request.client else "unknown"
    _check_rate(ip)

    cache_key = _cache_key(req.question, router)
    if _DEMO_MODE and cache_key in _answer_cache:
        return _answer_cache[cache_key]

    session = None
    if req.session_id:
        session = session_store.get(req.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session_id not found or expired (TTL 1 hour).")

    active_router = _v2 if router == "v2" else _v1
    decision = active_router.route(req.question)
    chosen = req.force_tier or decision.tier
    result = get_tier(chosen).run(req.question, session=session)

    response = QueryResponse(question=req.question, route=decision, result=result)

    if _DEMO_MODE:
        _answer_cache[cache_key] = response

    _record(chosen.value, result.cost_usd, result.latency_ms)

    trace_query(
        question=req.question,
        tier=chosen.value,
        router_version=router,
        routing_reason=decision.reasoning,
        answer=result.answer,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        num_llm_calls=result.num_llm_calls,
        citations=[{"doc_id": c.doc_id, "snippet": c.snippet} for c in result.citations],
        metadata={"ip_hash": hashlib.sha256(ip.encode()).hexdigest()[:8]},
    )

    return response
