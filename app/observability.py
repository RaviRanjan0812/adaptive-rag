"""Cost + latency tracking + Langfuse tracing.

Every tier run emits a Langfuse trace with:
  - input (question)
  - output (answer + citations)
  - cost_usd, latency_ms, num_llm_calls
  - tier name, router version, routing reasoning

Set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY to enable.
Tracing is a no-op when the keys are absent (local / CI runs).
"""
from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from typing import Any

# Langfuse is optional — degrades gracefully when not configured
try:
    from langfuse import Langfuse
    _lf = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
    ) if os.getenv("LANGFUSE_PUBLIC_KEY") else None
except Exception:
    _lf = None


# ── Illustrative per-1M-token prices (USD) ──────────────────────────────────
PRICING = {
    "claude-haiku-4-5-20251001": {"in": 0.80,  "out": 4.00},
    "claude-sonnet-4-6":         {"in": 3.00,  "out": 15.00},
    "gemini-flash":              {"in": 0.075, "out": 0.30},
    "cross-encoder-rerank":      {"in": 0.0,   "out": 0.0},
}


def token_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    p = PRICING.get(model, {"in": 0, "out": 0})
    return round((in_tokens / 1_000_000) * p["in"] + (out_tokens / 1_000_000) * p["out"], 6)


def simulate(base_cost: float, base_latency: float):
    """Stub helper — only used if a tier hasn't been fully implemented yet."""
    import random
    jitter = random.uniform(0.85, 1.15)
    return round(base_cost * jitter, 6), round(base_latency * jitter, 1)


@contextmanager
def track(label: str):
    start = time.perf_counter()
    yield
    print(f"[trace] {label}: {(time.perf_counter() - start) * 1000:.1f} ms")


# ── Langfuse helpers ─────────────────────────────────────────────────────────

def trace_query(
    question: str,
    tier: str,
    router_version: str,
    routing_reason: str,
    answer: str,
    cost_usd: float,
    latency_ms: float,
    num_llm_calls: int,
    citations: list[dict] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Emit a Langfuse trace for a completed query.  No-op if Langfuse is not configured."""
    if _lf is None:
        return
    try:
        trace = _lf.trace(
            name="adaptive-rag-query",
            input={"question": question},
            output={"answer": answer, "citations": citations or []},
            metadata={
                "tier": tier,
                "router": router_version,
                "routing_reason": routing_reason,
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
                "num_llm_calls": num_llm_calls,
                **(metadata or {}),
            },
        )
        trace.score(name="cost_usd", value=cost_usd)
        trace.score(name="latency_ms", value=latency_ms)
    except Exception:
        pass   # tracing must never crash the request


def trace_span(trace_id: str | None, name: str, input_: dict, output: dict,
               metadata: dict | None = None) -> None:
    """Emit a child span within an existing trace.  No-op if Langfuse is not configured."""
    if _lf is None or trace_id is None:
        return
    try:
        _lf.span(
            trace_id=trace_id,
            name=name,
            input=input_,
            output=output,
            metadata=metadata or {},
        )
    except Exception:
        pass


# ── Agentic notes parser ─────────────────────────────────────────────────────

def parse_agentic_notes(notes: str) -> dict:
    """Extract key=value pairs from an agentic TierResult.notes string."""
    result = {}
    for m in re.finditer(r"(\w+)=([^\s]+)", notes or ""):
        k, v = m.group(1), m.group(2)
        try:
            result[k] = int(v)
        except ValueError:
            result[k] = v
    return result
