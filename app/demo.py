"""No-server demo:  python -m app.demo

Shows router v1 (heuristic) vs v2 (LLM-judge) side by side.
Tiers run with real indexes when available; falls back gracefully if not built yet.
"""
import os

from app.router import AdaptiveRouter
from app.tiers.base import get_tier
import app.tiers.long_context, app.tiers.hybrid, app.tiers.graph_rag, app.tiers.agentic  # noqa: F401,E401

SAMPLES = [
    "What was Apple's total net revenue in FY2024?",
    "List NVIDIA's main reportable business segments.",
    "Compare Apple's disclosed supply-chain risks with the revenue miss its CFO discussed on the Q3 call.",
    "How did JPM's stated rate-sensitivity assumptions relate to its net interest income guidance and the actual Q2 result?",
    "What does the lunar calendar have to do with JPMorgan's tier-1 capital ratio?",
]


def main():
    v1 = AdaptiveRouter()

    # v2 only if API key is available
    v2 = None
    if os.getenv("ANTHROPIC_API_KEY"):
        from app.router_v2 import RouterV2
        v2 = RouterV2()

    for q in SAMPLES:
        d1 = v1.route(q)
        print(f"\nQ: {q}")
        print(f"  v1 -> {d1.tier.value:<14}  hops~{d1.estimated_hops}  | {d1.reasoning}")
        if v2:
            d2 = v2.route(q)
            conf = d2.features.get("confidence", "?")
            match = "✓" if d2.tier == d1.tier else "≠"
            print(f"  v2 -> {d2.tier.value:<14}  conf={conf:.2f}     {match} | {d2.reasoning}")

        # Run the v1-chosen tier to show cost/latency
        r = get_tier(d1.tier).run(q)
        print(f"       cost=${r.cost_usd:.5f}  latency={r.latency_ms:.0f}ms  llm_calls={r.num_llm_calls}")


if __name__ == "__main__":
    main()
