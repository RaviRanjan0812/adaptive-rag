"""Head-to-head benchmark.

  python -m eval.benchmark --queries eval/queries.jsonl [--skip-tiers] [--save-results eval/results.json]

Tables produced
---------------
1. Per-tier quality × cost × latency
2. Router v1 vs v2: routing accuracy + cost + faithfulness
3. Agentic failure-mode instrumentation
4. HEADLINE FINDING: router vs always-agentic

Exit codes
----------
0  all quality gates pass
1  faithfulness regression: router_v2 mean faith < FAITH_GATE threshold
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from app.observability import parse_agentic_notes
from app.router import AdaptiveRouter
from app.router_v2 import RouterV2
from app.schemas import TierName
from app.tiers.base import get_tier
from eval.metrics import (
    faithfulness,
    gold_tier,
    retrieval_precision_recall,
    routing_accuracy,
)
import app.tiers.long_context, app.tiers.hybrid, app.tiers.graph_rag, app.tiers.agentic  # noqa: F401,E401

FAITH_GATE = 0.60      # CI regression gate: exit(1) if mean router_v2 faith drops below this
FAITH_GATE_ENV = "FAITH_GATE"   # override via env var in CI


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _mean(xs: list) -> float:
    return statistics.mean(xs) if xs else 0.0


def _median(xs: list) -> float:
    return statistics.median(xs) if xs else 0.0


# --------------------------------------------------------------------------- #
# Per-tier table
# --------------------------------------------------------------------------- #

def run_tier_table(queries: list[dict]) -> tuple[dict, list]:
    """Returns (agg dict, list of per-query result records for save-results)."""
    agg: dict = defaultdict(lambda: defaultdict(list))
    agentic_loops, agentic_over, agentic_bad = [], [], []
    records: list[dict] = []

    for item in queries:
        gold_ids = item.get("gold_doc_ids", [])
        rec: dict = {
            "question_id": item.get("id", ""),
            "question": item["question"],
            "hop_type": item.get("hop_type", ""),
            "difficulty": item.get("difficulty", ""),
            "gold_tier": item.get("gold_tier", ""),
            "tier_results": {},
        }

        for tier in TierName:
            res = get_tier(tier).run(item["question"])
            p, r = retrieval_precision_recall([c.doc_id for c in res.citations], gold_ids)
            faith = faithfulness(res, item.get("gold_answer", ""))
            agg[tier]["faithfulness"].append(faith)
            agg[tier]["precision"].append(p)
            agg[tier]["recall"].append(r)
            agg[tier]["cost"].append(res.cost_usd)
            agg[tier]["latency"].append(res.latency_ms)
            rec["tier_results"][tier.value] = {
                "faith": faith, "precision": p, "recall": r,
                "cost_usd": res.cost_usd, "latency_ms": res.latency_ms,
                "num_llm_calls": res.num_llm_calls,
            }
            if tier == TierName.agentic:
                n = parse_agentic_notes(res.notes or "")
                agentic_loops.append(n.get("loops", 0))
                agentic_over.append(n.get("over_retrieval", 0))
                agentic_bad.append(n.get("bad_retrieval", 0))
                rec["agentic_notes"] = n

        records.append(rec)

    print(f"\n{'tier':<16}{'faith':>8}{'prec':>8}{'recall':>8}{'cost$':>11}{'p50ms':>9}")
    print("-" * 60)
    for tier in TierName:
        a = agg[tier]
        print(
            f"{tier.value:<16}{_mean(a['faithfulness']):>8.2f}"
            f"{_mean(a['precision']):>8.2f}{_mean(a['recall']):>8.2f}"
            f"{_mean(a['cost']):>11.5f}{_median(a['latency']):>9.1f}"
        )

    if agentic_loops:
        n = len(agentic_over)
        print("\n--- AGENTIC FAILURE-MODE INSTRUMENTATION ---")
        print(f"avg retrieval loops : {_mean(agentic_loops):.2f}")
        print(f"over-retrieval rate : {sum(1 for x in agentic_over if x > 0) / n:.0%}  "
              f"(loops that added 0 new chunks)")
        print(f"bad-retrieval rate  : {sum(1 for x in agentic_bad if x > 0) / n:.0%}  "
              f"(still insufficient after max loops)")

    return agg, records


# --------------------------------------------------------------------------- #
# Router v1 vs v2 comparison
# --------------------------------------------------------------------------- #

def run_router_comparison(queries: list[dict], records: list[dict]) -> tuple:
    v1 = AdaptiveRouter()
    v2 = RouterV2()

    v1_decisions, v2_decisions = [], []
    v1_results, v2_results = [], []
    v1_costs, v2_costs = [], []
    v2_fallbacks = 0

    for item, rec in zip(queries, records):
        gold = gold_tier(item)

        d1 = v1.route(item["question"])
        r1 = get_tier(d1.tier).run(item["question"])
        v1_decisions.append((d1.tier, gold))
        v1_results.append(r1)
        v1_costs.append(r1.cost_usd)
        rec["router_v1_tier"] = d1.tier.value
        rec["estimated_hops"] = d1.estimated_hops

        d2 = v2.route(item["question"])
        if d2.features.get("v2_fallback"):
            v2_fallbacks += 1
        r2 = get_tier(d2.tier).run(item["question"])
        v2_decisions.append((d2.tier, gold))
        v2_results.append(r2)
        v2_costs.append(r2.cost_usd)
        rec["router_v2_tier"] = d2.tier.value
        rec["v2_confidence"] = d2.features.get("confidence", None)

    v1_acc = routing_accuracy(v1_decisions)
    v2_acc = routing_accuracy(v2_decisions)
    v1_faith = _mean([faithfulness(r) for r in v1_results])
    v2_faith = _mean([faithfulness(r) for r in v2_results])
    v1_cost = _mean(v1_costs)
    v2_cost = _mean(v2_costs)

    print(f"\n{'router':<10}{'accuracy':>10}{'faith':>8}{'avg_cost$':>12}{'p50ms':>9}")
    print("-" * 49)
    for label, acc, faith, cost, results in [
        ("v1", v1_acc, v1_faith, v1_cost, v1_results),
        ("v2", v2_acc, v2_faith, v2_cost, v2_results),
    ]:
        print(
            f"{label:<10}{acc:>10.0%}{faith:>8.2f}"
            f"{cost:>12.5f}{_median([r.latency_ms for r in results]):>9.1f}"
        )
    if v2_fallbacks:
        print(f"  v2 fallbacks to v1: {v2_fallbacks}/{len(queries)}")

    print("\nTier distribution:")
    for label, decisions in [("v1", v1_decisions), ("v2", v2_decisions)]:
        counts: dict[str, int] = defaultdict(int)
        for chosen, _ in decisions:
            counts[chosen.value] += 1
        dist = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  {label}: {dist}")

    return v1_acc, v2_acc, v1_cost, v2_cost, v1_faith, v2_faith


# --------------------------------------------------------------------------- #
# Headline finding
# --------------------------------------------------------------------------- #

def print_headline(agg: dict, v1_faith: float, v2_faith: float,
                   v1_cost: float, v2_cost: float) -> float:
    """Returns router_v2 mean faithfulness for the CI gate."""
    always_agentic_cost = _mean(agg[TierName.agentic]["cost"])
    always_agentic_faith = _mean(agg[TierName.agentic]["faithfulness"])

    print("\n--- HEADLINE FINDING (router vs always-agentic) ---")
    for label, faith, cost in [
        ("router_v1", v1_faith, v1_cost),
        ("router_v2", v2_faith, v2_cost),
        ("always_agentic", always_agentic_faith, always_agentic_cost),
    ]:
        saving = ""
        if label != "always_agentic" and always_agentic_cost:
            pct = 100 * (1 - cost / always_agentic_cost)
            saving = f"  ({pct:+.0f}% cost vs agentic)"
        print(f"  {label:<18} faith={faith:.3f}  avg_cost=${cost:.5f}{saving}")
    return v2_faith


# --------------------------------------------------------------------------- #
# CI regression gate
# --------------------------------------------------------------------------- #

def check_gate(v2_faith: float, threshold: float) -> bool:
    import os
    gate = float(os.getenv(FAITH_GATE_ENV, threshold))
    if v2_faith < gate:
        print(f"\n[CI GATE FAILED] router_v2 faithfulness {v2_faith:.3f} < threshold {gate:.3f}")
        return False
    print(f"\n[CI GATE PASSED] router_v2 faithfulness {v2_faith:.3f} >= threshold {gate:.3f}")
    return True


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run(path: str, skip_tiers: bool = False, save_results: str | None = None):
    queries = load(path)
    print(f"Loaded {len(queries)} queries from {path}")

    agg: dict = defaultdict(lambda: defaultdict(list))
    records: list[dict] = [
        {"question_id": q.get("id",""), "question": q["question"],
         "hop_type": q.get("hop_type",""), "difficulty": q.get("difficulty",""),
         "gold_tier": q.get("gold_tier",""), "tier_results": {}}
        for q in queries
    ]

    if not skip_tiers:
        print("\n=== PER-TIER QUALITY × COST × LATENCY ===")
        agg, records = run_tier_table(queries)
    else:
        for item in queries:
            res = get_tier(TierName.agentic).run(item["question"])
            agg[TierName.agentic]["cost"].append(res.cost_usd)
            agg[TierName.agentic]["faithfulness"].append(faithfulness(res))

    print("\n=== ROUTER v1 vs v2 ===")
    v1_acc, v2_acc, v1_cost, v2_cost, v1_faith, v2_faith = run_router_comparison(
        queries, records
    )

    v2_faith_final = print_headline(agg, v1_faith, v2_faith, v1_cost, v2_cost)

    if save_results:
        Path(save_results).write_text(json.dumps(records, indent=2))
        print(f"\nResults saved to {save_results}")

    gate_passed = check_gate(v2_faith_final, FAITH_GATE)
    return gate_passed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="eval/queries.jsonl")
    ap.add_argument("--skip-tiers", action="store_true")
    ap.add_argument("--save-results", metavar="PATH",
                    help="Write per-query results JSON (for crossover analysis).")
    args = ap.parse_args()
    passed = run(args.queries, skip_tiers=args.skip_tiers, save_results=args.save_results)
    sys.exit(0 if passed else 1)
