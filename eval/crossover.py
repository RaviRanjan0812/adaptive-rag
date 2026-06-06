"""Crossover analysis: when does each tier earn its cost?

Reads a completed benchmark results JSON (produced by benchmark.py --save-results)
and produces the headline finding table + per-difficulty/per-hop-type breakdowns.

  python -m eval.crossover --results eval/results.json

Key finding this module tries to falsify:
  "The router matches always-agentic faithfulness within N% at M% lower cost.
   GraphRAG only beats hybrid above K hops. Long-context wins on easy single-hop."
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- #
# Per-stratum analysis
# --------------------------------------------------------------------------- #

def _mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def _stdev(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def crossover_table(records: list[dict]) -> None:
    """
    records: list of dicts with keys:
      question_id, hop_type, difficulty, gold_tier,
      tier_results: {tier_name: {faith, cost_usd, latency_ms, precision, recall}}
      router_v1_tier, router_v2_tier
    """
    tiers = ["long_context", "hybrid", "graph_rag", "agentic"]

    # ------------------------------------------------------------------ #
    # 1. Global tier comparison
    # ------------------------------------------------------------------ #
    global_agg: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        for tier, metrics in rec.get("tier_results", {}).items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    global_agg[tier][k].append(v)

    print("=== GLOBAL TIER COMPARISON ===")
    print(f"{'tier':<16}{'faith':>8}{'±':>6}{'prec':>7}{'recall':>8}{'cost$':>10}{'p50ms':>9}")
    print("-" * 64)
    for t in tiers:
        a = global_agg[t]
        print(
            f"{t:<16}"
            f"{_mean(a.get('faith',[0])):>8.3f}"
            f"{_stdev(a.get('faith',[0])):>6.3f}"
            f"{_mean(a.get('precision',[0])):>7.3f}"
            f"{_mean(a.get('recall',[0])):>8.3f}"
            f"{_mean(a.get('cost_usd',[0])):>10.5f}"
            f"{_mean(a.get('latency_ms',[0])):>9.0f}"
        )

    # ------------------------------------------------------------------ #
    # 2. Crossover by hop_type
    # ------------------------------------------------------------------ #
    print("\n=== FAITH × COST BY HOP TYPE ===")
    hop_agg: dict[str, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for rec in records:
        ht = rec.get("hop_type", "unknown")
        for tier, metrics in rec.get("tier_results", {}).items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    hop_agg[ht][tier][k].append(v)

    for hop_type, tier_data in sorted(hop_agg.items()):
        print(f"\n  hop_type={hop_type}")
        print(f"  {'tier':<16}{'faith':>8}{'cost$':>10}  winner")
        best_faith_tier = max(
            tier_data.keys(),
            key=lambda t: _mean(tier_data[t].get("faith", [0])),
        )
        cheapest_tier = min(
            tier_data.keys(),
            key=lambda t: _mean(tier_data[t].get("cost_usd", [float("inf")])),
        )
        for t in tiers:
            if t not in tier_data:
                continue
            a = tier_data[t]
            tags = []
            if t == best_faith_tier:
                tags.append("best-faith")
            if t == cheapest_tier:
                tags.append("cheapest")
            print(
                f"  {t:<16}"
                f"{_mean(a.get('faith',[0])):>8.3f}"
                f"{_mean(a.get('cost_usd',[0])):>10.5f}"
                f"  {','.join(tags)}"
            )

    # ------------------------------------------------------------------ #
    # 3. Crossover by difficulty
    # ------------------------------------------------------------------ #
    print("\n=== FAITH × COST BY DIFFICULTY ===")
    diff_agg: dict[str, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for rec in records:
        diff = rec.get("difficulty", "unknown")
        for tier, metrics in rec.get("tier_results", {}).items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    diff_agg[diff][tier][k].append(v)

    for difficulty in ["easy", "medium", "hard"]:
        tier_data = diff_agg.get(difficulty, {})
        if not tier_data:
            continue
        print(f"\n  difficulty={difficulty}")
        print(f"  {'tier':<16}{'faith':>8}{'cost$':>10}")
        for t in tiers:
            if t not in tier_data:
                continue
            a = tier_data[t]
            print(
                f"  {t:<16}"
                f"{_mean(a.get('faith',[0])):>8.3f}"
                f"{_mean(a.get('cost_usd',[0])):>10.5f}"
            )

    # ------------------------------------------------------------------ #
    # 4. Router accuracy
    # ------------------------------------------------------------------ #
    print("\n=== ROUTER ACCURACY (gold_tier vs chosen) ===")
    for router_key in ("router_v1_tier", "router_v2_tier"):
        total = correct = 0
        for rec in records:
            gold = rec.get("gold_tier")
            chosen = rec.get(router_key)
            if gold and chosen:
                total += 1
                if chosen == gold:
                    correct += 1
        label = router_key.replace("_tier", "")
        acc = correct / total if total else 0.0
        print(f"  {label}: {acc:.0%}  ({correct}/{total})")

    # ------------------------------------------------------------------ #
    # 5. Falsifiable headline
    # ------------------------------------------------------------------ #
    print("\n=== HEADLINE FINDING ===")
    agentic_faith = _mean(global_agg["agentic"].get("faith", [0]))
    agentic_cost = _mean(global_agg["agentic"].get("cost_usd", [0]))

    for router_key, label in [("router_v1_tier", "router_v1"), ("router_v2_tier", "router_v2")]:
        router_costs, router_faiths = [], []
        for rec in records:
            tier = rec.get(router_key)
            if tier and tier in rec.get("tier_results", {}):
                m = rec["tier_results"][tier]
                router_costs.append(m.get("cost_usd", 0))
                router_faiths.append(m.get("faith", 0))
        if not router_costs:
            continue
        rf = _mean(router_faiths)
        rc = _mean(router_costs)
        faith_gap = abs(rf - agentic_faith)
        cost_saving = 100 * (1 - rc / agentic_cost) if agentic_cost else 0
        print(
            f"  {label}: faith={rf:.3f} (gap vs agentic={faith_gap:+.3f})  "
            f"cost=${rc:.5f} ({cost_saving:.0f}% cheaper than always-agentic)"
        )

    # GraphRAG vs hybrid crossover
    hybrid_faith_by_hops: dict[int, list] = defaultdict(list)
    graph_faith_by_hops: dict[int, list] = defaultdict(list)
    for rec in records:
        hops = rec.get("estimated_hops", 1)
        tr = rec.get("tier_results", {})
        if "hybrid" in tr:
            hybrid_faith_by_hops[hops].append(tr["hybrid"].get("faith", 0))
        if "graph_rag" in tr:
            graph_faith_by_hops[hops].append(tr["graph_rag"].get("faith", 0))

    print("\n  GraphRAG vs hybrid faithfulness by hop count:")
    for h in sorted(set(hybrid_faith_by_hops) | set(graph_faith_by_hops)):
        hf = _mean(hybrid_faith_by_hops.get(h, [float("nan")]))
        gf = _mean(graph_faith_by_hops.get(h, [float("nan")]))
        winner = "graph_rag" if gf > hf else "hybrid   "
        print(f"    hops={h}: hybrid={hf:.3f}  graph_rag={gf:.3f}  -> {winner}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to results JSON from benchmark.py")
    args = ap.parse_args()
    records = load(args.results)
    print(f"Loaded {len(records)} benchmark records from {args.results}\n")
    crossover_table(records)


if __name__ == "__main__":
    main()
