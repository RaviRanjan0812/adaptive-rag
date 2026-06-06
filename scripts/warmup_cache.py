"""Pre-populate the demo-mode answer cache by hitting every example question once.

  python scripts/warmup_cache.py --api http://localhost:8000 --queries eval/queries.jsonl
"""
import argparse
import json
import time

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--queries", default="eval/queries.jsonl")
    ap.add_argument("--router", default="v2", choices=["v1", "v2"])
    ap.add_argument("--delay", type=float, default=1.0, help="Seconds between requests.")
    args = ap.parse_args()

    with open(args.queries) as f:
        queries = [json.loads(l) for l in f if l.strip()]

    print(f"Warming {len(queries)} queries against {args.api} (router={args.router})\n")
    for i, q in enumerate(queries, 1):
        question = q["question"]
        try:
            r = requests.post(
                f"{args.api}/query?router={args.router}",
                json={"question": question},
                timeout=90,
            )
            r.raise_for_status()
            data = r.json()
            tier = data["result"]["tier"]
            cost = data["result"]["cost_usd"]
            print(f"  [{i:02d}/{len(queries)}] {tier:<14} ${cost:.5f}  {question[:60]}")
        except Exception as exc:
            print(f"  [{i:02d}/{len(queries)}] FAILED: {exc}")
        time.sleep(args.delay)

    print("\nWarm-up complete.")


if __name__ == "__main__":
    main()
