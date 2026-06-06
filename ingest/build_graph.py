"""Extract entities + relations from chunks and build a NetworkX knowledge graph.

Run after build_index.py:
    python -m ingest.build_graph

Uses Gemini to extract (entity, relation, entity) triples from each chunk batch.

Outputs
-------
data/index/graph.pkl          NetworkX MultiDiGraph
data/index/graph_nodes.jsonl  node catalogue
"""

import json
import pickle
import re
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import networkx as nx

from app.llm import llm, strip_fences

INDEX_DIR   = Path("data/index")
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
GRAPH_PATH  = INDEX_DIR / "graph.pkl"
NODES_PATH  = INDEX_DIR / "graph_nodes.jsonl"

BATCH      = 2       # smaller batch → response fits in token budget
MAX_CHUNKS = 2000

EXTRACT_SYSTEM = """\
You extract a knowledge graph from SEC filing excerpts.
For the given chunk return a JSON array of triples.
Each triple: {"head": "...", "head_type": "...", "relation": "...", "tail": "...", "tail_type": "..."}

Entity types: COMPANY PERSON METRIC PRODUCT EVENT REGULATION RISK_FACTOR GEOGRAPHY
Relation examples: REPORTED EXCEEDED MISSED COMPARED_TO CAUSED IMPACTED_BY AFFILIATED_WITH
                   COMPETES_WITH ISSUED ACQUIRED DIVESTED

Rules:
- Normalise company names to tickers (Apple → AAPL, NVIDIA → NVDA, JPMorgan → JPM)
- Keep metric names specific ("revenue Q4 2024", not just "revenue")
- Return [] if no clear triples exist
- Return ONLY valid JSON, no prose
"""


def _extract_triples(chunks: list[dict], debug: bool = False) -> list[dict]:
    combined = "\n\n".join(f"[{c['chunk_id']}]\n{c['text'][:800]}" for c in chunks)
    prompt   = f"Extract triples from these filing excerpts:\n\n{combined}"
    raw, _, _ = llm(EXTRACT_SYSTEM, prompt, max_tokens=4096)
    raw = strip_fences(raw)

    if debug:
        print(f"\n    [debug] raw response: {raw[:200]!r}")

    # Find a JSON array anywhere in the response (Gemini sometimes adds preamble)
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        result = json.loads(m.group())
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"{CHUNKS_PATH} not found — run build_index.py first.")

    chunks = [json.loads(l) for l in CHUNKS_PATH.read_text().splitlines() if l.strip()]
    chunks = chunks[:MAX_CHUNKS]
    print(f"Processing {len(chunks)} chunks in batches of {BATCH}...")

    G = nx.MultiDiGraph()
    total_triples = 0

    for i in range(0, len(chunks), BATCH):
        batch   = chunks[i : i + BATCH]
        triples = _extract_triples(batch, debug=(i == 0))
        total_triples += len(triples)

        for t in triples:
            head = t.get("head", "").strip()
            tail = t.get("tail", "").strip()
            rel  = t.get("relation", "RELATED_TO").strip()
            if not head or not tail:
                continue

            for name, ntype in [(head, t.get("head_type", "UNKNOWN")),
                                 (tail, t.get("tail_type", "UNKNOWN"))]:
                if not G.has_node(name):
                    G.add_node(name, entity_type=ntype, chunk_ids=[], doc_ids=[])

            for c in batch:
                if head.lower() in c["text"].lower() or tail.lower() in c["text"].lower():
                    nd = G.nodes[head]
                    if c["chunk_id"] not in nd["chunk_ids"]:
                        nd["chunk_ids"].append(c["chunk_id"])
                    if c["doc_id"] not in nd["doc_ids"]:
                        nd["doc_ids"].append(c["doc_id"])

            G.add_edge(head, tail, relation=rel,
                       chunk_ids=[c["chunk_id"] for c in batch])

        if (i // BATCH) % 10 == 0:
            print(f"  batch {i//BATCH + 1}/{(len(chunks)-1)//BATCH + 1}"
                  f"  nodes={G.number_of_nodes()}  edges={G.number_of_edges()}"
                  f"  triples={total_triples}")
        time.sleep(0.05)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with GRAPH_PATH.open("wb") as f:
        pickle.dump(G, f)

    with NODES_PATH.open("w") as f:
        for n in G.nodes():
            f.write(json.dumps({"name": n, **G.nodes[n]}) + "\n")

    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges → {GRAPH_PATH}")


if __name__ == "__main__":
    main()
