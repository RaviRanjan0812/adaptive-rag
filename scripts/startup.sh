#!/bin/bash
# startup.sh — Download pre-built index from HF Hub on first boot if missing.
# Runs before uvicorn starts. Safe to re-run (skips if index already exists).

set -e

INDEX_DIR="data/index"
FAISS_FILE="$INDEX_DIR/faiss.index"

if [ -f "$FAISS_FILE" ]; then
    echo "✅ Index already exists — skipping download."
else
    echo "⚙️  No index found. Downloading from Hugging Face Hub..."

    mkdir -p "$INDEX_DIR"

    python - <<'PYEOF'
import os
from huggingface_hub import hf_hub_download

repo_id = "raviranjan0812/adaptive-rag-index"
files = [
    "faiss.index",
    "bm25.pkl",
    "chunks.jsonl",
    "corpus.txt",
    "graph.pkl",
    "graph_nodes.jsonl",
]

for fname in files:
    print(f"  Downloading {fname}...")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=fname,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
        local_dir="data/index",
    )
    print(f"  ✅ {fname} saved to {path}")

print("All index files downloaded.")
PYEOF

    echo "✅ Index download complete."
fi

# Start the API
echo "🚀 Starting uvicorn..."
exec sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 30"
