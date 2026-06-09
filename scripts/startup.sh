#!/bin/bash
# startup.sh — Auto-build index on first boot if data/index is empty.
# Runs before uvicorn starts. Safe to re-run (skips if index already exists).

set -e

INDEX_DIR="data/index"
FAISS_FILE="$INDEX_DIR/faiss.index"

if [ -f "$FAISS_FILE" ]; then
    echo "✅ Index already exists — skipping ingest."
else
    echo "⚙️  No index found. Building from scratch (~10-15 min)..."

    # Step 1 — Fetch filings from SEC EDGAR
    echo "--- Step 1/3: Fetching SEC filings ---"
    python -m ingest.fetch_filings

    # Step 2 — Chunk + embed + build FAISS & BM25
    echo "--- Step 2/3: Building FAISS + BM25 index ---"
    python -m ingest.build_index

    # Step 3 — Build knowledge graph
    echo "--- Step 3/3: Building knowledge graph ---"
    python -m ingest.build_graph

    echo "✅ Index build complete."
fi

# Start the API
echo "🚀 Starting uvicorn..."
exec sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 30"
