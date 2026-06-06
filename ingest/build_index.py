"""Chunk + embed the corpus, build FAISS + BM25, persist to data/index/.

Run after fetch_filings.py:
    python -m ingest.build_index

Outputs
-------
data/index/chunks.jsonl           all chunks with metadata
data/index/faiss.index            dense FAISS index (IVFFlat, inner-product)
data/index/bm25.pkl               serialised BM25Okapi object
data/index/corpus.txt             concatenated full corpus (for long-context tier)
"""

import json
import pickle
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

RAW_DIR = Path("data/raw")
INDEX_DIR = Path("data/index")
MANIFEST = RAW_DIR / "manifest.jsonl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast, free
CHUNK_TOKENS = 400       # target words per chunk (proxy; not true BPE tokens)
CHUNK_OVERLAP = 60       # overlap in words


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

# SEC section headers — used as natural split points
_SECTION_RE = re.compile(
    r"(?i)\b(ITEM\s+\d+[A-Z]?\.?\s+[A-Z][A-Z\s,;:]{4,})",
)


def _word_chunks(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return chunks


def chunk_document(doc_id: str, ticker: str, form: str, period: str, text: str) -> list[dict]:
    """Split on ITEM headers first, then by word count within each section."""
    sections = _SECTION_RE.split(text)
    chunks = []
    section_name = "preamble"
    idx = 0

    i = 0
    while i < len(sections):
        seg = sections[i].strip()
        if _SECTION_RE.match(seg):
            section_name = seg[:80]
            i += 1
            continue
        for chunk_text in _word_chunks(seg, CHUNK_TOKENS, CHUNK_OVERLAP):
            if len(chunk_text.split()) < 30:
                continue
            chunk_id = f"{doc_id}#{section_name.lower().replace(' ', '-')}#{idx}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "ticker": ticker,
                "form": form,
                "period": period,
                "section": section_name,
                "text": chunk_text,
            })
            idx += 1
        i += 1
    return chunks


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"{MANIFEST} not found — run python -m ingest.fetch_filings first."
        )

    manifest = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]

    # 1. Chunk all documents
    print("Chunking documents...")
    all_chunks: list[dict] = []
    corpus_parts: list[str] = []

    for rec in manifest:
        path = Path(rec["path"])
        if not path.exists():
            print(f"  WARN: {path} missing, skipping")
            continue
        text = path.read_text(encoding="utf-8")
        corpus_parts.append(f"=== {rec['doc_id']} ===\n{text}\n")
        chunks = chunk_document(
            rec["doc_id"], rec["ticker"], rec["form"], rec["period"], text
        )
        all_chunks.extend(chunks)
        print(f"  {rec['doc_id']}: {len(chunks)} chunks")

    if not all_chunks:
        raise RuntimeError("No chunks produced — check your raw corpus.")

    chunks_path = INDEX_DIR / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")
    print(f"\nTotal chunks: {len(all_chunks)} -> {chunks_path}")

    # 2. Write concatenated corpus for long-context tier
    corpus_path = INDEX_DIR / "corpus.txt"
    corpus_path.write_text("\n".join(corpus_parts), encoding="utf-8")
    print(f"Corpus: {corpus_path} ({corpus_path.stat().st_size // 1024} KB)")

    # 3. Embed -> FAISS
    print(f"\nEmbedding with {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    # IVFFlat with inner-product (cosine after normalize) — good for ~10k-100k chunks
    nlist = max(1, min(256, len(all_chunks) // 10))
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = 16

    faiss_path = INDEX_DIR / "faiss.index"
    faiss.write_index(index, str(faiss_path))
    print(f"FAISS: {faiss_path} ({index.ntotal} vectors, dim={dim})")

    # 4. BM25
    print("Building BM25...")
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_path = INDEX_DIR / "bm25.pkl"
    with bm25_path.open("wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25: {bm25_path}")

    print("\nIndex build complete.")


if __name__ == "__main__":
    main()
