"""Build an in-memory index from raw text (uploaded by the user).

Returns a SessionIndex ready to drop into the session store.
Reuses the same chunking logic as build_index.py.
"""
from __future__ import annotations

import pickle
import uuid
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.session_store import SessionIndex

EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_TOKENS  = 400
CHUNK_OVERLAP = 60


def _word_chunks(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return chunks


def _chunk_text(doc_id: str, text: str) -> list[dict]:
    chunks = []
    for i, chunk_text in enumerate(_word_chunks(text, CHUNK_TOKENS, CHUNK_OVERLAP)):
        if len(chunk_text.split()) < 30:
            continue
        chunks.append({
            "chunk_id": f"{doc_id}#chunk#{i}",
            "doc_id": doc_id,
            "ticker": "UPLOAD",
            "form": "upload",
            "period": "user",
            "section": "upload",
            "text": chunk_text,
        })
    return chunks


def build_session_index(text: str, filename: str) -> tuple[str, SessionIndex]:
    """Chunk, embed, and index `text`. Returns (session_id, SessionIndex)."""
    doc_id  = Path(filename).stem[:60]
    chunks  = _chunk_text(doc_id, text)

    if not chunks:
        raise ValueError("Document too short to index (< 30 words after chunking).")

    embed_model = SentenceTransformer(EMBED_MODEL)
    texts       = [c["text"] for c in chunks]
    embeddings  = embed_model.encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )
    embeddings = np.array(embeddings, dtype="float32")

    dim        = embeddings.shape[1]
    fi         = faiss.IndexFlatIP(dim)   # flat index — uploads are small
    fi.add(embeddings)

    tokenized = [t.lower().split() for t in texts]
    bm25      = BM25Okapi(tokenized)

    chunk_index = {c["chunk_id"]: c for c in chunks}
    corpus      = text

    session_id = str(uuid.uuid4())
    idx = SessionIndex(
        faiss_index=fi,
        bm25=bm25,
        chunks=chunks,
        corpus=corpus,
        chunk_index=chunk_index,
    )
    return session_id, idx
