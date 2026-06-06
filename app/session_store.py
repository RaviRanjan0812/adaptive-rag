"""In-memory session store: maps session_id → uploaded index data.

Each session holds a FAISS index, BM25 object, chunks list, and a plain-text
corpus string (for the long-context tier). Sessions expire after TTL_SECONDS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import faiss

TTL_SECONDS = 3600  # 1 hour


@dataclass
class SessionIndex:
    faiss_index: Any          # faiss.Index
    bm25: Any                 # BM25Okapi
    chunks: list[dict]
    corpus: str               # concatenated full text for long-context tier
    chunk_index: dict[str, dict] = field(default_factory=dict)  # chunk_id → chunk
    created_at: float = field(default_factory=time.time)

    def expired(self) -> bool:
        return (time.time() - self.created_at) > TTL_SECONDS


_store: dict[str, SessionIndex] = {}


def put(session_id: str, idx: SessionIndex) -> None:
    _purge_expired()
    _store[session_id] = idx


def get(session_id: str) -> SessionIndex | None:
    idx = _store.get(session_id)
    if idx is None or idx.expired():
        _store.pop(session_id, None)
        return None
    return idx


def _purge_expired() -> None:
    expired = [k for k, v in _store.items() if v.expired()]
    for k in expired:
        del _store[k]
