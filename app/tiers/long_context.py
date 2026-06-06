"""Tier 1 — LONG CONTEXT.

No retrieval: concatenate the entire bounded corpus and make one Gemini call.
Wins on simple/single-hop questions where the answer could be anywhere in the
corpus and retrieval would miss it.

Requires:
  data/index/corpus.txt   (built by ingest/build_index.py)
  GEMINI_API_KEY          env var
"""

import time
from functools import lru_cache
from pathlib import Path

from app.llm import llm, cost, MODEL
from app.tiers.base import register, RetrievalTier
from app.schemas import TierResult, TierName, Citation
from app.session_store import SessionIndex

CORPUS_PATH = Path("data/index/corpus.txt")

SYSTEM = (
    "You are a financial analyst assistant. "
    "Answer the question using ONLY the provided SEC filings corpus. "
    "Be precise, cite the specific filing section (e.g. AAPL-10K-2024#risk-factors). "
    "If the answer is not in the corpus, say so explicitly."
)


@lru_cache(maxsize=1)
def _load_corpus() -> str:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"{CORPUS_PATH} not found — run python -m ingest.build_index first."
        )
    return CORPUS_PATH.read_text(encoding="utf-8")


@register
class LongContextTier(RetrievalTier):
    name = TierName.long_context

    def run(self, question: str, session: SessionIndex | None = None) -> TierResult:
        corpus = session.corpus if session is not None else _load_corpus()

        # Gemini 2.0 Flash supports 1M token context; guard at ~180k words anyway
        max_words = 180_000
        words = corpus.split()
        if len(words) > max_words:
            corpus = " ".join(words[:max_words]) + "\n\n[corpus truncated]"

        prompt = f"<corpus>\n{corpus}\n</corpus>\n\nQuestion: {question}"

        t0 = time.perf_counter()
        answer, in_tok, out_tok = llm(SYSTEM, prompt, max_tokens=1024)
        latency_ms = (time.perf_counter() - t0) * 1000

        return TierResult(
            tier=self.name,
            answer=answer,
            citations=[Citation(doc_id="CORPUS", snippet="Full bounded corpus in context")],
            cost_usd=cost(in_tok, out_tok),
            latency_ms=round(latency_ms, 1),
            num_llm_calls=1,
            notes=f"model={MODEL} in={in_tok} out={out_tok}",
        )
