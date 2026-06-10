"""Shared LLM client + call helper used by all tiers and routers.

Provider is selected via LLM_PROVIDER env var (default: gemini).
  LLM_PROVIDER=gemini   → Google Gemini via google-genai SDK
  LLM_PROVIDER=groq     → Groq cloud API (llama-3.3-70b-versatile)

All LLM calls go through llm(). Switching providers means setting
LLM_PROVIDER — nothing else in the codebase needs to change.

Pricing (per 1M tokens, USD):
  Gemini 2.0 Flash:  input $0.10  output $0.40
  Groq llama-3.3-70b: effectively free on free tier
"""
from __future__ import annotations

import os
import re

from dotenv import load_dotenv
load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# ── Model names ──────────────────────────────────────────────────────────────
_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
_GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.3-70b-versatile")

MODEL = _GROQ_MODEL if PROVIDER == "groq" else _GEMINI_MODEL

# ── Pricing ───────────────────────────────────────────────────────────────────
# Cost is always reported at Gemini 2.0 Flash rates ($0.10 in / $0.40 out per 1M
# tokens) so per-tier cost comparison stays meaningful even when running on Groq's
# free tier. This is the cost the workload WOULD incur on a paid LLM.
_IN_PER_M  = 0.10
_OUT_PER_M = 0.40

# ── Lazy clients ─────────────────────────────────────────────────────────────
_gemini_client = None
_groq_client   = None


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _gemini_client


def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


# ── Public API ────────────────────────────────────────────────────────────────

def llm(system: str, user: str, max_tokens: int = 1024) -> tuple[str, int, int]:
    """Single LLM call. Returns (text, input_tokens, output_tokens)."""
    if PROVIDER == "groq":
        return _llm_groq(system, user, max_tokens)
    return _llm_gemini(system, user, max_tokens)


def _llm_gemini(system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
    from google.genai import types
    client = _get_gemini()
    response = client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        ),
    )
    text = response.text
    meta = getattr(response, "usage_metadata", None)
    in_tok  = getattr(meta, "prompt_token_count",     0) if meta else 0
    out_tok = getattr(meta, "candidates_token_count", 0) if meta else 0
    return text, in_tok, out_tok


def _llm_groq(system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
    client = _get_groq()
    response = client.chat.completions.create(
        model=_GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
    )
    text    = response.choices[0].message.content or ""
    in_tok  = getattr(response.usage, "prompt_tokens",     0)
    out_tok = getattr(response.usage, "completion_tokens", 0)
    return text, in_tok, out_tok


def cost(in_tok: int, out_tok: int) -> float:
    return round((in_tok / 1_000_000) * _IN_PER_M + (out_tok / 1_000_000) * _OUT_PER_M, 6)


def strip_fences(text: str) -> str:
    """Remove ```json … ``` wrappers that models sometimes add."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    return re.sub(r"\s*```$", "", text)
