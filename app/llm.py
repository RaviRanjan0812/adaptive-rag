"""Shared Gemini client + call helper used by all tiers and routers.

All LLM calls in this project go through _llm(). Switching models means
changing MODEL here — nothing else needs to change.

Gemini 2.0 Flash pricing (per 1M tokens, USD):
  input  $0.10   output  $0.40
"""
from __future__ import annotations

import os
import re

from dotenv import load_dotenv
load_dotenv()   # loads .env from the project root automatically

import google.generativeai as genai

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

_IN_PER_M  = 0.10
_OUT_PER_M = 0.40

_configured = False


def _client() -> None:
    """Configure the SDK once per process."""
    global _configured
    if not _configured:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        _configured = True


def llm(system: str, user: str, max_tokens: int = 1024) -> tuple[str, int, int]:
    """Single Gemini call. Returns (text, input_tokens, output_tokens)."""
    _client()
    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=system,
    )
    response = model.generate_content(
        user,
        generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens),
    )
    text = response.text

    # Token counts from usage_metadata (available on all non-streaming calls)
    meta = getattr(response, "usage_metadata", None)
    in_tok  = getattr(meta, "prompt_token_count",     0) if meta else 0
    out_tok = getattr(meta, "candidates_token_count", 0) if meta else 0
    return text, in_tok, out_tok


def cost(in_tok: int, out_tok: int) -> float:
    return round((in_tok / 1_000_000) * _IN_PER_M + (out_tok / 1_000_000) * _OUT_PER_M, 6)


def strip_fences(text: str) -> str:
    """Remove ```json … ``` wrappers that models sometimes add."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    return re.sub(r"\s*```$", "", text)
