"""Shared Gemini client + call helper used by all tiers and routers.

All LLM calls in this project go through llm(). Switching models means
changing MODEL here — nothing else needs to change.

Gemini 2.0 Flash pricing (per 1M tokens, USD):
  input  $0.10   output  $0.40
"""
from __future__ import annotations

import os
import re

from dotenv import load_dotenv
load_dotenv()   # loads .env from the project root automatically

from google import genai
from google.genai import types

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

_IN_PER_M  = 0.10
_OUT_PER_M = 0.40

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Create the SDK client once per process."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def llm(system: str, user: str, max_tokens: int = 1024) -> tuple[str, int, int]:
    """Single Gemini call. Returns (text, input_tokens, output_tokens)."""
    client = _get_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        ),
    )
    text = response.text

    # Token counts from usage_metadata
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
