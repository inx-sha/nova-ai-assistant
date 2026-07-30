"""
Thin wrapper around Ollama's HTTP API.

This is the ONLY file that knows Ollama exists. If we ever switch to a
different runtime later, this is the only file that needs to change.
"""
from __future__ import annotations

import httpx

from config import OLLAMA_HOST, LLM_MODEL, EMBED_MODEL, LLM_TEMPERATURE


class LLMError(RuntimeError):
    pass


def chat(messages: list[dict], temperature: float | None = None) -> str:
    """
    messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    Returns the full response text.
    """
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature if temperature is not None else LLM_TEMPERATURE},
    }
    try:
        resp = httpx.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=180.0)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise LLMError(f"Ollama chat request failed: {e}") from e

    data = resp.json()
    return data.get("message", {}).get("content", "")


def embed(text: str) -> list[float]:
    """Returns the embedding vector for a piece of text."""
    payload = {"model": EMBED_MODEL, "prompt": text}
    try:
        resp = httpx.post(f"{OLLAMA_HOST}/api/embeddings", json=payload, timeout=120.0)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise LLMError(f"Ollama embeddings request failed: {e}") from e

    data = resp.json()
    embedding = data.get("embedding")
    if not embedding:
        raise LLMError(f"No embedding returned for text (len={len(text)})")
    return embedding