"""
Thin wrapper around Ollama's HTTP API.

This is the ONLY file that knows Ollama exists. If we ever switch to a
different runtime later, this is the only file that needs to change.
"""
from __future__ import annotations

import json
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

def chat_stream(messages: list[dict], temperature: float | None = None):
    """
    Generator version of chat() -- yields text chunks as they arrive from
    Ollama, instead of waiting for the full response. Used for streaming
    responses to the UI so answers appear progressively, like ChatGPT's
    typing effect, rather than all at once after a long wait.
    """
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature if temperature is not None else LLM_TEMPERATURE},
    }
    try:
        with httpx.stream("POST", f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120.0) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
                if chunk.get("done"):
                    break
    except httpx.HTTPError as e:
        raise LLMError(f"Ollama streaming chat request failed: {e}") from e

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