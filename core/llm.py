from __future__ import annotations

import os
import json
import httpx

from config import OLLAMA_HOST, EMBED_MODEL, LLM_TEMPERATURE, LLM_CONTEXT_WINDOW
from core.memory import get_current_model


class LLMError(RuntimeError):
    pass


_http_client = httpx.Client(timeout=httpx.Timeout(300.0, connect=15.0, read=300.0, write=30.0), limits=httpx.Limits(max_keepalive_connections=20, max_connections=50))


def chat(messages: list[dict], temperature: float | None = None, model: str | None = None) -> str:
    """
    messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    Returns the full response text.
    """
    resolved_model = model or get_current_model()
    payload = {
        "model": resolved_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
            "num_ctx": LLM_CONTEXT_WINDOW,
            "num_thread": os.cpu_count() or 8,
        },
    }
    try:
        resp = _http_client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise LLMError(f"Ollama chat request failed: {e}") from e

    data = resp.json()
    content = data.get("message", {}).get("content", "")
    # Strip <think>...</think> block if present
    if "<think>" in content and "</think>" in content:
        content = content.split("</think>", 1)[1].strip()
    return content

def chat_stream(messages: list[dict], temperature: float | None = None, model: str | None = None):
    """
    Generator version of chat() -- yields text chunks as they arrive from
    Ollama, filtering internal reasoning tags so users get immediate clean output.
    """
    resolved_model = model or get_current_model()
    payload = {
        "model": resolved_model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
            "num_ctx": LLM_CONTEXT_WINDOW,
            "num_thread": os.cpu_count() or 8,
        },
    }
    try:
        with _http_client.stream("POST", f"{OLLAMA_HOST}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            in_think = False
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if not content:
                    if chunk.get("done"):
                        break
                    continue

                if "<think>" in content:
                    in_think = True
                    content = content.split("<think>", 1)[0]

                if in_think:
                    if "</think>" in content:
                        in_think = False
                        content = content.split("</think>", 1)[1]
                    else:
                        content = ""

                if content:
                    yield content
                if chunk.get("done"):
                    break
    except httpx.HTTPError as e:
        raise LLMError(f"Ollama streaming chat request failed: {e}") from e

def embed(text: str, model: str | None = None) -> list[float]:
    """Returns the embedding vector for a piece of text."""
    resolved_model = model or EMBED_MODEL
    payload = {"model": resolved_model, "prompt": text}
    try:
        resp = _http_client.post(f"{OLLAMA_HOST}/api/embeddings", json=payload)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise LLMError(f"Ollama embeddings request failed: {e}") from e

    data = resp.json()
    embedding = data.get("embedding")
    if not embedding:
        raise LLMError(f"No embedding returned for text (len={len(text)})")
    return embedding

def embed_many(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Returns embedding vectors for a list of text strings efficiently."""
    return [embed(t, model=model) for t in texts]