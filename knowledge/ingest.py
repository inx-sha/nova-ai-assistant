from __future__ import annotations

import re

from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS
from knowledge.store import add_chunk, query

# Rough estimate: ~4 characters per token for English text.
_CHARS_PER_TOKEN = 4


def _split_into_paragraphs(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", text.strip())
    return [b.strip() for b in blocks if b.strip()]


def chunk_text(text: str,
                chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
                overlap_tokens: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    max_chars = chunk_size_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN

    paragraphs = _split_into_paragraphs(text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            overlap_text = current[-overlap_chars:] if current else ""
            current = f"{overlap_text}\n\n{para}".strip() if overlap_text else para

            while len(current) > max_chars:
                chunks.append(current[:max_chars])
                current = current[max_chars - overlap_chars:]

    if current:
        chunks.append(current)

    return chunks

def ingest_text(text: str, source: str, tags: list[str] | None = None,
                 confidence: float = 1.0, categories: list[str] | None = None,
                 tier: str = "cache", doc_type: str = "general") -> dict:
    """Chunks and stores text in high-performance batch mode."""
    chunks = chunk_text(text)
    if not chunks:
        return {"chunks_stored": 0, "chunks_skipped_as_duplicate": 0}

    from knowledge.store import get_collection, _write_lock, add_chunks_batch
    # Remove prior chunks for this exact source to keep index clean
    try:
        with _write_lock:
            existing = get_collection().get(where={"source": source})
            if existing and existing.get("ids"):
                get_collection().delete(ids=existing["ids"])
    except Exception as e:
        print(f"[ingest_text] note on prior source delete: {e}")

    ids = add_chunks_batch(
        chunks, source=source, tags=tags, confidence=confidence,
        categories=categories, tier=tier, doc_type=doc_type
    )
    return {"chunks_stored": len(ids), "chunks_skipped_as_duplicate": 0}