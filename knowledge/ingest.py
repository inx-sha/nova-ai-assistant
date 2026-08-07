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

DUPLICATE_THRESHOLD = 0.95

def ingest_text(text: str, source: str, tags: list[str] | None = None,
                 confidence: float = 1.0, categories: list[str] | None = None,
                 tier: str = "cache", doc_type: str = "general") -> dict:
    """Chunks and stores text, skipping near-duplicates. Returns counts."""
    chunks = chunk_text(text)
    stored = 0
    skipped = 0

    for chunk in chunks:
        existing = query(chunk, top_k=1)
        if existing and existing[0]["similarity"] >= DUPLICATE_THRESHOLD:
            skipped += 1
            continue
        add_chunk(chunk, source=source, tags=tags, confidence=confidence,
                  categories=categories, tier=tier, doc_type=doc_type)
        stored += 1

    return {"chunks_stored": stored, "chunks_skipped_as_duplicate": skipped}