"""
ChromaDB wrapper — single persistent collection for Phase 1.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import chromadb

from config import CHROMA_PATH, CHROMA_COLLECTION
from core.llm import embed

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_chunk(text: str, source: str, tags: list[str] | None = None,
              confidence: float = 1.0, categories: list[str] | None = None,
              tier: str = "cache") -> str:
    """
    Embeds and stores a single chunk. Returns the generated chunk id.

    tier: "pack" (installed via a domain pack, never auto-pruned),
          "pinned" (user explicitly saved it, never auto-pruned),
          "cache" (default -- ordinary fetched knowledge, prunable later).
    """
    chunk_id = str(uuid.uuid4())
    vector = embed(text)
    get_collection().add(
        ids=[chunk_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[{
            "source": source,
            "tags": ",".join(tags or []),
            "confidence": confidence,
            "categories": ",".join(categories or ["general"]),
            "tier": tier,
            "date_collected": datetime.now(timezone.utc).isoformat(),
        }],
    )
    return chunk_id


def query(text: str, top_k: int = 5) -> list[dict]:
    """
    Returns [{"text", "metadata", "similarity"}, ...] sorted best-first.
    Chroma returns cosine *distance*; we convert to similarity (1 - distance).
    """
    vector = embed(text)
    results = get_collection().query(
        query_embeddings=[vector],
        n_results=top_k,
    )

    out = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        out.append({
            "text": doc,
            "metadata": meta,
            "similarity": 1 - dist,
        })
    return out


def count() -> int:
    return get_collection().count()

def set_tier(chunk_id: str, tier: str) -> bool:
    """Updates the tier of an existing chunk (e.g. cache -> pinned)."""
    collection = get_collection()
    existing = collection.get(ids=[chunk_id])
    if not existing or not existing.get("ids"):
        return False
    metadata = existing["metadatas"][0]
    metadata["tier"] = tier
    collection.update(ids=[chunk_id], metadatas=[metadata])
    return True


def find_chunks_by_source(source: str) -> list[dict]:
    """Returns all stored chunks whose metadata 'source' matches exactly."""
    collection = get_collection()
    results = collection.get(where={"source": source})
    out = []
    for chunk_id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
        out.append({"id": chunk_id, "text": doc, "metadata": meta})
    return out

from datetime import timedelta


def prune_cache(max_age_days: int = 30, dry_run: bool = True) -> dict:
    """
    Deletes cache-tier chunks older than max_age_days. NEVER touches
    'pinned' or 'pack' tier chunks, regardless of age.

    dry_run=True (default): reports what WOULD be deleted, deletes nothing.
    Set dry_run=False to actually delete.
    """
    collection = get_collection()
    all_chunks = collection.get()

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    to_delete = []

    for chunk_id, meta in zip(all_chunks["ids"], all_chunks["metadatas"]):
        if meta.get("tier") != "cache":
            continue  # pinned/pack are never eligible

        collected_str = meta.get("date_collected")
        if not collected_str:
            continue
        collected = datetime.fromisoformat(collected_str)
        if collected < cutoff:
            to_delete.append(chunk_id)

    if not dry_run and to_delete:
        collection.delete(ids=to_delete)

    return {
        "eligible_for_deletion": len(to_delete),
        "actually_deleted": 0 if dry_run else len(to_delete),
        "dry_run": dry_run,
    }

def backfill_missing_tiers(default_tier: str = "cache") -> int:
    """
    One-time fix: sets tier="cache" on any chunk stored before the tier
    field existed (where metadata.tier is missing/None). Returns count fixed.
    """
    collection = get_collection()
    all_chunks = collection.get()

    fixed = 0
    for chunk_id, meta in zip(all_chunks["ids"], all_chunks["metadatas"]):
        if meta.get("tier") is None:
            meta["tier"] = default_tier
            collection.update(ids=[chunk_id], metadatas=[meta])
            fixed += 1

    return fixed