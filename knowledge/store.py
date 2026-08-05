from __future__ import annotations

import uuid
from datetime import datetime, timezone

import chromadb

from config import CHROMA_PATH, CHROMA_COLLECTION
from core.llm import embed

import threading

_write_lock = threading.Lock()

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
    """..."""
    chunk_id = str(uuid.uuid4())
    vector = embed(text)
    with _write_lock:
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


def query(text: str, top_k: int = 5, category_filter: str | None = None) -> list[dict]:
    """
    Returns [{"text", "metadata", "similarity"}, ...] sorted best-first.
    If category_filter is set, only chunks whose categories include it
    are considered -- filtered in Python after retrieval, since Chroma's
    metadata filtering doesn't reliably substring-match our comma-joined
    categories field.
    """
    vector = embed(text)
    # Pull more candidates than requested when filtering, since some
    # will get discarded -- otherwise a filtered query could return
    # fewer than top_k results even when enough matches actually exist.
    fetch_k = top_k * 4 if category_filter else top_k

    with _write_lock:
        results = get_collection().query(
            query_embeddings=[vector],
            n_results=fetch_k,
        )

    out = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        if category_filter:
            categories = meta.get("categories", "").split(",")
            if category_filter not in categories:
                continue
        out.append({
            "text": doc,
            "metadata": meta,
            "similarity": 1 - dist,
        })
        if len(out) >= top_k:
            break

    return out
    """..."""
    vector = embed(text)
    with _write_lock:
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
    with _write_lock:
        return get_collection().count()

def set_tier(chunk_id: str, tier: str) -> bool:
    """..."""
    with _write_lock:
        collection = get_collection()
        existing = collection.get(ids=[chunk_id])
        if not existing or not existing.get("ids"):
            return False
        metadata = existing["metadatas"][0]
        metadata["tier"] = tier
        collection.update(ids=[chunk_id], metadatas=[metadata])
        return True


def find_chunks_by_source(source: str) -> list[dict]:
    """..."""
    with _write_lock:
        collection = get_collection()
        results = collection.get(where={"source": source})
    out = []
    for chunk_id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
        out.append({"id": chunk_id, "text": doc, "metadata": meta})
    return out

from datetime import timedelta


def prune_cache(max_age_days: int = 30, dry_run: bool = True) -> dict:

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

    collection = get_collection()
    all_chunks = collection.get()

    fixed = 0
    for chunk_id, meta in zip(all_chunks["ids"], all_chunks["metadatas"]):
        if meta.get("tier") is None:
            meta["tier"] = default_tier
            collection.update(ids=[chunk_id], metadatas=[meta])
            fixed += 1

    return fixed

from collections import Counter
from datetime import datetime, timedelta


def get_knowledge_stats() -> dict:
    """
    Aggregates stats across all stored knowledge -- tier breakdown,
    category breakdown, and growth over the last 14 days -- for the
    dashboard view.
    """
    with _write_lock:
        all_chunks = get_collection().get()

    metas = all_chunks["metadatas"]
    total = len(metas)

    tier_counts = Counter(m.get("tier", "cache") for m in metas)

    category_counts = Counter()
    for m in metas:
        for cat in m.get("categories", "general").split(","):
            if cat:
                category_counts[cat] += 1

    # Growth over the last 14 days, bucketed by date
    cutoff = datetime.now(timezone.utc) - timedelta(days=14)
    daily_counts = Counter()
    for m in metas:
        collected_str = m.get("date_collected")
        if not collected_str:
            continue
        collected = datetime.fromisoformat(collected_str)
        if collected >= cutoff:
            day_key = collected.strftime("%Y-%m-%d")
            daily_counts[day_key] += 1

    avg_confidence = (
        sum(m.get("confidence", 1.0) for m in metas) / total if total else 0
    )

    return {
        "total_chunks": total,
        "by_tier": dict(tier_counts),
        "by_category": dict(category_counts),
        "growth_last_14_days": dict(sorted(daily_counts.items())),
        "average_confidence": round(avg_confidence, 3),
    }