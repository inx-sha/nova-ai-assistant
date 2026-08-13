from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler

from core import memory
from knowledge.packs import AVAILABLE_PACKS
from knowledge.internet import research_topic
from knowledge.ingest import ingest_text
from knowledge.store import get_collection, _write_lock

logger = logging.getLogger("nova.scheduler")

REFRESH_INTERVAL_HOURS = 6      # how often the scheduler checks for stale topics
STALE_AFTER_DAYS = 7            # a topic older than this gets re-researched

_scheduler: BackgroundScheduler | None = None


def _topic_last_researched(pack_name: str, topic: str) -> datetime | None:
    source = f"pack:{pack_name}:{topic}"
    with _write_lock:
        results = get_collection().get(where={"source": source})

    if not results["metadatas"]:
        return None

    dates = [
        datetime.fromisoformat(m["date_collected"])
        for m in results["metadatas"]
        if m.get("date_collected")
    ]
    return max(dates) if dates else None


def refresh_stale_pack_topics() -> dict:
    
    installed = [p for p in memory.get_all_packs() if p["installed"]]
    if not installed:
        return {"packs_checked": 0, "topics_refreshed": 0}

    cutoff = datetime.now(timezone.utc) - timedelta(days=STALE_AFTER_DAYS)
    refreshed = 0

    for pack in installed:
        name = pack["name"]
        topics = AVAILABLE_PACKS.get(name, [])

        for topic in topics:
            last = _topic_last_researched(name, topic)
            if last is not None and last >= cutoff:
                continue  # still fresh, skip

            logger.info(f"Refreshing stale topic: {name} / {topic}")
            outcome = research_topic(topic)
            if outcome is None:
                logger.warning(f"Refresh failed (no internet or no results): {topic}")
                continue

            ingest_text(
                outcome.summary,
                source=f"pack:{name}:{topic}",
                tags=["pack", name],
                confidence=outcome.confidence,
                categories=[name],
                tier="pack",
            )
            refreshed += 1

    return {"packs_checked": len(installed), "topics_refreshed": refreshed}


def start_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        return  # already running, don't start twice

    now = datetime.now()
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        refresh_stale_pack_topics,
        "interval",
        hours=REFRESH_INTERVAL_HOURS,
        id="refresh_pack_topics",
        next_run_time=now + timedelta(hours=REFRESH_INTERVAL_HOURS),
    )

    _scheduler.add_job(
        lambda: __import__("knowledge.backup", fromlist=["create_backup"]).create_backup(),
        "interval",
        hours=24,
        id="daily_backup",
        next_run_time=now + timedelta(hours=24),
    )
    
    _scheduler.start()
    logger.info(f"Scheduler started: checking pack freshness every {REFRESH_INTERVAL_HOURS}h")


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None