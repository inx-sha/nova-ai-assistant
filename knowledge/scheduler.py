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


def _parse_iso_datetime(dt_str: str) -> datetime | None:
    if not dt_str:
        return None
    try:
        cleaned = str(dt_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _topic_last_researched(pack_name: str, topic: str) -> datetime | None:
    source = f"pack:{pack_name}:{topic}"
    try:
        with _write_lock:
            results = get_collection().get(where={"source": source})

        metadatas = results.get("metadatas") or []
        if not metadatas:
            return None

        dates = [
            _parse_iso_datetime(m.get("date_collected"))
            for m in metadatas
            if isinstance(m, dict) and m.get("date_collected")
        ]
        valid_dates = [d for d in dates if d is not None]
        return max(valid_dates) if valid_dates else None
    except Exception as e:
        logger.warning(f"Error checking last researched date for {source}: {e}")
        return None


def refresh_stale_pack_topics() -> dict:
    installed = [p for p in memory.get_all_packs() if p["installed"]]
    if not installed:
        return {"packs_checked": 0, "topics_refreshed": 0}

    cutoff = datetime.now(timezone.utc) - timedelta(days=STALE_AFTER_DAYS)
    total_refreshed = 0

    for pack in installed:
        name = pack["name"]
        topics = AVAILABLE_PACKS.get(name, [])
        pack_refreshed = 0

        for topic in topics:
            last = _topic_last_researched(name, topic)
            if last is not None and last >= cutoff:
                continue  # still fresh, skip

            logger.info(f"Refreshing stale topic: {name} / {topic}")
            try:
                outcome = research_topic(topic)
            except Exception as e:
                logger.warning(f"Refresh exception for {topic}: {e}")
                continue

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
            pack_refreshed += 1
            total_refreshed += 1

        if pack_refreshed > 0:
            current_researched = pack.get("topics_researched", 0) or 0
            memory.upsert_pack(
                name,
                installed=True,
                topics_researched=max(current_researched, pack_refreshed),
                topics_total=len(topics)
            )

    return {"packs_checked": len(installed), "topics_refreshed": total_refreshed}


def _run_daily_backup() -> None:
    try:
        from knowledge.backup import create_backup
        res = create_backup()
        logger.info(f"Daily backup completed: {res.get('backup_name')}")
    except Exception as e:
        logger.error(f"Daily backup failed: {e}")


def start_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        return  # already running, don't start twice

    now = datetime.now(timezone.utc)
    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        refresh_stale_pack_topics,
        "interval",
        hours=REFRESH_INTERVAL_HOURS,
        id="refresh_pack_topics",
        next_run_time=now + timedelta(hours=REFRESH_INTERVAL_HOURS),
    )

    _scheduler.add_job(
        _run_daily_backup,
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