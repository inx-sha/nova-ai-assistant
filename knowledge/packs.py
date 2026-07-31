"""
Domain knowledge packs -- bundles of seed topics that get proactively
researched (via the same research_topic() -> ingest_text() pipeline
used for live chat) so offline mode has a reliable base of knowledge
in a chosen domain, without waiting for the user to happen to ask
about each topic first.
"""
from __future__ import annotations

from core import memory
from knowledge.internet import research_topic
from knowledge.ingest import ingest_text

AVAILABLE_PACKS: dict[str, list[str]] = {
    "embedded_systems": [
        "STM32 clock configuration and timebase",
        "difference between UART SPI and I2C protocols",
        "RTOS task scheduling and priorities",
        "Verilog testbench basics",
        "ESP32 WiFi provisioning",
        "FreeRTOS queues and semaphores",
    ],
    "medical_devices_basics": [
        "how blood pressure is measured",
        "what an ECG measures",
        "how continuous glucose monitors work",
        "how closed-loop insulin pumps work",
    ],
}


def list_available_packs() -> list[str]:
    return list(AVAILABLE_PACKS.keys())


def install_pack(name: str) -> dict:
    """
    Researches every seed topic in the named pack (online, via
    research_topic) and stores results tagged tier='pack' so they're
    protected from pruning. Returns progress info.
    """
    if name not in AVAILABLE_PACKS:
        raise ValueError(f"Unknown pack '{name}'. Available: {list_available_packs()}")

    topics = AVAILABLE_PACKS[name]
    memory.upsert_pack(name, installed=True, topics_researched=0, topics_total=len(topics))

    researched = 0
    failed_topics = []

    for topic in topics:
        outcome = research_topic(topic)
        if outcome is None:
            failed_topics.append(topic)
            continue

        ingest_text(
            outcome.summary,
            source=f"pack:{name}:{topic}",
            tags=["pack", name],
            confidence=outcome.confidence,
            categories=[name],
            tier="pack",
        )
        researched += 1
        memory.upsert_pack(name, installed=True, topics_researched=researched, topics_total=len(topics))

    return {
        "pack": name,
        "topics_researched": researched,
        "topics_total": len(topics),
        "failed_topics": failed_topics,
    }