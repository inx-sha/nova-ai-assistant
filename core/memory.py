"""
SQLite-backed memory: conversations, personal memory, devices.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from config import SQLITE_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS personal_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL,
    category TEXT,
    confirmed_by_user INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    type TEXT,
    ip_or_topic TEXT,
    location TEXT,
    last_seen TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS packs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    installed INTEGER NOT NULL DEFAULT 0,
    topics_researched INTEGER NOT NULL DEFAULT 0,
    topics_total INTEGER NOT NULL DEFAULT 0,
    installed_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    wrong_info TEXT NOT NULL,
    correct_info TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'unverified',
    verification_note TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# --- Conversation memory ---

def add_message(session_id: str, role: str, content: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO conversations (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now()),
        )


def get_recent_messages(session_id: str, limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# --- Personal memory ---

def set_personal_fact(key: str, value: str, category: str = "general",
                       confirmed: bool = False) -> None:
    now = _now()
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM personal_memory WHERE key = ?", (key,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE personal_memory SET value = ?, category = ?, "
                "confirmed_by_user = ?, updated_at = ? WHERE key = ?",
                (value, category, int(confirmed), now, key),
            )
        else:
            conn.execute(
                "INSERT INTO personal_memory "
                "(key, value, category, confirmed_by_user, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, value, category, int(confirmed), now, now),
            )


def get_personal_fact(key: str) -> str | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT value FROM personal_memory WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else None


def get_all_personal_facts() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM personal_memory").fetchall()
    return [dict(r) for r in rows]


# --- Device memory ---

def upsert_device(name: str, type_: str, ip_or_topic: str, location: str = "") -> None:
    now = _now()
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM devices WHERE name = ?", (name,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE devices SET type = ?, ip_or_topic = ?, location = ?, "
                "last_seen = ?, updated_at = ? WHERE name = ?",
                (type_, ip_or_topic, location, now, now, name),
            )
        else:
            conn.execute(
                "INSERT INTO devices "
                "(name, type, ip_or_topic, location, last_seen, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (name, type_, ip_or_topic, location, now, now, now),
            )


def get_devices() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM devices").fetchall()
    return [dict(r) for r in rows]

# --- Pack tracking ---

def upsert_pack(name: str, installed: bool, topics_researched: int = 0,
                 topics_total: int = 0) -> None:
    now = _now()
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id, installed_at FROM packs WHERE name = ?", (name,)
        ).fetchone()
        if existing:
            installed_at = existing["installed_at"] if existing["installed_at"] else (now if installed else None)
            conn.execute(
                "UPDATE packs SET installed = ?, topics_researched = ?, "
                "topics_total = ?, installed_at = ?, updated_at = ? WHERE name = ?",
                (int(installed), topics_researched, topics_total, installed_at, now, name),
            )
        else:
            conn.execute(
                "INSERT INTO packs "
                "(name, installed, topics_researched, topics_total, installed_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, int(installed), topics_researched, topics_total,
                 now if installed else None, now),
            )


def get_pack(name: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM packs WHERE name = ?", (name,)).fetchone()
    return dict(row) if row else None


def get_all_packs() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM packs").fetchall()
    return [dict(r) for r in rows]

# --- Corrections ---

def add_correction(topic: str, wrong_info: str, correct_info: str,
                    status: str = "unverified", verification_note: str = "") -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO corrections (topic, wrong_info, correct_info, status, "
            "verification_note, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (topic, wrong_info, correct_info, status, verification_note, _now()),
        )


def get_all_corrections() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM corrections ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]

def get_verified_corrections() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM corrections WHERE status = 'verified' ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]