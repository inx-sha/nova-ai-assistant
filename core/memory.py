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

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL DEFAULT 'general',
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
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
        _migrate_add_column(conn, "conversations", "pinned", "INTEGER NOT NULL DEFAULT 0")
        _migrate_add_column(conn, "conversations", "starred", "INTEGER NOT NULL DEFAULT 0")


def _migrate_add_column(conn, table: str, column: str, definition: str) -> None:
    """Adds a column to an existing table if it doesn't already exist --
    safe to call every startup, unlike CREATE TABLE IF NOT EXISTS which
    can't add columns to a table that already exists."""
    existing_columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing_columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


# --- Conversation memory ---

def add_message(session_id: str, role: str, content: str) -> int:
    with get_conn() as conn:
        cursor = conn.execute(
            "INSERT INTO conversations (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now()),
        )
        return cursor.lastrowid


def get_recent_messages(session_id: str, limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

def get_session_history_with_ids(session_id: str, limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, role, content, pinned, starred FROM conversations "
            "WHERE session_id = ? ORDER BY id ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]

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

# --- Session listing ---

def get_all_sessions() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT session_id,
                   MIN(CASE WHEN role = 'user' THEN content END) AS first_message,
                   MAX(timestamp) AS last_activity
            FROM conversations
            GROUP BY session_id
            ORDER BY last_activity DESC
            """
        ).fetchall()

    sessions = []
    for r in rows:
        title = (r["first_message"] or "New chat")[:60]
        sessions.append({
            "session_id": r["session_id"],
            "title": title,
            "last_activity": r["last_activity"],
        })
    return sessions


def delete_session(session_id: str) -> int:
    """Deletes all messages for a session. Returns count deleted."""
    with get_conn() as conn:
        cursor = conn.execute(
            "DELETE FROM conversations WHERE session_id = ?", (session_id,)
        )
        return cursor.rowcount

# --- Session metadata ---

def ensure_session(session_id: str, mode: str = "general") -> None:
    """Creates a session row if it doesn't exist yet. No-op otherwise."""
    now = _now()
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO sessions (session_id, mode, archived, created_at, updated_at) "
                "VALUES (?, ?, 0, ?, ?)",
                (session_id, mode, now, now),
            )


def get_session_mode(session_id: str) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT mode FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return row["mode"] if row else "general"


def set_session_archived(session_id: str, archived: bool) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET archived = ?, updated_at = ? WHERE session_id = ?",
            (int(archived), _now(), session_id),
        )

def set_session_mode(session_id: str, mode: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET mode = ?, updated_at = ? WHERE session_id = ?",
            (mode, _now(), session_id),
        )

def get_all_sessions(include_archived: bool = False) -> list[dict]:
    """
    Returns one entry per distinct session with title, last activity,
    mode, and archived status -- everything the sidebar needs.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT c.session_id,
                   MIN(CASE WHEN c.role = 'user' THEN c.content END) AS first_message,
                   MAX(c.timestamp) AS last_activity
            FROM conversations c
            GROUP BY c.session_id
            ORDER BY last_activity DESC
            """
        ).fetchall()

        session_meta = {
            r["session_id"]: {"mode": r["mode"], "archived": r["archived"]}
            for r in conn.execute("SELECT session_id, mode, archived FROM sessions").fetchall()
        }

    sessions = []
    for r in rows:
        meta = session_meta.get(r["session_id"], {"mode": "general", "archived": 0})
        if not include_archived and meta["archived"]:
            continue
        title = (r["first_message"] or "New chat")[:60]
        sessions.append({
            "session_id": r["session_id"],
            "title": title,
            "last_activity": r["last_activity"],
            "mode": meta["mode"],
            "archived": bool(meta["archived"]),
        })
    return sessions

def get_all_distinct_modes() -> list[str]:
    with get_conn() as conn:
        rows = conn.execute("SELECT DISTINCT mode FROM sessions").fetchall()
    return [r["mode"] for r in rows]

def delete_messages_by_ids(message_ids: list[int]) -> int:
    if not message_ids:
        return 0
    placeholders = ",".join("?" for _ in message_ids)
    with get_conn() as conn:
        cursor = conn.execute(
            f"DELETE FROM conversations WHERE id IN ({placeholders})", message_ids
        )
        return cursor.rowcount

# --- Message pin/star/edit ---

def set_message_pinned(message_id: int, pinned: bool) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET pinned = ? WHERE id = ?",
            (int(pinned), message_id),
        )


def set_message_starred(message_id: int, starred: bool) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET starred = ? WHERE id = ?",
            (int(starred), message_id),
        )


def get_pinned_messages(session_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, role, content, timestamp FROM conversations "
            "WHERE session_id = ? AND pinned = 1 ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_starred_messages_by_mode() -> dict:
    """
    Returns starred messages grouped by their session's mode -- e.g.
    {"embedded_systems": [...], "general": [...]}, matching the
    "star according to their mode" requirement.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.role, c.content, c.timestamp, c.session_id,
                   COALESCE(s.mode, 'general') AS mode
            FROM conversations c
            LEFT JOIN sessions s ON c.session_id = s.session_id
            WHERE c.starred = 1
            ORDER BY c.id ASC
            """
        ).fetchall()

    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r["mode"], []).append({
            "id": r["id"], "role": r["role"], "content": r["content"],
            "timestamp": r["timestamp"], "session_id": r["session_id"],
        })
    return grouped


def edit_message(message_id: int, new_content: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET content = ? WHERE id = ?",
            (new_content, message_id),
        )


def delete_messages_after(session_id: str, message_id: int) -> int:
    """
    Deletes every message in a session that comes AFTER the given one --
    used for 'retry': edit/resend a question, discard everything that
    followed the original answer, since it's no longer valid context.
    """
    with get_conn() as conn:
        cursor = conn.execute(
            "DELETE FROM conversations WHERE session_id = ? AND id > ?",
            (session_id, message_id),
        )
        return cursor.rowcount