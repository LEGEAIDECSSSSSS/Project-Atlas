"""SQLite storage for conversations and messages (survives restarts)."""
import json
import sqlite3
import time
from contextlib import contextmanager

import config


@contextmanager
def _db():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New chat',
                project_id TEXT,
                project_name TEXT,
                mode TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                meta TEXT,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
            """
        )


def _conv(row):
    return dict(row) if row else None


def get_conversation(conversation_id):
    with _db() as conn:
        return _conv(conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone())


def get_or_create_conversation(conversation_id):
    existing = get_conversation(conversation_id)
    if existing:
        return existing
    now = time.time()
    with _db() as conn:
        conn.execute(
            "INSERT INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
            (conversation_id, now, now),
        )
    return get_conversation(conversation_id)


def update_conversation(conversation_id, **fields):
    allowed = {"title", "project_id", "project_name", "mode"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    fields["updated_at"] = time.time()
    assignments = ", ".join(f"{k} = ?" for k in fields)
    with _db() as conn:
        conn.execute(
            f"UPDATE conversations SET {assignments} WHERE id = ?",
            (*fields.values(), conversation_id),
        )


def list_conversations():
    with _db() as conn:
        rows = conn.execute(
            "SELECT id, title, project_id, project_name, mode, updated_at "
            "FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_conversation(conversation_id):
    with _db() as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def project_in_use(project_id):
    with _db() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE project_id = ? LIMIT 1", (project_id,)
        ).fetchone()
    return row is not None


def add_message(conversation_id, role, content, meta=None):
    now = time.time()
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, meta, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, json.dumps(meta) if meta else None, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
        )
        return cur.lastrowid


def _message(row):
    data = dict(row)
    data["meta"] = json.loads(data["meta"]) if data.get("meta") else {}
    return data


def get_messages(conversation_id, limit=None):
    """Oldest-first. With `limit`, returns the most recent `limit` messages."""
    with _db() as conn:
        if limit:
            rows = conn.execute(
                "SELECT * FROM (SELECT * FROM messages WHERE conversation_id = ? "
                "ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                (conversation_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()
    return [_message(r) for r in rows]


def pop_last_turn(conversation_id):
    """Remove the trailing assistant message (if any) and the user message before it.

    Returns the removed user message's text (used by "regenerate"), or None.
    """
    messages = get_messages(conversation_id)
    to_delete = []
    if messages and messages[-1]["role"] == "assistant":
        to_delete.append(messages.pop())
    user_text = None
    if messages and messages[-1]["role"] == "user":
        last = messages.pop()
        to_delete.append(last)
        user_text = last["content"]
    with _db() as conn:
        for m in to_delete:
            conn.execute("DELETE FROM messages WHERE id = ?", (m["id"],))
    return user_text
