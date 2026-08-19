"""
session_store.py — SQLite-backed session persistence.

Sessions (AnalyticsEngine, ClusterEngine, classifiers dict, history, domain
context) are pickled and stored in a local SQLite file, replacing the old
in-memory-only SESSIONS dict. This means a server restart no longer silently
wipes every active user's uploaded data and trained models.

Chosen over Redis specifically because this runs as a single uvicorn process
with modest concurrent load — SQLite needs zero extra infrastructure (no
separate server, no Windows compatibility issues), while giving the same
core guarantee: session state survives a restart. If this ever needs to
scale across multiple server processes, migrating to Redis later is a
contained change — same "pickle -> store bytes -> unpickle" pattern, just a
different backend.

NOTE ON WHAT GETS PICKLED: AnalyticsEngine holds a live DuckDB connection,
and ClusterEngine/SupervisedClassifierEngine hold a lazily-created Groq
client and a SHAP explainer — none of these are reliably picklable (they
wrap C-level/network resources). Each of those classes implements
__getstate__/__setstate__ to strip the non-picklable pieces out before
pickling and cheaply rebuild them after unpickling (see analytics_engine.py
and prediction_engine.py) — this module doesn't need to know about that,
it just pickles whatever object it's given.
"""

import sqlite3
import pickle
import time
import threading
from pathlib import Path

DB_PATH = Path(__file__).parent / "sessions.db"
_lock = threading.Lock()   # guards against concurrent writers stepping on each other

# A fresh connection is opened per operation rather than one long-lived
# module-level connection reused for the app's whole lifetime. A single
# shared connection held open across many unrelated operations (heavy
# numpy/sklearn/UMAP work in between writes, multiple request threads, etc.)
# is a known-fragile pattern for sqlite3 — short-lived per-operation
# connections are the standard, safer approach and cost very little given
# how infrequently these calls happen relative to a single query's runtime.

def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            data BLOB NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

_init_db()


def load_session(session_id: str):
    """Returns the unpickled session dict, or None if not found or corrupt.
    A corrupt/unreadable row is treated as "not found" rather than raised —
    callers fall back to creating a fresh session, which is the same
    behavior as a first-ever visit."""
    with _lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            row = conn.execute(
                "SELECT data FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        finally:
            conn.close()
    if row is None:
        return None
    try:
        return pickle.loads(row[0])
    except Exception as e:
        print(f"[session_store] Failed to unpickle session {session_id}: {e}")
        return None


def save_session(session_id: str, session: dict) -> bool:
    """Pickles and upserts the session dict. Returns False (and logs) on
    failure rather than raising — a failed save shouldn't break the request
    that triggered it; the in-memory copy is still correct for this
    process's lifetime, it just won't survive a restart."""
    try:
        blob = pickle.dumps(session)
    except Exception as e:
        print(f"[session_store] Failed to pickle session {session_id}: {e}")
        return False
    try:
        with _lock:
            conn = sqlite3.connect(DB_PATH)
            try:
                conn.execute(
                    "INSERT INTO sessions (session_id, data, updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
                    (session_id, blob, time.time()),
                )
                conn.commit()
            finally:
                conn.close()
        return True
    except Exception as e:
        print(f"[session_store] Failed to save session {session_id}: {e}")
        return False


def delete_session(session_id: str):
    with _lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()


def cleanup_old_sessions(max_age_seconds: int = 7 * 24 * 3600):
    """Optional housekeeping — call on startup (or periodically) to purge
    sessions untouched for longer than max_age_seconds, so sessions.db
    doesn't grow unbounded from abandoned sessions."""
    cutoff = time.time() - max_age_seconds
    with _lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            deleted = conn.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,)).rowcount
            conn.commit()
        finally:
            conn.close()
    if deleted:
        print(f"[session_store] Cleaned up {deleted} session(s) older than {max_age_seconds}s")