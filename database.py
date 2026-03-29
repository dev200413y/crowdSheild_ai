"""
CrowdShield AI - Incident Database
SQLite-backed store for persisting crowd incidents and agent actions.
"""

import sqlite3
import os
import threading
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "crowdshield.db")

# Thread-local storage ensures each thread gets its own connection,
# preventing race conditions when multiple threads access the DB.
_local = threading.local()


def _get_connection() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating it if needed."""
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return _local.conn


def init_db() -> None:
    """Create tables if they do not already exist."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                zone        TEXT    NOT NULL,
                people_count INTEGER,
                pressure    REAL,
                risk_label  TEXT,
                action_taken TEXT,
                ai_plan     TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                event_type  TEXT    NOT NULL,
                zone        TEXT,
                details     TEXT
            )
        """)
        conn.commit()


def log_incident(
    zone: str,
    people_count: int,
    pressure: float,
    risk_label: str,
    action_taken: str = "",
    ai_plan: str = "",
) -> int:
    """Insert a new incident record and return its row ID."""
    ts = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO incidents
                (timestamp, zone, people_count, pressure, risk_label, action_taken, ai_plan)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, zone, people_count, pressure, risk_label, action_taken, ai_plan),
        )
        conn.commit()
        return cur.lastrowid


def log_agent_event(event_type: str, zone: str = "", details: str = "") -> None:
    """Record an agent decision or action for auditability."""
    ts = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_logs (timestamp, event_type, zone, details)
            VALUES (?, ?, ?, ?)
            """,
            (ts, event_type, zone, details),
        )
        conn.commit()


def get_recent_incidents(limit: int = 20) -> list[dict]:
    """Return the most recent incidents as a list of dicts."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_agent_logs(limit: int = 50) -> list[dict]:
    """Return the most recent agent log entries."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM agent_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# Initialise the database on import
init_db()
