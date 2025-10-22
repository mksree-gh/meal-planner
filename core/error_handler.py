# core/error_handler.py

"""
core/error_handler.py

Centralized error logging and recovery utilities for the meal planner.

Responsibilities:
- Log structured errors into the SQLite `errors` table
- Provide lightweight helpers for retry, rollback, or safe_run
"""

import sqlite3
import traceback
from typing import Optional, Dict, Any
from config import DB_PATH
from core.utils import utc_now

# ---------------------------------------------------------------------
# Low-level helper
# ---------------------------------------------------------------------
def log_error(run_id: str, agent_name: str, step: str, summary: str, hint: Optional[str] = None):
    """
    Log an error entry into the `errors` table with structured details.
    If DB logging fails, print to console as fallback.
    """
    ts = utc_now()
    hint = hint or ""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO errors (run_id, agent_name, step, summary, hint, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, agent_name, step[:200], summary[:400], hint[:400], ts),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ErrorHandler] Failed to log error to DB: {e}")
        print(f"[ErrorHandler] Original error ({agent_name}:{step}): {summary}")

# ---------------------------------------------------------------------
# Safe execution wrapper
# ---------------------------------------------------------------------
def safe_run(agent_name: str, step: str, run_id: str, func, *args, **kwargs) -> Optional[Any]:
    """
    Execute a callable safely; log error on failure and return None.
    Returns function result on success.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        tb = traceback.format_exc()
        log_error(run_id, agent_name, step, str(e), hint=tb)
        return None

# ---------------------------------------------------------------------
# Optional: clear or fetch recent errors
# ---------------------------------------------------------------------
def get_recent_errors(limit: int = 10) -> list[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM errors ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]
