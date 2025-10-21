# core/utils.py

import uuid
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------
# ID + Time helpers
# ----------------------------------------------------------
def generate_id(prefix: str) -> str:
    """Generate short unique ID like plan_4af31c9e."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def utc_now() -> str:
    """Return ISO-8601 UTC timestamp (timezone-aware)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

# ----------------------------------------------------------
# JSON safety
# ----------------------------------------------------------
def safe_json(obj):
    """Safe JSON serialization for logging or DB."""
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

# ----------------------------------------------------------
# Logging utility
# ----------------------------------------------------------
LOG_PATH = Path(__file__).resolve().parent.parent / "logs"
LOG_PATH.mkdir(exist_ok=True)

def log_event(agent: str, message: str, data: dict | None = None):
    """Append structured log line per agent."""
    record = {
        "timestamp": utc_now(),
        "agent": agent,
        "message": message,
        "data": data or {},
    }
    logfile = LOG_PATH / f"{agent}.log"
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def log_weight_diff(agent: str, axis: str, before: dict, after: dict, user_id: str = ""):
    """Quick structured diff for debugging weight evolution."""
    changes = {}
    for k, v in (after or {}).items():
        old = before.get(k)
        if not old:
            changes[k] = {"change": "added", "new": v}
        elif old != v:
            changes[k] = {"change": "updated", "old": old, "new": v}
    for k in (before or {}):
        if k not in (after or {}):
            changes[k] = {"change": "removed", "old": before[k]}
    if changes:
        log_event(agent, "weight_diff", {"user_id": user_id, "axis": axis, "changes": changes})

# ----------------------------------------------------------
# Simple logger setup
# ----------------------------------------------------------
def get_logger(name: str):
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    return logger
