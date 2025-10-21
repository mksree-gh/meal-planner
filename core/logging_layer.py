# core/logging_layer.py

"""
Unified lightweight logging + observability layer for Posha Assistant.
Designed for clarity, structure, and easy extensibility.
"""

import json
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Root folder for all logs
LOG_ROOT = Path(__file__).resolve().parent.parent / "logs"
LOG_ROOT.mkdir(exist_ok=True)

# Thread lock for safe concurrent writes
_lock = threading.Lock()


# ---------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------
# Core writers
# ---------------------------------------------------------------------
def _write_jsonl(path: Path, record: Dict[str, Any]):
    """Thread-safe JSONL append."""
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Unified logging entrypoints
# ---------------------------------------------------------------------
def log_event(agent: str, event_type: str, data: Dict[str, Any]):
    """
    Structured event log for all agents.
    Each record is a JSONL with fields:
      - timestamp
      - agent
      - type
      - data
    """
    record = {
        "timestamp": utc_now(),
        "agent": agent,
        "type": event_type,
        "data": data or {},
    }
    _write_jsonl(LOG_ROOT / "agents" / f"{agent}.jsonl", record)


def log_session_event(run_id: str, phase: str, data: Dict[str, Any]):
    """
    Low-level trace log for a single run or conversation.
    Each run_id gets its own file.
    """
    record = {"timestamp": utc_now(), "run_id": run_id, "phase": phase, "data": data or {}}
    _write_jsonl(LOG_ROOT / "sessions" / f"session_{run_id}.jsonl", record)


def log_summary(user_id: str, section: str, text: str):
    """
    Append a concise Markdown summary for human review.
    Each user_id gets a single file per section type.
    """
    path = LOG_ROOT / "summaries" / f"{user_id}_{section}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n## {utc_now()}\n{text.strip()}\n")


# ---------------------------------------------------------------------
# Helper: human-readable diff printout
# ---------------------------------------------------------------------
def print_pretty_diff(axis: str, changes: Dict[str, Any]):
    """Print concise console summary of preference/weight diffs."""
    parts = []
    for k, v in changes.items():
        ctype = v["change"]
        if ctype == "added":
            new_val = v.get("new")
            if isinstance(new_val, dict):
                new_val = json.dumps(new_val)
            parts.append(f"+{k} ↑{new_val}")
        elif ctype == "updated":
            delta = round(v["new"] - v["old"], 3)
            arrow = "↑" if delta > 0 else "↓"
            parts.append(f"{k} {arrow}{abs(delta):.2f}")
        elif ctype == "removed":
            parts.append(f"-{k}")
    if parts:
        print(f"⚖️  {axis}: {', '.join(parts)}")


# ---------------------------------------------------------------------
# Optional global runtime log (for quick tail -f)
# ---------------------------------------------------------------------
def log_console_line(message: str):
    """Mirror key events to a single runtime.log file."""
    path = LOG_ROOT / "runtime.log"
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{utc_now()} {message.strip()}\n")
