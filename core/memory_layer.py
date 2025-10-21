# core/memory_layer.py

"""
core/memory_layer.py

SQLite-based memory abstraction for Posha.

Includes:
- init_db() and seeding helpers
- JSON helpers
- MemoryLayer class used by agents to read/write structured data
"""

import sqlite3
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

from config import DB_PATH  # project config path
from core.utils import utc_now

# ---------------------------------------------------------------------
# Paths (DB_PATH from config)
# ---------------------------------------------------------------------
DB_PATH = Path(DB_PATH)

# ---------------------------------------------------------------------
# Simple connection / schema helpers
# ---------------------------------------------------------------------
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "recipes.json"

def init_db():
    """Create tables from core/schema.sql."""
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file missing: {SCHEMA_PATH}")
    conn = get_connection()
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

def seed_recipes_if_empty():
    """Seed recipes table from data/recipes.json if empty."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM recipes;")
    count = cur.fetchone()[0]
    if count > 0:
        conn.close()
        return

    if not DATA_PATH.exists():
        print(f"[memory_layer] No recipe data at {DATA_PATH}; skipping seed.")
        conn.close()
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        recipes = json.load(f)

    for r in recipes:
        cur.execute(
            """
            INSERT OR REPLACE INTO recipes (
                recipe_id, name, main_ingredients, tags, cuisine,
                prep_time_min, cook_time_min, nutrition_facts, cooking_style, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r.get("recipe_id"),
                r.get("name"),
                json.dumps(r.get("main_ingredients", []), ensure_ascii=False),
                json.dumps(r.get("tags", []), ensure_ascii=False),
                r.get("cuisine"),
                r.get("prep_time_min"),
                r.get("cook_time_min"),
                json.dumps(r.get("nutrition_facts", {}), ensure_ascii=False),
                r.get("cooking_style"),
                r.get("description"),
            ),
        )
    conn.commit()
    conn.close()
    print(f"[memory_layer] Seeded {len(recipes)} recipes.")


# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------
def to_json(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False)

def from_json(s: Optional[str]) -> Any:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


# ---------------------------------------------------------------------
# Thread-safe session logging
# ---------------------------------------------------------------------
_log_lock = threading.Lock()
def log_session_event(event: Dict[str, Any], plan_id: Optional[str] = None):
    plan_id = plan_id or event.get("plan_id") or "session"
    log_file = Path(__file__).resolve().parent.parent / "logs" / f"session_{plan_id}.jsonl"
    with _log_lock:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# MemoryLayer class - thin wrapper around DB operations
# ---------------------------------------------------------------------
class MemoryLayer:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else DB_PATH

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ----- Profiles -----
    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        out = dict(row)
        # parse JSON fields if present
        for col in ("preferences_json", "weightages_json", "session_overlay_json", "rejection_history_json"):
            if col in out and out[col] is not None:
                out[col] = from_json(out[col])
            else:
                out[col] = out.get(col) or {}
        return out

    def upsert_profile(
        self,
        user_id: str,
        name: str,
        preferences: Dict[str, Any],
        source: str = "agent",
        profile_version: int = 1,
        updated_at: Optional[str] = None,
        weightages: Optional[Dict[str, Any]] = None,
        session_overlay: Optional[Dict[str, Any]] = None,
        rejection_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Insert or update a profile record. This accepts partial components so callers
        don't need to provide all JSON blobs every time.
        """
        updated_at = updated_at or utc_now()
        conn = self._connect()
        cur = conn.cursor()

        # Fetch existing to merge if needed
        cur.execute("SELECT preferences_json, weightages_json, session_overlay_json, rejection_history_json, profile_version FROM profiles WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        if row:
            existing = dict(row)
            existing_prefs = from_json(existing.get("preferences_json"))
            existing_weights = from_json(existing.get("weightages_json"))
            existing_overlay = from_json(existing.get("session_overlay_json"))
            existing_rejections = from_json(existing.get("rejection_history_json"))

            # Simple merge strategy: overwrite with provided, keep others
            merged_prefs = {**existing_prefs, **(preferences or {})}
            merged_weights = {**existing_weights, **(weightages or {})}
            merged_overlay = (session_overlay or existing_overlay)
            new_version = (existing.get("profile_version") or 0) + 1

            # Only merge if explicit new rejections were passed
            if rejection_history:
                existing = existing_rejections or []
                new_unique = [
                    r for r in rejection_history
                    if not any(e.get("plan_id") == r.get("plan_id") and e.get("reason") == r.get("reason") for e in existing)
                ]
                merged_rejections = existing + new_unique
            else:
                merged_rejections = existing_rejections or []

        else:
            merged_prefs = preferences or {}
            merged_weights = weightages or {}
            merged_overlay = session_overlay or {}
            merged_rejections = rejection_history or []
            new_version = profile_version

        cur.execute(
            """
            INSERT INTO profiles (user_id, name, profile_version, preferences_json, weightages_json, session_overlay_json, rejection_history_json, updated_at, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                name = excluded.name,
                profile_version = excluded.profile_version,
                preferences_json = excluded.preferences_json,
                weightages_json = excluded.weightages_json,
                session_overlay_json = excluded.session_overlay_json,
                rejection_history_json = excluded.rejection_history_json,
                updated_at = excluded.updated_at,
                source = excluded.source
            """,
            (
                user_id,
                name or "user",
                new_version,
                to_json(merged_prefs),
                to_json(merged_weights),
                to_json(merged_overlay),
                to_json(merged_rejections),
                updated_at,
                source,
            ),
        )
        conn.commit()
        conn.close()

        return {
            "user_id": user_id,
            "name": name,
            "profile_version": new_version,
            "preferences_json": merged_prefs,
            "weightages_json": merged_weights,
            "session_overlay_json": merged_overlay,
            "rejection_history_json": merged_rejections,
            "updated_at": updated_at,
            "source": source,
        }

    # ----- Recipes -----
    def get_top_recipes(self, top_k: int = 50) -> List[Dict[str, Any]]:
        """Return up to top_k recipes (no ranking applied)."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM recipes LIMIT ?", (top_k,))
        rows = cur.fetchall()
        conn.close()
        out = []
        for r in rows:
            rec = dict(r)
            # parse JSON fields
            for key in ("main_ingredients", "tags", "nutrition_facts"):
                rec[key] = from_json(rec.get(key))
            out.append(rec)
        return out

    def get_recipe_by_id(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM recipes WHERE recipe_id = ?", (recipe_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        rec = dict(row)
        for key in ("main_ingredients", "tags", "nutrition_facts"):
            rec[key] = from_json(rec.get(key))
        return rec

    # ----- Plans & Run State -----
    def save_plan(self, user_id: str, plan_id: str, plan_json: Dict[str, Any], rationale: str, created_at: str):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO plans (plan_id, user_id, plan_json, candidate_recipe_ids, rationale, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'draft', ?, ?)
            """,
            (
                plan_id,
                user_id,
                to_json(plan_json),
                to_json([m for day in plan_json.get("plan", []) for m in day.get("meals", [])]) if plan_json else None,
                rationale,
                created_at,
                created_at,
            ),
        )
        conn.commit()
        conn.close()

    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM plans WHERE plan_id = ?", (plan_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        out = dict(row)
        out["plan_json"] = from_json(out.get("plan_json"))
        out["candidate_recipe_ids"] = from_json(out.get("candidate_recipe_ids"))
        return out

    def get_latest_plan_for_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM plans WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (user_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        out = dict(row)
        out["plan_json"] = from_json(out.get("plan_json"))
        out["candidate_recipe_ids"] = from_json(out.get("candidate_recipe_ids"))
        return out

    # ----- Run State -----
    def get_run_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM run_state WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1", (user_id,))
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None

    def update_run_state(self, run_id: str, user_id: str, stage: Optional[str], status: str, plan_id: Optional[str] = None, last_step: Optional[str] = None, error_message: Optional[str] = None):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO run_state
            (run_id, user_id, stage, last_step, status, current_plan_id, error_message, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (run_id, user_id, stage, last_step, status, plan_id, error_message),
        )
        conn.commit()
        conn.close()

    # ----- Plan status update -----
    def update_plan_status(self, plan_id: str, new_status: str, reason: Optional[str] = None):
        conn = self._connect()
        cur = conn.cursor()
        if reason:
            cur.execute(
                """
                UPDATE plans
                SET status = ?, rejection_reason = ?, updated_at = ?
                WHERE plan_id = ?
                """,
                (new_status, reason, utc_now(), plan_id),
            )
        else:
            cur.execute(
                """
                UPDATE plans
                SET status = ?, updated_at = ?
                WHERE plan_id = ?
                """,
                (new_status, utc_now(), plan_id),
            )
        conn.commit()
        conn.close()

    # ----- Unified Rejection Recording -----
    def append_rejection(self, user_id: str, plan_id: str, reason: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Record a rejection consistently:
        - Deduplicate by (plan_id, reason)
        - Update both 'plans' and 'profiles'
        - Log structured event to logs/session_*.jsonl
        """
        timestamp = utc_now()
        conn = self._connect()
        cur = conn.cursor()

        # --- 1. Update plan record ---
        cur.execute("""
            UPDATE plans
            SET status = 'rejected',
                rejection_reason = ?,
                updated_at = ?
            WHERE plan_id = ?
        """, (reason, timestamp, plan_id))

        # --- 2. Fetch existing profile history ---
        cur.execute("SELECT rejection_history_json, name FROM profiles WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        history = []
        name = "user"
        if row:
            history = from_json(row["rejection_history_json"])
            name = row["name"]

        # --- 3. Add new entry if not duplicate ---
        new_entry = {"plan_id": plan_id, "reason": reason, "metadata": metadata or {}, "timestamp": timestamp}
        already_exists = any(
            e.get("plan_id") == plan_id and e.get("reason") == reason for e in (history or [])
        )
        if not already_exists:
            history.append(new_entry)

        # --- 4. Update profile record ---
        cur.execute("""
            UPDATE profiles
            SET rejection_history_json = ?, updated_at = ?
            WHERE user_id = ?
        """, (to_json(history), timestamp, user_id))

        conn.commit()
        conn.close()

        # --- 5. Structured log ---
        log_session_event({
            "event": "plan_rejected",
            "user_id": user_id,
            "plan_id": plan_id,
            "reason": reason,
            "metadata": metadata or {},
            "timestamp": timestamp,
        }, plan_id)

        return history
    

    # ----- Errors -----
    def log_error(self, run_id: str, step: str, summary: str, hint: str = ""):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO errors (run_id, agent_name, step, summary, hint, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, "memory_layer", step[:200], summary[:400], hint[:400], utc_now()),
        )
        conn.commit()
        conn.close()


# ---------------------------------------------------------------------
# Module-level init -- make it easy to initialize DB when running this file
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("🔧 Initializing DB and seeding recipes...")
    init_db()
    seed_recipes_if_empty()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM recipes")
    print("🍽 Recipes in DB:", cur.fetchone()[0])
    conn.close()
