# core/reasoning.py

""""

Shared reasoning helpers for Posha.

Responsibilities:
- Merge user preference layers: base preferences, adaptive weightages, and session overlays
- Update and normalize weightages when new signals arrive
- Decay weights over time (time-based decay / aging)
- Manage session overlay TTL and expiration
- Record rejection feedback as structured signals for later learning
- Small helper: convert preference signals into a simple scoring function (used by Planner)

This module is intentionally *stateless*: it operates on Python dicts and returns
new dicts so callers (agents / memory layer) control persistence.
"""

from __future__ import annotations
from typing import Dict, Any, Iterable, List, Tuple
from datetime import datetime, timezone, timedelta
import math

from core.utils import utc_now

# -------------------------
# Constants / Defaults
# -------------------------
DEFAULT_DECAY = 0.92       # multiplicative decay per "decay" call (e.g., per day or per plan)
MIN_WEIGHT = 0.0
MAX_WEIGHT = 1.0

SESSION_TTL_DAYS = 7       # default session overlay validity (if not provided)

# -------------------------
# Utilities
# -------------------------
def _clamp(v: float, lo: float = MIN_WEIGHT, hi: float = MAX_WEIGHT) -> float:
    try:
        fv = float(v)
    except Exception:
        return lo
    if math.isnan(fv):
        return lo
    return max(lo, min(hi, fv))

def _now_dt() -> datetime:
    return datetime.now(timezone.utc)

def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

# -------------------------
# Weight utilities
# -------------------------
def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize a dict of weights so max becomes 1.0 (if any positive),
    otherwise returns the clipped weights.
    """
    if not weights:
        return {}
    cleaned = {k: _clamp(v) for k, v in weights.items()}
    max_v = max(cleaned.values()) if cleaned else 0.0
    if max_v <= 0:
        return cleaned
    factor = 1.0 / max_v
    return {k: _clamp(v * factor) for k, v in cleaned.items()}


def decay_weights(weights: Dict[str, float], decay: float = DEFAULT_DECAY) -> Dict[str, float]:
    """
    Apply multiplicative decay to all numeric weights.
    Keep keys and return a new dict.
    """
    if not weights:
        return {}
    return {k: _clamp(v * decay) for k, v in weights.items()}




def update_weights(
    existing: dict[str, Any],
    deltas: dict[str, float],
    learning_rate: float = 0.5,
    recency_half_life_days: int = 14,
    category_type: str = "generic",  # e.g., "cuisine", "diet", "allergen"
) -> dict[str, Any]:
    """
    Merge new weight signals into existing ones using:
      • adaptive learning rate based on recency
      • controlled decay (category-aware)
      • recency bias for latest preference
      • no normalization (absolute preferences allowed)

    Args:
        existing: current stored weights (with 'value' and 'last_updated')
        deltas: new preference values (0.0–1.0)
        learning_rate: how fast to adapt to new input (default 0.5)
        recency_half_life_days: how long before preference halves naturally
        category_type: used to adjust decay (e.g. 'diet' = low decay)

    Example:
        {"italian": {"value": 0.8, "last_updated": "2025-10-20T14:32:00Z"}}
    """

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    out: dict[str, Any] = {}

    # --- 1. Normalize existing structure ---
    for k, v in (existing or {}).items():
        if isinstance(v, dict) and "value" in v:
            out[k] = {
                "value": float(v["value"]),
                "last_updated": v.get("last_updated", now.isoformat().replace("+00:00", "Z")),
            }
        else:
            out[k] = {"value": float(v), "last_updated": now.isoformat().replace("+00:00", "Z")}

    # --- 2. Apply category-aware recency decay ---
    decay_modifiers = {
        "cuisine": 1.0,    # decays normally
        "diet": 0.1,       # almost stable
        "allergen": 0.0,   # no decay at all
        "generic": 1.0,
    }
    decay_factor_base = decay_modifiers.get(category_type, 1.0)

    for k, meta in list(out.items()):
        try:
            t_prev = datetime.fromisoformat(meta["last_updated"].replace("Z", "+00:00"))
            age_days = (now - t_prev).days
            # half-life formula
            decay_factor = (0.5 ** (age_days / recency_half_life_days)) ** decay_factor_base
            meta["value"] *= decay_factor
        except Exception:
            pass  # ignore invalid timestamps

    # --- 3. Merge incoming deltas with adaptive learning ---
    for key, delta_val in (deltas or {}).items():
        delta_val = max(0.0, min(1.0, float(delta_val)))
        prev_meta = out.get(key, {"value": 0.0, "last_updated": now.isoformat().replace("+00:00", "Z")})
        prev_val = prev_meta["value"]

        # dynamic learning rate: boost if last update was long ago
        try:
            t_prev = datetime.fromisoformat(prev_meta["last_updated"].replace("Z", "+00:00"))
            age_days = (now - t_prev).days
            recency_boost = 1.0 + min(1.0, age_days / recency_half_life_days)
        except Exception:
            recency_boost = 1.0
        lr = learning_rate * recency_boost

        merged_val = prev_val * (1 - lr) + delta_val * lr
        out[key] = {
            "value": round(max(0.0, min(1.0, merged_val)), 3),
            "last_updated": now.isoformat().replace("+00:00", "Z"),
        }

    # --- 4. Recency bias: boost the latest updated preference slightly ---
    if deltas:
        latest_key = list(deltas.keys())[-1]
        if latest_key in out:
            out[latest_key]["value"] = round(min(1.0, out[latest_key]["value"] * 1.15), 3)

    return out


# -------------------------
# Preference merging
# -------------------------
def merge_preferences(
    base_prefs: Dict[str, Any],
    weightages: Dict[str, Any],
    session_overlay: Dict[str, Any],
    now_iso: str | None = None,
) -> Dict[str, Any]:
    """
    Produce a merged preference view that the Planner can consume.
    - base_prefs: canonical long-term preferences (diet, allergens, dislikes, goals)
    - weightages: adaptive numeric axes (cuisines, ingredients, novelty, variety_bias)
    - session_overlay: ephemeral requests (requested_cuisines, requested_ingredients, avoid, valid_until)
    Returns a dict with keys: base, weights, overlay, merged_timestamp
    Note: this does NOT persist anything; agents should write results to DB.
    """
    now_iso = now_iso or utc_now()

    # Shallow copies to avoid mutating inputs
    base = dict(base_prefs or {})
    w = dict(weightages or {})
    overlay = dict(session_overlay or {})

    # If overlay has valid_until, and it's expired, drop it
    if overlay:
        valid_until = overlay.get("valid_until")
        if valid_until:
            dt = _parse_iso(valid_until)
            if dt and dt < _now_dt():
                overlay = {}

    merged = {
        "base_preferences": base,
        "weightages": w,
        "session_overlay": overlay,
        "merged_at": now_iso,
    }
    return merged

def merge_base_preferences(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """
    Combine long-term base preferences.
    - Union additive lists (allergens, dislikes, goals)
    - Replace diet if explicitly changed
    - Lower-case everything
    """
    merged = dict(existing or {})
    for k, v in (new or {}).items():
        if isinstance(v, list):
            existing_vals = [str(x).lower() for x in merged.get(k, [])]
            merged[k] = sorted(set(existing_vals + [str(x).lower() for x in v]))
        elif k == "diet":
            if v and v.lower() != merged.get("diet"):
                merged[k] = v.lower()
        else:
            merged[k] = v
    return merged


# -------------------------
# Session overlay helpers
# -------------------------
def create_session_overlay(
    requested_cuisines: Iterable[str] | None = None,
    requested_ingredients: Iterable[str] | None = None,
    avoid: Iterable[str] | None = None,
    ttl_days: int | None = None,
) -> Dict[str, Any]:
    """
    Create a session overlay dict with an explicit expiry timestamp (ISO).
    """
    ttl_days = ttl_days if ttl_days is not None else SESSION_TTL_DAYS
    expiry = (_now_dt() + timedelta(days=ttl_days)).isoformat().replace("+00:00", "Z")
    overlay: Dict[str, Any] = {"valid_until": expiry}
    if requested_cuisines:
        overlay["requested_cuisines"] = list(dict.fromkeys([c.lower() for c in requested_cuisines]))
    if requested_ingredients:
        overlay["requested_ingredients"] = list(dict.fromkeys([c.lower() for c in requested_ingredients]))
    if avoid:
        overlay["avoid"] = list(dict.fromkeys([a.lower() for a in avoid]))
    overlay["created_at"] = utc_now()
    return overlay


# -------------------------
# Simple scoring helper
# -------------------------
def score_recipe_by_preferences(
    recipe: Dict[str, Any],
    weightages: Dict[str, Any],
    base_prefs: Dict[str, Any] | None = None,
) -> float:
    """
    Produce a simple score (0..1) for a recipe based on weightages and base preferences.
    This is intentionally lightweight: the PlannerAgent may replace with a learned model later.

    Factors considered:
    - cuisine weight (weightages.get('cuisines', {}))
    - ingredient hits (weightages.get('ingredients', {}))
    - prep time bias: if weightages contains 'prep_effort_importance', prefer lower prep_time
    - penalize if recipe contains an allergen or a base dislike (score -> 0)
    """

    if not recipe or not weightages:
        return 0.0

    # Basic allergy/dislike filter
    allergens = set((base_prefs or {}).get("allergens", []) or [])
    dislikes = set((base_prefs or {}).get("dislikes", []) or [])
    ingredients = [str(i).lower() for i in (recipe.get("main_ingredients") or [])]

    if any(a.lower() in ing for a in allergens for ing in ingredients):
        return 0.0
    if any(d.lower() in ing for d in dislikes for ing in ingredients):
        return 0.0

    score = 0.0
    # cuisine
    cuisine = (recipe.get("cuisine") or "").lower()
    cuisine_weights = weightages.get("cuisines", {}) or {}
    score += _clamp(cuisine_weights.get(cuisine, 0.0))

    # ingredients: sum top matches
    ingredient_weights = weightages.get("ingredients", {}) or {}
    ingredient_score = 0.0
    for ing in ingredients:
        ingredient_score = max(ingredient_score, _clamp(ingredient_weights.get(ing, 0.0)))
    score += ingredient_score

    # goals / tags
    tags = [t.lower() for t in (recipe.get("tags") or [])]
    goal_weights = weightages.get("goals", {}) or {}
    for g, wgt in goal_weights.items():
        if g.lower() in tags:
            score += _clamp(wgt)

    # prep effort bias (reduce score if prep_time large and importance high)
    prep_importance = _clamp(weightages.get("prep_effort_importance", 0.0))
    prep_time = recipe.get("prep_time_min") or 0
    # simple mapping: prefer prep < 25
    time_penalty = max(0.0, (prep_time - 25) / 60.0)  # scale
    score -= prep_importance * time_penalty

    # Normalize heuristically: clamp and scale to 0..1
    return _clamp(score / 3.0)  # denominator chosen empirically

# -------------------------
# Helpful debug printer
# -------------------------
def explain_weights(weights: Dict[str, Any]) -> str:
    """Return a short human-friendly summary of weight dicts."""
    parts = []
    for axis, mapping in (weights or {}).items():
        if isinstance(mapping, dict):
            top = sorted(mapping.items(), key=lambda kv: kv[1], reverse=True)[:5]
            parts.append(f"{axis}: " + ", ".join(f"{k}({v:.2f})" for k, v in top))
        else:
            parts.append(f"{axis}: {mapping}")
    return " | ".join(parts)
