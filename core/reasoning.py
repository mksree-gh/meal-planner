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
    existing: Dict[str, float],
    deltas: Dict[str, float],
    learning_rate: float = 0.5,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Merge 'deltas' into 'existing' using a simple additive update:
      new = existing * (1 - lr) + delta * lr
    - existing: current weights (0..1)
    - deltas: new signals (0..1), expected same keyspace (missing keys considered 0)
    - learning_rate: how strongly to accept deltas (0..1)
    Returns normalized (optional) dictionary.
    """
    out = dict(existing or {})
    for k, dv in (deltas or {}).items():
        old = _clamp(out.get(k, 0.0))
        delta_v = _clamp(dv)
        out[k] = _clamp(old * (1.0 - learning_rate) + delta_v * learning_rate)
    if normalize:
        return normalize_weights(out)
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
# Rejection handling
# -------------------------
def record_rejection(
    rejection_history: List[Dict[str, Any]] | None,
    plan_id: str,
    reason: str,
    metadata: Dict[str, Any] | None = None,
    timestamp_iso: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Append a structured rejection entry to rejection_history (list) and return new list.
    metadata can include which cuisines were disliked or other structured fields.
    """
    ts = timestamp_iso or utc_now()
    entry = {"plan_id": plan_id, "reason": reason, "metadata": metadata or {}, "timestamp": ts}
    out = list(rejection_history or [])
    out.append(entry)
    return out

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
