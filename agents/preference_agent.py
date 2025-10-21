# agents/preference_agent.py

"""
PreferenceAgent — interprets natural-language user input and updates
the user's preference profile in structured form.

"""

import json
import time
from typing import Dict, Any, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from config import get_agent_config, get_prompt_path, DB_PATH
from core.memory_layer import MemoryLayer, log_session_event
from core.reasoning import (
    merge_preferences,
    merge_base_preferences,
    update_weights,
    create_session_overlay,
)
from core.utils import utc_now, generate_id, log_event, safe_json, log_weight_diff, get_logger
from core.logging_layer import log_summary, log_console_line
from core.error_handler import log_error

logger = get_logger("PreferenceAgent")

# --- Helper normalizer ---
def _normalize_axes(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lowercase keys for nested maps like cuisines/ingredients, and ensure numeric weights are floats.
    Input e.g. {"cuisines": {"Italian": 0.9}, "ingredients": {"Spinach": 0.8}}
    """
    out = {}
    for axis, val in (d or {}).items():
        if isinstance(val, dict):
            normalized = {}
            for k, v in val.items():
                if isinstance(v, dict):
                    # try to extract numeric field if nested
                    numeric_candidates = [vv for vv in v.values() if isinstance(vv, (int, float))]
                    v = numeric_candidates[0] if numeric_candidates else v
                try:
                    fv = float(v)
                except Exception:
                    continue
                normalized[str(k).lower()] = max(0.0, min(1.0, fv))
            out[str(axis).lower()] = normalized
        else:
            out[str(axis).lower()] = val
    return out


class PreferenceOutput(BaseModel):
    base_preferences: Dict[str, Any] = Field(default_factory=dict)
    # nested axes, e.g. {"cuisines": {"italian":0.9}, "ingredients":{"spinach":0.8}}
    weightages: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    session_overlay: Dict[str, Any] = Field(default_factory=dict)
    rationale: Optional[str] = None


class PreferenceAgent:
    def __init__(self, agent_name: str = "preference"):
        self.agent_name = agent_name
        self.agent_cfg = get_agent_config(agent_name)
        self.prompt_path = get_prompt_path(agent_name)
        self.client = genai.Client()
        self.memory = MemoryLayer(DB_PATH)

        self.primary_model = self.agent_cfg["primary_model"]
        self.fallback_model = self.agent_cfg["fallback_model"]

    def process_user_text(self, user_id: str, user_text: str) -> Dict[str, Any]:
        logger.info(f"🧠 Processing preferences for user={user_id}")

        run_id = generate_id("run")
        base_prompt = self.prompt_path.read_text(encoding="utf-8")
        prompt = f"{base_prompt.strip()}\n\nUser message:\n{user_text.strip()}"

        # Log prompt (truncated) to session log for debugging
        log_session_event({"run_id": run_id, "agent": self.agent_name, "phase": "prompt", "prompt": prompt[:4000]})

        # Call LLM (primary -> fallback)
        parsed_raw = self._call_llm(prompt)

        # Log raw LLM response (truncated)
        log_session_event({"run_id": run_id, "agent": self.agent_name, "phase": "llm_response", "response": safe_json(parsed_raw)[:4000]})


        # Validate structured output
        try:
            structured = PreferenceOutput.model_validate(parsed_raw)
        except ValidationError as ve:
            logger.error("LLM produced invalid structure: %s", ve)
            raise RuntimeError(f"Invalid LLM output: {ve}")
        
        # Normalize axes
        normalized_weightages = _normalize_axes(structured.weightages or {})

        # Build session overlay properly
        so = structured.session_overlay or {}
        requested_cuisines = None
        requested_ingredients = None
        avoid = None
        ttl_days = None

        # support multiple variant keys gracefully
        if so:
            requested_cuisines = so.get("requested_cuisines") or so.get("cuisines") or so.get("requested_cuisine")
            requested_ingredients = so.get("requested_ingredients") or so.get("ingredients")
            avoid = so.get("avoid") or so.get("avoid_list") or so.get("avoids")
            ttl_days = so.get("ttl_days") or so.get("ttl") or None

        if requested_cuisines:
            requested_cuisines = [str(x).lower() for x in requested_cuisines]
        if requested_ingredients:
            requested_ingredients = [str(x).lower() for x in requested_ingredients]
        if avoid:
            avoid = [str(x).lower() for x in avoid]

        if requested_cuisines or requested_ingredients or avoid:
            merged_overlay = create_session_overlay(
                requested_cuisines=requested_cuisines,
                requested_ingredients=requested_ingredients,
                avoid=avoid,
                ttl_days=ttl_days,
            )
        else:
            # keep existing overlay if none provided
            profile = self.memory.get_profile(user_id) or {}
            merged_overlay = profile.get("session_overlay_json", {}) or {}

        # Fetch existing profile for merging
        profile = self.memory.get_profile(user_id) or {}
        existing_prefs = profile.get("preferences_json", {}) or {}
        existing_weights = profile.get("weightages_json", {}) or {}

        # Extract only the base_preferences from the merged view
        if "base_preferences" in existing_prefs:
            existing_base_prefs = existing_prefs["base_preferences"]
        else:
            existing_base_prefs = existing_prefs

        # --- Merge weights per axis (with recency/timestamp support) ---
        merged_weights = {}
        for axis, deltas in normalized_weightages.items():
            existing_axis = existing_weights.get(axis, {}) or {}
            merged_weights[axis] = update_weights(existing_axis, deltas)
            log_weight_diff(self.agent_name, axis, existing_axis, merged_weights[axis], user_id)

        # --- Merge base preferences intelligently (union lists) ---
        normalized_base = {}
        for k, v in (structured.base_preferences or {}).items():
            if isinstance(v, list):
                normalized_base[k] = [str(x).lower() for x in v]
            elif isinstance(v, str):
                normalized_base[k] = v.lower()
            else:
                normalized_base[k] = v

        merged_base = merge_base_preferences(existing_base_prefs, normalized_base)

        # --- Combine full merged view ---
        merged_view = merge_preferences(merged_base, merged_weights, merged_overlay, now_iso=utc_now())


        # Persist to DB: pass explicit blobs to upsert_profile so memory_layer stores cleanly
        profile_version = (profile.get("profile_version", 0) or 0) + 1
        self.memory.upsert_profile(
            user_id=user_id,
            name=profile.get("name", "user"),
            preferences=merged_view,
            source=self.agent_name,
            profile_version=profile_version,
            updated_at=utc_now(),
            weightages=merged_weights,
            session_overlay=merged_overlay,
            rejection_history=profile.get("rejection_history_json", []) or [],
        )

        # --- Structured log ---
        event = {
            "run_id": run_id,
            "user_id": user_id,
            "agent": self.agent_name,
            "input_text": user_text,
            "merged_view": merged_view,
            "timestamp": utc_now(),
        }
        log_event(self.agent_name, "preference_update", event)
        log_console_line(f"{self.agent_name}: {user_id} -> {event}")

        # --- Human summary for developer trace ---
        summary_lines = []
        for axis, new_axis in merged_weights.items():
            diffs = []
            old_axis = existing_weights.get(axis, {})
            for k, new_val in new_axis.items():
                old_val = old_axis.get(k)
                try:
                    new_f = float(new_val) if not isinstance(new_val, dict) else None
                    old_f = float(old_val) if not isinstance(old_val, dict) else None
                except Exception:
                    new_f, old_f = None, None

                # Case 1: new numeric only
                if old_val is None and new_f is not None:
                    diffs.append(f"+{k} ↑{new_f:.2f}")
                # Case 2: both numeric
                elif old_f is not None and new_f is not None and old_f != new_f:
                    delta = new_f - old_f
                    sym = "↑" if delta > 0 else "↓"
                    diffs.append(f"{k} {sym}{abs(delta):.2f}")
                # Case 3: complex nested dicts or invalid data
                elif isinstance(new_val, dict):
                    diffs.append(f"{k} [complex update]")
            if diffs:
                summary_lines.append(f"⚖️ **{axis}**: {', '.join(diffs)}")

        summary_text = f"""
| ### 🧠 Preference Update — {user_id}
| **Input:** {user_text}
| {chr(10).join(summary_lines) or 'No notable weight changes.'}
| ✅ Profile version {profile_version}
"""
        log_summary(user_id, "preference", summary_text)
        logger.info("✅ Preferences updated successfully for %s", user_id)

        return merged_view

    def _build_prompt(self, user_text: str) -> str:
        base_prompt = self.prompt_path.read_text(encoding="utf-8")
        return f"{base_prompt.strip()}\n\nUser message:\n{user_text.strip()}"

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        # Attempt primary then fallback
        last_exc = None
        for model_name in [self.primary_model, self.fallback_model]:
            try:
                logger.info("🤖 Calling %s for structured preference parsing", model_name)
                start = time.time()
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.agent_cfg["temperature"],
                        max_output_tokens=self.agent_cfg["max_output_tokens"],
                        response_mime_type=self.agent_cfg.get("response_mime_type", "application/json"),
                    ),
                )
                latency = int((time.time() - start) * 1000)
                logger.info("🕒 LLM response in %d ms", latency)

                parsed = getattr(response, "parsed", None)
                if parsed is not None:
                    return parsed

                # Fall back to text parsing if structured parse missing
                txt = getattr(response, "text", None)
                if txt:
                    try:
                        return json.loads(txt)
                    except Exception:
                        # try a safe JSON extraction
                        import re
                        m = re.search(r"\{[\s\S]*\}", txt)
                        if m:
                            return json.loads(m.group(0))
                        raise

                raise RuntimeError("LLM returned neither parsed JSON nor text.")

            except Exception as e:
                last_exc = e
                logger.warning("LLM model %s failed: %s", model_name, e)
                # If fallback also fails, log and re-raise
                if model_name == self.fallback_model:
                    run_id = generate_id("run")
                    log_error(run_id, self.agent_name, "llm_call", str(e))
                    raise
                # otherwise try next model
                continue

        # if we exit loop with no return, raise last exception
        raise RuntimeError(f"LLM calls failed: {last_exc}")

# If run as module for quick test, keep the same CLI usage
if __name__ == "__main__":
    import argparse, pprint

    parser = argparse.ArgumentParser("PreferenceAgent CLI")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    agent = PreferenceAgent()
    out = agent.process_user_text(user_id=args.user_id, user_text=args.text)
    print("\n✅ Final merged preferences:\n")
    pprint.pp(out)
