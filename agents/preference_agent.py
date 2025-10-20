# agents/preference_agent.py

"""
PreferenceAgent — interprets natural-language user input and updates
the user's preference profile in structured form.

"""

import json
import logging
import time
from typing import Dict, Any, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from config import get_agent_config, get_prompt_path, DB_PATH
from core.memory_layer import MemoryLayer, log_session_event
from core.reasoning import (
    merge_preferences,
    update_weights,
    create_session_overlay,
)
from core.utils import utc_now, generate_id, log_event, safe_json
from core.error_handler import log_error

logger = logging.getLogger("preference_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

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
                try:
                    fv = float(v)
                except Exception:
                    fv = 0.0
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

        # print(f"\n\nParsed Raw from LLM: {parsed_raw}\n\n")

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
        print("\n\nDebug Info:")
        print(f"Existing Weights: {existing_weights}")

        # Merge weights axis-wise (existing_weights can be nested; update per axis)
        merged_weights = dict(existing_weights or {})
        for axis, deltas in normalized_weightages.items():
            axis_existing = merged_weights.get(axis, {}) or {}
            merged_axis = update_weights(axis_existing, deltas)
            merged_weights[axis] = merged_axis

        print(f"Merged Weights: {merged_weights}\n\n")

        # Merge base preferences (structured.base_preferences can contain diet/allergens/dislikes/goals)
        base_prefs_from_llm = structured.base_preferences or {}
        # normalize base_prefs keys and lower-case lists (where applicable)
        normalized_base = {}
        for k, v in base_prefs_from_llm.items():
            if isinstance(v, list):
                normalized_base[k] = [str(x).lower() for x in v]
            elif isinstance(v, str):
                normalized_base[k] = v.lower()
            else:
                normalized_base[k] = v

        # Produce merged_view (for planner and UI)
        merged_view = merge_preferences(existing_prefs, merged_weights, merged_overlay, now_iso=utc_now())
        # Ensure the base_preferences part is updated with new base prefs
        # (we'll merge normalized_base into merged_view.base_preferences)
        merged_view_base = dict(merged_view.get("base_preferences") or {})
        merged_view_base.update(normalized_base)
        merged_view["base_preferences"] = merged_view_base

        # Persist to DB: pass explicit blobs to upsert_profile so memory_layer stores cleanly
        profile_version = (profile.get("profile_version", 0) or 0) + 1
        upserted = self.memory.upsert_profile(
            user_id=user_id,
            name=profile.get("name", "user"),
            preferences=merged_view_base,
            source=self.agent_name,
            profile_version=profile_version,
            updated_at=utc_now(),
            weightages=merged_weights,
            session_overlay=merged_overlay,
            rejection_history=profile.get("rejection_history_json", []) or [],
        )

        # Log session event (structured)
        event = {
            "run_id": run_id,
            "user_id": user_id,
            "agent": self.agent_name,
            "input_text": user_text,
            "llm_parsed": safe_json(parsed_raw),
            "merged_view": merged_view,
            "timestamp": utc_now(),
        }
        log_event(self.agent_name, "preference_update", event)

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
