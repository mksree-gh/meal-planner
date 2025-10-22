# agents/planner_agent.py

"""
PlannerAgent — selects candidate recipes and asks an LLM to produce a 3-day,
2-meals-per-day plan (MealPlanOutput). Saves a draft plan to memory.

"""

import json
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

from google import genai
from google.genai import types

from config import get_agent_config, get_prompt_path, DB_PATH, should_log_prompts
from core.memory_layer import MemoryLayer, log_session_event
from core.reasoning import (
    merge_preferences,
    score_recipe_by_preferences,
)
from core.utils import utc_now, generate_id, log_event, safe_json, get_logger
from core.logging_layer import log_event, log_summary, log_console_line
from core.error_handler import log_error
from agents import PreferenceAgent

logger = get_logger("planner_agent")

# ------------------------------
# Output schema
# ------------------------------
class MealPlanDay(BaseModel):
    day: str
    meals: List[str] = Field(..., description="Exactly two meal names per day")

class MealPlanOutput(BaseModel):
    plan: List[MealPlanDay]
    rationale: str
    preference_add: bool

class PlannerChatOutput(BaseModel):
    reply: str
    preference_add: bool


# ------------------------------
# PlannerAgent
# ------------------------------
class PlannerAgent:
    def __init__(self, agent_name: str = "planner"):
        self.agent_name = agent_name
        self.agent_cfg = get_agent_config(agent_name)
        self.prompt_path = get_prompt_path(agent_name)
        self.client = genai.Client()
        self.memory = MemoryLayer(DB_PATH)

        self.primary_model = self.agent_cfg["primary_model"]
        self.fallback_model = self.agent_cfg["fallback_model"]

    # --------------------------
    def _filter_recipes(self, recipes: List[Dict[str, Any]], base_prefs: Dict[str, Any], overlay: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply hard filters: allergens, dislikes, overlay avoid list."""
        allergens = set((base_prefs.get("allergens") or []) or [])
        dislikes = set((base_prefs.get("dislikes") or []) or [])
        avoid = set((overlay.get("avoid") or []) or [])
        # print(f"sent receipes for filtering: {recipes}")
        def bad(r):
            ingredients = [str(i).lower() for i in (r.get("main_ingredients") or [])]
            tags = [t.lower() for t in (r.get("tags") or [])]
            # allergens or dislikes present in ingredients or tags?
            for a in allergens:
                if a and any(a.lower() in ing for ing in ingredients):
                    return True
            for d in dislikes:
                if d and any(d.lower() in ing for ing in ingredients):
                    return True
                if d and d.lower() in tags:
                    return True
            for av in avoid:
                if av and (av.lower() in tags or any(av.lower() in ing for ing in ingredients)):
                    return True
            return False

        filtered = [r for r in recipes if not bad(r)]
        # print(f"filterred recipes: {filtered}")
        return filtered

    # --------------------------
    def _greedy_diverse_sample(self, scored: List[Tuple[Dict[str, Any], float]], choose_k: int) -> List[Dict[str, Any]]:
        """
        Greedy unique sampling that prefers higher score but penalizes repeated cuisines.
        Algorithm:
          - start with highest-scored item
          - for next selections, compute adjusted_score = raw_score - penalty * (count of cuisine already chosen)
          - pick max adjusted_score, repeat until choose_k items
        """
        if not scored:
            return []

        # copy and sort descending by raw score
        pool = list(scored)
        pool.sort(key=lambda x: x[1], reverse=True)

        chosen: List[Dict[str, Any]] = []
        cuisine_counts: Dict[str, int] = {}
        penalty = 5.0  # magnitude relative to scores; tuned qualitatively

        while len(chosen) < min(choose_k, len(pool)):
            best_idx = None
            best_adj = -float("inf")
            for idx, (r, s) in enumerate(pool):
                cuisine = (r.get("cuisine") or "").lower()
                count = cuisine_counts.get(cuisine, 0)
                adj = s - penalty * count
                # small random jitter to break ties
                adj += random.random() * 1e-6
                if adj > best_adj:
                    best_adj = adj
                    best_idx = idx
            if best_idx is None:
                break
            best_recipe, best_score = pool.pop(best_idx)
            chosen.append(best_recipe)
            cuisine = (best_recipe.get("cuisine") or "").lower()
            cuisine_counts[cuisine] = cuisine_counts.get(cuisine, 0) + 1

        return chosen
    
    # --------------------------
    def _score_and_sample(self, candidates: List[Dict[str, Any]], base_prefs: Dict[str, Any], weightages: Dict[str, Any], sample_k: int = 20, choose_k: int = 12) -> List[Dict[str, Any]]:
        """
        Score candidates, keep top sample_k by score, then sample choose_k with weighted randomness.
        This balances preference alignment and novelty.
        """
        scored = []
        for r in candidates:
            score = score_recipe_by_preferences(r, weightages or {}, base_prefs or {})
            scored.append((r, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: max(0, min(len(scored), sample_k))]

        chosen = self._greedy_diverse_sample(top, choose_k)
        return chosen

    # --------------------------
    def _build_prompt(self, session_text: Optional[str], merged_view: Dict[str, Any], candidates: List[Dict[str, Any]], chat_history) -> str:
        """
        Build the planner prompt by combining the base prompt file and structured context:
        - session_text: user's immediate message
        - merged_view: result of merge_preferences
        - candidates: list of recipe snippets (max ~12)
        """
        base_prompt = self.prompt_path.read_text(encoding="utf-8")

        prefs = json.dumps(merged_view.get("base_preferences", {}), indent=2, ensure_ascii=False)
        weights = json.dumps(merged_view.get("weightages", {}), indent=2, ensure_ascii=False)
        overlay = json.dumps(merged_view.get("session_overlay", {}), indent=2, ensure_ascii=False)

        # compact recipe snippets
        snippets = []
        for r in candidates:
            snippets.append({
                "recipe_id": r.get("recipe_id"),
                "name": r.get("name"),
                "cuisine": r.get("cuisine"),
                "main_ingredients": r.get("main_ingredients"),
                "prep_time_min": r.get("prep_time_min"),
                "tags": r.get("tags"),
                "description": r.get("description"),
            })
        recipes_text = json.dumps(snippets, indent=2, ensure_ascii=False)

#         prompt = f"""{base_prompt}

# === Session Context ===
# {session}

# === Persistent Preferences (base) ===
# {prefs}

# === Weightages (adaptive) ===
# {weights}

# === Session Overlay (ephemeral) ===
# {overlay}

# === Candidate Recipes (subset) ===
# {recipes_text}

# Now generate a JSON output strictly following the MealPlanOutput schema: 3 days, exactly 2 meals per day.
# """
        instruction = f"""{base_prompt}

=== Persistent Preferences (base) ===
{prefs}

=== Weightages (adaptive) ===
{weights}

=== Session Overlay (ephemeral) ===
{overlay}

=== Candidate Recipes (subset) ===
{recipes_text}

=== Users Message ===
{chat_history[0]['parts'][0]['text']}

Now generate a JSON output strictly following the MealPlanOutput schema: 3 days, exactly 2 meals per day.
"""
        # chat_history_with_system = [{"role":"user","parts":[{"text": prompt}]}] + chat_history[1:]
        return instruction

    # --------------------------
    def _call_llm_for_plan(self, chat_history,instruction, run_id: str):
        last_exc = None
        for model_name in [self.primary_model, self.fallback_model]:
            try:
                logger.info("🤖 Calling %s to think...", model_name)
                start = time.time()
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=chat_history,
                    config=types.GenerateContentConfig(
                        temperature=self.agent_cfg["temperature"],
                        max_output_tokens=self.agent_cfg["max_output_tokens"],
                        response_mime_type=self.agent_cfg.get("response_mime_type", "application/json"),
                        system_instruction=instruction
                    ),
                )
                latency = int((time.time() - start) * 1000)
                logger.info("🕒 LLM response in %d ms", latency)

                # print(f"\n\n_call_llm_for_plan LLM response: {response.model_dump_json()}\n\n")

                parsed = getattr(response, "parsed", None)
                if parsed is not None:
                    return parsed,response

                # fallback to text extraction
                txt = getattr(response, "text", None)
                if txt:
                    import re
                    m = re.search(r"\{[\s\S]*\}", txt)
                    if m:
                        return json.loads(m.group(0)),response
                    return json.loads(txt),response
                raise RuntimeError("LLM returned no structured content")

            except Exception as e:
                last_exc = e
                logger.warning("LLM model %s failed: %s", model_name, e)
                if model_name == self.fallback_model:
                    log_error(run_id, self.agent_name, "llm_plan_call", str(e))
                    raise
                continue

        raise RuntimeError(f"LLM planning failed: {last_exc}")

    # --------------------------
    def generate_plan(
            self, 
            user_id: str, 
            session_text: Optional[str] = None, 
            top_k_candidates: int = 50,
            preferences_override: Optional[Dict[str, Any]] = None,
            memory: Optional[MemoryLayer] = None
            ) -> Dict[str, Any]:
        """
        Main planner entrypoint.
        - user_id: which user's preferences to use
        - session_text: immediate instruction (optional)
        - top_k_candidates: how many recipes to consider from DB before sampling
        Returns a dict with plan metadata and saved plan_id.
        """
        memory = memory or self.memory
        logger.info("📋 Generating plan for user=%s", user_id)
        run_id = generate_id("run")
        chat_history = []
        chat_history.append({"role":"user","parts":[{"text": session_text or "No immediate session request provided."}]})
        preference_agent_need = False
        feedback_text = session_text

        while True:
            # Load user profile and merged view
            

            # If weightages look stale, you can decay them here (optional)
            # weightages = decay_weights(weightages, decay=0.98)

            # Fetch candidate recipes and filter
            # all_recipes = self.memory.get_top_recipes(top_k_candidates)
            profile = self.memory.get_profile(user_id) or {}
            base_prefs = profile.get("preferences_json") or {}
            weightages = profile.get("weightages_json") or {}
            overlay = profile.get("session_overlay_json") or {}
            merged_view = merge_preferences(base_prefs, weightages, overlay, now_iso=utc_now())
            all_recipes = self.memory.get_top_recipes(100)
            filtered = self._filter_recipes(all_recipes, base_prefs, overlay)
            if not filtered:
                raise RuntimeError("No recipes available after filtering — check preferences/recipes dataset.")

            # Score & sample candidates
            # sampled = self._score_and_sample(filtered, base_prefs, weightages, sample_k=top_k_candidates, choose_k=30)
            sampled = filtered

            # Build prompt for LLM with sampled candidates
            merged_view = merge_preferences(base_prefs, weightages, overlay, now_iso=utc_now())
            instruction = self._build_prompt(session_text, merged_view, sampled,chat_history)

            # Log prompt (truncated)
            if should_log_prompts():
                log_session_event({"run_id": run_id, "agent": self.agent_name, "phase": "prompt", "prompt": chat_history[:8000]})

            # Call LLM to generate final plan
            plan_parsed, raw_response = self._call_llm_for_plan(chat_history,instruction, run_id)
            # Log raw LLM response
            if should_log_prompts():
                # store a trimmed representation of raw response (avoid huge dumped binary)
                resp_text = getattr(raw_response, "text", None) or str(getattr(raw_response, "candidates", None) or "")
                log_session_event({"run_id": run_id, "agent": self.agent_name, "phase": "llm_response", "response": resp_text[:8000]})


            if not isinstance(plan_parsed, dict):
                txt = getattr(raw_response, "text", None)
                if txt:
                    import re, json as _json
                    m = re.search(r"\{[\s\S]*\}", txt)
                    if m:
                        try:
                            plan_parsed = _json.loads(m.group(0))
                        except Exception:
                            pass
            plan = None
            try:
                plan_model = MealPlanOutput.model_validate(plan_parsed)
                plan = "Plan"
            except Exception as e:
                try: 
                    chat_model = PlannerChatOutput.model_validate(plan_parsed)
                    plan = "Chat"
                except ValidationError as ve:
                    run_id_err = generate_id("run")
                    log_error(run_id_err, self.agent_name, "validation", str(ve))
                    logger.error("❌ Invalid plan structure returned by LLM: %s", plan_parsed)
                    raise RuntimeError(
                        f"Plan validation failed — the model didn't return proper JSON. "
                        "You can inspect logs/session_*.jsonl for the exact response."
                    )
            if plan =="Chat":
                chat_history.append({"role":"model","parts":[{"text": f"{plan_parsed}"}]})
                print(f"AI Reply: {chat_model.reply}")
                preference_agent_need = chat_model.preference_add
            else:
                plan_json = plan_model.model_dump(exclude_none=True)
                rationale = plan_model.rationale

                chat_history.append({"role":"model","parts":[{"text": f"{plan_parsed}"}]})

                # Persist plan as draft
                plan_id = generate_id("plan")
                created_at = utc_now()
                # store the plan_json (structure) and rationale
                self.memory.save_plan(user_id, plan_id, plan_json, rationale, created_at)

                # Log structured event
                event = {
                    "user_id": user_id,
                    "plan_id": plan_id,
                    "candidate_count": len(sampled),
                    "created_at": created_at,
                    "days": len(plan_json.get("plan", []))
                }
                log_event(self.agent_name, "plan_generated", event)
                log_console_line(f"{self.agent_name}: {user_id} -> {event}")
                
                # Human-readable markdown summary
                meals_summary = "\n".join(
                    [f"- **{day['day']}**: {', '.join(day['meals'])}" for day in plan_json.get("plan", [])]
                )
                summary_text = f"""
    | ## 📋 Meal Plan — {user_id}
    | **Plan ID:** {plan_id}
    | **Candidates:** {len(sampled)}  
    | **Rationale:** {rationale[:200]}...

    {meals_summary}
                """
                log_summary(user_id, "planner", summary_text)

                logger.info("✅ Plan %s generated for %s", plan_id, user_id)
                print("✅ Draft meal plan created.\n")

                # Pretty print summary
                for day in plan_json.get("plan", []):
                    print(f"  📅 {day['day']}:")
                    for meal in day["meals"]:
                        print(f"    • {meal}")
                print(f"\n💡 Rationale: {rationale}\n")
                preference_agent_need = plan_model.preference_add

            
            if preference_agent_need:
                if feedback_text:
                    print("\n🧠 Updating preferences...")
                    pref_agent = PreferenceAgent()
                    merged_view = pref_agent.process_user_text(user_id=user_id, user_text=feedback_text, run_id= run_id)
                    print("✅ Preferences updated.\n")
                else:
                    # load existing profile to fill merged_view
                    profile = memory.get_profile(user_id) or {}
                    merged_view = {
                        "base_preferences": profile.get("preferences_json", {}) or {},
                        "weightages": profile.get("weightages_json", {}) or {},
                        "session_overlay": profile.get("session_overlay_json", {}) or {},
                    }
                    print("ℹ️ Using existing preferences.\n")
                
                preferences_override = merged_view


            if preferences_override:
                base_prefs = preferences_override.get("base_preferences", {}) or {}
                weightages = preferences_override.get("weightages", {}) or {}
                overlay = preferences_override.get("session_overlay", {}) or {}
                merged_view = preferences_override
            else:
                profile = self.memory.get_profile(user_id) or {}
                base_prefs = profile.get("preferences_json") or {}
                weightages = profile.get("weightages_json") or {}
                overlay = profile.get("session_overlay_json") or {}
                merged_view = merge_preferences(base_prefs, weightages, overlay, now_iso=utc_now())

            action = input(
        "Let us know if you like the plan (Enter A), "
        "want to request changes (type feedback), "
        "press R to reject, or Enter E to exit: ").strip().lower()

            feedback_text = None
            run_id = generate_id("run")

            if action in ("a", "approve"):
                # --- APPROVE ---

                log_event("planner", "plan_approved", {
                    "user_id": user_id,
                    "plan_id": plan_id,
                    "timestamp": utc_now(),
                })

                memory.update_plan_status(plan_id, "approved")
                memory.update_run_state(run_id, user_id, stage="approval", status="approved", plan_id=plan_id)

                print(f"\n🎉 Plan approved and finalized (plan_id={plan_id}).\n")
                return "Exit" 
            
            elif action in ("r", "reject"):
                # --- REJECT ---
                reason = input("💬 Why are you rejecting this plan?\n> ").strip()
                if not reason:
                    reason = "Rejected without a reason provided."

                memory.append_rejection(user_id, plan_id, reason)
                memory.update_run_state(generate_id("run"), user_id, stage="approval", status="rejected", plan_id=plan_id)

                print(f"\n❌ Plan rejected. Feedback saved: “{reason}”\n")

                feedback_text = f"Please revise the plan. Avoid / change: {reason}"

                log_event("planner", "feedback_cycle", {
                        "user_id": user_id,
                        "previous_plan_id": plan_id,
                        "feedback": reason,
                        "timestamp": utc_now(),
                    })

            elif action in ("e", "exit"):
                # --- EXIT ---
                print("\n🛑 Exiting per user request.\n")
                return "Exit"
            
            else:
                # --- FREE TEXT FEEDBACK ---
                if not action:
                    print("\n⚠️ No feedback provided, exiting.\n")
                    return "Exit"
                feedback_text = action

            
            # --- CONTINUE LOOP WITH FEEDBACK ---
            chat_history.append({"role":"user","parts":[{"text": feedback_text}]})


# ------------------------------
# CLI for quick testing
# ------------------------------
if __name__ == "__main__":
    import argparse
    import pprint

    parser = argparse.ArgumentParser("PlannerAgent CLI")
    parser.add_argument("--user-id", default="user_001", help="User ID")
    parser.add_argument("--text", default=None, help="Session-level text for this planning round")
    args = parser.parse_args()

    agent = PlannerAgent()
    try:
        plan = agent.generate_plan(user_id=args.user_id, session_text=args.text)
        print("\n📋 Draft Meal Plan:\n")
        pprint.pp(plan)
    except Exception as e:
        logger.exception("PlannerAgent failed: %s", e)
        raise
