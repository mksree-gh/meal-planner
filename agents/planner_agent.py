# agents/planner_agent.py

"""
PlannerAgent — Refactored to work with Streamlit's stateless architecture.
Key changes:
1. Removed while True loop
2. Added state-based processing
3. Returns intermediate results instead of blocking for input
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
from core.reasoning import merge_preferences, score_recipe_by_preferences
from core.utils import utc_now, generate_id, log_event, safe_json, get_logger
from core.logging_layer import log_event, log_summary, log_console_line
from core.error_handler import log_error
from agents import PreferenceAgent

logger = get_logger("planner_agent")

# ------------------------------
# Output schema (unchanged)
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
# NEW: Result types for state management
# ------------------------------
class PlannerResult(BaseModel):
    """Result object returned by generate_plan_step"""
    status: str  # "chat", "plan_ready", "completed", "error"
    message: Optional[str] = None
    plan_id: Optional[str] = None
    plan_json: Optional[Dict[str, Any]] = None
    rationale: Optional[str] = None
    chat_reply: Optional[str] = None
    preference_add: bool = False
    error: Optional[str] = None


# ------------------------------
# PlannerAgent (Refactored)
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
    # Helper methods (unchanged)
    # --------------------------
    def _filter_recipes(self, recipes: List[Dict[str, Any]], base_prefs: Dict[str, Any], overlay: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply hard filters: allergens, dislikes, overlay avoid list."""
        allergens = set((base_prefs.get("allergens") or []) or [])
        dislikes = set((base_prefs.get("dislikes") or []) or [])
        avoid = set((overlay.get("avoid") or []) or [])
        
        def bad(r):
            ingredients = [str(i).lower() for i in (r.get("main_ingredients") or [])]
            tags = [t.lower() for t in (r.get("tags") or [])]
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
        return filtered

    def _greedy_diverse_sample(self, scored: List[Tuple[Dict[str, Any], float]], choose_k: int) -> List[Dict[str, Any]]:
        """Greedy unique sampling that prefers higher score but penalizes repeated cuisines."""
        if not scored:
            return []

        pool = list(scored)
        pool.sort(key=lambda x: x[1], reverse=True)

        chosen: List[Dict[str, Any]] = []
        cuisine_counts: Dict[str, int] = {}
        penalty = 5.0

        while len(chosen) < min(choose_k, len(pool)):
            best_idx = None
            best_adj = -float("inf")
            for idx, (r, s) in enumerate(pool):
                cuisine = (r.get("cuisine") or "").lower()
                count = cuisine_counts.get(cuisine, 0)
                adj = s - penalty * count
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
    
    def _score_and_sample(self, candidates: List[Dict[str, Any]], base_prefs: Dict[str, Any], weightages: Dict[str, Any], sample_k: int = 20, choose_k: int = 12) -> List[Dict[str, Any]]:
        """Score candidates, keep top sample_k by score, then sample choose_k with weighted randomness."""
        scored = []
        for r in candidates:
            score = score_recipe_by_preferences(r, weightages or {}, base_prefs or {})
            scored.append((r, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: max(0, min(len(scored), sample_k))]
        chosen = self._greedy_diverse_sample(top, choose_k)
        return chosen

    def _build_prompt(self, session_text: Optional[str], merged_view: Dict[str, Any], candidates: List[Dict[str, Any]], chat_history) -> str:
        """Build the planner prompt."""
        base_prompt = self.prompt_path.read_text(encoding="utf-8")

        prefs = json.dumps(merged_view.get("base_preferences", {}), indent=2, ensure_ascii=False)
        weights = json.dumps(merged_view.get("weightages", {}), indent=2, ensure_ascii=False)
        overlay = json.dumps(merged_view.get("session_overlay", {}), indent=2, ensure_ascii=False)

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

        print(chat_history)

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
        return instruction

    def _call_llm_for_plan(self, chat_history, instruction, run_id: str):
        """Call LLM to generate plan or chat response."""
        last_exc = None
        for model_name in [self.primary_model, self.fallback_model]:
            try:
                logger.info("🤖 Calling %s to generate meal plan", model_name)
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

                parsed = getattr(response, "parsed", None)
                if parsed is not None:
                    return parsed, response

                txt = getattr(response, "text", None)
                if txt:
                    import re
                    m = re.search(r"\{[\s\S]*\}", txt)
                    if m:
                        return json.loads(m.group(0)), response
                    return json.loads(txt), response
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
    # NEW: State-based generation
    # --------------------------
    def generate_plan_step(
        self, 
        user_id: str, 
        user_message: str,
        chat_history: List[Dict[str, Any]],
        preferences_override: Optional[Dict[str, Any]] = None,
        memory: Optional[MemoryLayer] = None,       
        run_id: str = None,
    ) -> PlannerResult:
        """
        Single step of plan generation - called once per user message.
        Returns a PlannerResult indicating what happened and what to do next.
        
        Args:
            user_id: User identifier
            user_message: Current user message/feedback
            chat_history: Full conversation history in LLM format
            preferences_override: Override preferences if provided
            memory: Memory layer instance
        
        Returns:
            PlannerResult with status and next action
        """
        memory = memory or self.memory
        # run_id = generate_id("run")
        
        try:
            # Load or use override preferences
            if preferences_override:
                base_prefs = preferences_override.get("base_preferences", {}) or {}
                weightages = preferences_override.get("weightages", {}) or {}
                overlay = preferences_override.get("session_overlay", {}) or {}
                merged_view = preferences_override
            else:
                profile = memory.get_profile(user_id) or {}
                base_prefs = profile.get("preferences_json") or {}
                weightages = profile.get("weightages_json") or {}
                overlay = profile.get("session_overlay_json") or {}
                merged_view = merge_preferences(base_prefs, weightages, overlay, now_iso=utc_now())

            # Fetch and filter recipes
            all_recipes = memory.get_top_recipes(100)
            filtered = self._filter_recipes(all_recipes, base_prefs, overlay)
            
            if not filtered:
                return PlannerResult(
                    status="error",
                    error="No recipes available after filtering — check preferences/recipes dataset."
                )

            # Build prompt and call LLM
            instruction = self._build_prompt(user_message, merged_view, filtered, chat_history)
            
            if should_log_prompts():
                log_session_event({
                    "run_id": run_id, 
                    "agent": self.agent_name, 
                    "phase": "prompt", 
                    "prompt": str(chat_history)
                })

            plan_parsed, raw_response = self._call_llm_for_plan(chat_history, instruction, run_id)
            
            if should_log_prompts():
                resp_text = getattr(raw_response, "text", None) or str(getattr(raw_response, "candidates", None) or "")
                log_session_event({
                    "run_id": run_id, 
                    "agent": self.agent_name, 
                    "phase": "llm_response", 
                    "response": resp_text
                })

            # Parse response as dict if needed
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

            # Try to validate as plan or chat
            try:
                plan_model = MealPlanOutput.model_validate(plan_parsed)
                
                # Save plan to database
                plan_json = plan_model.model_dump(exclude_none=True)
                rationale = plan_model.rationale
                plan_id = generate_id("plan")
                created_at = utc_now()

                memory.update_run_state(run_id, user_id, stage="approval", status="in_progress", plan_id=plan_id, last_step="generate_plan_step")

                # print(f"Plan JSON: {plan_json}")
                # print(f"Rationale: {rationale}")
                # print(f"Plan ID: {plan_id}")
                # print(f"Created At: {created_at}")
                
                memory.save_plan(user_id, plan_id, plan_json, rationale, created_at)
                print(f"Plan saved to database: {plan_id}")
                
                # Log event
                event = {
                    "user_id": user_id,
                    "plan_id": plan_id,
                    "candidate_count": len(filtered),
                    "created_at": created_at,
                    "days": len(plan_json.get("plan", []))
                }
                log_event(self.agent_name, "plan_generated", event)
                
                logger.info("✅ Plan %s generated for %s", plan_id, user_id)
                
                return PlannerResult(
                    status="plan_ready",
                    plan_id=plan_id,
                    plan_json=plan_json,
                    rationale=rationale,
                    preference_add=plan_model.preference_add,
                    message="Plan generated successfully"
                )
                
            except ValidationError:
                # Try as chat response
                try:
                    chat_model = PlannerChatOutput.model_validate(plan_parsed)
                    
                    return PlannerResult(
                        status="chat",
                        chat_reply=chat_model.reply,
                        preference_add=chat_model.preference_add,
                        message="Chat response generated"
                    )
                    
                except ValidationError as ve:
                    log_error(run_id, self.agent_name, "validation", str(ve))
                    logger.error("❌ Invalid structure returned by LLM: %s", plan_parsed)
                    
                    return PlannerResult(
                        status="error",
                        error=f"Plan validation failed — the model didn't return proper JSON. Error: {str(ve)}"
                    )

        except Exception as e:
            logger.exception("Error in generate_plan_step: %s", e)
            log_error(run_id, self.agent_name, "generate_step", str(e))
            
            return PlannerResult(
                status="error",
                error=str(e)
            )

    # --------------------------
    # NEW: Handle approval/rejection
    # --------------------------
    def handle_approval(
        self,
        user_id: str,
        plan_id: str,
        memory: Optional[MemoryLayer] = None,
        run_id: str = None,
    ) -> PlannerResult:
        """Handle plan approval."""
        memory = memory or self.memory
        # run_id = generate_id("run")
        
        try:
            log_event("planner", "plan_approved", {
                "user_id": user_id,
                "plan_id": plan_id,
                "timestamp": utc_now(),
            })
            
            memory.update_plan_status(plan_id, "approved")
            memory.update_run_state(run_id, user_id, stage="approval", status="approved", plan_id=plan_id)
            
            return PlannerResult(
                status="completed",
                plan_id=plan_id,
                message=f"Plan {plan_id} approved and finalized."
            )
        except Exception as e:
            return PlannerResult(
                status="error",
                error=f"Failed to approve plan: {str(e)}"
            )

    def handle_rejection(
        self,
        user_id: str,
        plan_id: str,
        reason: str,
        memory: Optional[MemoryLayer] = None,
        run_id: str = None,
    ) -> PlannerResult:
        """Handle plan rejection."""
        memory = memory or self.memory
        # run_id = generate_id("run")
        
        try:
            memory.append_rejection(user_id, plan_id, reason)
            memory.update_run_state(run_id, user_id, stage="approval", status="rejected", plan_id=plan_id)
            
            log_event("planner", "feedback_cycle", {
                "user_id": user_id,
                "previous_plan_id": plan_id,
                "feedback": reason,
                "timestamp": utc_now(),
            })
            
            return PlannerResult(
                status="chat",
                message="Plan rejected. Please provide feedback for revision.",
                chat_reply=f"I understand. I'll revise the plan based on: {reason}"
            )
        except Exception as e:
            return PlannerResult(
                status="error",
                error=f"Failed to reject plan: {str(e)}"
            )



# ------------------------------
# CLI for quick testing
# ------------------------------
if __name__ == "__main__":
    print("⚠️  This module is refactored for Streamlit.")
    print("Use the Streamlit app or call generate_plan_step() directly.")