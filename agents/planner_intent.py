"""
PlannerIntentAgent — lightweight conversational layer for PlannerAgent.

- Interprets user follow-ups about existing meal plans.
- Decides whether to respond conversationally or regenerate the plan.
- Uses a smaller LLM (flash → pro fallback) configured in settings.py.
"""

import json
import re
import time
from typing import Dict, Any

from google import genai
from google.genai import types

from config import get_agent_config, get_prompt_path, should_log_prompts
from core.utils import generate_id, utc_now, get_logger
from core.error_handler import log_error
from core.memory_layer import MemoryLayer, log_session_event
from agents.planner_agent import PlannerAgent

logger = get_logger("planner_intent_agent")




class PlannerIntentAgent:

    def __init__(self, agent_name: str = "planner_intent"):
        self.agent_name = agent_name
        self.agent_cfg = get_agent_config(agent_name)
        self.prompt_path = get_prompt_path(agent_name)
        self.client = genai.Client()
        self.memory = MemoryLayer()
        self.primary_model = self.agent_cfg["primary_model"]
        self.fallback_model = self.agent_cfg["fallback_model"]

    def build_context_from_memory(self, user_id: str) -> str:
        """
        Build a conversation starter context from the user's last plan and recent chats.
        Returns a human-readable summary string.
        """
        last_plan = self.memory.get_latest_plan_for_user(user_id)
        if not last_plan:
            return "No previous meal plan found."

        plan_json = last_plan.get("plan_json", {})
        rationale = last_plan.get("rationale", "(no rationale available)")
        plan_id = last_plan.get("plan_id", "unknown")
        created_at = last_plan.get("created_at", "unknown")

        plan_summary = "\n".join(
            [f"- {day['day']}: {', '.join(day['meals'])}" for day in plan_json.get("plan", [])]
        )

        # Optional: recent conversation (from logs/sessions)
        recent_msgs = self.memory.get_recent_chat_history(user_id, limit=3) if hasattr(self.memory, "get_recent_chat_history") else []

        recent_context = "\n".join(
            [f"User: {msg['input_message']}\nBot: {msg['llm_response']}" for msg in recent_msgs]
        ) if recent_msgs else ""

        context_str = (
            f"Last plan (ID: {plan_id}, created {created_at}):\n{plan_summary}\n\n"
            f"Rationale: {rationale}\n\n"
        )
        if recent_context:
            context_str += f"Recent chat:\n{recent_context}\n"

        return context_str


    

    # --------------------------
    def handle_message(self, user_id: str, message: str) -> Dict[str, Any]:
        
        """
        Main conversational entrypoint.
        Understands user feedback and decides:
          - "respond" → return conversational reply
          - "regenerate" → call PlannerAgent.generate_plan()
        """
        context = self.build_context_from_memory(user_id)

        logger.info("💬 PlannerIntentAgent handling user=%s", user_id)
        run_id = generate_id("run")

        # --- 1️⃣ Fetch existing plan
        last_plan = self.memory.get_latest_plan_for_user(user_id)
        if not last_plan:
            return {
                "response": "I couldn’t find an existing meal plan for you. Would you like me to create one?",
                "action": "regenerate",
            }

        plan_json = last_plan.get("plan_json", {})
        rationale = last_plan.get("rationale", "(no rationale available)")
        plan_id = last_plan.get("plan_id", "unknown")

        plan_summary = "\n".join(
            [f"- {day['day']}: {', '.join(day['meals'])}" for day in plan_json.get("plan", [])]
        )

        # --- 2️⃣ Build LLM prompt
        base_prompt = self.prompt_path.read_text(encoding="utf-8")
        system_instruction = f"""{base_prompt}
=== CONTEXT ===
{context}

=== CURRENT PLAN SUMMARY ===
{plan_summary}

=== RATIONALE ===
{rationale}

=== USER MESSAGE ===
{message}
"""

        contents = [
            {"role": "user", "parts": [{"text": system_instruction}]},
        ]

        # --- 3️⃣ Call small model (planner_intent)
        parsed_result = None
        network_error = None

        for model_name in [self.primary_model, self.fallback_model]:
            try:
                logger.info("🧠 Calling %s for conversational reasoning", model_name)
                start = time.time()
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=self.agent_cfg["temperature"],
                        max_output_tokens=self.agent_cfg["max_output_tokens"],
                        response_mime_type="application/json",
                    ),
                )
                latency = int((time.time() - start) * 1000)
                logger.info("🕒 LLM response in %d ms", latency)

                parsed_result = getattr(response, "parsed", None)
                if not parsed_result:
                    txt = getattr(response, "text", "") or ""
                    m = re.search(r"\{[\s\S]*\}", txt)
                    if m:
                        parsed_result = json.loads(m.group(0))
                    else:
                        parsed_result = {"response": txt.strip(), "action": "respond"}
                break

            except OSError as net_err:
                # Handle [Errno 8] etc.
                network_error = str(net_err)
                log_error(run_id, self.agent_name, "network_error", str(net_err))
                logger.warning("⚠️ Network error: %s", net_err)
                continue

            except Exception as e:
                log_error(run_id, self.agent_name, "intent_llm_call", str(e))
                logger.warning("⚠️ %s failed: %s", model_name, e)
                continue

        if not parsed_result:
            if network_error:
                logger.error("💔 Network failure during conversation. Ending session.")
                return {
                    "response": "I lost connection while processing your request. Let's continue next time!",
                    "action": "end_session"
                }
            raise RuntimeError("PlannerIntentAgent LLM failed for all models.")

        # --- 4️⃣ Interpret result
        response_text = parsed_result.get("response", "").strip()
        action = parsed_result.get("action", "respond").lower()

        # --- 5️⃣ Log
        log_session_event(
            {
                "event": "planner_intent",
                "user_id": user_id,
                "plan_id": plan_id,
                "input_message": message,
                "response": response_text,
                "action": action,
                "timestamp": utc_now(),
            },
            plan_id,
        )

        # --- 6️⃣ If regenerate → delegate to PlannerAgent
        if action == "regenerate":
            logger.info("🔁 IntentAgent triggered regeneration.")
            planner = PlannerAgent()
            new_plan = planner.generate_plan(user_id=user_id, session_text=message)
            return {
                "response": response_text or "Sure, I’ve regenerated your plan as requested.",
                "action": "regenerate",
                "new_plan": new_plan,
            }

        # --- 7️⃣ Else respond conversationally
        logger.info("💬 Returning conversational response for user=%s", user_id)
        return {"response": response_text, "action": "respond"}

# ------------------------------
# CLI for quick manual testing
# ------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("PlannerIntentAgent Interactive CLI")
    parser.add_argument("--user-id", default="user_001", help="User ID")
    args = parser.parse_args()

    user_id = args.user_id
    agent = PlannerIntentAgent()

    print("\n💬 Starting Planner Chat. Type 'E' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("e", "exit"):
            print("👋 Exiting Planner Chat.")
            break

        result = agent.handle_message(user_id=user_id, message=user_input)
        response = result.get("response")
        print(f"\nPosha: {response}\n")

        if result.get("action") == "regenerate":
            print("🔁 (A new plan was generated based on your feedback.)\n")

