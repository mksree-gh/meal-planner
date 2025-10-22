# main.py

"""
main.py — 

Coordinates the multi-agent workflow
Implements pause/resume using run_state table in core.memory_layer.
"""

import argparse
import sys
import json
from typing import Optional

from config import DB_PATH
from core.memory_layer import MemoryLayer
from core.error_handler import log_error
from core.utils import utc_now, generate_id, log_event
from agents import PreferenceAgent, PlannerAgentCLI as PlannerAgent

from typing import Dict, Any


def resume_session(user_id: str, ctx: Dict[str, Any], memory: MemoryLayer):
    """
    Resume an interrupted session using context from run_state + logs.
    """
    plan_data = ctx.get("plan_data")
    chat_history = ctx.get("chat_history") or []
    stage = ctx["run_state"].get("stage")

    print(f"\n🔁 Resuming from stage: {stage}")

    if stage == "preference":
        print("🧠 Last step was updating preferences...")
        pref_agent = PreferenceAgent()
        merged_view = pref_agent.process_user_text(user_id, user_text=None)
        planner = PlannerAgent()
        planner.generate_plan(user_id, session_text=None, preferences_override=merged_view, memory=memory)

    elif stage == "planning":
        print("📋 Resuming meal plan generation...")
        planner = PlannerAgent()
        last_text = chat_history[-1]["parts"][0]["text"] if chat_history else None
        planner.generate_plan(user_id, session_text=last_text, memory=memory)

    elif stage == "approval" and plan_data:
        print("🗒️ You had a pending plan awaiting feedback.")
        print(f"Plan ID: {plan_data['plan_id']}")
        print("Would you like to review it again?")
        choice = input("(r)eview / (a)pprove / (n)ew session: ").strip().lower()
        if choice == "a":
            memory.update_plan_status(plan_data["plan_id"], "approved")
            memory.update_run_state(generate_id("run"), user_id, stage="approval", status="approved", plan_id=plan_data["plan_id"])
            print("✅ Plan approved and finalized.")
            return
        elif choice == "r":
            planner = PlannerAgent()
            planner.generate_plan(user_id, session_text="Please revise the previous plan.", memory=memory)
        else:
            print("🆕 Starting new session.")

    else:
        print("❓ Unknown recovery stage. Starting fresh.")
        orchestrate(user_id)



def orchestrate(user_id: str, user_text: Optional[str] = None):
    memory = MemoryLayer(DB_PATH)

    print("\n👋 Hello! I am your meal planning assistant.")
    print(f"📘 Using database: {DB_PATH}")

    # --- STEP 1: Detect unfinished session ---
    last_state = memory.get_run_state(user_id)
    if last_state and last_state["status"] in ("paused", "failed"):
        print("\n⚠️ Previous session detected.")
        print(f"   Stage: {last_state.get('stage')} | Status: {last_state.get('status')}")
        resume_choice = input("Do you want to resume it? (y/n): ").strip().lower()
        if resume_choice == "y":
            ctx = memory.get_last_session_context(user_id)
            if ctx:
                print("🔄 Resuming previous session...")
                run_id = generate_id("run")
                memory.mark_session_resumed(run_id, user_id, plan_id=ctx["run_state"].get("current_plan_id"))
                return resume_session(user_id, ctx, memory)
            else:
                print("⚠️ No valid context found. Starting fresh.")
        else:
            print("🆕 Starting a new session.")
            memory.update_run_state(generate_id("run"), user_id, stage="init", status="new")

    # --- STEP 2: Proceed with normal flow ---
    try:
        while True:
            # Accept user preference text if provided or ask
            if not user_text:
                user_text = input("\nEnter a new preference (or press Enter to skip): ").strip() or None

            # merged_view = None
            # if user_text:
            #     print("\n🧠 Updating preferences...")
            #     pref_agent = PreferenceAgent()
            #     merged_view = pref_agent.process_user_text(user_id=user_id, user_text=user_text)
            #     print("✅ Preferences updated.\n")
            # else:
            #     # load existing profile to fill merged_view
            #     profile = memory.get_profile(user_id) or {}
            #     merged_view = {
            #         "base_preferences": profile.get("preferences_json", {}) or {},
            #         "weightages": profile.get("weightages_json", {}) or {},
            #         "session_overlay": profile.get("session_overlay_json", {}) or {},
            #     }
            #     print("ℹ️ Using existing preferences.\n")

            # Generate plan with merged_view passed
            planner = PlannerAgent()
            print("📋 Generating meal plan...")
            plan_data = planner.generate_plan(user_id=user_id, session_text=user_text,memory = memory)

            if plan_data == "Exit":
                return

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user. Saving session state...")
        memory.update_run_state(generate_id("run"), user_id, stage="planning", status="paused")
        print("💾 Session paused. You can resume later.")
        sys.exit(0)

    except Exception as e:
        print(f"\n💥 Error: {e}")
        log_error(generate_id("run"), "orchestrator", "main_flow", str(e))
        memory.update_run_state(generate_id("run"), user_id, stage="planning", status="failed", error_message=str(e))
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Orchestrator CLI")
    parser.add_argument("--user-id", required=True, help="Unique user ID")
    parser.add_argument("--text", default=None, help="Optional preference text to process directly")

    args = parser.parse_args()

    orchestrate(user_id=args.user_id, user_text=args.text)
