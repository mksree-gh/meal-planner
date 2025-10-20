# main.py

"""
main.py — Posha Assistant Orchestrator

Coordinates the multi-agent workflow:
1. PreferenceAgent → parses and updates structured preferences.
2. PlannerAgent → generates a 3-day meal plan draft.
3. Human approval → accept or reject plan.

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
from agents import PreferenceAgent, PlannerAgent


def orchestrate(user_id: str, user_text: Optional[str] = None):
    memory = MemoryLayer(DB_PATH)
    print("\n👋 Hello! I am Posha's meal planning assistant.")
    print(f"📘 Using database: {DB_PATH}")

    try:
        while True:
            # Accept user preference text if provided or ask
            if not user_text:
                user_text = input("\nEnter a new preference (or press Enter to skip): ").strip() or None

            merged_view = None
            if user_text:
                print("\n🧠 Updating preferences...")
                pref_agent = PreferenceAgent()
                merged_view = pref_agent.process_user_text(user_id=user_id, user_text=user_text)
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

            # Generate plan with merged_view passed
            planner = PlannerAgent()
            print("📋 Generating meal plan...")
            plan_data = planner.generate_plan(user_id=user_id, session_text=user_text, preferences_override=merged_view,memory = memory)

            if plan_data == "Exit":
                print("\n🛑 Exiting as per request.\n")
                return
            elif plan_data== "New Session":
                print("\n🔄 Starting a new session as per request.\n")
                user_text = None
                continue

            # # Approval loop - mandatory user decision or explicit exit
            # while True:
            #     action = input("Choose action — approve (a), reject (r), regenerate with feedback (g), or exit (e): ").strip().lower()
            #     if action in ("a", "approve"):
            #         memory.update_plan_status(plan_data["plan_id"], "approved")
            #         memory.update_run_state(generate_id("run"), user_id, stage="approval", status="approved", plan_id=plan_data["plan_id"])
            #         print(f"\n🎉 Plan approved and finalized (plan_id={plan_data['plan_id']}).\n")
            #         log_event("orchestrator", "plan_approved", {"user_id": user_id, "plan_id": plan_data["plan_id"], "timestamp": utc_now()})
            #         return
            #     elif action in ("r", "reject"):
            #         reason = input("💬 Why are you rejecting this plan?\n> ").strip() or "User rejected without reason"
            #         memory.update_plan_status(plan_data["plan_id"], "rejected", reason)
            #         memory.update_run_state(generate_id("run"), user_id, stage="approval", status="rejected", plan_id=plan_data["plan_id"])
            #         # append rejection history with dedupe (simple check)
            #         profile = memory.get_profile(user_id) or {}
            #         old_hist = profile.get("rejection_history_json", []) or []
            #         new_entry = {"plan_id": plan_data["plan_id"], "reason": reason, "timestamp": utc_now()}
            #         # dedupe by (plan_id, reason)
            #         if not any(e.get("plan_id") == new_entry["plan_id"] and e.get("reason") == new_entry["reason"] for e in old_hist):
            #             old_hist.append(new_entry)
            #         memory.upsert_profile(
            #             user_id=user_id,
            #             name=profile.get("name", "user"),
            #             preferences=profile.get("preferences_json", {}) or {},
            #             source="orchestrator",
            #             profile_version=(profile.get("profile_version", 0) or 0) + 1,
            #             weightages=profile.get("weightages_json", {}) or {},
            #             session_overlay=profile.get("session_overlay_json", {}) or {},
            #             rejection_history=old_hist,
            #         )
            #         print(f"\n❌ Plan rejected. Feedback saved: “{reason}”\n")
            #         # Ask whether to regenerate immediately
            #         retry = input("🔁 Would you like me to try again with that feedback? (y/n): ").strip().lower()
            #         if retry in ("y", "yes"):
            #             feedback_text = f"Please revise the plan. Avoid / change: {reason}"
            #             # regenerate, loop back into outer while to create a new plan with feedback
            #             user_text = feedback_text
            #             break  # break inner loop and regenerate
            #         else:
            #             # ask whether to exit or continue with new preference
            #             cont = input("Do you want to continue (c) with new preferences, or exit (e)? ").strip().lower()
            #             if cont in ("e", "exit"):
            #                 print("\n🛑 Ending session. You can rerun later to continue.\n")
            #                 return
            #             else:
            #                 # allow user to type new preferences next loop
            #                 user_text = None
            #                 break
            #     elif action in ("g", "regenerate"):
            #         feedback_text = input("Enter concise feedback for regeneration (e.g., 'more American, less Indian'):\n> ").strip()
            #         if not feedback_text:
            #             print("No feedback entered, aborting regeneration request.")
            #             continue
            #         user_text = f"Previous_requests: {user_text} \n feedback: {feedback_text}"
            #         break  # break inner loop to regenerate outside
            #     elif action in ("e", "exit"):
            #         print("\n🛑 Exiting per user request.\n")
            #         return
            #     else:
            #         print("Unknown action. Please enter 'a' (approve), 'r' (reject), 'g' (regenerate), or 'e' (exit).")
            # outer while continues to generate new plan with updated user_text or merged_view
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
        memory.update_run_state(generate_id("run"), user_id, stage="interrupted", status="paused")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error: {e}")
        log_error(generate_id("run"), "orchestrator", "main_flow", str(e))
        memory.update_run_state(generate_id("run"), user_id, stage="error", status="failed", error_message=str(e))
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Posha Orchestrator CLI")
    parser.add_argument("--user-id", required=True, help="Unique user ID")
    parser.add_argument("--text", default=None, help="Optional preference text to process directly")
    args = parser.parse_args()

    orchestrate(user_id=args.user_id, user_text=args.text)
