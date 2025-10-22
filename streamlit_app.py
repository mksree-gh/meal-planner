# streamlit_app.py

import streamlit as st
import json
from typing import Optional, Dict, Any, List
from agents.planner_agent import PlannerAgent, PlannerResult
from agents import PreferenceAgent
from core.memory_layer import MemoryLayer
from config import DB_PATH
from core.utils import generate_id, utc_now
from core.logging_layer import log_event
from core.reasoning import merge_preferences

# --- Page Configuration ---
st.set_page_config(
    page_title="Meal Planner",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    defaults = {
        "messages": [],
        "user_id": "user_001",
        "memory": MemoryLayer(DB_PATH),
        "planner_agent": PlannerAgent(),
        "chat_history": [],
        "session_initialized": False,
        "pending_resume": None,
        "current_plan_id": None,
        "awaiting_action": False,
        "preferences_override": None,
        "current_plan_json": None,
        "rejection_mode": False,
        "feedback_mode": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- UI Helper Functions ---
def display_plan(plan_json: Dict[str, Any], rationale: str) -> str:
    """Formats and displays a meal plan in a structured way."""
    plan_display = ""
    for day in plan_json.get("plan", []):
        plan_display += f"**📅 {day['day']}**\n"
        for meal in day["meals"]:
            plan_display += f"- {meal}\n"
        plan_display += "\n"
    
    if rationale:
        plan_display += f"**💡 Rationale:** *{rationale}*\n"
        
    return plan_display

def clear_session():
    """Clears the session state for a fresh start."""
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.session_initialized = False
    st.session_state.pending_resume = None
    st.session_state.current_plan_id = None
    st.session_state.awaiting_action = False
    st.session_state.preferences_override = None
    st.session_state.current_plan_json = None
    st.session_state.memory.update_run_state(
        generate_id("run"), 
        st.session_state.user_id, 
        stage="init", 
        status="new"
    )

# --- Sidebar ---
with st.sidebar:
    st.title("🍽️ Meal Planner")
    st.markdown("Your AI-powered meal planning assistant.")

    st.header("User Settings")
    new_user_id = st.text_input("Current User ID", value=st.session_state.user_id)
    if new_user_id != st.session_state.user_id:
        st.session_state.user_id = new_user_id
        clear_session()
        st.rerun()

    if st.button("🗑️ Start New Session", use_container_width=True, type="secondary"):
        clear_session()
        st.rerun()
        
    st.divider()

    st.header("Session Info")
    st.caption(f"**Database:** `{DB_PATH}`")
    if st.session_state.current_plan_id:
        st.caption(f"**Current Plan:** `{st.session_state.current_plan_id}`")
    
    with st.expander("How to Use"):
        st.markdown("""
        1. **Share Preferences:** Tell me your diet, allergies, and favorite foods.
        2. **Request a Plan:** Ask for a meal plan (e.g., "a 3-day high-protein plan").
        3. **Review & Refine:** Approve, reject, or request changes to the generated plan.
        """)

    st.divider()
    st.caption("© Keerthi Sree Marrapu")

# --- Main App Logic ---

# Check for unfinished session on first load
if not st.session_state.session_initialized:
    st.session_state.session_initialized = True
    last_state = st.session_state.memory.get_run_state(st.session_state.user_id)
    
    if last_state and last_state["status"] in ("paused", "failed"):
        st.session_state.pending_resume = last_state
        welcome_msg = f"""👋 **Welcome back!**
Detected a previous session that was *{last_state.get('status', 'unknown')}*. 
Would you like to resume or start fresh?"""
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    else:
        welcome_msg = "👋 **Hello!** I'm your meal planning assistant. How can I help you today?"
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

# --- Chat Interface ---
st.header("Meal Plan Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# Display resume buttons if pending
if st.session_state.pending_resume:
    col1, col2 = st.columns(2)
    if col1.button("🔁 Resume Previous Session", use_container_width=True, type="primary"):
        st.session_state.messages.append({
               "role": "user",
               "content": "Resume previous session"
           })
          
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            status_placeholder.markdown("🔄 Resuming previous session...")
            
            try:
                ctx = st.session_state.memory.get_last_session_context(st.session_state.user_id)
                if ctx:
                    run_id = generate_id("run")
                    st.session_state.memory.mark_session_resumed(
                        run_id,
                        st.session_state.user_id,
                        plan_id=ctx["run_state"].get("current_plan_id")
                    )
                    
                    # Restore chat history if available
                    if ctx.get("chat_history"):
                        st.session_state.chat_history = ctx["chat_history"]
                    
                    # Handle resume based on stage
                    stage = ctx["run_state"].get("stage")
                    plan_data = ctx.get("plan_data")
                    
                    if stage == "approval" and plan_data:
                        # Restore plan state
                        st.session_state.current_plan_id = plan_data["plan_id"]
                        st.session_state.current_plan_json = plan_data.get("plan_json")
                        st.session_state.awaiting_action = True
                        
                        # Show the pending plan
                        plan_display = f"""## 🗒️ Your Pending Plan


**Plan ID:** `{plan_data['plan_id']}`


Here's the plan that was awaiting your approval:"""
                        
                        if plan_data.get("plan_json"):
                            for day in plan_data["plan_json"].get("plan", []):
                                plan_display += f"\n\n### 📅 {day['day']}\n"
                                for meal in day["meals"]:
                                    plan_display += f"- {meal}\n"
                        
                        if plan_data.get("rationale"):
                            plan_display += f"\n\n**💡 Rationale:** {plan_data['rationale']}\n"
                        
                        plan_display += "\n\n---\n**What would you like to do?**"
                        
                        status_placeholder.markdown(plan_display)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": plan_display
                        })
                    else:
                        msg = f"✅ Session resumed from stage: **{stage}**\n\nPlease continue by sharing your preferences or requesting a plan."
                        status_placeholder.markdown(msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": msg
                        })
                else:
                    status_placeholder.markdown("⚠️ No valid context found. Please start fresh.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "⚠️ No valid context found. Please start fresh."
                    })
                
            except Exception as e:
                status_placeholder.error(f"❌ Error resuming session: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error resuming session: {str(e)}"
                })
            
            st.session_state.pending_resume = None
            st.rerun()

    if col2.button("🆕 Start New Session", use_container_width=True):
        st.session_state.messages.append({
            "role": "user",
            "content": "Start new session"
        })
        st.session_state.memory.update_run_state(
            generate_id("run"),
            st.session_state.user_id,
            stage="init",
            status="new"
        )
        
        msg = "🆕 Starting a fresh session. What would you like to do?"
        st.session_state.messages.append({
            "role": "assistant",
            "content": msg
        })
        st.session_state.pending_resume = None
        st.rerun()

# Display action buttons for a generated plan
if st.session_state.awaiting_action and st.session_state.current_plan_id:
    st.markdown("---")
    st.info("A plan is awaiting your feedback. Please choose an option below.")
    cols = st.columns([1, 1, 1.2])
    
    if cols[0].button("✅ Approve Plan", use_container_width=True):
        plan_id = st.session_state.current_plan_id
          
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            status_placeholder.markdown("✅ Approving plan...")
            
            result = st.session_state.planner_agent.handle_approval(
                user_id=st.session_state.user_id,
                plan_id=plan_id,
                memory=st.session_state.memory
            )
            
            if result.status == "completed":
                response = f"""🎉 **Plan Approved!**


Plan ID: `{plan_id}` has been finalized and saved.


Your meal plan is ready to use! Would you like to create another plan or modify your preferences?"""
                status_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.awaiting_action = False
                st.session_state.current_plan_id = None
                st.session_state.current_plan_json = None
            else:
                status_placeholder.error(f"❌ {result.error}")
                st.session_state.messages.append({"role": "assistant", "content": f"❌ {result.error}"})
            
            st.rerun()

    if cols[1].button("❌ Reject & Explain", use_container_width=True):
        st.session_state.awaiting_action = False
        msg = "I see. Please tell me what you didn't like so I can improve."
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.session_state.rejection_mode = True
        st.rerun()

    if cols[2].button("✏️ Request Specific Changes", use_container_width=True):
        st.session_state.awaiting_action = False
        msg = "Of course. What specific changes would you like to make?"
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.session_state.feedback_mode = True
        st.rerun()

# --- Chat Input Processing ---
if not st.session_state.pending_resume:
    if prompt := st.chat_input("What are your dietary needs?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "user", "parts": [{"text": prompt}]})

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            status_placeholder.markdown("Thinking...")
            
            try:
                # --- Agent Logic ---
                memory = st.session_state.memory
                user_id = st.session_state.user_id
                planner = st.session_state.planner_agent
                
                # Simplified logic flow for demonstration
                if st.session_state.get("rejection_mode") or st.session_state.get("feedback_mode"):
                    # Handle rejection/feedback logic
                    st.session_state.rejection_mode = False
                    st.session_state.feedback_mode = False
                    # Fall through to generate a new plan with the new context
                    
                # Generate plan
                result = planner.generate_plan_step(
                    user_id=user_id,
                    user_message=prompt,
                    chat_history=st.session_state.chat_history,
                    preferences_override=st.session_state.preferences_override,
                    memory=memory
                )

                # --- Result Handling ---
                if result.status == "plan_ready":
                    st.session_state.chat_history.append({"role": "model", "parts": [{"text": json.dumps(result.plan_json)}]})
                    
                    plan_header = f"### 📋 Here's Your Meal Plan (ID: `{result.plan_id}`)"
                    plan_details = display_plan(result.plan_json, result.rationale)
                    
                    response_content = f"{plan_header}\n{plan_details}"
                    status_placeholder.markdown(response_content)
                    st.session_state.messages.append({"role": "assistant", "content": response_content})

                    st.session_state.current_plan_id = result.plan_id
                    st.session_state.current_plan_json = result.plan_json
                    st.session_state.awaiting_action = True
                    st.rerun()
                
                elif result.status == "chat":
                    st.session_state.chat_history.append({"role": "model", "parts": [{"text": result.chat_reply}]})
                    status_placeholder.markdown(result.chat_reply)
                    st.session_state.messages.append({"role": "assistant", "content": result.chat_reply})

                else: # Handles completed, error, etc.
                    final_message = result.message or f"❌ **Error:** {result.error}"
                    status_placeholder.markdown(final_message)
                    st.session_state.messages.append({"role": "assistant", "content": final_message})
                    st.session_state.awaiting_action = False
            
            except Exception as e:
                error_msg = f"An unexpected error occurred: {e}"
                st.error(error_msg)
                log_event("streamlit_error", {"error": str(e)})
                st.session_state.messages.append({"role": "assistant", "content": f"❌ {error_msg}"})