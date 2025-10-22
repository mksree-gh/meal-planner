import streamlit as st
import json
from typing import Optional, Dict, Any, List
from agents.planner_agent import PlannerAgent, PlannerResult
from agents import PreferenceAgent
from core.memory_layer import MemoryLayer, log_session_event
from config import DB_PATH
from core.utils import generate_id, utc_now
from core.logging_layer import log_event
from core.reasoning import merge_preferences
import ast

# Page config
st.set_page_config(page_title="Meal Planner", page_icon="🍽️", layout="wide")



# Initialize session state defaults
default_session_state = {
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
    "resume_process_last_message": False,
    "run_id": None,
}

for key, value in default_session_state.items():
    st.session_state.setdefault(key, value)


# Title
st.title("🍽️ Meal Planner")

# Inject custom CSS to center and limit width of main content
st.markdown("""
    <style>
    /* Target the main content block */
    .block-container {
        max-width: 1200px;
        padding-left: 2rem;
        padding-right: 2rem;
        margin: 0 auto;
    }
    
    /* Alternative targeting for different Streamlit versions */
    section.main > div {
        max-width: 1200px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Ensure proper spacing on smaller screens */
    @media (max-width: 1400px) {
        .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # User ID input
    new_user_id = st.text_input("User ID", value=st.session_state.user_id)
    if new_user_id != st.session_state.user_id:
        st.session_state.user_id = new_user_id
        st.session_state.session_initialized = False
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.current_plan_id = None
        st.session_state.awaiting_action = False
        st.rerun()
    
    st.divider()
    
    # Database info
    st.caption(f"📘 Database: {DB_PATH}")
    st.caption(f"💬 Messages: {len(st.session_state.messages)}")
    if st.session_state.current_plan_id:
        st.caption(f"📋 Current Plan: {st.session_state.current_plan_id}")
    
    st.divider()
    
    # Testing: Random error trigger
    st.subheader("🧪 Testing")
    if "error_enabled" not in st.session_state:
        st.session_state.error_enabled = False
    
    st.session_state.error_enabled = st.checkbox(
        "Enable Random Errors", 
        value=st.session_state.error_enabled,
        help="Randomly trigger errors to test session resume functionality"
    )
    
    if st.session_state.error_enabled:
        error_chance = st.slider("Error Probability (%)", 0, 100, 30)
        st.session_state.error_chance = error_chance
        st.caption(f"⚠️ {error_chance}% chance of error per message")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat & Start Fresh", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.session_initialized = False
        st.session_state.pending_resume = None
        st.session_state.current_plan_id = None
        st.session_state.awaiting_action = False
        st.session_state.preferences_override = None
        st.session_state.current_plan_json = None
        # Mark as new session in DB
        st.rerun()

# Check for unfinished session on first load
if not st.session_state.session_initialized:
    st.session_state.session_initialized = True
    memory = st.session_state.memory
    user_id = st.session_state.user_id
    
    last_state = memory.get_run_state(user_id)
    # print(f"Last state: {last_state}")
    if last_state and last_state["status"] in ("paused", "failed", "in_progress"):
        st.session_state.pending_resume = last_state
        run_id = last_state.get("run_id")
        # print(f"Pending resume: {st.session_state.pending_resume}")
        # print(f"Pending resume: {st.session_state.pending_resume}")
        # Show resume prompt
        stage = last_state.get('stage', 'unknown')
        status = last_state.get('status', 'unknown')
        
        welcome_msg = f"""👋 **Welcome back!**

⚠️ I detected a previous session that was **{status}** at stage: **{stage}**

Would you like to resume where you left off, or start fresh?"""
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_msg
        })
    else:
        # Fresh start
        # print(f"Fresh start")
        run_id = generate_id("run")
        st.session_state.run_id = run_id
        st.session_state.memory.update_run_state(
            run_id, 
            st.session_state.user_id, 
            stage="init", 
            status="new"
        )
        # print(f"Updated run state: {st.session_state.memory.get_run_state(st.session_state.user_id)}")
        welcome_msg = """👋 **Hello! I am your meal planning assistant.**

I can help you create personalized meal plans based on your preferences!

You can:
- Share your dietary preferences and restrictions
- Request a meal plan
- Modify and refine plans until you're happy

What would you like to do today?"""
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_msg
        })

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Show resume buttons if pending
if st.session_state.pending_resume:
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔁 Resume Previous Session", use_container_width=True):
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
                        
                        run_id = ctx["run_state"].get("run_id")
                        st.session_state.run_id = run_id
                        # st.session_state.memory.mark_session_resumed(
                        #     run_id, 
                        #     st.session_state.user_id, 
                        #     plan_id=ctx["run_state"].get("current_plan_id")
                        # )
                        # Restore chat history if available
                        if ctx.get("chat_history"):
                            st.session_state.chat_history = ctx.get("chat_history")

                        stage = ctx["run_state"].get("stage")
                        plan_data = ctx.get("plan_data")
                        if stage == "approval" and plan_data:
                            # Restore plan state
                            st.session_state.current_plan_id = plan_data["plan_id"]
                            st.session_state.current_plan_json = plan_data.get("plan_json")
                            st.session_state.awaiting_action = True
                            
                            # Show the pending plan
                            def plan_display_format(plan_data):
                                plan_display = f"""## Here's the plan that was awaiting your approval:"""
                                if plan_data.get("plan"):
                                    for day in plan_data.get("plan", []):
                                        plan_display += f"\n\n### 📅 {day['day']}\n"
                                        for meal in day["meals"]:
                                            plan_display += f"- {meal}\n"
                                
                                if plan_data.get("rationale"):
                                    plan_display += f"\n\n**💡 Rationale:** {plan_data['rationale']}\n"
                                
                                plan_display += "\n\n---\n**What would you like to do?**"
                                return plan_display
                            
                            # status_placeholder.markdown(plan_display_format(plan_data))
                            for message in st.session_state.chat_history:
                                text_part = message["parts"][0]["text"]
                                json_text_part = None
                                literal_text_part = None
                                try:
                                    json_text_part = json.loads(text_part)
                                except Exception:
                                    pass
                                try:
                                    literal_text_part = ast.literal_eval(text_part)
                                except Exception:
                                    pass
                                if isinstance(text_part, dict):
                                    content = plan_display_format(text_part)
                                elif isinstance(json_text_part, dict):
                                    content = plan_display_format(json_text_part)
                                elif isinstance(literal_text_part, dict):
                                    content = plan_display_format(literal_text_part)
                                else:
                                    content = text_part
                                
                            
                                st.session_state.messages.append({
                                    "role": message["role"] if message["role"]=="user" else "assistant",
                                    "content": content
                                })
                        elif stage == "planning":
                            if len(st.session_state.chat_history) > 0:
                                # st.session_state.messages.extend(st.session_state.chat_history)
                                
                                last_role = st.session_state.chat_history[-1]["role"]
                                last_content = st.session_state.chat_history[-1]["parts"][0]["text"]
                                st.session_state.messages.append({
                                "role": last_role,
                                "content": last_content
                            })
                                

                                if last_role == "user":
                                    st.session_state["resume_process_last_message"] = True
                                    status_placeholder.markdown("✅ Session resumed. Processing your last message...")
                                    print(f"Resume process last message: {True}")
                                else:
                                    # Last message was from assistant, just restore and wait
                                    msg = "✅ Session resumed. How can I help you further?"
                                    status_placeholder.markdown(msg)
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": msg
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
    
    with col2:
        if st.button("🆕 Start New Session", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Start new session"
            })
            run_id = generate_id("run")
            st.session_state.run_id = run_id
            st.session_state.memory.update_run_state(
                run_id, 
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

# Action buttons when awaiting user decision on plan
if st.session_state.awaiting_action and st.session_state.current_plan_id:
    st.divider()
    st.markdown("### 🎯 Plan Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Approve Plan", use_container_width=True, type="primary"):
            plan_id = st.session_state.current_plan_id
            
            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                status_placeholder.markdown("✅ Approving plan...")
                
                result = st.session_state.planner_agent.handle_approval(
                    user_id=st.session_state.user_id,
                    plan_id=plan_id,
                    memory=st.session_state.memory,
                    run_id=st.session_state.run_id
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
    
    with col2:
        if st.button("❌ Reject Plan", use_container_width=True):
            st.session_state.awaiting_action = False
            msg = "💬 Please tell me what you didn't like about this plan so I can create a better one."
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.session_state["rejection_mode"] = True
            st.rerun()
    
    with col3:
        if st.button("✏️ Request Changes", use_container_width=True):
            st.session_state.awaiting_action = False
            msg = "💬 Please describe the specific changes you'd like me to make to the plan."
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.session_state["feedback_mode"] = True
            st.rerun()

# Chat input (disabled when showing resume prompt)
# if not st.session_state.pending_resume:
prompt = None
if not st.session_state.pending_resume or st.session_state["resume_process_last_message"]:
    if st.session_state.get("resume_process_last_message"):
        prompt = st.session_state.chat_history[-1]["parts"][0]["text"]
        print(f"Prompt: {prompt}")
        st.session_state["resume_process_last_message"] = False
    if prompt or (prompt := st.chat_input("Share preferences or request a meal plan...")):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Add to LLM chat history
        st.session_state.chat_history.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        
        # 🧪 TESTING: Random error injection
        if st.session_state.get("error_enabled", False):
            import random
            error_chance = st.session_state.get("error_chance", 30)
            if random.randint(1, 100) <= error_chance:
                # Mark session as paused
                # run_id = generate_id("run")
                current_stage = "planning" if not st.session_state.awaiting_action else "approval"
                
                st.session_state.memory.update_run_state(
                    st.session_state.run_id,
                    st.session_state.user_id,
                    stage=current_stage,
                    status="failed",
                    plan_id=st.session_state.current_plan_id,
                    error_message="Simulated error for testing"
                )

                log_session_event({"run_id": st.session_state.run_id, "agent": "streamlit_app", "phase": "prompt", "prompt": prompt})
                
                error_msg = f"""💥 **Simulated Error (Testing Mode)**

The session has been paused at stage: **{current_stage}**)

This is a test error to verify session resume functionality.

**To test resume:**
1. Refresh the page (F5 or reload)
2. You should see the resume prompt
3. Click "Resume Previous Session"
4. The conversation should continue from this point

*Disable "Enable Random Errors" in sidebar to stop this behavior.*"""
                
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.stop()
        
        # Process with planner agent
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            
            try:
                memory = st.session_state.memory
                user_id = st.session_state.user_id
                planner = st.session_state.planner_agent
                
                # Handle rejection mode
                if st.session_state.get("rejection_mode"):
                    status_placeholder.markdown("📝 Recording feedback...")
                    
                    plan_id = st.session_state.current_plan_id
                    if plan_id:
                        result = planner.handle_rejection(
                            user_id=user_id,
                            plan_id=plan_id,
                            reason=prompt,
                            memory=memory,
                            run_id=st.session_state.run_id
                        )
                        
                        # Add LLM response to chat history
                        st.session_state.chat_history.append({
                            "role": "model",
                            "parts": [{"text": result.chat_reply or result.message}]
                        })
                        
                        status_placeholder.markdown("✅ Feedback recorded. Generating revised plan...")
                        st.session_state["rejection_mode"] = False
                        
                        # Continue to generate new plan (fall through to plan generation)
                    else:
                        status_placeholder.error("❌ No current plan to reject.")
                        st.stop()
                
                # Handle feedback mode  
                elif st.session_state.get("feedback_mode"):
                    status_placeholder.markdown("📝 Processing your feedback...")
                    st.session_state["feedback_mode"] = False
                    # Fall through to plan generation with feedback
                
                # Get or update preferences
                status_placeholder.markdown("🧠 Analyzing preferences...")
                
                if st.session_state.preferences_override:
                    merged_view = st.session_state.preferences_override
                else:
                    # Check if we should update preferences
                    pref_agent = PreferenceAgent()
                    merged_view = pref_agent.process_user_text(
                        user_id=user_id, 
                        user_text=prompt,
                        run_id=st.session_state.run_id
                    )
                    st.session_state.preferences_override = merged_view
                
                # Generate plan step
                status_placeholder.markdown("🍳 Fetching recipes and generating plan...")
                
                result = planner.generate_plan_step(
                    user_id=user_id,
                    user_message=prompt,
                    chat_history=st.session_state.chat_history,
                    preferences_override=merged_view,
                    memory=memory,
                    run_id=st.session_state.run_id
                )

                
                # log_session_event({"run_id": st.session_state.run_id, "agent": "streamlit_app", "phase": "generate_plan_step", "result": result})
                
                # Handle result based on status
                if result.status == "plan_ready":
                    # Add LLM response to chat history
                    st.session_state.chat_history.append({
                        "role": "model",
                        "parts": [{"text": json.dumps(result.plan_json)}]
                    })
                    
                    # Display plan
                    plan_display = f"""## 📋 Your Meal Plan

**Plan ID:** `{result.plan_id}`

"""
                    
                    for day in result.plan_json.get("plan", []):
                        plan_display += f"### 📅 {day['day']}\n"
                        for meal in day["meals"]:
                            plan_display += f"- {meal}\n"
                        plan_display += "\n"
                    
                    plan_display += f"**💡 Rationale:** {result.rationale}\n\n"
                    plan_display += "---\n**What would you like to do with this plan?**"
                    
                    status_placeholder.markdown(plan_display)
                    st.session_state.messages.append({"role": "assistant", "content": plan_display})
                    
                    # Store plan data
                    st.session_state.current_plan_id = result.plan_id
                    st.session_state.current_plan_json = result.plan_json
                    st.session_state.awaiting_action = True
                    
                    # Update preferences if needed
                    if result.preference_add and prompt:
                        pref_agent = PreferenceAgent()
                        merged_view = pref_agent.process_user_text(user_id=user_id, user_text=prompt, run_id=st.session_state.run_id)
                        st.session_state.preferences_override = merged_view
                    
                    st.rerun()
                
                elif result.status == "chat":
                    # Add LLM response to chat history
                    st.session_state.chat_history.append({
                        "role": "model",
                        "parts": [{"text": result.chat_reply}]
                    })
                    
                    # Display chat response
                    status_placeholder.markdown(result.chat_reply)
                    st.session_state.messages.append({"role": "assistant", "content": result.chat_reply})
                    
                    # Update preferences if needed
                    if result.preference_add and prompt:
                        pref_agent = PreferenceAgent()
                        merged_view = pref_agent.process_user_text(user_id=user_id, user_text=prompt, run_id=st.session_state.run_id)
                        st.session_state.preferences_override = merged_view
                
                elif result.status == "completed":
                    status_placeholder.markdown(result.message)
                    st.session_state.messages.append({"role": "assistant", "content": result.message})
                    st.session_state.awaiting_action = False
                
                elif result.status == "error":
                    status_placeholder.error(f"❌ **Error:** {result.error}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"❌ **Error:** {result.error}\n\nPlease try again or rephrase your request."
                    })
                
            except Exception as e:
                error_msg = f"❌ **Unexpected Error:** {str(e)}\n\nPlease try again."
                status_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
                # Log error
                from core.error_handler import log_error
                log_error(st.session_state.run_id, "streamlit_app", "chat_processing", str(e))
                st.session_state.memory.update_run_state(
                    st.session_state.run_id,
                    st.session_state.user_id,
                    stage="planning",
                    status="failed",
                    plan_id=st.session_state.current_plan_id,
                    last_step="chat_processing",
                    error_message=str(e)
                )

# Info section at bottom
with st.expander("ℹ️ How to use Meal Planner", expanded=False):
    st.markdown("""
    ### Getting Started
    
    **1. Share Your Preferences**
    - Tell me about your dietary restrictions (e.g., "I'm vegetarian")
    - Mention allergies (e.g., "I'm allergic to nuts and shellfish")
    - Share your favorite cuisines (e.g., "I love Italian and Thai food")
    
    **2. Request a Meal Plan**
    - Simply ask: "Create a meal plan for me"
    - Or be specific: "I want a low-carb meal plan"
    
    **3. Review & Refine**
    - Approve plans you like with the ✅ button
    - Request changes by describing what to adjust
    - Reject plans and explain why for better results
    
    ### Features
    
    - 🔄 **Session Resume**: If you leave mid-conversation, pick up where you left off
    - 🧠 **Smart Preferences**: The system learns from your feedback
    - 📋 **3-Day Plans**: Get 2 meals per day, personalized to your taste
    - 💾 **Plan History**: All approved plans are saved to your profile
    
    ### Tips
    
    - Be specific about what you want to avoid
    - Mention time constraints (e.g., "quick meals under 30 minutes")
    - Share cooking skill level if relevant
    - The more details, the better the plan!
    """)

# Footer
st.divider()
st.caption("🍽️ AI Meal Planner")