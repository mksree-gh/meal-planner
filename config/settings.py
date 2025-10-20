# config/settings.py

"""
config/settings.py

Central configuration for the Posha Assistant project.
Defines:
- Paths (DB, prompts, data)
- Model configurations per agent
- Fallback models
- Common constants (token limits, temperature)
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ------------------------------------------------------------
# Base Paths
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data"
PROMPTS_PATH = ROOT / "prompts"
DB_PATH = Path(os.getenv("DATABASE_PATH", ROOT / "posha.db"))
LOG_PATH = ROOT / "logs"

LOG_PATH.mkdir(exist_ok=True)
PROMPTS_PATH.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Prompt Files
# ------------------------------------------------------------
PROMPT_PATHS = {
    "preference": PROMPTS_PATH / "preference_v1.txt",
    "planner": PROMPTS_PATH / "planner_v1.txt",
    "decision": PROMPTS_PATH / "decision_v1.txt",
}

# ------------------------------------------------------------
# Model Configurations
# ------------------------------------------------------------
AGENT_CONFIGS = {
    "preference": {
        "primary_model": "gemini-2.5-flash",
        "fallback_model": "gemini-2.5-pro",
        "temperature": 0.25,
        "max_output_tokens": 4000,
        "response_mime_type": "application/json",
        "retries": 1,
    },
    "planner": {
        "primary_model": "gemini-2.5-pro",
        "fallback_model": "gemini-2.5-flash",
        "temperature": 0.35,
        "max_output_tokens": 6000,
        "response_mime_type": "application/json",
        "retries": 1,
    },
    "decision": {
        "primary_model": "gemini-2.5-flash",
        "fallback_model": "gemini-2.5-pro",
        "temperature": 0.0,
        "max_output_tokens": 1024,
        "response_mime_type": "application/json",
        "retries": 1,
    },
}

# Toggle prompt/response logging (useful for debugging)
_LOG_PROMPTS_RAW = os.getenv("LOG_PROMPTS", "false").lower()
LOG_PROMPTS = _LOG_PROMPTS_RAW in ("1", "true", "yes", "on")

# ------------------------------------------------------------
# Utility Accessors
# ------------------------------------------------------------
def get_agent_config(agent_name: str) -> dict:
    """Return configuration dictionary for the given agent."""
    if agent_name not in AGENT_CONFIGS:
        raise KeyError(f"No configuration found for agent '{agent_name}'.")
    return AGENT_CONFIGS[agent_name]

def get_prompt_path(agent_name: str) -> Path:
    """Return the prompt path for the given agent."""
    if agent_name not in PROMPT_PATHS:
        raise KeyError(f"No prompt path found for agent '{agent_name}'.")
    return PROMPT_PATHS[agent_name]

def should_log_prompts() -> bool:
    """Return whether to store prompts/responses for debugging."""
    return LOG_PROMPTS