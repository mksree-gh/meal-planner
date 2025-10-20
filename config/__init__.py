# config/__init__.py

"""
config package

This module initializes global configuration for the Posha Assistant project.

It exposes key paths, model configurations, and utility accessors from settings.py.
"""

from .settings import (
    ROOT,
    DB_PATH,
    DATA_PATH,
    LOG_PATH,
    PROMPTS_PATH,
    PROMPT_PATHS,
    AGENT_CONFIGS,
    LOG_PROMPTS,
    get_agent_config,
    get_prompt_path,
    should_log_prompts,
)

__all__ = [
    "ROOT",
    "DB_PATH",
    "DATA_PATH",
    "LOG_PATH",
    "PROMPTS_PATH",
    "PROMPT_PATHS",
    "AGENT_CONFIGS",
    "LOG_PROMPTS",
    "get_agent_config",
    "get_prompt_path",
    "should_log_prompts",
]
