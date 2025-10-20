# agents/__init__.py

"""
agents package

Contains all autonomous agents that make up Posha's multi-agent system:
- PreferenceAgent: parses natural language into structured preferences.
- PlannerAgent: generates draft meal plans using recipe data and preferences.
"""

from .preference_agent import PreferenceAgent
from .planner_agent import PlannerAgent

__all__ = ["PreferenceAgent", "PlannerAgent"]
