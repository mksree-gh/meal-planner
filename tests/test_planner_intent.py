"""
tests/test_planner_intent.py

Offline unit tests for PlannerIntentAgent (conversational layer).

Covers:
- Conversational response flow
- Regeneration trigger
- Missing plan fallback
- Invalid LLM output recovery
- Logging consistency
"""

import pytest
import sqlite3
from unittest.mock import MagicMock, patch
from core.memory_layer import MemoryLayer
from agents.planner_intent import PlannerIntentAgent


# ======================================================================
# 🔒 GLOBAL AUTOUSE FIXTURE — disables all real Gemini calls
# ======================================================================
@pytest.fixture(autouse=True)
def patch_genai(monkeypatch):
    """
    Force every genai.Client() call to return a fake local client.
    Prevents any external network requests or API key use.
    Autouse=True ensures it applies to every test automatically.
    """

    def fake_generate_content(model, contents, config):
        """Return fake structured LLM responses."""
        user_text = contents[-1]["parts"][0]["text"].lower()
        if any(k in user_text for k in ["protein", "change", "avoid"]):
            return MagicMock(parsed={"response": "Sure, I can adjust that!", "action": "regenerate"})
        else:
            return MagicMock(parsed={"response": "Yes, this plan is balanced and nutritious.", "action": "respond"})

    fake_client = MagicMock()
    fake_client.models.generate_content = fake_generate_content

    # Patch both possible import paths
    monkeypatch.setattr("agents.planner_intent.genai.Client", lambda: fake_client)
    monkeypatch.setattr("google.genai.Client", lambda: fake_client)

    return fake_generate_content


# ======================================================================
# 🧱 Fixtures — create in-memory DB with Posha schema and one plan
# ======================================================================
@pytest.fixture
def mock_memory(tmp_path):
    """Creates a temporary SQLite DB with schema + one fake plan."""
    from core.memory_layer import SCHEMA_PATH

    db_path = tmp_path / "test_posha.db"
    mem = MemoryLayer(str(db_path))

    # Initialize schema
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        sql = f.read()
    conn = sqlite3.connect(db_path)
    conn.executescript(sql)
    conn.commit()
    conn.close()

    # Insert a fake plan
    mem.save_plan(
        user_id="user_001",
        plan_id="plan_abc",
        plan_json={
            "plan": [
                {"day": "Day 1", "meals": ["Salad Bowl", "Grilled Chicken"]},
                {"day": "Day 2", "meals": ["Pasta", "Tofu Stir Fry"]},
                {"day": "Day 3", "meals": ["Smoothie", "Veggie Curry"]},
            ]
        },
        rationale="Balanced plan with moderate protein.",
        created_at="2025-10-21T00:00:00Z",
    )

    return mem


@pytest.fixture
def intent_agent(mock_memory):
    """Return a PlannerIntentAgent instance using mock memory."""
    agent = PlannerIntentAgent()
    agent.memory = mock_memory
    return agent


# ======================================================================
# 🧪 TEST CASES
# ======================================================================
def test_conversational_response(intent_agent):
    """Should return a conversational reply for neutral queries."""
    result = intent_agent.handle_message("user_001", "Is this plan balanced?")
    assert result["action"] == "respond"
    assert "balanced" in result["response"].lower()


def test_triggers_regeneration(intent_agent):
    """Should trigger regeneration when modification is requested."""
    with patch("agents.planner_agent.PlannerAgent.generate_plan", return_value={"plan": "new"}) as mock_gen:
        result = intent_agent.handle_message("user_001", "Make it higher in protein")
        assert result["action"] == "regenerate"
        assert "sure" in result["response"].lower()
        mock_gen.assert_called_once()


def test_missing_plan_graceful(intent_agent):
    """If no plan exists, agent should prompt to create one."""
    # Delete plans
    conn = sqlite3.connect(intent_agent.memory.db_path)
    conn.execute("DELETE FROM plans")
    conn.commit()
    conn.close()

    result = intent_agent.handle_message("user_001", "What meals do I have?")
    assert result["action"] == "regenerate"
    assert "create" in result["response"].lower()


def test_invalid_llm_output_recovery(intent_agent, monkeypatch):
    """Handles invalid LLM output gracefully by using fallback parsing."""
    def bad_generate(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.text = "This is not valid JSON, just a text answer."
        mock_resp.parsed = None
        return mock_resp

    # Patch the constructor for both possible paths
    monkeypatch.setattr(
        "agents.planner_intent.genai.Client",
        lambda: MagicMock(models=MagicMock(generate_content=bad_generate))
    )
    monkeypatch.setattr(
        "google.genai.Client",
        lambda: MagicMock(models=MagicMock(generate_content=bad_generate))
    )

    result = intent_agent.handle_message("user_001", "What's in day 1?")
    assert result["action"] == "respond"
    assert "day" in result["response"].lower()


def test_logging_and_action_consistency(intent_agent, tmp_path):
    """Ensure planner_intent logs events with correct structure."""
    result = intent_agent.handle_message("user_001", "Avoid rice in my plan")
    assert "response" in result
    assert result["action"] in ("respond", "regenerate")

    # Verify log file exists
    from pathlib import Path
    logs_path = Path("logs")
    files = list(logs_path.glob("session_*.jsonl"))
    assert files, "No session logs found"
