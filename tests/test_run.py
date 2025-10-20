# tests/test_run.py

"""
tests/test_run.py

Integration smoke test:
- sets DATABASE_PATH to a temporary sqlite file
- initializes DB (executes core/schema.sql)
- inserts a small recipe set
- runs PreferenceAgent then PlannerAgent
- asserts a plan was created and profile updated
"""

import os
import tempfile
import sqlite3
import json
from pathlib import Path

# Set test DB path BEFORE importing config modules so they pick it up
tmp_db = tempfile.NamedTemporaryFile(prefix="posha_test_", suffix=".db", delete=False)
TEST_DB_PATH = tmp_db.name
os.environ["DATABASE_PATH"] = TEST_DB_PATH

import config.settings as settings  # now picks up DATABASE_PATH
from core.memory_layer import init_db, seed_recipes_if_empty, get_connection
from core.memory_layer import to_json
from agents.preference_agent import PreferenceAgent
from agents.planner_agent import PlannerAgent

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "core" / "schema.sql"

def setup_test_db():
    # initialize schema
    init_db()
    # insert minimal recipes directly
    conn = sqlite3.connect(TEST_DB_PATH)
    cur = conn.cursor()
    # insert a couple of test recipes
    recipes = [
        {
            "recipe_id": "t_rec_1",
            "name": "Test Burrito",
            "main_ingredients": json.dumps(["tortilla", "beans", "cheese"]),
            "tags": json.dumps(["mexican", "spicy"]),
            "prep_time_min": 10,
            "cook_time_min": 10,
            "nutrition_facts": json.dumps({"protein": 15}),
            "cuisine": "Mexican",
            "cooking_style": "bowl",
            "description": "A test burrito"
        },
        {
            "recipe_id": "t_rec_2",
            "name": "Test Burger",
            "main_ingredients": json.dumps(["beef", "bun"]),
            "tags": json.dumps(["american"]),
            "prep_time_min": 15,
            "cook_time_min": 10,
            "nutrition_facts": json.dumps({"protein": 25}),
            "cuisine": "American",
            "cooking_style": "grilled",
            "description": "A test burger"
        }
    ]
    for r in recipes:
        cur.execute("""
            INSERT OR REPLACE INTO recipes (recipe_id, name, main_ingredients, tags, prep_time_min, cook_time_min, nutrition_facts, cuisine, cooking_style, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (r["recipe_id"], r["name"], r["main_ingredients"], r["tags"], r["prep_time_min"], r["cook_time_min"], r["nutrition_facts"], r["cuisine"], r["cooking_style"], r["description"]))
    conn.commit()
    conn.close()

def test_preference_and_planner_flow():
    # Setup DB
    setup_test_db()

    user_id = "test_user_1"
    pref_agent = PreferenceAgent()
    planner = PlannerAgent()

    # 1. Provide preference
    merged = pref_agent.process_user_text(user_id=user_id, user_text="I want mexican food, avoid beef")
    assert isinstance(merged, dict)
    assert "session_overlay" in merged or "base_preferences" in merged

    # 2. Generate plan
    plan = planner.generate_plan(user_id=user_id, session_text="Please make a 3-day plan focused on Mexican")
    assert "plan_id" in plan
    assert plan["status"] == "draft"
    assert isinstance(plan["plan_json"], dict)
    assert "plan" in plan["plan_json"]
    # each day should have 2 meals
    for d in plan["plan_json"]["plan"]:
        assert len(d["meals"]) == 2

    # 3. Verify profile written in DB
    conn = sqlite3.connect(TEST_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT preferences_json, weightages_json FROM profiles WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    assert row is not None

if __name__ == "__main__":
    test_preference_and_planner_flow()
    print("✔ test_run passed")
