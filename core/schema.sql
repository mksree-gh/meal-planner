-- core/schema.sql
-- ==========================================================
-- Posha Database Schema (v1)
-- ==========================================================

PRAGMA foreign_keys = ON;

-- ----------------------------------------------------------
-- USERS / PROFILES
-- Stores base preferences, adaptive weightages, session overlays, and rejection history.
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS profiles (
    user_id TEXT PRIMARY KEY,
    name TEXT,
    profile_version INTEGER DEFAULT 1,
    preferences_json TEXT,        -- base, long-term preferences (diet, dislikes, allergens, etc.)
    weightages_json TEXT,         -- adaptive weights (cuisines, diet_type, ingredients, biases)
    session_overlay_json TEXT,    -- temporary session-specific wishes
    rejection_history_json TEXT,  -- structured history of rejected plans and reasons
    updated_at TEXT,
    source TEXT
);

-- ----------------------------------------------------------
-- RECIPES
-- The complete recipe dataset.
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS recipes (
    recipe_id TEXT PRIMARY KEY,
    name TEXT,
    main_ingredients TEXT,        -- JSON list of key ingredients
    tags TEXT,                    -- JSON list (dietary/nutritional tags)
    cuisine TEXT,
    prep_time_min INTEGER,
    cook_time_min INTEGER,
    nutrition_facts TEXT,         -- JSON (optional)
    cooking_style TEXT,
    description TEXT
);

-- ----------------------------------------------------------
-- PLANS
-- Stores each generated meal plan, draft or approved.
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS plans (
    plan_id TEXT PRIMARY KEY,
    user_id TEXT,
    plan_json TEXT,               -- structured MealPlanOutput
    candidate_recipe_ids TEXT,    -- JSON list of recipes considered
    rationale TEXT,
    status TEXT,                  -- draft | approved | rejected
    rejection_reason TEXT,
    created_at TEXT,
    approved_at TEXT,
    rejected_at TEXT,
    updated_at TEXT
);

-- ----------------------------------------------------------
-- RUN STATE
-- Tracks progress, enables pause/resume.
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS run_state (
    run_id TEXT PRIMARY KEY,
    user_id TEXT,
    stage TEXT,                   -- current stage (preference, planning, approval)
    last_step TEXT,
    status TEXT,                  -- in_progress | completed | error
    current_plan_id TEXT,
    error_message TEXT,
    updated_at TEXT
);

-- ----------------------------------------------------------
-- ERRORS
-- Compact error logs (used by Orchestrator).
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS errors (
    error_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    agent_name TEXT,
    step TEXT,
    summary TEXT,
    hint TEXT,
    timestamp TEXT
);
