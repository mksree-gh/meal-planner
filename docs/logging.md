# 📘 `docs/logging.md` — Developer Guide

# 🧭 Logging & Observability System (v1)

This document describes the new structured logging system used across **PreferenceAgent**, **PlannerAgent**, and the Orchestrator.

---

## 📂 Directory Structure

```

/logs/
│
├── agents/           ← JSONL per agent (structured events)
│   ├── preference.jsonl
│   ├── planner.jsonl
│   └── orchestrator.jsonl
│
├── sessions/         ← JSONL per LLM call or workflow (debug traces)
│   ├── session_run_ab12cd34.jsonl
│   └── ...
│
├── summaries/        ← Markdown summaries for human review
│   ├── user_001_preference.md
│   ├── user_001_planner.md
│   └── ...
│
└── runtime.log       ← One-line live traces for CLI tailing

````

---

## 🧱 Log Layers Overview

| Layer | Purpose | Example |
|-------|----------|---------|
| **Agent Logs** (`/agents/`) | Persistent structured events | Plan generation, preference updates |
| **Session Logs** (`/sessions/`) | Fine-grained step traces | LLM prompt + response |
| **Summary Logs** (`/summaries/`) | Human-readable overviews | Markdown snapshots |
| **Runtime Log** (`runtime.log`) | Simple live feedback | `"planner: user_001 -> plan_generated"` |

---

## 🔍 Event Schema (Agent Logs)

Every JSON line follows:
```json
{
  "timestamp": "2025-10-21T09:30:00Z",
  "agent": "preference",
  "type": "weight_diff",
  "data": {
    "user_id": "user_001",
    "axis": "cuisines",
    "changes": {
      "thai": {"change": "added", "new": 0.8},
      "italian": {"change": "updated", "old": 0.6, "new": 0.9, "delta": 0.3}
    }
  }
}
````

---

## 📑 Common Event Types

| Event Type                    | Logged By                   | Meaning                       |
| ----------------------------- | --------------------------- | ----------------------------- |
| `preference_update`           | PreferenceAgent             | Final merged preference state |
| `weight_diff`                 | PreferenceAgent             | Per-axis change diff          |
| `plan_generated`              | PlannerAgent                | New meal plan draft           |
| `plan_approved`               | PlannerAgent / Orchestrator | User accepted plan            |
| `plan_rejected`               | PlannerAgent / Orchestrator | User rejected plan            |
| `feedback_cycle`              | PlannerAgent                | Plan regeneration attempt     |
| `llm_prompt` / `llm_response` | Any agent                   | Raw model trace events        |

---

## 🧠 Summary Logs (Markdown)

Summaries appear under `/logs/summaries/` for quick reading.

Example — `user_001_preference.md`:

```
## 🧠 Preference Update — user_001
**Input:** I love Italian and Thai food  
⚖️ **cuisines**: +thai ↑0.8, italian ↑0.3  
✅ Profile version 3
```

Example — `user_001_planner.md`:

```
## 📋 Meal Plan — user_001
**Plan ID:** plan_8a7cde12
**Rationale:** Balanced Thai + Italian menu

- **Day 1:** Pad Thai, Caprese Salad
- **Day 2:** Green Curry, Pasta Primavera
```

---

## 🕵️ Querying Logs

### View recent events per agent

```bash
jq '.[].data.user_id' logs/agents/preference.jsonl | tail -10
```

### Inspect all plans generated for a user

```bash
jq 'select(.data.user_id=="user_001" and .type=="plan_generated")' logs/agents/planner.jsonl
```

### Tail live runtime activity

```bash
tail -f logs/runtime.log
```

### Correlate a full run

```bash
cat logs/sessions/session_run_*.jsonl | jq .
```

---

## ⚙️ Implementation Notes

* **Thread-safe:** All file writes use a shared lock.
* **Timestamps:** UTC ISO-8601 for global traceability.
* **No external dependencies:** Pure Python + JSON.
* **Backwards compatible:** `log_event` and `log_session_event` remain callable from anywhere.
* **Human-friendly console:** Key changes printed inline (⚖️, 🧠, 📋).

---

## 💡 Extending the System

You can easily add more log types:

```python
log_event("planner", "plan_latency", {"ms": latency})
log_summary("analytics", "report", "Generated weekly metrics.")
```

Or create metrics collectors by tailing JSONL files.

---

## ✅ Quick Checklist for Developers

* [ ] Verify `/logs/agents/` JSONL is well-formed with `jq .`
* [ ] Ensure every run_id has a `/logs/sessions/session_*.jsonl`
* [ ] Summaries must exist for user profiles and plans
* [ ] Runtime log should mirror main flow
* [ ] Keep LLM payloads under 4KB (truncate via `[:4000]`)

---

**Version:** Logging v1.0 (2025-10-21)
```

---
