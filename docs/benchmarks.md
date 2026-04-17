# Benchmarks

Provides real manual prompts and actions to try during development, along with expected behaviors and source files to check when things go wrong.

---

## What this is for

The goal is for this file to act as a place to document results from real manual runs to be evaluated for QA.

Prefer recording real observed behavior here instead of assumptions from reading code alone.
Keep entries short and comparable so multiple runs can be reviewed side by side.

---

## Manual QA Runs

Use this table for prompt-driven validation.
Add one row per scenario or manual check, and record what actually happened in the app.
If a run fails, point `Source` at the first code path you would inspect.

| Version | Date | Backend | Scenario | Prompt / action | Expected behavior | Observed behavior | Pass | Notes | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

---

## Timing / Performance Observations

Use this table only for measured timings from real runs.
Prefer values taken from the session log in `logs/` when available.
Leave timing cells blank rather than guessing.

| Version | Date | Backend | Model | Scenario | Cold/Warm | Generation ms | Tool ms | ctx_create ms | tokenize ms | prefill ms | generation stage ms | Log file | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

---

## Environment

Use this table to capture the config and machine context behind timing results.
This makes runs easier to compare when the model, token limits, or hardware change.

| Version | Backend | Model | context_tokens | batch_tokens | max_tokens | Machine notes |
| --- | --- | --- | --- | --- | --- | --- |
