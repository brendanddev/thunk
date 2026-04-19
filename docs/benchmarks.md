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

### Phase 8.2 Baseline (Backend: llama.cpp qwen2.5-3b q4_k_m, Machine: M2 Air 8GB)

| Version | Date       | Backend                                   | Scenario             | Prompt / action                                                         | Expected behavior                                                                 | Observed behavior                                                                   | Tool rounds | Answer mode   | Pass   | Notes                                                                | Source |
|---------|------------|-------------------------------------------|----------------------|-------------------------------------------------------------------------|-----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|-------------|---------------|--------|----------------------------------------------------------------------|--------|
| 0.8.10  | 2026-04-19 | qwen2.5-3b q4_k_m                         | create file          | Create a file test_phase82.txt with the content hello world             | write_file proposed, approval required, file created, grounded confirmation       | write_file emitted, approval required, file created successfully, correct synthesis | 1           | ToolAssisted  | PASS   | Clean execution, no formatting drift                                 | manual |
| 0.8.10  | 2026-04-19 | qwen2.5-3b q4_k_m                         | reject mutation      | Create a file reject_test_phase75.txt with the content should not exist | write_file proposed, reject handled, no file created                              | write_file emitted, approval requested, reject handled, no file created, Direct ack | 1           | Direct        | PASS   | Slightly weak post-reject messaging                                  | manual |
| 0.8.10  | 2026-04-19 | qwen2.5-3b q4_k_m                         | edit file            | Edit test_phase82.txt and change hello world to hello params            | malformed attempts corrected, final valid edit executes                           | malformed edit_file emitted twice, no tool execution triggered                      | 0           | Direct        | FAIL   | Parser does not recognize old/new content format                     | manual |
| 0.8.10  | 2026-04-19 | qwen2.5-3b q4_k_m                         | missing read         | Read missing_file_phase75.rs                                            | read_file attempted, failure surfaced cleanly, no fabricated content              | read_file attempted twice, both fail, correct conclusion                            | 2           | ToolAssisted  | PASS   | Slight redundant retry but bounded                                   | manual |
| 0.8.10  | 2026-04-19 | qwen2.5-3b q4_k_m                         | existing read        | Read test_phase82.txt                                                   | read_file executes, returns content, grounded answer                              | read_file executes, correct file content returned                                   | 1           | ToolAssisted  | PASS   | Clean and correct                                                    | manual |
| 0.8.10  | 2026-04-19 | qwen2.5-3b q4_k_m                         | search natural lang  | Find where logging is initialized                                       | bounded search, keyword-based, no retry narration                                 | search_code used once, query simplified, grounded answer                            | 1           | ToolAssisted  | PASS   | Search behavior fixed (no spiral)                                    | manual |

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
