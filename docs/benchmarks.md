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

### Phase 8.2 Current Checks

> Backend: llama.cpp qwen2.5-3b-instruct-q4_k_m, Machine: M2 Air 8GB

The rows below reflect the current expected behavior after the final Phase 8.2 stabilization fixes. Some values are source/test-validated rather than fresh live CLI observations; replace them with live observations during the next manual pass.

| Version | Date       | Backend                                            | Scenario             | Prompt / action                                                         | Expected behavior                                                                 | Observed behavior                                                                   | Tool rounds | Answer mode   | Pass   | Notes                                                                 | Source                                                |
|---------|------------|----------------------------------------------------|----------------------|-------------------------------------------------------------------------|-----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|-------------|---------------|--------|-----------------------------------------------------------------------|-------------------------------------------------------|
| 0.8.10  | 2026-04-19 | qwen2.5-3b-instruct q4_k_m                         | create file          | Create a file test_phase82.txt with the content hello world             | write_file proposed, approval required, file created, grounded confirmation       | write_file emitted, approval required, file created successfully, correct synthesis | 1           | ToolAssisted  | PASS   | Clean execution, no formatting drift                                  | manual                                                |
| 0.8.10  | 2026-04-19 | qwen2.5-3b-instruct q4_k_m                         | reject mutation      | Create a file reject_test_phase75.txt with the content should not exist | write_file proposed, reject handled, no file created, runtime-owned cancellation  | Runtime path now emits cancellation without model synthesis                         | 1           | ToolAssisted  | PASS   | Source/test validated; refresh with live CLI                          | `src/runtime/engine.rs`                               |
| 0.8.10  | 2026-04-19 | qwen2.5-3b-instruct q4_k_m                         | edit file            | Edit test_phase82.txt and change hello world to hello params            | valid or narrowly tolerated edit format executes through approval                 | `old content:` / `new content:` format now parses and requests approval             | 1           | ToolAssisted  | PASS   | Edit may still need multiple model attempts; quality, not correctness | `src/runtime/tool_codec.rs`, `src/runtime/engine.rs`  |
| 0.8.10  | 2026-04-19 | qwen2.5-3b-instruct q4_k_m                         | missing read         | Read missing_file_phase75.rs                                            | read_file attempted, failure surfaced cleanly, no retry loop                      | Runtime path now emits terminal failed-read answer after tool error                 | 1           | ToolAssisted  | PASS   | Source/test validated; refresh with live CLI                          | `src/runtime/engine.rs`                               |
| 0.8.10  | 2026-04-19 | qwen2.5-3b-instruct q4_k_m                         | existing read        | Read test_phase82.txt                                                   | read_file executes, returns content, grounded answer                              | read_file executes, correct file content returned                                   | 1           | ToolAssisted  | PASS   | Clean and correct                                                     | manual                                                |
| 0.8.10  | 2026-04-19 | qwen2.5-3b-instruct q4_k_m                         | search natural lang  | Find where logging is initialized                                       | bounded search, keyword-based, no retry narration                                 | search_code used once, query simplified, grounded answer                            | 1           | ToolAssisted  | PASS   | Search behavior fixed (no spiral)                                     | manual                                                |

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
