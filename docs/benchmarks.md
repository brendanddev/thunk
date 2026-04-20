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

### Phase 9.0 Baseline

> Backend: llama.cpp qwen2.5-coder-3b-instruct-q4_k_m, Machine: M2 Air 8GB

| Version | Date       | Backend                                            | Scenario              | Prompt / action                                                         | Expected behavior                                                                 | Observed behavior                                                                   | Tool rounds | Answer mode     | Pass   | Notes                                                                 | Source                                                |
|---------|------------|----------------------------------------------------|-----------------------|-------------------------------------------------------------------------|-----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|-------------|-----------------|--------|-----------------------------------------------------------------------|-------------------------------------------------------|
| 0.8.12  | 2026-04-20 | qwen2.5-coder-3b-instruct q4_k_m                   | create file           | Create a file test_phase9.txt with the content hello world              | write_file proposed, approval required, file created, grounded confirmation       | write_file emitted, approval required, file created, follow-up read confirms        | 2           | ToolAssisted    | PASS   | Clean execution; includes validation read step                        | manual                                                |
| 0.8.12  | 2026-04-20 | qwen2.5-coder-3b-instruct q4_k_m                   | reject mutation       | Create a file reject_test_phase9.txt with the content should not exist  | write_file proposed, reject handled, no file created, runtime-owned cancellation  | Runtime cancels cleanly; no file created; no model-side synthesis                   | 1           | RuntimeTerminal | PASS   | Correct rejection path; no hallucinated follow-up                     | manual                                                |
| 0.8.12  | 2026-04-20 | qwen2.5-coder-3b-instruct q4_k_m                   | edit file             | Edit test_phase9.txt and change hello world to hello params             | edit_file proposed, approval required, change applied, grounded confirmation      | edit_file executed with approval; content updated correctly                         | 1           | ToolAssisted    | PASS   | Clean edit execution; no retry needed                                 | manual                                                |
| 0.8.12  | 2026-04-20 | qwen2.5-coder-3b-instruct q4_k_m                   | missing read          | Read missing_file_phase100.rs                                           | read_file attempted, failure surfaced cleanly, no retry loop                      | read_file fails; runtime returns terminal failure; no retry or hallucination        | 1           | RuntimeTerminal | PASS   | Correct failure handling                                              | manual                                                |
| 0.8.12  | 2026-04-20 | qwen2.5-coder-3b-instruct q4_k_m                   | existing read         | Read test_phase9.txt                                                    | read_file executes, returns content, grounded answer                              | read_file executes; correct content returned                                        | 1           | ToolAssisted    | PASS   | Clean grounded read                                                   | manual                                                |
| 0.8.12  | 2026-04-20 | qwen2.5-coder-3b-instruct q4_k_m                   | search + investigate  | Find where logging is initialized in sandbox/                           | search_code → read_file → grounded answer; prefer relevant source file            | search_code used; read sandbox/cli/commands.py; plausible grounded explanation      | 2           | ToolAssisted    | PASS   | Correct flow; file selection reasonable; answer slightly generic      | manual                                                |
| 0.8.12  | 2026-04-20 | qwen2.5-coder-3b-instruct q4_k_m                   | definition lookup     | Where is TaskStatus defined in sandbox/                                 | search_code → read_file; prefer source definition site                            | read sandbox/models/enums.py; correct definition location returned                  | 2           | ToolAssisted    | PASS   | Strong Phase 9.0 signal; correct file prioritization                  | manual                                                |
| 0.8.12  | 2026-04-20 | qwen2.5-coder-3b-instruct q4_k_m                   | file explanation      | What does sandbox/services/task_service.py do?                          | read_file (or search + read); grounded explanation of file                        | search_code → read_file; correct summary of TaskService responsibilities            | 2           | ToolAssisted    | PASS   | Good grounded explanation; search step used before direct read        | manual                                                |
| 0.8.12  | 2026-04-20 | qwen2.5-coder-3b-instruct q4_k_m                   | usage lookup          | Where are completed tasks filtered in sandbox/                          | search_code → read_file; identify relevant implementation                         | read sandbox/services/task_service.py; correct filtering explanation                | 2           | ToolAssisted    | PASS   | Correct flow; answer slightly high-level vs exact code reference      | manual                                                |

### Phase 9.0.x Single-step Investigation Stabilization (v0.8.13)

> Backend: llama.cpp qwen2.5-coder-3b-instruct-q4_k_m, Machine: M2 Air 8GB
> Phase 9 remains active. This section records the completed Phase 9.0.x stabilization slice only; Phase 9.1 multi-step investigation has not started.

| Version | Date       | Backend                                            | Scenario              | Prompt / action                                                         | Expected behavior                                                                 | Observed behavior                                                                   | Tool rounds | Answer mode     | Pass   | Notes                                                                                     | Source                                                |
|---------|------------|----------------------------------------------------|-----------------------|-------------------------------------------------------------------------|-----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|-------------|-----------------|--------|-------------------------------------------------------------------------------------------|-------------------------------------------------------|
| 0.8.13  | 2026-04-20 | qwen2.5-coder-3b-instruct q4_k_m                   | definition lookup     | Where is TaskStatus defined in sandbox/                                 | search_code → read_file; definition file read is sufficient                       | search_code → read_file; read sandbox/models/enums.py; grounded answer succeeds     | 2           | ToolAssisted    | PASS   | Definition lookup accepts definition-file evidence                                        | manual                                                |
| 0.8.13  | 2026-04-20 | qwen2.5-coder-3b-instruct q4_k_m                   | usage lookup          | Where is TaskStatus used in sandbox/                                    | list_dir blocked before search; definition-only read rejected; usage file read    | list_dir blocked; search_code; read enums.py; targeted recovery; read usage file    | 4           | ToolAssisted    | PASS   | Runtime recovers from definition-first bias with concrete usage-file target               | manual                                                |
| 0.8.13  | 2026-04-20 | qwen2.5-coder-3b-instruct q4_k_m                   | search + investigate  | Find where logging is initialized in sandbox/                           | search_code → read_file; select correct implementation                            | search_code → read_file; read sandbox/logging_setup.py                              | 2           | ToolAssisted    | PASS   | Correct file selected; grounded answer                                                    | manual                                                |
| 0.8.13  | 2026-04-20 | qwen2.5-coder-3b-instruct q4_k_m                   | usage lookup          | Where are completed tasks filtered in sandbox/                          | list_dir blocked before search; search_code → read_file; identify implementation | list_dir blocked; search_code → read_file; grounded filtering answer                | 3           | ToolAssisted    | PASS   | Investigation trigger covers `filtered`; no directory-listing answer                      | manual                                                |
| 0.8.13  | 2026-04-20 | qwen2.5-coder-3b-instruct q4_k_m                   | broad search          | Search for "task" in sandbox/                                           | search_code → read_file; reasonable file selection (not necessarily optimal)      | search_code → read_file; read sandbox/cli/commands.py                               | 2           | ToolAssisted    | PASS   | Behavior unchanged; still shallow but expected for broad query                            | manual                                                |

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
