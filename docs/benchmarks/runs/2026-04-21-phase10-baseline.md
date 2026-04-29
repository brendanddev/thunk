# Benchmark Run — 2026-04-21 — Phase 10.0

Date: 2026-04-21  
Version: 0.8.16  
Backend: llama.cpp  
Model: qwen2.5-coder-3b-instruct-q4_k_m  
Machine: M2 Air 8GB  

---

## Context

This section validates the completed Basic Anchor slices of Phase 10.0:

- last-read file anchor and last-search replay, both runtime-owned and structurally enforced through exact phrase matching
- Anchor behavior is strictly explicit and non-semantic; pronouns, ordinals, and fuzzy references are intentionally unsupported
- Anchor replay is bounded to a single typed tool call and does not trigger investigation flows or candidate reads
- Phase 9 invariants (search → read → answer, read caps, path scoping) must remain preserved

---

## Results

| Version | Date       | Backend                                            | Scenario                       | Prompt / action                                                                    | Expected behavior                                                           | Observed behavior                                                   | Tool rounds | Answer mode     | Pass | Notes                                                                               | Source |
| ------- | ---------- | ---------------------------------------------------| ------------------------------ | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------- | ----------- | --------------- | ---- | ----------------------------------------------------------------------------------- | ------ |
| 0.8.16  | 2026-04-21 | qwen2.5-coder-3b-instruct q4_k_m                   | mutation regression            | Create a file test.txt with the content hello world in sandbox/                    | write_file proposed; approval required; file created; grounded confirmation | write_file → approve → read_file → grounded confirmation            | 2           | ToolAssisted    | PASS | Mutation flow preserved; anchor_updated triggered only after successful read        | manual |
| 0.8.16  | 2026-04-21 | qwen2.5-coder-3b-instruct q4_k_m                   | mutation rejection             | Create a file phase10_test.txt with the content hello anchors (reject)             | write_file proposed; rejection cancels mutation                             | write_file → reject → deterministic runtime cancellation            | 1           | RuntimeTerminal | PASS | Clean rejection path; no side effects                                               | manual |
| 0.8.16  | 2026-04-21 | qwen2.5-coder-3b-instruct q4_k_m                   | edit regression                | Edit sandbox/test.txt changing hello world to hello params                         | edit_file proposed; approval required; edit applied                         | edit_file → approve → grounded confirmation                         | 1           | ToolAssisted    | PASS | Edit flow unchanged by anchors                                                      | manual |
| 0.8.16  | 2026-04-21 | qwen2.5-coder-3b-instruct q4_k_m                   | usage investigation regression | Find where TaskStatus is used in sandbox/                                          | search → read → grounded usage answer                                       | search_code → read_file → grounded answer                           | 2           | ToolAssisted    | PASS | Phase 9 investigation behavior preserved                                            | manual |
| 0.8.16  | 2026-04-21 | qwen2.5-coder-3b-instruct q4_k_m                   | last-read anchor               | Read sandbox/main.py → read that file again → open the last file                   | anchor resolves to last_read_file; repeated read_file                       | read_file → anchor replay → anchor replay                           | 1 per step  | ToolAssisted    | PASS | Exact phrase matching works; anchor_resolved + anchor_updated logged                | manual |
| 0.8.16  | 2026-04-21 | qwen2.5-coder-3b-instruct q4_k_m                   | last-read no-anchor            | read that file (new session)                                                       | deterministic failure; no tool call                                         | runtime terminal: No previous file is available to read             | 0           | RuntimeTerminal | PASS | anchor_missing triggered; correct isolation across sessions                         | manual |
| 0.8.16  | 2026-04-21 | qwen2.5-coder-3b-instruct q4_k_m                   | last-search anchor             | Find logging init → search that again → repeat the last search → search again      | exact search replay; one search_code per prompt                             | search_code → anchor replay → anchor replay → anchor replay         | 1 per step  | ToolAssisted    | PASS | Query + scope preserved; no candidate reads triggered                               | manual |
| 0.8.16  | 2026-04-21 | qwen2.5-coder-3b-instruct q4_k_m                   | last-search no-anchor          | search that again (new session)                                                    | deterministic failure; no tool call                                         | runtime terminal: No previous search is available                   | 0           | RuntimeTerminal | PASS | anchor_missing correctly handled                                                    | manual |
| 0.8.16  | 2026-04-21 | qwen2.5-coder-3b-instruct q4_k_m                   | search anchor overwrite        | logging search → TaskStatus search → repeat the last search                        | last search replaces previous; replay new query                             | search_code(logging) → search_code(TaskStatus) → replay TaskStatus  | 1           | ToolAssisted    | PASS | Anchor overwrite works correctly; state updated only on successful search           | manual |
| 0.8.16  | 2026-04-21 | qwen2.5-coder-3b-instruct q4_k_m                   | unsupported anchor phrases     | search it again → search for that thing again → search again → read that → open it | no anchor resolution; fallback to normal runtime/model behavior             | normal search/read flows triggered; no anchor_prompt_matched events | variable    | Mixed           | PASS | Correct non-resolution; confirms strict structural matching (no pronouns/semantics) | manual |

---

## Notes

- Phase 10 introduces runtime-owned anchor behavior
- Anchor resolution is strictly structural (no semantic interpretation)
- Investigation invariants from Phase 9 remain preserved