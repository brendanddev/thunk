# Benchmark Run — 2026-04-20 — Phase 9.0

Date: 2026-04-20  
Version: 0.8.12  
Backend: llama.cpp  
Model: qwen2.5-coder-3b-instruct-q4_k_m  
Machine: M2 Air 8GB  

---

## Context

Phase 9.0 baseline validating investigation behavior.

This run captures early investigation flows including:
- search → read chaining
- definition vs usage lookup behavior
- basic file explanation and retrieval patterns

---

## Results

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

---

## Notes

- Phase 9 introduces investigation flow (search → read chaining)
- Candidate selection is still shallow but structurally correct
- This run serves as the baseline before investigation refinements