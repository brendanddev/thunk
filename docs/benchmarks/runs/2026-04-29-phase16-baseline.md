# Benchmark Run — 2026-04-29 — Pre-Phase 16 Baseline

Date: 2026-04-29  
Version: 0.8.30  
Backend: llama.cpp  
Model: qwen2.5-coder-1.5b-instruct-q4_k_m  
Machine: M2 Air 8GB 

---

## Context

This run captures the behavior of the system immediately before Phase 16.

System state at this point:

- Runtime modularization complete (project, protocol, investigation, orchestration)
- Search → read → answer gating enforced
- Tool surface restrictions active
- Investigation modes and path scoping active
- Anchors implemented (last-read, last-search)
- Retrieval uses substring-based search (`search_code`)

Known limitations at this stage:

- No strict candidate enforcement after search
- Weak semantic ranking of search results
- Model can select incorrect files despite correct candidates
- Tool formatting fragile under small models
- Context window easily exceeded in multi-step flows

---

## Key Behaviors Being Measured

- retrieval correctness (file selection quality)
- search → read discipline
- handling of weak / broad queries
- failure behavior (search budget, terminals)
- mutation flow stability
- direct read behavior
- investigation flow correctness

---

## Results
| Version | Date | Backend | Scenario | Prompt / action | Expected behavior | Observed behavior | Tool rounds | Answer mode | Pass | Notes | Source |
|--------|------|---------|----------|-----------------|------------------|------------------|-------------|-------------|------|------|--------|
| 0.8.30 | 2026-04-29 | qwen2.5-coder-1.5b-instruct q4_k_m | initialization lookup | Find where logging is initialized in sandbox/ | search → read candidate in sandbox/ → grounded answer | search scoped correctly, but model attempted read on `.github/ISSUE_TEMPLATE.md`; read failed; runtime terminated | 2 | RuntimeTerminal | FAIL | Non-candidate read after scoped search; breaks retrieval discipline | manual/log |
| 0.8.30 | 2026-04-29 | qwen2.5-coder-1.5b-instruct q4_k_m | definition lookup | Where is TaskStatus defined in sandbox/ | search → read correct definition file → grounded answer | correctly read sandbox/models/enums.py and returned definition | 2 | ToolAssisted | PASS | Clean definition lookup | manual |
| 0.8.30 | 2026-04-29 | qwen2.5-coder-1.5b-instruct q4_k_m | usage lookup | Where is TaskStatus used in sandbox/ | search → read usage sites → grounded usage answer | read correct files but answered definition instead of usage | 3 | ToolAssisted | FAIL | Usage vs definition confusion; synthesis error despite correct reads | manual/log |
| 0.8.30 | 2026-04-29 | qwen2.5-coder-1.5b-instruct q4_k_m | filtering lookup | Where are completed tasks filtered in sandbox/ | search → read relevant service file → correct location | read README instead of source file; hallucinated correct location | 2 | ToolAssisted | FAIL | Wrong candidate selection; answer not grounded in read file | manual/log |
| 0.8.30 | 2026-04-29 | qwen2.5-coder-1.5b-instruct q4_k_m | file explanation | What does sandbox/services/task_service.py do? | read target file → grounded explanation | read correct file but marked as non-candidate; later read unrelated benchmark file | 3 | ToolAssisted | FAIL | Retrieval discipline broken; candidate rejection incorrect; drift to unrelated file | manual/log |
| 0.8.30 | 2026-04-29 | qwen2.5-coder-1.5b-instruct q4_k_m | direct read | Read sandbox/main.py | direct read → return file content | correct file read and returned | 1 | ToolAssisted | PASS | Direct read works but flagged as non-candidate internally | manual |
| 0.8.30 | 2026-04-29 | qwen2.5-coder-1.5b-instruct q4_k_m | direct read | Read sandbox/services/task_service.py | direct read → return file content | correct file read and returned | 1 | ToolAssisted | PASS | Same non-candidate classification issue as previous direct read | manual |
| 0.8.30 | 2026-04-29 | qwen2.5-coder-1.5b-instruct q4_k_m | missing read | Read missing_file_xyz.rs | read_file fails → clean terminal | correctly failed with RuntimeTerminal | 0 | RuntimeTerminal | PASS | Proper failure handling | manual |
| 0.8.30 | 2026-04-29 | qwen2.5-coder-1.5b-instruct q4_k_m | git surface | Show git status → Show git diff → git | bounded git tool usage → stable response | git works, but final prompt exceeds context window and fails | 1 | Mixed | LIMITATION | Context overflow on chained git usage | manual/log |
| 0.8.30 | 2026-04-29 | qwen2.5-coder-1.5b-instruct q4_k_m | create file | Create a file baseline_test.txt with the content hello world | write_file → approval → file created | correct approval flow and creation | 1 | ToolAssisted | PASS | Mutation flow working correctly | manual |
| 0.8.30 | 2026-04-29 | qwen2.5-coder-1.5b-instruct q4_k_m | edit file | Edit baseline_test.txt and change hello world to hello thunk | edit_file → approval → update applied | model produced invalid tool format; operation failed | 2 | RuntimeTerminal | FAIL | Tool formatting fragility; weak model failure | manual/log |
| 0.8.30 | 2026-04-29 | qwen2.5-coder-1.5b-instruct q4_k_m | anchor behavior | Read → read again → open the last file | anchor reuse → repeated reads only | anchor works but triggers unnecessary search and extra tool calls | 2 | ToolAssisted | LIMITATION | Anchor correctness but inefficient flow | manual/log |

---

## Summary

| Result | Count |
|--------|------:|
| PASS | 5 |
| FAIL | 5 |
| LIMITATION | 2 |

---

## Notes

Key failures observed:

- Retrieval discipline is broken:
  - non-candidate reads allowed after search (initialization lookup, file explanation)
  - model can escape scoped search results

- Candidate selection is weak:
  - incorrect files chosen despite relevant candidates (filtering lookup)
  - drift to unrelated files after correct reads

- Grounding is inconsistent:
  - answers not aligned with read content (usage lookup, filtering lookup)

- Mutation reliability issues:
  - edit_file fails due to invalid tool formatting (small model limitation)

- Context limitations:
  - chained git operations exceed context window

- Anchor system:
  - functionally correct but inefficient (extra tool calls, unnecessary search)

This baseline defines targets for Phase 16:
- retrieval discipline enforcement
- candidate selection improvement
- grounding guarantees
- tool formatting robustness