# Benchmark Run — 2026-05-01 — Post-Phase 17 / Pre-Phase 18

Date: 2026-05-01  
Version: 0.8.33  
Backend: llama.cpp  
Model: qwen2.5-coder-1.5b-instruct q4_k_m  
Machine: M2 Air 8GB  

---

## Context

This run evaluates the system after completion of Phase 16 (retrieval discipline, runtime strategy)
and Phase 17 (external project usage, root handling, bounded enumeration, noisy-directory handling).

Goal:
- validate improvements over the pre-Phase 16 baseline
- identify remaining runtime failure modes
- define the scope of Phase 18

---

## Key Behaviors Being Measured

- search → read → answer discipline
- candidate enforcement and recovery
- answer grounding / evidence correctness
- behavior under weak model outputs
- failure handling and termination conditions
- direct read behavior
- mutation reliability (write/edit)
- environment independence (Phase 17)

---

## Results

| Version | Date       | Backend                            | Scenario              | Prompt / action                                              | Expected behavior                                       | Observed behavior                                                                                                               | Tool rounds | Answer mode     | Pass | Notes                                                                 | Source     |
| ------- | ---------- | ---------------------------------- | --------------------- | ------------------------------------------------------------ | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------- | --------------- | ---- | --------------------------------------------------------------------- | ---------- |
| 0.8.33  | 2026-05-01 | qwen2.5-coder-1.5b-instruct q4_k_m | initialization lookup | Find where logging is initialized in sandbox/                | search → read candidate in sandbox/ → grounded answer   | search scoped correctly; non-candidate read `.github/ISSUE_TEMPLATE.md` rejected; model retried search after closure → terminal | 4           | RuntimeTerminal | FAIL | No recovery after rejected non-candidate read; falls into search loop | manual/log |
| 0.8.33  | 2026-05-01 | qwen2.5-coder-1.5b-instruct q4_k_m | definition lookup     | Where is TaskStatus defined in sandbox/                      | search → read correct definition file → grounded answer | correctly read sandbox/models/enums.py and returned definition                                                                  | 3           | ToolAssisted    | PASS | Stable definition lookup                                              | manual     |
| 0.8.33  | 2026-05-01 | qwen2.5-coder-1.5b-instruct q4_k_m | usage lookup          | Where is TaskStatus used in sandbox/                         | search → read usage sites → grounded usage answer       | read correct files but attempted to reference unread enums.py; answer guard rejected; terminal                                  | 4           | RuntimeTerminal | FAIL | No bounded recovery after answer guard rejection                      | manual/log |
| 0.8.33  | 2026-05-01 | qwen2.5-coder-1.5b-instruct q4_k_m | filtering lookup      | Where are completed tasks filtered in sandbox/               | search → read relevant service file → correct location  | initial bad read redirected; correct file read; grounded answer returned                                                        | 4           | ToolAssisted    | PASS | Candidate redirect worked correctly                                   | manual     |
| 0.8.33  | 2026-05-01 | qwen2.5-coder-1.5b-instruct q4_k_m | file explanation      | What does sandbox/services/task_service.py do?               | read target file → grounded explanation                 | correct read and answer; read classified as non-candidate                                                                       | 2           | ToolAssisted    | PASS | Direct read works but evidence classification is misleading           | manual     |
| 0.8.33  | 2026-05-01 | qwen2.5-coder-1.5b-instruct q4_k_m | direct read           | Read sandbox/main.py                                         | direct read → return file content                       | correct file read and returned                                                                                                  | 1           | ToolAssisted    | PASS | Clean direct read path                                                | manual     |
| 0.8.33  | 2026-05-01 | qwen2.5-coder-1.5b-instruct q4_k_m | direct read           | Read sandbox/services/task_service.py                        | direct read → return file content                       | correct file read and returned                                                                                                  | 1           | ToolAssisted    | PASS | Same classification issue as other direct reads                       | manual     |
| 0.8.33  | 2026-05-01 | qwen2.5-coder-1.5b-instruct q4_k_m | missing read          | Read missing_file_xyz.rs                                     | read_file fails → clean terminal                        | correctly failed with RuntimeTerminal                                                                                           | 1           | RuntimeTerminal | PASS | Proper failure handling                                               | manual     |
| 0.8.33  | 2026-05-01 | qwen2.5-coder-1.5b-instruct q4_k_m | create file           | Create a file baseline_test.txt with the content hello world | write_file → approval → file created                    | correct approval flow and creation                                                                                              | 1           | ToolAssisted    | PASS | Mutation flow stable                                                  | manual     |
| 0.8.33  | 2026-05-01 | qwen2.5-coder-1.5b-instruct q4_k_m | edit file             | Edit baseline_test.txt and change hello world to hello thunk | edit_file → approval → update applied                   | malformed tool syntax; repeated correction; terminal                                                                            | 2           | RuntimeTerminal | FAIL | Weak model tool formatting; no recovery path                          | manual/log |

---

## Summary

| Result | Count |
|--------|------:|
| PASS   | 6 |
| FAIL   | 3 |
| N/A    | 1 |

---

## Notes

### Improvements from baseline (pre-Phase 16)
- non-candidate reads are now rejected (no silent drift)
- answer guard prevents ungrounded answers
- direct reads are deterministic and fast
- mutation create flow is stable
- environment independence works (Phase 17)

### Remaining failure modes

1. **Non-candidate read recovery is missing**
   - runtime rejects invalid read but does not redirect to valid candidate
   - leads to repeated search violations and terminal

2. **Answer recovery after guard rejection is missing**
   - model reads correct files but attempts synthesis using unread file
   - runtime terminates instead of forcing bounded answer from existing evidence

3. **Direct read evidence classification is unclear**
   - valid reads marked as `not_search_candidate`
   - does not break behavior but weakens evidence model

4. **Edit tool is unreliable with small models**
   - malformed tool syntax leads to terminal
   - likely requires protocol-level mitigation

### Conclusion

The system has improved correctness and safety but lacks bounded recovery paths.

Failures are no longer due to lack of enforcement, but due to:
- insufficient runtime-controlled recovery strategies
- reliance on model to self-correct after rejection