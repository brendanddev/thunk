# Benchmarks

Manual QA and performance benchmark records for `thunk`.

These benchmarks capture real prompts, observed runtime behavior, tool usage, regressions, and performance notes across project phases.

The goal is to record what actually happened during real runs so behavior can be compared over time.

---

## Structure

```
docs/benchmarks/
├── README.md
└── runs/
    └── YYYY-MM-DD-phase-name.md
```

- README.md explains the system and rules.
- runs/ contains individual benchmark runs.
- Each run is isolated, dated, and tied to a specific phase or validation pass.

---

## Run File Naming

Use:

YYYY-MM-DD-phase-or-purpose.md

Examples:

- 2026-04-23-phase-11-1-3.md
- 2026-04-27-phase-13-1-4.md
- 2026-04-29-runtime-refactor-baseline.md

---

## Benchmark Rules

- Record actual behavior, not intended behavior
- Keep rows comparable across runs
- Use LIMITATION for known weaknesses
- Use FAIL only when invariants break
- Do not paste large logs — reference them

---

## Standard Values

Pass column:
- PASS
- FAIL
- LIMITATION

Answer mode:
- ToolAssisted
- RuntimeTerminal
- RuntimeCommand
- Mixed