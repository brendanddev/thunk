# Benchmark Evaluator

## Purpose

Use this skill to evaluate `params-cli` behavior against the tracked benchmark set.

This skill is for:
- recording benchmark runs
- grading quality
- spotting regressions
- comparing versions
- turning ad hoc testing into repeatable evaluation

## Use When

Use this skill when the user asks to:
- benchmark the current version
- evaluate answer quality
- compare before/after behavior
- update `docs/BENCHMARKS.md`
- judge whether a fix actually worked

## Source Of Truth

Primary benchmark doc:
- `docs/BENCHMARKS.md`

Use the benchmark cases and exit criteria in that file as the standard.

## Core Benchmark Set

Evaluate these prompts unless the user asks for a different set:

1. `Can you see my project?`
2. `/read src/main.rs` then `What does this file do?`
3. `Tell me more`
4. `What calls load_most_recent`
5. `What uses SessionStore`
6. `Explain how session restore works`
7. `Where is eco mode configured?`

## What To Record

For each benchmark case, record:
- prompt
- version
- backend
- model
- eco on/off
- reflection on/off
- time to final answer
- whether the answer visibly streamed
- answer text
- correctness
- grounding
- latency
- UX notes

## Scoring

Use the project’s 0-2 scale.

### Correctness
- `0` wrong
- `1` partially right
- `2` correct

### Grounding
- `0` vague or irrelevant
- `1` some real evidence
- `2` clearly grounded in relevant source files/lines

### Latency
- `0` unacceptable
- `1` borderline
- `2` good

### UX
- `0` broken or confusing
- `1` usable with rough edges
- `2` smooth

## Workflow

1. Read `docs/BENCHMARKS.md`.
2. Capture the run metadata.
3. Evaluate each benchmark prompt one by one.
4. Compare output against expected behavior.
5. Mark each case as:
   - Pass
   - Partial
   - Fail
6. Summarize:
   - what improved
   - what regressed
   - what still blocks trust

## Evaluation Rules

### Caller / Usage Lookup
Fail if:
- answer returns insufficient evidence when real source evidence exists
- answer points to tests, strings, fixtures, or unrelated helpers
- latency is far too high for a simple lookup

### File Summary
Fail if:
- answer is too shallow to be useful
- follow-up repeats instead of expanding
- answer loses the loaded-file anchor

### Flow Trace
Fail if:
- answer is just raw lines
- answer misses the main branch behavior
- answer uses tests or internals when source exists

### Repo Overview
Fail if:
- it asks the user to provide files manually
- it gives generic prose with no grounding
- it does not inspect the repo itself

## Output Style

When reporting results:
- be specific
- cite the prompt
- include timing
- call out whether it streamed
- distinguish “better” from “fixed”

## Updating The Benchmark Doc

When updating `docs/BENCHMARKS.md`:
- append or revise carefully
- keep prior baselines visible when useful
- do not overwrite historical benchmark notes without reason
- keep placeholders if the user has not filled in final benchmark results yet

## Avoid

- vague claims like “seems better”
- marking a case as fixed just because routing improved
- grading based only on tests without checking live output
- rewriting benchmark history to look cleaner than reality

## Good Summary Shape

- Repo overview improved and feels more natural.
- Caller lookup still fails despite real source evidence.
- Flow trace is more readable but still too shallow.
- Anchored follow-up remains weak.

## Bad Summary Shape

- “Everything looks better now.”
- “Tests pass so this is fixed.”
- “Probably fine.”
