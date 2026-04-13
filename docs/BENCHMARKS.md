# Benchmarks

This document defines how `params-cli` should be evaluated for answer quality, grounding, latency, and user-facing behavior.

The goal is not just to measure raw speed. The main question is:

`Does the app investigate repositories like a useful coding agent, and does it answer correctly and quickly enough to be trusted?`

---

## What To Measure

For this project, the important benchmark dimensions are:

- correctness
- grounding quality
- latency
- convergence behavior
- streaming / UX behavior
- regression stability

---

## Benchmark Categories

### 1. Repo Overview

Examples:
- `Can you see my project?`
- `Do you understand this repo?`
- `Inspect this project`

Checks:
- enters the technical repo investigation path
- does not ask the user to manually provide files for an accessible repo
- answer is short and structural
- answer mentions real repo files such as `Cargo.toml`, `README.md`, `src/main.rs`

### 2. File Summary

Examples:
- `/read src/main.rs` then `What does this file do?`
- `Tell me more`

Checks:
- stays anchored to the loaded file
- does not pivot into unrelated files without a clear reason
- no wrapper-text leakage
- answer is user-facing prose, not an internal evidence dump

### 3. Implementation Lookup

Examples:
- `Where is session restore implemented?`
- `What does load_most_recent do?`

Checks:
- finds the right file
- cites real line numbers
- does not collapse branches incorrectly
- does not rely on docs/tests when source is available

### 4. Caller / Usage Lookup

Examples:
- `What calls load_most_recent`
- `What uses SessionStore`

Checks:
- returns real non-definition source call-sites/usages
- excludes string literals, helper labels, tests, fixtures, and internal runtime files
- converges quickly

### 5. Flow Trace

Examples:
- `Explain how session restore works`
- `Trace the logging functionality`

Checks:
- uses real cross-file evidence
- does not leak tests or tool-loop internals unless explicitly asked
- answers as an ordered flow
- returns insufficient evidence instead of guessing when needed

### 6. UX / Runtime Behavior

Checks:
- final answer streams live
- raw tool tags never appear in assistant output
- no separate `Thinking:` transcript row during active generation
- no unnecessary iteration-limit fallback on simple technical prompts

---

## Benchmark Case Template

Use a simple case schema like this:

```md
### Case: caller_lookup_load_most_recent

- Prompt: `What calls load_most_recent`
- Intent: `CallSiteLookup`
- Expected files:
  - `src/inference/session/runtime.rs`
- Forbidden output:
  - tests
  - `auto_inspect`
  - raw tool tags
  - string literal matches
- Latency target:
  - under 20s ideal
  - under 40s acceptable
- Notes:
  - should stop after confirmed caller evidence
```

---

## Recommended Benchmark Set

Start with a compact tracked set of “must-pass” prompts:

1. `Can you see my project?`
2. `/read src/main.rs` then `What does this file do?`
3. `Tell me more`
4. `What calls load_most_recent`
5. `What uses SessionStore`
6. `Explain how session restore works`
7. `Where is eco mode configured?`

---

## Scoring Ideas

You can score each case on a 0-2 or 0-3 scale.

Suggested dimensions:

- Correctness
  - `0` wrong
  - `1` partially right
  - `2` correct

- Grounding
  - `0` vague / ungrounded
  - `1` some real evidence
  - `2` clearly grounded with real file/line references

- Latency
  - `0` unacceptable
  - `1` borderline
  - `2` good

- UX
  - `0` leaks tool syntax / broken streaming / confusing state
  - `1` usable with rough edges
  - `2` smooth

---

## Current Baseline

The first recorded baseline for this document is:

- version: `0.7.5`
- backend: `llama_cpp`
- model: `qwen2.5-3b-instruct-q4_k_m.gguf`
- eco: `off`
- reflection: `off`
- date: `2026-04-12`

This baseline is useful because it captures the project after a large repo-navigation/runtime refactor, but before the behavior feels consistently trustworthy in live use.

Treat it as a recorded behavioral baseline, not a guarantee that the latest worktree still behaves exactly the same. Structural cleanup and modularization can land between benchmark runs without changing the version string.

### High-Level Summary

- Repo overview is fast and grounded, but it feels templated and does not visibly stream.
- Anchored file summary works in the narrow sense, but it is too terse and weak on follow-up expansion.
- Caller and usage lookup are currently the weakest benchmark area.
- Flow trace can return grounded lines, but the answer shape is still too mechanical and did not stream in this run.
- Config-location quality is not reliable enough yet.

---

## Latest Benchmark Run

### Run Metadata

| Field | Value |
|---|---|
| Version | `0.7.5` |
| Backend | `llama_cpp` |
| Model | `qwen2.5-3b-instruct-q4_k_m.gguf` |
| Eco | `off` |
| Reflection | `off` |
| Date | `2026-04-12` |

### Status Matrix

Use this table as the quick-glance summary for the current release.

| Case | Intent | Status | Why |
|---|---|---|---|
| `Can you see my project?` | Repo Overview | Partial | Correct and grounded, but too instant/template-like and not visibly streamed |
| `What does this file do?` | File Summary | Partial | Anchored correctly, but too shallow to be genuinely useful |
| `Tell me more` | File Summary Follow-up | Fail | Repeats the same answer instead of deepening the explanation |
| `What calls load_most_recent` | Caller Lookup | Fail | Timed out into insufficient evidence after `1m33s` |
| `What uses SessionStore` | Usage Lookup | Fail | Still failed and did not stream |
| `Explain how session restore works` | Flow Trace | Partial | Grounded, but mechanical and not really a broader trace |
| `Where is eco mode configured?` | Config Lookup | Fail | Returned irrelevant config evidence |

### Aggregate Scorecard

Scoring uses the 0-2 scale defined above.

| Case | Correctness | Grounding | Latency | UX | Notes |
|---|---:|---:|---:|---:|---|
| `Can you see my project?` | 2 | 2 | 2 | 1 | Correct and grounded, but felt instant/template-like rather than streamed |
| `What does this file do?` | 1 | 2 | 2 | 1 | Anchored to `src/main.rs`, but too shallow |
| `Tell me more` | 0 | 1 | 2 | 0 | Repeated the same answer instead of expanding |
| `What calls load_most_recent` | 0 | 0 | 0 | 0 | Failed after 1m33s |
| `What uses SessionStore` | 0 | 0 | 1 | 0 | Failed after 27.5s |
| `Explain how session restore works` | 1 | 2 | 2 | 1 | Grounded lines, but no streaming and no real cross-file explanation |
| `Where is eco mode configured?` | 0 | 0 | 2 | 1 | Returned irrelevant config evidence |

### Baseline Findings

- Overall reliability is not yet good enough for caller/use-site questions.
- Follow-up quality is weaker than first-answer quality.
- Visible streaming behavior is inconsistent or absent in this run.
- Several good answers feel “hardcoded” because they arrive immediately as rigid evidence-shaped text instead of natural streamed prose.

---

## Repo Overview Results

### Case: `Can you see my project?`

- Prompt: `Can you see my project?`
- Time: near-instant
- Streaming: no visible streaming
- Result:

```text
This repo is a Rust CLI project defined in `Cargo.toml:2` `name = "params-cli"`. Startup runs through `src/main.rs:90` `fn main() -> Result<()> {`. Top-level modules visible so far are cache, commands, config, and debug_log from `src/main.rs:1`. The project description is summarized in `README.md:1` `# params-cli`.
```

- Assessment:
  - correctly identified the repo as a Rust CLI project
  - used real file/line grounding
  - felt immediate/template-like rather than naturally investigated

---

## File Summary Results

### Case: `/read src/main.rs` then `What does this file do?`

- Prompt: `What does this file do?`
- Time: near-instant
- Streaming: no visible streaming
- Result:

```text
`src/main.rs:1` declares modules cache, commands, config, debug_log, error, and events.
```

- Assessment:
  - correctly anchored to `src/main.rs`
  - too shallow for a useful file summary
  - did not mention the CLI struct, subcommands, or `main`

### Case: `Tell me more`

- Prompt: `Tell me more`
- Time: near-instant
- Streaming: no visible streaming
- Result:

```text
`src/main.rs:1` declares modules cache, commands, config, debug_log, error, and events.
```

- Assessment:
  - failed as a follow-up expansion
  - repeated the same answer instead of deepening the file summary

---

## Caller / Usage Results

### Case: `What calls load_most_recent`

- Prompt: `What calls load_most_recent`
- Time: `1m33s`
- Streaming: no
- Result:

```text
I couldn't gather enough source evidence to answer CallSiteLookup within the current investigation budget.
```

- Assessment:
  - benchmark failure
  - latency is unacceptable for a simple caller lookup
  - should have returned the real runtime caller or at least a bounded partial answer

### Case: `What uses SessionStore`

- Prompt: `What uses SessionStore`
- Time: `27.5s`
- Streaming: no
- Result:

```text
I couldn't gather enough source evidence to answer UsageLookup within the current investigation budget.
```

- Assessment:
  - benchmark failure
  - better than the 1m33s caller failure, but still too slow and still incorrect

---

## Flow Trace Results

### Case: `Explain how session restore works`

- Prompt: `Explain how session restore works`
- Time: `19.6s`
- Streaming: no
- Result:

```text
1. `src/session/mod.rs:262` `pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {`
2. `src/session/mod.rs:263` `let Some(summary) = self.list_sessions()?.into_iter().next() else {`
3. `src/session/mod.rs:264` `return Ok(None);`
4. `src/session/mod.rs:265` `};`
5. `src/session/mod.rs:266` `self.load_session_by_id(&summary.id)`
```

- Assessment:
  - grounded on real source lines
  - structurally correct for the function body itself
  - still too mechanical
  - did not feel like a full “trace” across the broader restore flow

---

## Config Lookup Results

### Case: `Where is eco mode configured?`

- Prompt: `Where is eco mode configured?`
- Time: near-instant
- Streaming: no visible streaming
- Result:

```text
`src/config/profile.rs:14` `pub struct ProjectProfile {` is part of the relevant config evidence. `src/config/profile.rs:15` `pub backend: Option<String>,` is part of the relevant config evidence. `src/config/profile.rs:17` `#[serde(default)]` is part of the relevant config evidence.
```

- Assessment:
  - benchmark failure
  - grounded in a real file, but the lines are not the answer to the question
  - indicates weak config-evidence selection

---

## Latency Results

### Observed Latency Snapshot

| Case | Observed Time | Notes |
|---|---:|---|
| `Can you see my project?` | near-instant | suspiciously immediate / no visible streaming |
| `What does this file do?` | near-instant | fast but too shallow |
| `Tell me more` | near-instant | repeated answer |
| `What calls load_most_recent` | `1m33s` | unacceptable |
| `What uses SessionStore` | `27.5s` | too slow for a failed answer |
| `Explain how session restore works` | `19.6s` | acceptable only because it returned a grounded answer |
| `Where is eco mode configured?` | near-instant | wrong but fast |

### Latency Takeaways

- Instant answers currently correlate with rigid/template-like response shapes.
- The slowest cases are the ones that matter most for coding-agent credibility: caller/use-site lookup.
- The current system is not yet balancing convergence and answer quality well.

---

## UX Regression Results

### Streaming / Presentation

Observed in this run:

- several answers appeared immediately with no visible streaming
- some answers felt “hardcoded” or over-rendered rather than naturally generated
- no raw tool tags leaked in the recorded prompts here, which is good

### Follow-Up Coherence Regression

The following exploratory follow-up sequence showed a serious coherence problem:

```text
/read src/main.rs
Is this file defining something? Is it Rust?
What does it do tho
WHat?
```

Observed behavior:

1. `Is this file defining something? Is it Rust?`
   - answered with the same shallow `src/main.rs:1` module-declaration line
2. `What does it do tho`
   - incorrectly answered:

```text
The implementation is in `src/safety.rs` at line 15.
```

3. `What?`
   - then tried to recover by saying there was a misunderstanding

Assessment:
- anchored follow-up handling is still unstable in natural conversation
- the app can jump to an unrelated file even after an explicit `/read src/main.rs`

---

## What This Baseline Says

Version `0.7.5` has:

- a decent starting point for repo overview
- partial grounding for file summaries and local function traces
- poor caller/use-site reliability
- weak natural follow-up behavior
- inconsistent or missing visible streaming on several benchmark prompts

This is a useful baseline, but not yet a strong repo-navigation release.

## Exit Criteria For The Next Baseline

These are the concrete conditions the next recorded version should meet before it can be called an improvement.

### Repo Overview

- `Can you see my project?` should still be grounded in real files
- the answer should feel naturally generated, not pre-rendered
- visible streaming should occur on the final answer
- the answer should describe startup plus main subsystems, not just list one module line

### File Summary

- `What does this file do?` after `/read src/main.rs` should mention:
  - that `src/main.rs` is the Rust entrypoint
  - the CLI shape
  - the `main` function / command routing role
- `Tell me more` should expand the answer rather than repeat it
- anchored follow-ups should not jump to unrelated files

### Caller / Usage Lookup

- `What calls load_most_recent` should return at least one real non-definition source caller
- `What uses SessionStore` should return at least one real non-definition source usage
- neither case should fail with an investigation-budget message when source evidence exists
- target latency should be:
  - under `20s` ideal
  - under `40s` acceptable

### Flow Trace

- `Explain how session restore works` should describe the control flow in plain language
- it should include the branch split correctly
- it should use real source files and avoid tests/internal runtime files unless explicitly requested
- it should stream the final answer

### Config Lookup

- `Where is eco mode configured?` should point to the actual eco-mode config fields or parsing logic
- generic nearby config structs should not count as a passing answer

### UX

- final answers should stream on benchmark prompts
- no raw tool tags should appear
- no investigation-budget failure on simple caller/use-site questions when evidence exists
- no contradictory follow-up behavior like `src/main.rs` suddenly pivoting to `src/safety.rs`

---

## Suggested Release Gate

For the next version, a reasonable minimum release gate would be:

- no `Fail` on repo overview or basic file summary
- caller lookup and usage lookup must return real evidence at least once each
- no major follow-up contradiction in the anchored file-summary path
- visible final-answer streaming present on at least one repo overview case and one deeper technical case

---

## Manual Benchmark Checklist

When you run a manual benchmark pass, record:

- prompt
- backend
- eco on/off
- reflection on/off
- whether the answer streamed
- time to first token
- time to final answer
- answer text
- whether the answer was correct
- whether forbidden content appeared

---

## Future Work

Later, this should grow into a real `evals/` system with:

- tracked benchmark cases
- fixtures
- expected/forbidden patterns
- latency budgets
- report generation

For now, this file should act as the source of truth for what “good” means.
