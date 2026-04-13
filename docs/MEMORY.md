# Memory

`params-cli` uses multiple forms of memory so longer sessions can stay useful without stuffing the entire transcript into every turn.

This is still evolving, but the current model is already layered enough to document.

---

## Current Memory Layers

### 1. Session Compression

Relevant files:
- `src/memory/compression.rs`
- `src/inference/session/investigation.rs`

Purpose:
- keep recent turns usable without sending the full transcript forever
- preserve active goals, decisions, recent repo investigation state, grounded facts, and open questions

Current behavior:
- older history is collapsed into structured context
- the newest few turns stay verbatim
- technical investigation context can be carried forward separately from freeform chat

### 2. Project Index

Relevant files:
- `src/memory/index.rs`
- `src/inference/session/memory.rs`

Purpose:
- maintain lightweight project summaries that can be recalled when useful
- support long-running projects without re-reading everything every turn

Current behavior:
- incremental indexing
- changed/new files get refreshed
- deleted files get removed
- oversized files are skipped

### 3. Durable Facts

Relevant files:
- `src/memory/facts.rs`
- `src/memory/facts/store.rs`
- `src/memory/facts/quality.rs`
- `src/memory/facts/prompting.rs`
- `src/inference/session/memory.rs`

Purpose:
- preserve cross-session facts that are stable and worth keeping

Current behavior:
- facts are project-scoped
- verified facts are preferred over legacy ones
- irrelevant/generic facts can be pruned
- TTL and project caps apply

Implementation note:
- `src/memory/facts.rs` is now a facade; storage, fact-quality filtering, and extraction-prompt construction live in the split `src/memory/facts/` submodules

---

## Investigation Context

Technical turns also maintain a lighter “investigation state” that is not the same as long-term memory.

Tracked examples:
- recent loaded files
- recent directories
- recent searches
- last technical intent
- top anchor for vague follow-ups like `Tell me more`

This helps questions like:
- `What does this file do?`
- `Tell me more`
- `What about that?`

---

## Memory Commands

Use these to inspect what the system currently knows:

- `/memory status`
- `/memory facts`
- `/memory last`
- `/memory recall <query>`
- `/memory prune`

---

## What Should Go Into Memory

Good candidates:
- stable project facts
- accepted user preferences
- validated implementation facts
- durable architectural decisions

Bad candidates:
- speculative explanations
- temporary wrong answers
- raw tool noise
- generic coding advice not specific to the project
- wrapper text from injected context

---

## Current Design Principles

- reliability over volume
- structured context over freeform summary blobs
- grounded facts over fuzzy recollections
- project-scoped storage over global noise

---

## Known Limitations

- memory quality is only as good as the evidence that produced it
- repo-navigation quality still matters upstream because bad technical answers can poison later turns if they are treated as truth
- benchmark coverage for long-session memory behavior still needs to improve

---

## What To Watch During Development

When working on memory, check:

- does compression preserve the current task correctly?
- do vague follow-ups stay anchored to the right file or repo context?
- are verified facts meaningfully better than legacy ones?
- does pruning remove stale/generic junk without deleting useful project facts?

---

## Future Direction

The long-term direction should be:

- stronger fact validation
- better contradiction handling
- better long-session benchmarks
- clearer observability around why a fact was recalled, stored, skipped, or pruned
