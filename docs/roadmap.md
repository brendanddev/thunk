# ROADMAP.md

Long-term phase plan for `params-cli`.

---

## Current Status

- Phase 8.4.x: COMPLETE (runtime stabilization achieved)
- Phase 8.5: COMPLETE (context efficiency / read bounding delivered)
- Phase 9: COMPLETE / CLOSED (investigation quality)
- Phase 9.1.1: COMPLETE (bounded second matched-file read)
- Phase 9.1.2: COMPLETE (path-scoped investigation)
- Phase 9.1.3: COMPLETE (import-only weak candidate rejection)
- Phase 9.1.4: COMPLETE (prompt scope as search upper bound)
- Phase 9.2: COMPLETE (structured investigation modes)
- Phase 9.2.1: COMPLETE (trace-validated InvestigationMode enum + config lookup mode)
- Phase 9.2.2: COMPLETE + VALIDATED (initialization lookup)
- Phase 9.2.3: COMPLETE + VALIDATED (CreateLookup)
- Phase 9.2.4: COMPLETE + VALIDATED (RegisterLookup)
- Phase 9.2.5: COMPLETE + VALIDATED (LoadLookup)
- Phase 9.2.6: COMPLETE + VALIDATED (SaveLookup)
- Phase 10: PAUSED (bounded Multi-Turn UX checkpoint after 10.1.1)
- Phase 10.0: EFFECTIVELY COMPLETE (Basic Anchors; 10.0.3 deferred)
- Phase 10.0.1: COMPLETE + VALIDATED (Last Read File Anchor)
- Phase 10.0.2: COMPLETE + VALIDATED (Last Search Anchor)
- Phase 10.0.3: DEFERRED (Single Search Candidate File Anchor; only if justified by real need)
- Phase 10.1: PAUSED (Investigation Continuity checkpoint)
- Phase 10.1.1: COMPLETE + VALIDATED (Explicit Same-Scope Investigation Continuity)
- Phase 11: ACTIVE (Tool Expansion)
- Phase 11.0: COMPLETE + VALIDATED (Git read-only)
- Phase 11.0.1: COMPLETE + VALIDATED (Git Status read-only)
- Phase 11.0.2: COMPLETE + VALIDATED (Git Diff read-only)
- Phase 11.0.3: COMPLETE + VALIDATED (Git Log read-only)
- Phase 11.1: COMPLETE + VALIDATED (Runtime Tool Surface Policy)
- Phase 11.1.0: COMPLETE + VALIDATED (RetrievalFirst / GitReadOnly per-turn surface enforcement)
- Phase 11.1.1: COMPLETE + VALIDATED (Retrieval Weakness Guardrails)
- Phase 11.1.2: COMPLETE + VALIDATED (Empty Search Termination)
- Runtime refactor checkpoint: FUNCTIONALLY COMPLETE (code extraction complete + validated; test decomposition functionally complete; only intentional coupled/cross-cutting `engine.rs` tests remain)

---

## Recently Completed Phases

### Phase 8.3 — Search → Read Chaining (COMPLETE)

Status:

* COMPLETE
* Architecture Review COMPLETE
* Validation COMPLETE

Delivered:

* enforced search → read behavior before grounded synthesis
* established the first runtime grounding invariant after search

### Phase 8.4 — Evidence Awareness (COMPLETE)

Status:

* COMPLETE
* Architecture Review COMPLETE
* Validation COMPLETE

Delivered:

* identifier-triggered investigation enforcement
* runtime distinction between enough evidence vs not enough evidence
* stronger runtime-owned investigation control

Note:

* additional coder-model stabilization continued in Phase 8.4.x and is now complete

---

## Phase 8.4.x — Runtime Stabilization (COMPLETE)

Purpose:

* enforce correct behavior under stronger (coder) model
* remove remaining model-trust assumptions

Scope:

* answer admission gate (no answer before evidence-ready)
* single synthesis per investigation turn
* read-only vs mutation tool policy
* search results require at least one read
* bounded correction loops
* deterministic terminal outcomes

Reason:

* coder model exposed enforcement gaps not caught by instruct model

**Closure Criteria**:

All runtime-enforced investigation invariants are implemented and validated:

- no answer before evidence-ready
- search → read enforcement
- read-only vs mutation separation
- bounded correction loops
- deterministic terminal outcomes

**Remaining issues were reclassified as**:

- Phase 8.5: context efficiency / latency
- Phase 9: investigation quality

No further work should be added to Phase 8.4.x.

---

## Phase 8.5 — Context Efficiency / Read Bounding (COMPLETE)

**Purpose**:
- optimize prompt size and latency after Phase 8 correctness work

**Scope**:
- prevent duplicate reads within a single turn
- introduce per-turn read bounding (after validation)
- reduce context growth per turn
- reduce unnecessary large prompt injection

**Non-Goals**:
- no investigation quality improvements
- no ranking or selection heuristics
- no architecture redesign
- no model-specific behavior

**Delivered**:
- duplicate read_file dedup: second read of the same path within one turn is blocked with a
  tool error; path is normalized before tracking; state is per-turn and resets each call to
  run_turns; `reads_this_turn: HashSet<String>` follows the SearchBudget pattern in engine.rs
- per-turn read bounding: `MAX_READS_PER_TURN = 3`; the 4th+ unique read in a turn is blocked
  with a tool error
- search result output bounding: `search_code` now separates walk cap from output cap;
  `MAX_RESULTS_SHOWN = 15`, `total_matches` is preserved, and truncation is surfaced clearly
  in the rendered tool output
- reduced context growth from rejected intermediate output: premature synthesis before
  `READ_BEFORE_ANSWERING` is discarded from context before the correction is injected
- investigation-trigger coverage expanded for explicit search / occurrence phrasing so valid
  repo-search prompts do not fall through to stale Direct answers

**Notes**:
- does not change investigation correctness
- does not introduce Phase 9 quality improvements
- focuses only on reducing prompt size and latency

### Closure Criteria:

**Phase 8.5 is complete when**:
- duplicate reads are bounded within a turn
- per-turn reads are capped
- large search result injection is bounded
- unnecessary rejected assistant output is not retained in context
- manual traces confirm a cleaner search → read → answer flow with materially reduced prefill growth

**Result**:
- the wasted pre-read synthesis round is eliminated
- search/result/read flow is materially faster and more stable
- remaining issues now fall into investigation quality or later backend/system work

No further Phase 8.5 work should be added unless new evidence shows unresolved context-efficiency
problems within the current architecture.

---

## Deferred / Reclassified Work

### From Phase 8.x

* File selection quality
  → Phase 9 (completed structural investigation quality)

* Definition vs usage ranking
  → Phase 9 (resolved structurally without ranking)

* Advanced multi-read optimization beyond the bounded second matched-file read
  → deferred; reopen only if new evidence justifies it

* TUI correction visibility
  → Phase 14

* Runtime observability cleanup
  → Phase 13

---

## Phase 9 — Investigation Quality (COMPLETE / CLOSED)

### Phase 9.0 — Single-step Investigation

* simple definition / location queries
* current search results prioritize source files, then config/data files, then docs/text files
* rendered search output is grouped by file with match counts and up to 3 representative lines
* definition-like match content can add a conservative first-read hint when exactly one source file contains a definition pattern
* runtime blocks `list_dir` before `search_code` on investigation-required turns
* Phase 9.0 preserved the single-step search -> read -> answer flow
* Phase 9.0 did not introduce multi-read investigation, a broad ranking system, or query-intent classification

#### Phase 9.0.x — Single-step Investigation Stabilization (COMPLETE)

Delivered:

* definition lookup correctness:
  * `Where is TaskStatus defined in sandbox/` searches, reads the definition file, and accepts that read as sufficient evidence
* usage lookup correctness:
  * `Where is TaskStatus used in sandbox/` searches before reading
  * definition-only reads do not satisfy usage evidence when usage candidates exist
  * runtime injects a bounded correction naming a concrete matched usage file
* natural-language lookup coverage:
  * prompts such as `Where are completed tasks filtered in sandbox/` trigger investigation
  * `list_dir` is blocked before search for these investigation-required turns
* location lookup stability:
  * `Find where logging is initialized in sandbox/` selects the grounded implementation file

Important boundaries:

* this completes only the Phase 9.0.x stabilization slice
* Phase 9 is now closed; reopen only for regressions in validated investigation behavior
* Phase 9.1 completed separately; do not relabel Phase 9.0.x work as Phase 9.1
* no ranking system, query-intent classifier, or multi-read planner was introduced by Phase 9.0.x

Status:

* Phase 9.0.x is closed
* reopen only for regressions in the validated single-step behavior

### Phase 9.1 — Multi-step Investigation (COMPLETE)

Purpose:

* support bounded investigations where one read is not enough to answer confidently
* preserve current runtime-owned evidence discipline
* build on existing search -> read -> answer behavior without introducing a planner architecture

Current scope:

* search -> read -> optional bounded second candidate read when the first matched read is structurally insufficient
* path-scoped investigation for clear relative path scopes in investigation-required prompts
* import-only weak candidate rejection for structurally insufficient candidate reads
* keep follow-up reads bounded and drawn from current search candidates unless a later slice proves another search is required
* preserve duplicate-read blocking and per-turn read caps
* maintain one grounded final answer after evidence is ready

Non-goals:

* no broad ranking system
* no query-intent classifier
* no semantic search or embeddings
* no LSP/reference tooling
* no hidden global investigation state

#### Phase 9.1.1 — Bounded Second Matched-file Read (COMPLETE)

Delivered:

* allows one additional distinct read from current search candidates when the first matched read is structurally insufficient
* keeps candidate reads bounded; after two candidate reads, evidence still not ready terminates cleanly with insufficient evidence
* preserves duplicate-read blocking and the existing per-turn read cap
* preserves runtime-owned evidence discipline; tools still return typed outputs only
* does not introduce tool changes, a planner, a ranking system, or a query-intent classifier

Narrow intended scope:

* solves structurally insufficient first reads
* canonical case: a usage lookup first reads a definition-only file while usage candidates also exist
* definition lookups still accept definition-file reads
* search -> read -> answer remains the overall structure, with one optional bounded second candidate read in this slice

Out of scope for Phase 9.1.1:

* broad semantic sufficiency judgment
* query-intent classification
* scoped/path-qualified query satisfaction such as `in sandbox/cli` was not part of this slice; it was addressed later by Phase 9.1.2 for clear prompt-derived path scopes
* richer semantic qualifier matching such as `used to format report output`
* semantic search, embeddings, LSP/reference tooling, or a multi-read planner

#### Slice 9.1.2 — Path-scoped Investigation (COMPLETE)

Delivered:

* extracts a clear relative path scope from investigation-required prompts using conservative `in <path>` / `within <path>` patterns
* injects that scope into `SearchCode` dispatch when the model did not already provide a path
* narrows the candidate set by construction at search dispatch time
* prevents results outside the scoped path from becoming current search candidates in that dispatch path
* reuses existing path normalization
* preserves unscoped investigation behavior
* preserves Phase 9.0.x and Phase 9.1.1 invariants

Implementation boundaries:

* localized to `engine.rs`
* no tool changes
* no `tool_codec` changes
* no evidence-gating redesign
* no changes to `record_read_result`
* no changes to `evidence_ready`
* no changes to candidate read counting
* no changes to correction or termination logic
* model-specified search paths are respected and not overridden by this slice
* does not introduce a planner, ranking system, semantic search, embeddings, LSP/reference tooling, or broad query-intent classifier

Deferred note:

* prompts like `Find where logging is initialized in sandbox/services/` are not failures of path scoping when search dispatch is narrowed; they exposed a separate issue later addressed narrowly by Phase 9.2.2 for initialization lookup. This remains out of scope for Phase 9.1.2.

Broader semantic qualifier evidence gating beyond completed structured lookup modes remains deferred and is not started.

#### Slice 9.1.3 — Candidate Selection Quality (import-only weak candidate rejection) (COMPLETE)

Delivered:

* classifies a search candidate as import-only when every matched line in that file is an import declaration
* treats an import-only candidate read as structurally insufficient when at least one non-import candidate exists
* injects one bounded recovery pointing to a non-import candidate
* accepts the read as useful evidence when all candidates are import-only
* preserves runtime-owned evidence discipline
* preserves Phase 9.0.x, Phase 9.1.1, and Phase 9.1.2 invariants

Implementation boundaries:

* localized to `engine.rs`
* no tool changes
* no `tool_codec` changes
* no ranking or scoring system
* no planner behavior
* no semantic qualifier handling
* no broad query-intent classifier
* candidate read counting remains unchanged
* correction paths remain bounded and non-looping
* termination behavior remains unchanged

Validation:

* unit tests cover `looks_like_import`
* integration tests cover import-only rejection and fallback acceptance
* manual testing preserved `Where is TaskStatus used?` search -> definition read -> usage read -> grounded answer behavior
* manual testing preserved `write_file` and `edit_file` approval flows
* implementation summary reported 249 passing tests after this slice

Deferred note:

* import-only weak candidate rejection is structural. It did not solve `logging is initialized` style prompts in this slice; initialization lookup was later addressed by Phase 9.2.2.
* non-import weak candidates can still exist and remain an accepted limitation.

#### Slice 9.1.4 — Prompt Scope as Search Upper Bound (COMPLETE)

Purpose:

* closes the remaining structural gap in path-scoped investigation: when a prompt has an explicit path scope and the model supplies its own search path, the prompt scope remains the upper bound for candidate construction

Why this slice:

* it remains a structural follow-up to Phase 9.1.2 path-scoped investigation
* it keeps investigation quality work in runtime dispatch/enforcement
* it improves path-scope determinism without introducing semantic qualifier handling

Scope:

* Phase 9.1.2 behavior is preserved when the model omits a search path
* model-specified search paths that are equal to or beneath the prompt-derived scope are preserved
* broader or unrelated model-specified search paths are clamped to the prompt-derived scope
* all scope checks remain based on normalized relative paths
* candidate sets remain narrowed by construction before evidence gating
* implemented in runtime (`engine.rs`)
* validated via tests

Implementation note:

* current codec behavior emits `path: None` for `search_code`
* clamping remains a defensive runtime invariant until codec support for explicit search paths exists

Non-goals:

* no semantic qualifier evidence gating
* no action-specific lookup satisfaction
* no planner
* no ranking or scoring system
* no broad query-intent classifier
* no semantic search, embeddings, or LSP/reference tooling
* no hidden global investigation state

Invariants to preserve:

* runtime owns orchestration and policy
* tools return typed outputs only
* tool parsing and model-facing rendering remain in `tool_codec`
* unscoped prompts preserve current behavior
* explicit model paths without a prompt-derived scope preserve current behavior
* duplicate-read blocking, per-turn read caps, candidate read counting, and insufficient-evidence termination remain unchanged

Validation prompts:

* `Where is TaskStatus used in sandbox/cli/`
* `Where is TaskStatus defined within sandbox/models/`
* `Find where completed tasks are filtered in sandbox/services/`
* `Where is run_turns used in src/runtime/`
* `Find TaskStatus within sandbox/cli/commands/`
* `Find where logging is initialized in sandbox/services/` — validate path-scope enforcement only; initialization lookup is handled separately by Phase 9.2.2

### Phase 9.2 — Structured Investigation Modes (COMPLETE)

* definition lookup
* usage lookup
* config lookup
* initialization lookup
* create lookup
* register lookup
* load lookup
* save lookup

Closure note:

* Phase 9.2 includes `ConfigLookup`, `InitializationLookup`, `CreateLookup`, `RegisterLookup`, `LoadLookup`, and `SaveLookup`
* no further `InvestigationMode` additions are planned without new observed failures

#### Slice 9.2.1 — InvestigationMode + Config Lookup Mode (COMPLETE)

Delivered:

* `InvestigationMode` enum with `General`, `UsageLookup`, `DefinitionLookup`, and `ConfigLookup`
* mode computed once per turn and enforced in runtime evidence handling
* `ConfigLookup` classifies config candidates by extension only
* non-config candidate reads are structurally insufficient when config candidates exist
* one bounded recovery can direct the model to a matched config file
* if no config candidates exist, config lookup falls back to existing candidate-read behavior

Validation notes:

* manual validation found a scoped config lookup bug for `Find where database is configured in the sandbox/ folder`
* root cause: `extract_investigation_path_scope` only checked the token immediately after `in`, so it saw `the` instead of `sandbox/`
* fix: scope extraction now supports optional `the` in `in the <path>` and `within the <path>`
* validation after the fix confirmed that search stayed scoped to `sandbox/` and out-of-scope files were excluded
* runtime tracing is gated by `PARAMS_TRACE_RUNTIME=1`, observational only, and routed through the application session log
* traced `UsageLookup` validation: `Where is TaskStatus used in sandbox/` detected `UsageLookup`, extracted `sandbox/`, rejected definition-only `sandbox/models/enums.py`, issued `DefinitionOnly` recovery to `sandbox/cli/commands.py`, and accepted `sandbox/cli/commands.py`
* traced `ConfigLookup` validation: `Find where database is configured in the sandbox/ folder` detected `ConfigLookup`, extracted `sandbox/`, rejected non-config `sandbox/database.py`, issued `ConfigFile` recovery to `sandbox/database.yaml`, and accepted `sandbox/database.yaml`
* trace points cover mode, scope, search clamp, candidate classification, read acceptance/rejection, recovery, and terminal insufficient-evidence

#### Slice 9.2.2 — Initialization Lookup (COMPLETE + VALIDATED)

Delivered:

* detects initialization lookup using exact substrings only:
  * `initialize`
  * `initialized`
  * `initialization`
* classifies search candidates as initialization or non-initialization based on matched lines
* non-initialization candidate reads are structurally insufficient when initialization candidates exist
* one bounded recovery can direct the model to a matched initialization candidate
* if no initialization candidates exist, initialization lookup falls back to existing candidate-read behavior

Validation:

* forced scenario validated wrong-file-first behavior:
  * model read a usage file first
  * runtime rejected the non-initialization read
  * runtime issued bounded recovery to the initialization candidate
  * runtime accepted the initialization file as evidence
  * final answer was grounded

Boundaries:

* no synonyms
* no fuzzy matching
* no semantic interpretation
* no planner, ranking, or scoring system
* no tool or `tool_codec` changes

#### Slice 9.2.3 — CreateLookup (COMPLETE)

Purpose:

* structured investigation mode for creation-style queries
* ensures create candidates are preferred over weaker mentions

Scope:

* exact substring trigger detection:
  * `create`
  * `created`
  * `creation`
* structural classification from matched lines
* rejection of non-create reads when create candidates exist
* single bounded recovery
* fallback when no create candidates exist
* no changes to tools or `tool_codec`
* implemented in runtime (`engine.rs`)
* validated via tests

Implementation note:

* create-term matching is structural and may be noisy
* fallback may admit weaker but grounded answers when no create candidates exist

#### Slice 9.2.4 — RegisterLookup (COMPLETE)

Purpose:

* structured investigation mode for registration-style queries
* ensures register candidates are preferred over weaker mentions

Scope:

* exact substring trigger detection:
  * `register`
  * `registered`
  * `registration`
* structural classification from matched lines
* rejection of non-register reads when register candidates exist
* single bounded recovery
* fallback when no register candidates exist
* no changes to tools or `tool_codec`
* implemented in runtime (`engine.rs`)
* validated via tests and manual checks

Implementation note:

* register-term matching is structural and may be noisy
* fallback may admit weaker but grounded answers when no register candidates exist

#### Slice 9.2.5 — LoadLookup (COMPLETE)

Purpose:

* structured investigation mode for load-style queries
* ensures load candidates are preferred over weaker mentions

Scope:

* exact substring trigger detection:
  * `load`
  * `loaded`
  * `loading`
* structural classification from matched lines
* rejection of non-load reads when load candidates exist
* single bounded recovery
* fallback when no load candidates exist
* no changes to tools or `tool_codec`
* implemented in runtime (`engine.rs`)
* validated via tests and manual checks

Implementation note:

* load-term matching is structural and may be noisy
* fallback may admit weaker but grounded answers when no load candidates exist

#### Slice 9.2.6 — SaveLookup (COMPLETE)

Delivered:

* `InvestigationMode::SaveLookup`
* exact substring trigger detection:
  * `save`
  * `saved`
  * `saving`
* matched-line structural save candidate classification
* wrong-first-read rejection only when save candidates exist
* one bounded recovery to a matched save candidate
* fallback to normal candidate-read behavior when no save candidates exist
* preservation of max 2 candidate reads
* preservation of runtime architecture invariants

Implementation note:

* save-term matching is structural and may be noisy
* fallback may admit weaker but grounded answers when no save candidates exist

### Phase 9 Closure Note

* final closure fix enforces the max-2 candidate-read invariant pre-dispatch
* a third distinct search-candidate `read_file` after two insufficient candidate reads is blocked before dispatch
* synthesis-time insufficient-evidence termination remains as a backstop
* no tool or `tool_codec` changes were required
* accepted limitations remain: no qualifier-bearing evidence gating, structural/noisy action-term matching, fallback to weaker grounded answers when no stronger mode-specific candidates exist, and no semantic interpretation, ranking, planner behavior, fuzzy matching, embeddings, LSP/reference tooling, or broad query-intent classifier

---

## Phase 10 — Multi-Turn UX (PAUSED)

Phase 10 is paused after Phase 10.1.1 as a sufficient bounded continuity checkpoint. Further Phase 10 work requires real usage evidence, not parity or symmetry.

### Phase 10.0 — Basic Anchors (EFFECTIVELY COMPLETE)

Phase 10.0 is intentionally split into narrow numbered slices to preserve bounded structural behavior and avoid scope creep.

#### Slice 10.0.1 — Last Read File Anchor (COMPLETE)

Delivered:

* runtime-owned `last_read_file` anchor
* updated only from successful typed `read_file` output
* explicit structural follow-up phrases only:
  * `read that file`
  * `read that file again`
  * `read the last file`
  * `open that file`
  * `open that file again`
  * `open the last file`
* no semantic resolution
* no pronoun resolution
* no ordinal result references
* no transcript parsing
* no tool changes
* no `tool_codec` ownership drift
* reset clears the anchor
* anchored read goes through the normal typed read path
* anchored read remains bounded and counts correctly against same-turn read behavior

Validation:

* explicit anchored reread validated
* no-anchor deterministic failure validated
* anchor overwrite behavior validated
* unsupported phrases do not resolve as anchors

Implementation caveat:

* no-anchor terminal currently uses the existing `ReadFileFailed` runtime terminal reason; acceptable for now and does not imply semantic anchor resolution or transcript recovery

#### Slice 10.0.2 — Last Search Anchor (COMPLETE)

Delivered:

* runtime-owned search anchor:
  * `last_search_query`
  * `last_search_scope`
* updated only from successful typed `search_code` execution
* stores effective runtime-dispatched values after query simplification and scope clamping
* explicit structural replay prompts only, such as:
  * `search that again`
  * `repeat the last search`
* replay dispatches exactly one typed `search_code`
* no auto-read of candidates
* no result selection or ranking
* no semantic query rewriting
* no pronoun handling
* no transcript parsing
* reset clears the search anchor
* no tool changes
* no `tool_codec` involvement

Validation:

* replay behavior validated for exact query and scope
* no-anchor deterministic failure validated
* anchor overwrite behavior confirmed
* unsupported phrases do not resolve as anchors
* Phase 10.0.1 last-read anchor behavior remains preserved

Implementation caveat:

* replay currently emits a simple runtime-owned acknowledgement: `Repeated the last search`; acceptable and not required for runtime correctness

#### Slice 10.0.3 — Single Search Candidate File Anchor (DEFERRED)

* only if later justified by real need

Phase 10.0 status:

* effectively complete after Phase 10.0.1 and Phase 10.0.2
* Phase 10.0.3 remains deferred, not skipped, and requires real usage justification before implementation

### Runtime Refactor Checkpoint (COMPLETE)

Completed after Phase 10.0.2:

* `anchors.rs` extraction is complete
* `investigation.rs` extraction is complete
* recent runtime refactor audit identified one real coupling issue: `investigation.rs` depended on `tool_codec::looks_like_definition`
* coupling issue resolved: `looks_like_definition` is now investigation-owned
* `detect_investigation_mode` moved into `investigation.rs`
* `engine.rs` imports `detect_investigation_mode` from `investigation.rs`
* behavior remained unchanged
* validation passed:
  * `cargo check`
  * `cargo test` (`359 passed`)
  * `cargo fmt --check`
  * `git diff --check`

### Phase 10.1 — Investigation Continuity (PAUSED)

#### Slice 10.1.1 — Explicit Same-Scope Investigation Continuity (COMPLETE)

Delivered:

* exact same-scope phrase matching:
  * `in the same folder`
  * `within the same folder`
  * `in the same directory`
  * `within the same directory`
  * `in the same scope`
  * `within the same scope`
* reuses the last successful scoped search's effective path scope
* integrates with existing Phase 9 path-scope enforcement
* explicit concrete path in the current prompt takes precedence
* deterministic failure when no prior scoped search exists
* deterministic failure after an unscoped previous search
* no transcript parsing
* no semantic interpretation
* no candidate selection or ranking
* no tool or `tool_codec` changes
* implemented in `anchors.rs` and `engine.rs`; `investigation.rs` unchanged for this slice
* hardening rejects parent-path widening through `..`
* blank or whitespace-only stored scopes are treated as unscoped

Validation:

* same-scope reuse validated
* no-anchor and unscoped-anchor deterministic failures validated
* explicit concrete path precedence validated
* unsupported phrases do not resolve
* broader model-supplied paths clamp to the reused same-scope path
* validation passed: `cargo check`, `cargo test` (`368 passed`), `cargo fmt --check`, `git diff --check`

---

## Phase 11 — Tool Expansion + Runtime Integration (ACTIVE)

### Phase 11.0 — Git (read-only) (COMPLETE)

Delivered:

* typed read-only Git tools: `git_status`, `git_diff`, `git_log`
* bounded process capture and rendered/structured output
* no shell passthrough
* no user-supplied args
* no anchor updates
* no Phase 9 investigation evidence satisfaction
* retrieval-first investigation behavior remains intact

#### Slice 11.0.1 — Git Status (read-only) (COMPLETE)

Delivered:

* typed `git_status` read-only tool
* runs fixed `git status --short --branch`
* no shell passthrough
* no user-supplied args
* no mutation-capable options
* bounded process capture and structured output
* no anchor updates
* no Phase 9 investigation evidence satisfaction
* no `investigation.rs` changes

Validation:

* validation passed: `cargo check`, `cargo test` (`381 passed`), `cargo fmt --check`, `git diff --check`

#### Slice 11.0.2 — Git Diff (read-only) (COMPLETE)

Delivered:

* typed `git_diff` read-only tool
* runs fixed `git diff --no-ext-diff --no-textconv --no-color --`
* no shell passthrough
* no user-supplied args
* no path or revision options
* external diff helpers disabled
* textconv filters disabled
* bounded process capture and rendered output
* no anchor updates
* no Phase 9 investigation evidence satisfaction
* no `investigation.rs` or `anchors.rs` changes

Validation:

* implemented, manually tested, audited, hardened, and validated
* validation passed: `cargo check`, `cargo test` (`394 passed`), `cargo fmt --check`, `git diff --check`

#### Slice 11.0.3 — Git Log (read-only) (COMPLETE)

Delivered:

* typed `git_log` read-only tool
* fixed recent-history inspection only
* bounded process capture and structured output
* no shell passthrough
* no user-supplied args
* no path, revision, range, or patch options
* no anchor updates
* no Phase 9 investigation evidence satisfaction
* no `investigation.rs` or `anchors.rs` changes

Validation:

* implemented, validated, and audited
* validation passed: `cargo check`, `cargo test` (`407 passed`), `cargo fmt --check`, `git diff --check`

### Phase 11.1 — Runtime Tool Surface Policy (COMPLETE)

#### Slice 11.1.0 — Retrieval-First Default (COMPLETE)

Delivered:

* runtime-owned per-turn tool surface selection
* supported surfaces:
  * `RetrievalFirst`
  * `GitReadOnly`
* `RetrievalFirst` allows `search_code`, `read_file`, `list_dir`
* `GitReadOnly` allows `git_status`, `git_diff`, `git_log`
* disallowed tools are rejected pre-dispatch
* first disallowed tool attempt emits a surface-specific runtime correction
* second disallowed tool attempt terminates deterministically with `RuntimeTerminalReason::RepeatedDisallowedTool`
* policy violations remain separate from investigation/evidence failures, tool execution failures, and malformed-tool corrections
* runtime owns policy and enforcement
* tools remain dumb executors
* `tool_codec` remains parse/render only and policy-free
* anchors are not updated by rejected or Git tool attempts

Validation:

* wrong-surface first strike produces surface-specific correction
* wrong-surface second strike terminates with `RepeatedDisallowedTool`
* GitReadOnly happy path works
* allowed tool failures retain their own failure reasons
* anchor behavior remains preserved
* validation passed: `cargo check`, `cargo test` (`417 passed`), `cargo fmt --check`, `git diff --check`

#### Slice 11.1.1 — Retrieval Weakness Guardrails (COMPLETE)

Delivered:

* runtime-owned weak-query guard for `RetrievalFirst` investigation-required `search_code`
* rejected weak queries:
  * empty / whitespace-only
  * very short queries
  * exact token `git`
* first weak-query attempt emits a runtime correction
* second weak-query attempt terminates deterministically with `RuntimeTerminalReason::RepeatedWeakSearchQuery`
* lockfile evidence guard rejects lockfile reads only when a matched source candidate exists
* lockfile recovery points to a matched source candidate
* lockfiles remain acceptable grounded evidence when they are the only matched candidate
* structural investigation trigger list includes `rendered`, so prompts such as `where is git status rendered` enter the investigation-required flow
* no new `InvestigationMode`
* no tool changes
* no `tool_codec` changes
* no ranking, semantic query rewriting, planner, embeddings, LSP, or semantic filtering

Validation:

* original `git` -> `Cargo.lock` -> bad-answer failure fixed
* weak-query termination works
* repeated disallowed-tool enforcement remains preserved
* normal retrieval remains preserved
* lockfile fallback works when it is the only candidate
* validation passed: `cargo check`, `cargo test` (`425 passed`), `cargo fmt --check`, `git diff --check`

#### Slice 11.1.2 — Empty Search Termination (COMPLETE)

Delivered:

* runtime-owned terminal path for empty-search exhaustion
* applies when search attempts produced no matches, no file was read, and the model repeats `search_code` after the allowed empty retry path
* duplicate empty-search retries terminate deterministically with `RuntimeTerminalReason::InsufficientEvidence`
* terminal answer uses `insufficient_evidence_final_answer()`
* no weak-query guard changes
* no lockfile guard changes
* no tool-surface policy changes
* no investigation mode changes
* no tool, `tool_codec`, or anchor changes

Validation:

* empty-search duplicate retry now terminates cleanly instead of looping through repeated tool errors
* Phase 11.1.1 `where is git status rendered` behavior still terminates with `RepeatedWeakSearchQuery`
* existing search-budget, weak-query, lockfile, and disallowed-tool tests passed
* validation passed: `cargo check`, `cargo test` (`426 passed`), `cargo fmt --check`, `git diff --check`

### Phase 11.2 — Runtime Turn Finalization (COMPLETE + VALIDATED)

**Delivered:**
- unified `answer_phase` finalization gate
- mutation bypasses generation entirely
- investigation evidence-ready integrated into `answer_phase`
- duplicate post-evidence enforcement removed
- runtime-owned correction + terminal paths preserved

**Validation:**
- full test suite passes
- manual validation confirmed:
  - direct read flow
  - investigation (definition + initialization)
  - recovery paths
  - post-evidence tool blocking behavior

**Result:**
- turn finalization now follows a single unified lifecycle
- no behavior regressions introduced

#### Slice 11.2.1 — Answer-Phase Introduction (COMPLETE + VALIDATED)

Delivered:

- introduced runtime-owned `answer_phase`
- after `answer_phase == true`, tool calls are rejected pre-dispatch
- first post-answer-phase tool attempt emits a correction
- repeated attempt terminates with `RuntimeTerminalReason::RepeatedToolAfterAnswerPhase`

Validation:

- automated tests passed: `cargo test` (`442 passed`)
- manual direct-read validation passed

#### Slice 11.2.2 — Non-Investigation Finalization (COMPLETE + VALIDATED)

Delivered:

- general retrieval:
  - first successful read enters `answer_phase`
  - prevents search → read → search loops

- direct read:
  - successful `read_file` enters `answer_phase`
  - prevents post-read tool drift

- detection uses:
  - `reads_this_turn`
  - not `InvestigationState`

Validation:

- `Read sandbox/main.py` performed one read and answered
- `Find where logging is initialized in sandbox/` preserved investigation behavior

#### Slice 11.2.3 — Mutation Finalization (COMPLETE + VALIDATED)

Delivered:

- successful approved mutation now calls `finish_with_runtime_answer`
- no generation re-entry after approval
- no post-mutation tool drift
- runtime-owned final output such as:
  - `write_file result: ...`
  - `edit_file result: ...`

Validation:

- manual create/edit validation passed
- automated tests passed

#### Slice 11.2.4 — Unified Finalization Gate (COMPLETE + VALIDATED)

**Delivered:**
- `answer_phase` gate exists for non-investigation finalization
- mutation bypasses generation entirely
- investigation evidence-ready handling now enters the unified `answer_phase` model
- the old separate post-evidence enforcement branch was removed
- investigation-specific correction text and terminal reasons are preserved

**Validation:**
- automated tests passed
- manual validation passed for:
  - direct read
  - definition lookup
  - initialization lookup with recovery

**Result:**
- turn finalization now uses one unified answer-phase enforcement path
- investigation semantics remain unchanged

#### Slice 11.2.5 — Investigation Integration: COMPLETE + VALIDATED

**Goal:**
- fold investigation lifecycle into unified AnswerPhase model

**Current state:**
- investigation still uses a separate enforcement gate

**Target:**
- `investigation.evidence_ready()` becomes an `answer_phase` trigger
- existing investigation gate is removed

**Constraints:**
- investigation semantics MUST remain unchanged
- only enforcement point is unified

**Notes:**
- Behavior unchanged
- investigation now uses the unified answer-phase enforcement path while preserving investigation-specific correction and terminal semantics.

#### Runtime Refactor Checkpoint — Code Extraction: COMPLETE + VALIDATED

**Purpose:**
- reduce `engine.rs` size and responsibility load
- reduce token cost for future model-assisted work
- split behavior-preserving runtime helpers into focused modules
- split large runtime tests by behavior area

**Scope:**
- correction/final-answer messages
- search budget logic
- runtime tracing helpers
- path/scope helpers
- tool surface policy
- query simplification / weak-query detection
- test decomposition

**Non-goals:**
- no behavior changes
- no lifecycle changes
- no tool changes
- no `tool_codec` changes
- no Phase 11.2.5 investigation integration during this checkpoint

#### Runtime Refactor Checkpoint — Test Decomposition (FUNCTIONALLY COMPLETE)

**Purpose:**
- split the 8k+ line engine.rs test suite by behavior area
- preserve all coverage
- reduce engine.rs size and future token cost

**Current:**
- completed behavior test modules: `tool_round`, `approval`, `read_bounds`, `search_budget`, `finalization`, `tool_surface`, `search_guardrails`, `anchors`, `git_acquisition`, `path_scope`, `investigation`, `investigation_modes`, `integration_misc`
- completed inline unit-test moves: `search_query.rs`, `paths.rs`, `prompt_analysis.rs`, `investigation.rs`
- `engine.rs` is reduced to ~2163 lines
- remaining `engine.rs` tests are intentional coupled terminal / cross-cutting system coverage
- no further decomposition is required unless future maintenance pain justifies it

**Non-goals:**
- no assertion weakening
- no behavior changes
- no runtime logic changes

### Phase 11.3 — Runtime Performance Observability

**Purpose:**
- measure runtime/model cost before starting memory or performance architecture changes
- establish reliable baselines for slow prompts
- identify whether latency is dominated by model rounds, prompt size, prefill, generation, cold start, or tool execution

**Scope:**
- per-turn model round count
- prefill/generation duration summary
- prompt/token size measurement if available
- correction / terminal reason summary
- tool-time vs model-time summary

**Non-goals:**
- no prompt cache yet
- no history compression yet
- no bounded synthesis redesign yet
- no tool changes
- no runtime behavior changes

#### Slice 11.3.0 — Model Round Instrumentation (COMPLETE + VALIDATED)
- count model generations per user turn
- label each round (initial, post-tool, post-evidence retry, etc.)

#### Slice 11.3.1 — Timing Breakdown (COMPLETE + VALIDATED)
- aggregate per-turn:
  - prefill total
  - generation total
  - ctx_create/tokenize time

#### Slice 11.3.2 — Prompt / Context Size Tracking (COMPLETE + VALIDATED)
- estimate prompt size per generation (tokens or string length)
- track growth across rounds

#### Slice 11.3.3 — Correction / Terminal Cause Tracking (COMPLETE + VALIDATED)
- record which runtime path caused additional rounds:
  - post_evidence_tool_call_rejected
  - recovery
  - search retry
- correlate with round count

#### Slice 11.3.4 — Cold Start Visibility
- log model load duration
- distinguish cold vs warm turns

#### Slice 11.3.5 — Tool vs Model Cost Split
- aggregate tool time vs model time per turn

---

## Phase 12 — Answer-Path Performance

### Phase 12.0 — Bounded Answer Synthesis
- after accepted evidence, enter a bounded final-answer path
- prevent additional tool calls after evidence is ready
- eliminate post_evidence_tool_call_rejected retry rounds
- preserve runtime-owned enforcement

### Phase 12.1 — Recovery Round Reduction
- reduce wrong-first-read recovery loops
- improve candidate selection / recovery targeting
- keep bounded recovery semantics

### Phase 12.2 — Answer Quality Validation
- benchmark answer correctness after bounded synthesis
- ensure no regression in grounded answers
- update benchmark docs

---

## Phase 13 - Memory & Context Performance

### Phase 13.0 — Prompt / KV cache investigation
### Phase 13.1 — Context/history compaction
### Phase 13.2 — Structured memory
### Phase 13.3 — Retrieval / indexing

---

## Phase 14 — Capability Expansion
### Phase 14.0 — LSP tools
* definition, hover, references
### Phase 14.1 — Safe web fetch
### Phase 14.2 — Shell/run with safety model

---

## Phase 15 — System Stabilization
* runtime cleanup
* reduce policy concentration
* unify patterns
* remove hacks
* clean up Phase 9.0.x maintainability debt when it becomes worthwhile:
  * clarify investigation correction flag names if more recovery paths are added
  * unify duplicated source-tier/file-class extension lists between search rendering and search ordering
  * keep investigation vocabulary narrow and avoid scattered trigger growth

---

## Phase 16 — UI / UX
### Phase 16.0 — Event System
### Phase 16.1 — UX Improvements
### Phase 16.2 — Polish / Animations

---

## Phase 17 — Backends & Optimization
### Phase 17.0 — Additional Providers
* Ollama
* OpenAI-compatible
### Phase 17.1 — Caching Improvements

---

## Phase 18 — TBD

Future expansion based on real usage.

---

## Guiding Rule

Do NOT pursue roadmap parity.

Each phase must:

* be justified by real observed behavior
* align with runtime-owned architecture
* avoid reintroducing old system coupling

---

## Principle

- Build correct behavior first.
- Then improve quality.
- Then expand capabilities.
