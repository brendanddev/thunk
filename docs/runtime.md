# Runtime

Describes the current runtime layer: what it owns, how a turn is processed, how tool calls are handled, and where the main control-flow boundaries live.

---

## Purpose

The runtime is the execution loop that sits between the UI, the model backend, and the tool layer.

It owns:

- the in-memory conversation
- the system prompt
- model generation
- tool-call parsing
- tool dispatch
- per-turn tool-surface policy and enforcement
- investigation and anchor state
- approval pause/resume state
- runtime corrections and runtime-owned final answers
- runtime events emitted to the UI

It does not own:

- terminal rendering
- session storage
- raw tool implementation logic

Those responsibilities stay in `tui/`, `app/` + `storage/`, and `tools/`.

---

## Main Types

### `Runtime`

`Runtime` in `src/runtime/engine.rs` owns the active conversation, the selected `ModelBackend`, the `ToolRegistry`, and the single optional `pending_action`.

### `Conversation`

`src/runtime/conversation.rs` stores the ordered message list sent to the model.

- the first message is always the system prompt
- user messages are appended directly
- assistant output is streamed into a single assistant message
- restored history is appended only once at startup

### `RuntimeRequest`

The runtime handles four requests:

- `Submit { text }`
- `Reset`
- `Approve`
- `Reject`

### `RuntimeEvent`

The runtime communicates outward only through events, including:

- activity changes
- assistant streaming start/chunk/finish
- tool start/finish
- approval required
- answer ready
- failure

The TUI renders these events but does not control runtime internals.

---

## Startup State

`Runtime::new()` builds a fresh system prompt from:

- the configured app name
- the discovered project root
- the registered tool specs
- the tool protocol instructions from `tool_codec`

`AppContext::build()` may then call `load_history()` once to append restored user and assistant messages after the new system prompt.

The runtime always starts from a fresh system prompt, even when conversation history is restored from storage.

Before each normal model generation, the runtime also injects an additional system message describing the active tool surface for that turn. That hint is part of the backend request only; it is not persisted in `Conversation` history. It narrows the current retrieval-vs-Git read-only family; mutation permission is enforced separately by the runtime.

---

## Runtime-Owned Turn Policy

Before tool dispatch, the runtime derives bounded per-turn policy state from the current user prompt:

- the active tool surface for the surface-owned read-only tools: `RetrievalFirst` or `GitReadOnly`
- whether mutating tools are allowed, based on conservative mutation-intent detection
- whether the prompt requires a bounded investigation flow
- the structural investigation mode and optional path scope
- whether the user explicitly requested reading one concrete file path
- whether an exact anchor prompt should replay the last successful read or search

The model does not own these classifications. They are runtime decisions used to constrain the turn.

---

## Turn Lifecycle

### 1. Submit

On `Submit`:

- the runtime rejects empty input
- the runtime rejects new input if a tool approval is already pending
- the user message is appended to `Conversation`
- activity changes to `processing`
- exact anchor prompts such as `read that file` and `search that again` can be resolved by the runtime without asking the model to rediscover the tool call
- otherwise the runtime enters the generate/tool loop

### 2. Generate

`run_generate_turn()` sends a full snapshot of the current conversation to the active backend as `GenerateRequest`.

Backend output is streamed back as `BackendEvent`s:

- backend status becomes runtime activity
- the first text chunk starts a new assistant message
- later chunks append to that same assistant message
- backend timing events are forwarded as advisory runtime timing events

If the backend produces no text at all, the runtime treats that as a failure.

### 3. Parse

After generation finishes, the full assistant response is scanned by `tool_codec::parse_all_tool_inputs()`.

- if no tool calls are found, the turn ends as a direct answer or a tool-assisted answer
- if tool calls are found, the runtime executes them in document order

### 4. Execute Tools

`run_tool_round()` dispatches each parsed `ToolInput` through `ToolRegistry`.

- immediate results are accumulated as runtime-owned `=== tool_result: name ===` blocks
- tool failures are accumulated as runtime-owned `=== tool_error: name ===` blocks
- the first tool that requires approval stops the round immediately

If the round finishes without needing approval, the accumulated result blocks are appended to the conversation as a user message.

Before dispatch, the runtime enforces structural turn policy:

- wrong-surface tools are rejected before they execute
- explicit `read path/to/file` requests must resolve to that exact path
- mutating tools are blocked when the conservative mutation-intent check does not classify the current prompt as a mutation request
- weak or repeated `search_code` calls are corrected or terminated by runtime policy rather than prompt wording alone

Search result blocks are rendered by `tool_codec` before they are appended. Current `search_code`
results are grouped by file in that rendered text, with per-file match counts and up to
`MAX_LINES_PER_FILE = 3` representative lines per file. This is presentation-only: the runtime still
receives typed `SearchResultsOutput` data and does not parse grouped text for decisions.

Some tool outcomes end with a runtime-owned assistant answer instead of another model generation. Current examples include failed `read_file` calls, rejected mutations, insufficient-evidence terminals, and completed Git read-only rounds.

`search_code` has extra runtime enforcement because prompt-only rules were not reliable enough with small local models:

- the model-facing prompt asks for one plain literal keyword or identifier
- before dispatch, runtime search calls are simplified to a single literal token
- the first search in a user turn is allowed
- a second search is allowed only if the first search returned no matches
- search closes after a non-empty result or after the one empty retry
- later search attempts are removed from the model context and replaced with a runtime correction that tells the model to answer from the available evidence

The runtime currently classifies investigation prompts into these structural modes, in this priority order:

`UsageLookup` > `ConfigLookup` > `InitializationLookup` > `CreateLookup` > `RegisterLookup` > `LoadLookup` > `SaveLookup` > `DefinitionLookup` > `General`

Investigation-required turns also have a post-evidence boundary:

- once sufficient evidence has been read, further tool calls in the same turn are structurally invalid
- the runtime corrects once
- a repeated violation ends the turn with `RuntimeTerminalReason::RepeatedToolAfterEvidenceReady`

This keeps the search -> read -> answer lifecycle runtime-owned instead of model-owned.

### Initialization Lookup

For prompts that ask where something is initialized, the runtime gives extra care to the file it accepts as evidence.

If search results include files that look like initialization matches, reading a different kind of match is not enough. The runtime can ask the model once to read a matched initialization file instead.

If no initialization match exists in the search results, the runtime falls back to the normal search-result read behavior.

### Save Lookup

For prompts that ask where something is saved, the runtime detects exact substring matches for `save`, `saved`, and `saving`.

Save candidates are classified only from matched search-result lines. If save candidates exist, reading a non-save match is not enough. The runtime can ask the model once to read a matched save candidate instead.

If no save match exists in the search results, the runtime falls back to the normal search-result read behavior.

### Git Read-Only Turns

`GitReadOnly` turns use a different bounded lifecycle from retrieval turns.

- among the surface-owned read-only tools, the turn allows only `git_status`, `git_diff`, and `git_log`
- one completed Git acquisition round is allowed
- that acquisition round may contain multiple Git tools in the same assistant response
- after that completed round, the runtime produces the visible answer directly from rendered Git output
- the runtime does not call the model again for Git answer synthesis

This keeps Git inspection deterministic and prevents post-tool looping or drift into retrieval.

### Direct Read Requests

If the original user prompt explicitly asks to read one concrete file path, the runtime tracks that request structurally.

- if the model reads that exact path, the turn continues normally from real file evidence
- if the model reads a different path, the runtime terminals with `ReadFileFailed`
- if no matching `read_file` result is ever produced, the runtime terminals instead of accepting an ungrounded answer

### 5. End or Pause

The current runtime behavior keeps tool evidence inside the same user turn:

- successful immediate retrieval rounds append results and usually re-enter generation for synthesis
- successful Git read-only acquisition rounds append results and end immediately with a runtime-produced visible answer
- approved mutations append the approved result and re-enter generation for a follow-up model response
- rejected mutations append a terminal tool error and a runtime-owned cancellation answer without re-entering model generation
- failed `read_file` calls append a tool error and a runtime-owned failure answer without re-entering model generation
- approval execution failures append a tool error and re-enter generation so the model can recover

The runtime has a hard cap of `10` tool rounds per turn, plus narrower runtime guards for repeated tool cycles and repeated searches.

---

## Approval Flow

Mutating tools do not write during their initial `run()` call. Instead they return `ToolRunResult::Approval(PendingAction)`.

When that happens:

1. the runtime stores the `PendingAction`
2. it emits `RuntimeEvent::ApprovalRequired`
3. it returns to idle
4. new user submissions are blocked until `/approve` or `/reject`

`Approve`:

- calls `ToolRegistry::execute_approved()`
- appends a runtime-owned tool result block on success
- appends a runtime-owned tool error block on failure
- re-enters model generation after either approved execution outcome

`Reject`:

- clears the pending action
- appends a runtime-owned tool error block noting user rejection
- emits a runtime-owned cancellation answer
- does not ask the model to synthesize the rejection, which prevents false claims that the mutation happened

Only one pending action can exist at a time.

---

## Tool Protocol Boundary

The runtime does not parse tool syntax itself. `src/runtime/tool_codec.rs` owns the wire protocol between assistant text and the tool layer.

That module is responsible for:

- parsing assistant text into typed `ToolInput` values
- formatting `ToolOutput` and tool failures back into conversation text
- providing the protocol instructions inserted into the system prompt
- shaping model-facing tool output for readability, such as grouped `search_code` result rendering

This keeps tool parsing centralized and keeps individual tools text-free.

---

## Conversation Rules

The runtime conversation is broader than what the user sees in the TUI.

It contains:

- the system prompt
- user prompts
- assistant text
- runtime-injected tool result and tool error blocks
- internal correction messages when the model violates the tool protocol
- runtime-owned terminal assistant answers for outcomes the runtime can state authoritatively

Notable correction paths today:

- if the assistant fabricates a `tool_result` or `tool_error` block instead of making a real tool call, the runtime removes that assistant message, injects a correction message, and retries once
- if the assistant emits a malformed tool block with the wrong opening tag but a recognizable closing tag, the runtime corrects and retries instead of treating the prose as a valid answer
- if an `edit_file` repair attempt follows an edit tool error but is still malformed, the runtime injects an edit-specific correction instead of silently accepting the malformed retry as a direct answer
- if `search_code` exceeds the per-turn search budget, the runtime discards that retry from conversation context and injects a search-closed correction

Runtime-owned final answers are streamed through the same assistant-message events as model text. Deterministic failure / rejection paths report `AnswerSource::RuntimeTerminal`. Completed Git read-only turns currently report `AnswerSource::ToolAssisted { rounds }` even though the visible answer text is runtime-produced, because `AnswerSource` still groups successful tool-completed paths together.

---

## Anchors And Traces

The runtime owns a small amount of explicit, in-memory continuity state:

- last successful `read_file`
- last successful `search_code` query and scope
- same-scope reuse for exact phrases such as `in the same folder` or `within the same scope`

Anchor matching is exact and structural only. There is no pronoun resolution, ranking layer, or semantic memory.

When `PARAMS_TRACE_RUNTIME` is set, the runtime also emits advisory `RuntimeTrace` events describing decisions such as anchor resolution, evidence corrections, and Git acquisition completion. These traces are for logging only; they must not drive control flow or UI behavior.

---

## Interaction With Other Layers

### With `llm/`

The runtime depends only on the `ModelBackend` trait and backend stream events. It does not know whether the active backend is `mock` or `llama_cpp`.

### With `tools/`

The runtime only sees typed `ToolInput`, `ToolOutput`, `ToolRunResult`, and `PendingAction`. Tool payloads stay opaque to the runtime.

### With `app/`

`AppContext` calls into the runtime and handles autosave/logging around it. The runtime does not talk to SQLite directly.

### With `tui/`

The runtime emits `RuntimeEvent`s. The TUI renders them and routes slash commands back as `RuntimeRequest`s.

---

## Current Limitations

- The runtime always sends the full in-memory conversation snapshot to the backend.
- Live context trimming is not implemented before generation.
- `AnswerSource::ToolAssisted` still covers both model-authored synthesis and runtime-authored successful Git answers.
- Successful mutation turns still rely on a post-approval model response. There is no runtime-owned completion invariant for `edit_file` / `write_file` yet.
- `edit_file` may still require multiple model attempts before producing a valid exact edit; that is a model-output quality issue, not a tool-execution correctness issue.
- Pending approval state is in memory only and is lost on restart.
- The visible TUI transcript is not rebuilt from restored runtime history on startup.
