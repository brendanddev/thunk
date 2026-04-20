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
- approval pause/resume state
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

---

## Turn Lifecycle

### 1. Submit

On `Submit`:

- the runtime rejects empty input
- the runtime rejects new input if a tool approval is already pending
- the user message is appended to `Conversation`
- activity changes to `processing`
- the runtime enters the generate/tool loop

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

Search result blocks are rendered by `tool_codec` before they are appended. Current `search_code`
results are grouped by file in that rendered text, with per-file match counts and up to
`MAX_LINES_PER_FILE = 3` representative lines per file. This is presentation-only: the runtime still
receives typed `SearchResultsOutput` data and does not parse grouped text for decisions.

Some tool outcomes end with a runtime-owned assistant answer instead of another model generation. Today that terminal path is used when `read_file` fails, so missing-file reads surface the tool error and stop cleanly instead of looping through repeated failed reads.

`search_code` has extra runtime enforcement because prompt-only rules were not reliable enough with small local models:

- the model-facing prompt asks for one plain literal keyword or identifier
- before dispatch, runtime search calls are simplified to a single literal token
- the first search in a user turn is allowed
- a second search is allowed only if the first search returned no matches
- search closes after a non-empty result or after the one empty retry
- later search attempts are removed from the model context and replaced with a runtime correction that tells the model to answer from the available evidence

### 5. End or Pause

The current runtime behavior keeps tool evidence inside the same user turn:

- successful immediate tool rounds append results and re-enter generation for synthesis
- approved mutations append the approved result and re-enter generation for synthesis
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
- resumes model generation after either approved execution outcome

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

Runtime-owned terminal answers are streamed through the same assistant-message events as model text and currently report `AnswerSource::ToolAssisted`.

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
- `AnswerSource` does not yet distinguish model synthesis from deterministic runtime-owned terminal answers.
- `edit_file` may still require multiple model attempts before producing a valid exact edit; that is a model-output quality issue, not a tool-execution correctness issue.
- Pending approval state is in memory only and is lost on restart.
- The visible TUI transcript is not rebuilt from restored runtime history on startup.
