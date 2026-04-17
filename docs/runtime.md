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

- immediate results are accumulated as runtime-owned `[tool_result: ...]` blocks
- tool failures are accumulated as runtime-owned `[tool_error: ...]` blocks
- the first tool that requires approval stops the round immediately

If the round finishes without needing approval, the accumulated result blocks are appended to the conversation as a user message.

### 5. End or Pause

The current runtime behavior is intentionally simple:

- successful tool rounds end the turn immediately
- approved mutations also end the turn immediately
- rejected mutations trigger another generation pass so the model can react
- approval execution failures also trigger another generation pass

The runtime has a hard cap of `10` tool rounds per turn.

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

`Reject`:

- clears the pending action
- appends a runtime-owned tool error block noting user rejection
- resumes model generation

Only one pending action can exist at a time.

---

## Tool Protocol Boundary

The runtime does not parse tool syntax itself. `src/runtime/tool_codec.rs` owns the wire protocol between assistant text and the tool layer.

That module is responsible for:

- parsing assistant text into typed `ToolInput` values
- formatting `ToolOutput` and tool failures back into conversation text
- providing the protocol instructions inserted into the system prompt

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

One notable correction path exists today: if the assistant fabricates a `[tool_result:]` or `[tool_error:]` block instead of making a real tool call, the runtime removes that assistant message, injects a correction message, and retries once. If the model repeats the behavior, the turn fails.

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

- Successful tool rounds do not automatically trigger a follow-up assistant response.
- There is no cycle detection beyond the hard `MAX_TOOL_ROUNDS = 10` cap.
- The runtime always sends the full in-memory conversation snapshot to the backend.
- Live context trimming is not implemented before generation.
- Pending approval state is in memory only and is lost on restart.
- The visible TUI transcript is not rebuilt from restored runtime history on startup.
