# Architecture

## System Overview

`params-cli` is a local-first Rust TUI application built around six main layers:

- `app` bootstraps the system and owns session persistence.
- `runtime` owns the conversation, model interaction, tool dispatch, and approval state.
- `tools` exposes typed tool operations behind a registry.
- `storage` persists sessions in SQLite.
- `llm` hides model-provider details behind a common backend trait.
- `tui` handles terminal input, rendering, and control commands.

The boot path is straightforward:

1. `src/main.rs` calls `app::run()`.
2. `app::run()` discovers the project root from `config.toml`, ensures `data/` and `logs/` exist, loads config, builds the selected model backend, builds the default tool registry, opens logging, restores the most recent session, then launches the TUI.
3. `AppContext` sits between the TUI and the runtime so the UI never talks to storage or tool implementations directly.

This document follows the current code, not older status notes. In the current implementation, `edit_file`, `write_file`, and approval flow are implemented.

## Layer Breakdown

| Layer | Owns | Does Not Own |
| --- | --- | --- |
| `app` | startup, path/config discovery, backend + tool assembly, session restore/save, reset, optional session logging | tool parsing, tool execution logic, terminal rendering |
| `runtime` | conversation state, system prompt, backend generation, tool-call parsing via `tool_codec`, tool dispatch, pending approval state, runtime events | SQLite, TUI state, file rendering |
| `tools` | typed tool contracts, tool registry, path resolution, read/write/search/edit implementations, approval proposals for mutating tools | parsing raw model text, UI behavior, session storage |
| `storage` | SQLite schema and session CRUD | runtime/tool semantics, prompt construction |
| `llm` | `ModelBackend` abstraction, provider selection, model-specific prompt formatting, streaming backend events | tool logic, persistence, UI |
| `tui` | raw terminal lifecycle, input editing, slash command parsing, rendering transcript and status, mapping runtime events to visible messages | business logic, tool execution, persistence |

There is also a small `logging` utility. It is advisory rather than foundational: `AppContext` can attach a per-session log file, but correctness does not depend on it.

## Responsibilities By Layer

### `app`

- Finds the project root by walking upward until `config.toml` is found.
- Resolves relative config paths, notably `llama_cpp.model_path`, against that root.
- Creates `AppContext`, which is the boundary object used by the TUI.
- Auto-saves session history after `RuntimeRequest::Submit`.
- Starts a new session on reset.

`AppContext` deliberately does not execute tools. It forwards `Submit`, `Approve`, `Reject`, and `Reset` into the runtime and handles persistence around that.

### `runtime`

- Builds the system prompt from the app name, project root, tool specs, and tool format instructions.
- Stores the conversation as ordered `Message` values, always starting with the system prompt.
- Sends conversation snapshots to the active `ModelBackend`.
- Streams assistant output into the conversation while emitting UI-facing `RuntimeEvent`s.
- Parses assistant text into typed `ToolInput` values by delegating to `tool_codec`.
- Dispatches tools through `ToolRegistry`.
- Owns the single in-memory `pending_action` used for approval flow.

The runtime is intentionally unaware of both the TUI and SQLite. It only speaks in `RuntimeRequest` and `RuntimeEvent`.

### `tools`

- Define the typed contract: `ToolInput`, `ToolOutput`, `ToolRunResult`, and `ToolSpec`.
- Register tool implementations in `ToolRegistry`.
- Use `ToolContext` so relative paths resolve against the project root instead of process CWD.
- Separate read-only execution from mutating execution.

Current default tools:

- `read_file`
- `list_dir`
- `search_code`
- `edit_file`
- `write_file`

Read-only tools return immediate typed outputs. Mutating tools return approval requests first and only write during `execute_approved()`.

### `storage`

- Stores sessions in `data/sessions.db`.
- Owns the SQLite schema (`sessions` and `session_messages`).
- Loads the most recently updated session at startup.
- Replaces stored messages on save instead of appending deltas.

The explicit conversion between runtime `Message` values and stored records happens in `src/app/session.rs`. That boundary is intentional: storage stays decoupled from runtime message types.

### `llm`

- Defines `ModelBackend`, `GenerateRequest`, and streamed `BackendEvent`s.
- Supports two providers today: `mock` and `llama_cpp`.
- Formats `llama_cpp` requests into a ChatML-style prompt.
- Lazy-loads the llama.cpp model on first generation.

The runtime never knows whether the active backend is mock or llama.cpp.

### `tui`

- Owns terminal setup/teardown and raw-mode lifecycle.
- Edits the input buffer and cursor state.
- Parses only control commands: `/help`, `/clear`, `/quit`, `/approve`, `/reject`.
- Renders assistant streaming, compact tool summaries, approval prompts, and status text.

The TUI intentionally keeps control flow thin. It does not parse tool syntax and does not decide what tools do.

## Data Flow

### Startup

1. Paths are discovered from the current working directory upward.
2. Config is loaded from `config.toml`.
3. The model backend and default tool registry are created.
4. The most recent session is restored from SQLite.
5. `Runtime` is built and receives restored history after the freshly built system prompt.
6. The TUI starts with a fresh `AppState`.

One important current-state detail: restored history is loaded into the runtime, but not replayed into the TUI transcript. On startup, the model has prior context immediately, while the visible TUI transcript still starts from the welcome message.

### Normal Submit

1. The user types in the TUI.
2. Slash commands are handled locally; all other input becomes `RuntimeRequest::Submit`.
3. `AppContext` forwards the request to `Runtime`.
4. `Runtime` rejects empty prompts and blocks submission if an approval is already pending.
5. The user message is appended to `Conversation`.
6. The backend receives a full snapshot of the conversation and streams text back as `BackendEvent`s.
7. The runtime appends streamed assistant chunks and emits `RuntimeEvent`s for the TUI.
8. When generation ends, `tool_codec` parses the assistant text for tool calls.
9. If no tool calls are found, the assistant message is the final answer.
10. After the submit finishes, `AppContext` saves the runtime message snapshot to SQLite.

### Tool Round

If the assistant output contains tool calls:

1. `tool_codec::parse_all_tool_inputs()` extracts typed `ToolInput` values in document order.
2. `ToolRegistry::dispatch()` routes each input to the correct tool.
3. Read-only tools return `ToolRunResult::Immediate(ToolOutput)`.
4. The runtime emits a compact `ToolCallFinished` summary for the TUI.
5. The full tool result is formatted as `[tool_result: name]...[/tool_result]` and appended back into the conversation as a user message.

Current UX rule: after a completed tool round, the runtime ends the turn instead of automatically asking the model for a follow-up answer. This matches the Phase 6 direction in `CLAUDE.md`: tool results are surfaced as UI events rather than immediately triggering another assistant narration.

The runtime is structured as a loop and still has a `MAX_TOOL_ROUNDS` guard of `10`, but in the current implementation a successful tool round ends the turn immediately.

Because generation is streamed before parsing, the raw tool call itself is also present as assistant text in the conversation and visible transcript today. The compact tool summary is an extra UI event layered on top of that.

## Tool Execution Model

### Core Types

- `ToolInput`: typed request for one tool.
- `ToolOutput`: typed result of a completed tool.
- `ToolRunResult::Immediate(ToolOutput)`: no approval needed.
- `ToolRunResult::Approval(PendingAction)`: mutation proposed but not yet executed.
- `PendingAction`: `{ tool_name, summary, risk, payload }`.

`PendingAction.payload` is opaque to the runtime. The runtime stores it and passes it back to the tool on approval without inspecting it.

### Two-Phase Mutation Flow

Mutating tools (`edit_file` and `write_file`) follow a strict two-step contract:

1. `run()`
   - validates input
   - checks the current file state
   - creates a human-readable summary
   - returns `ToolRunResult::Approval(PendingAction)`
2. `execute_approved()`
   - receives the tool-owned payload
   - re-validates anything that can go stale
   - performs the file mutation
   - returns `ToolOutput`

This keeps mutation out of the initial model-driven phase. `run()` proposes; `execute_approved()` mutates.

### Approval Flow

Approval is runtime-owned and TUI-triggered:

1. A mutating tool returns `Approval(PendingAction)`.
2. The runtime stores exactly one pending action and emits `RuntimeEvent::ApprovalRequired`.
3. While a pending action exists, new submits are rejected.
4. The user resolves it with `/approve` or `/reject`.

If earlier tool calls in the same assistant response already completed immediately, their formatted results are preserved before the pause.

On `/approve`:

- `ToolRegistry::execute_approved()` calls the original tool by `tool_name`.
- Success produces a compact TUI summary and a `[tool_result: ...]` conversation message.
- The turn ends immediately after the approved mutation; there is no automatic follow-up generation.

On `/reject`:

- The runtime appends a `[tool_error: ...]` block noting user rejection.
- The runtime does run generation again so the model can react or choose another action.

On approval execution failure:

- The runtime appends a `[tool_error: ...]` block.
- The runtime resumes model generation so the model can recover.

### Current Tool-Specific Behavior

- `read_file` truncates at `100_000` bytes and preserves UTF-8 boundaries.
- `search_code` caps results at `50` matches and skips common build/output directories.
- `edit_file` requires exact search text, replaces only the first occurrence, and re-checks that the search text is still present at approval time.
- `write_file` does not create missing parent directories.

### Path Rules

- Relative tool paths resolve against the discovered project root.
- Mutating tools reject `..` traversal and reject absolute paths outside the project root.
- Read-only tools are project-root aware, but absolute paths pass through unchanged.

## Session Persistence Model

Session persistence is simple and transcript-based:

- Storage lives in `data/sessions.db`.
- `SessionStore::load_most_recent()` restores the latest session on startup.
- `ActiveSession::save()` writes the full current transcript for the active session.
- Save replaces prior stored messages for that session instead of appending deltas.

What is stored:

- user messages
- assistant messages

What is not stored:

- system prompt
- TUI-only system/status messages
- pending approval state

The system prompt is rebuilt at runtime from config and tool specs, so storage intentionally avoids keeping a stale copy.

Two current limitations are worth calling out:

- `AppContext` only auto-saves after `Submit`, not after `Approve` or `Reject`. That means approval outcomes and rejection-triggered follow-up messages can exist only in memory until the next submit.
- Pending approvals are not persisted. Restarting the app clears in-flight approval state.

`/clear` is a real session boundary: it resets the runtime conversation and creates a new session row in SQLite.

## Tool Protocol (`runtime/tool_codec.rs`)

`tool_codec` is the sole owner of the model/tool wire format.

Its responsibilities are:

- parse model text into typed `ToolInput` values
- format `ToolOutput` back into conversation text
- define the exact instructions injected into the system prompt

This is a key architectural rule: tools never parse raw model text themselves.

### Inbound Formats

Single-line tools:

```text
[read_file: src/main.rs]
[list_dir: src/]
[search_code: fn main]
```

Multi-line tools:

```text
[write_file]
path: src/new.rs
---content---
full file contents
[/write_file]
```

```text
[edit_file]
path: src/lib.rs
---search---
exact text to replace
---replace---
replacement text
[/edit_file]
```

Parser behavior is intentionally strict but small:

- malformed or incomplete calls are skipped rather than heuristically fixed
- mixed tool call types are returned in document order
- `list_dir` defaults to `"."` when given an empty argument
- block parsing trims only the structural newline around content blocks
- no normalization is done inside tool implementations

### Outbound Formats

Successful results:

```text
[tool_result: read_file]
...
[/tool_result]
```

Errors:

```text
[tool_error: write_file]
...
[/tool_error]
```

The TUI does not render these full bodies directly. Instead, the runtime also derives a compact one-line summary for display. This keeps the user-facing transcript smaller while still preserving structured tool context inside the runtime conversation.

### Current Protocol Gap

`ToolInput::SearchCode` supports an optional scoped path, but the current text protocol only exposes `[search_code: query]`. In practice, model-issued `search_code` calls currently search from the project root.

## Key Architectural Rules And Invariants

- Lower layers do not depend on higher layers.
- The runtime does not know about the TUI or SQLite.
- The TUI does not execute tools or contain business logic.
- Parsing of model tool calls belongs only in `runtime/tool_codec.rs`.
- Tools operate on typed inputs and outputs, not raw model text.
- The first runtime message is always the current system prompt.
- Relative tool paths resolve from project root, not process CWD.
- There is at most one pending approval at a time.
- Mutating tools do not perform writes during `run()`.
- Tool payloads are opaque outside the owning tool.

These rules match the intent in `CLAUDE.md` and are reflected in the current code structure.

## What Is Intentionally Deferred

The code and `CLAUDE.md` make a few current boundaries explicit:

- No shell execution, web fetch, LSP integration, or external integrations yet.
- No memory system beyond persisted conversation transcripts.
- No persisted pending-action state across restarts.
- No advanced session UX such as browsing or restoring transcript history in the TUI itself.
- No rich tool-result UI yet: the TUI shows compact summaries, not previews, diffs, or collapsible content.
- No fully structured model-facing protocol yet: tool execution is typed internally, but the model boundary is still centralized text via `tool_codec`.
- No scoped `search_code` syntax in the current wire format.
- Observability is present only as lightweight per-session file logging, not the fuller structured timing/logging plan described in later phases.
