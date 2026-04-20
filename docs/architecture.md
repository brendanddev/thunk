# Architecture

Defines the high-level architecture and design decisions of the app, including the layered structure, data flow, tool execution model, protocol design, persistence model, and known limitations.

---

## System Overview

`params-cli` is a local-first Rust TUI coding assistant. It runs a conversation loop against a selected model backend, lets the model request a small set of file-system tools through a constrained text protocol, and requires explicit user approval before mutating files.

At startup, `src/main.rs` calls `app::run()`. The app layer discovers the project root from `config.toml`, loads config, builds the model backend and tool registry, opens optional session logging, restores the most recent session from SQLite, and launches the TUI. After that, the TUI talks only to `AppContext`; `AppContext` forwards requests into the runtime and persists the runtime transcript.

The core problem the project solves is running an AI coding assistant locally without collapsing the system into one text-driven loop. The current implementation keeps model generation, tool execution, approval, persistence, and UI rendering in separate layers with explicit boundaries.

---

## Layered Architecture

### `app/`

- Responsibilities: startup, project-root discovery, config loading, backend and tool assembly, session restore/save, reset orchestration, optional logging hookup.
- Owns: `AppContext`, `AppPaths`, `ActiveSession`, and the boot path that constructs `Runtime`.
- Must not: parse tool syntax, execute tool logic, or render terminal UI.

### `runtime/`

- Responsibilities: system prompt construction, conversation state, backend generation loop, tool-call parsing through `tool_codec`, tool dispatch, approval pause/resume, and runtime events.
- Owns: `Conversation`, `pending_action`, `RuntimeRequest`, `RuntimeEvent`, and the decision to inject tool results back into the conversation.
- Must not: talk to SQLite directly, manipulate terminal state, or let tools parse raw model text.

### `tools/`

- Responsibilities: typed tool contracts, tool registry, project-root-aware path resolution, and concrete tool behavior for reading, listing, searching, editing, and writing files.
- Owns: `ToolInput`, `ToolOutput`, `ToolRunResult`, `PendingAction` payload encoding, and path-safety checks for mutating tools.
- Must not: parse assistant text, manage approval state after returning a `PendingAction`, render UI, or persist sessions.

### `storage/`

- Responsibilities: SQLite schema management and session CRUD.
- Owns: `SessionStore`, the `sessions` / `session_messages` tables, and stored session/message types.
- Must not: know about runtime control flow, tool semantics, prompts, or UI behavior.

### `llm/`

- Responsibilities: model backend abstraction, provider selection, provider-specific prompt formatting, streaming backend events, and llama.cpp execution details.
- Owns: `ModelBackend`, `GenerateRequest`, `BackendEvent`, `BackendStatus`, `mock`, and `llama_cpp`.
- Must not: know about tools, persistence, slash commands, or terminal rendering.

### `tui/`

- Responsibilities: terminal lifecycle, raw input editing, slash-command parsing, transcript rendering, and mapping `RuntimeEvent`s into visible UI messages.
- Owns: `AppState`, `/help`, `/clear`, `/quit`, `/approve`, `/reject`, and alternate-screen / raw-mode handling.
- Must not: execute tools, save sessions, parse tool calls, or contain runtime business logic.

### Supporting Utility: `logging/`

- Responsibilities: best-effort per-session log files under `logs/`.
- Owns: timestamped log creation and elapsed-time logging helpers.
- Must not: affect correctness; the app continues if logging cannot be opened.

---

## Data Flow

1. The user types into the TUI.
2. Slash commands are handled in `tui/commands`; normal prompts become `RuntimeRequest::Submit`.
3. `AppContext` forwards the request into `Runtime` and logs request/event timing if a session log is open.
4. The runtime appends the user message to `Conversation`, snapshots the full message list, and sends it to the active `ModelBackend`.
5. The backend streams `BackendEvent`s. The runtime converts status updates into `Activity` changes and text deltas into an assistant message in the conversation.
6. When generation finishes, `tool_codec::parse_all_tool_inputs()` scans the full assistant response and returns typed `ToolInput` values in document order.
7. `ToolRegistry` dispatches each `ToolInput` to its tool implementation.
8. Immediate tool results are rendered two ways by the runtime: a compact one-line summary for the TUI, and a `=== tool_result: name ===` block appended back into the conversation as a user message.
9. If a tool returns `Approval(PendingAction)`, the runtime stores that single pending action, emits `ApprovalRequired`, and stops the turn until the user chooses `/approve` or `/reject`.
10. If no approval is pending, the runtime normally re-enters generation with the injected tool results so the assistant can produce a same-turn answer grounded in actual tool output.
11. If the runtime already knows the terminal outcome, such as a rejected mutation or failed `read_file`, it can emit a runtime-owned assistant answer instead of asking the model to synthesize.
12. The TUI renders events only. It never sees typed tool payloads and never calls tool implementations directly.

One important current behavior: successful tool rounds do not end the turn immediately. The runtime normally calls the model again with the tool results so the final answer can synthesize what was actually found or changed. Terminal runtime-owned answers are reserved for cases where model synthesis would be less reliable than the runtime state, such as rejection or missing-file read failures.

---

## Tool Execution Model

### `ToolRunResult`

`ToolRunResult` is the runtime-facing result of dispatching a tool:

- `Immediate(ToolOutput)` means the tool finished synchronously and no approval is required.
- `Approval(PendingAction)` means the tool proposed a mutation and the turn must pause.

### `PendingAction`

`PendingAction` is pure data:

- `tool_name`
- `summary`
- `risk`
- `payload`

The runtime owns the pending action lifecycle, but it does not interpret `payload`. That payload is opaque tool-owned data passed back into `execute_approved()`.

### Approval Flow

1. A mutating tool returns `Approval(PendingAction)` from `run()`.
2. The runtime stores it in `pending_action` and emits `RuntimeEvent::ApprovalRequired`.
3. While `pending_action` is set, `Submit` is rejected.
4. `/approve` calls `ToolRegistry::execute_approved()`.
5. `/reject` appends a `=== tool_error: name ===` block and emits a runtime-owned cancellation answer.

On approval success, the runtime appends a `=== tool_result: name ===` block and resumes generation for synthesis. On approval failure, it appends a `=== tool_error: name ===` block and resumes generation so the model can recover. On rejection, the runtime does not re-enter model generation because it already knows no mutation occurred.

### Two-Phase Execution

Mutating tools use an explicit two-phase contract:

1. `run()`
   Validates input, inspects current file state, and returns either `Immediate` or `Approval`.
2. `execute_approved()`
   Re-validates anything that may have gone stale, performs the mutation, and returns `ToolOutput`.

Current mutating tools:

- `edit_file` proposes an exact-text replacement, replaces only the first match, and re-checks that the search text still exists at approval time.
- `write_file` proposes create/overwrite, writes only after approval, and does not create missing parent directories.

---

## Tool Protocol

`runtime/tool_codec.rs` owns the wire protocol between model text and tool execution. It has three jobs:

- parse assistant text into typed `ToolInput` values
- format `ToolOutput` / tool errors back into runtime-owned conversation text
- provide the tool-use instructions embedded in the system prompt

It also owns model-facing output shaping for tool results. For example, `search_code` returns typed
search matches, while `tool_codec` renders those matches grouped by file with per-file match counts
and a small per-file line cap. That grouping is for model interpretability only; it does not change
the typed search result data or runtime orchestration.

Supported model-facing call formats:

```text
[read_file: path/to/file.rs]
[list_dir: src/]
[search_code: keyword]

[edit_file]
path: path/to/file.rs
---search---
old content
---replace---
new content
[/edit_file]

[write_file]
path: path/to/file.rs
---content---
full file content
[/write_file]
```

Protocol rules in the current implementation:

- `tool_codec` is the only parser for raw assistant tool syntax.
- Single-line bracket calls must close on the same line.
- Multi-line `edit_file` / `write_file` blocks must contain the required delimiters and closing tag.
- `search_code` is model-facing as a single literal keyword or identifier, not a regex, method call, or phrase query.
- `search_code` still accepts narrow legacy block forms such as `pattern:` and `query:` for model-drift tolerance, but those names are parser compatibility only; the tool performs literal substring matching.
- `search_code` input is simplified by the runtime to one literal token before dispatch when the model emits a phrase or method-shaped query.
- `edit_file` remains model-facing as the canonical `---search---` / `---replace---` block, but the parser accepts narrow observed drift forms including `old content:` / `new content:` labels and generic triple-dash search/replace delimiter pairs.
- Malformed wrong-open-tag tool blocks are detected and corrected instead of silently becoming normal assistant prose.
- Malformed `edit_file` retries after an edit error receive an edit-specific runtime correction instead of being accepted as a final answer.
- Mixed tool-call formats are executed in the order they appear in the assistant response.

The system prompt tells the model that when a tool is needed, the reply should contain tool call tags only. Direct plain-text answers are still allowed when no tool is needed. Tool results are never model-authored: the runtime injects `=== tool_result: name ===` and `=== tool_error: name ===` blocks after execution, and those result formats are intentionally not described back to the model.

Prompt-only behavioral rules are not treated as sufficient for loop safety. For `search_code`, the runtime also enforces a per-turn budget: one search is always allowed, one retry is allowed only if the first search returned no matches, and later search attempts are blocked with a runtime correction.

---

## Session & Persistence Model

Sessions are stored in `data/sessions.db` through `storage/session`.

- `sessions` stores session metadata.
- `session_messages` stores ordered messages for each session.
- `SessionStore::load_most_recent()` restores the most recently updated session at startup.
- `ActiveSession::save()` rewrites the stored messages for the current session instead of appending deltas.

The stored transcript is derived from the runtime conversation:

- system messages are never stored
- user and assistant messages are stored as plain role/content rows
- runtime-injected tool result/error blocks are stored like any other user message

Restore behavior is intentionally narrower than storage:

- only the most recent `RESTORE_WINDOW` messages are injected back into the runtime
- the current `RESTORE_WINDOW` is `10`
- user messages that start with `=== tool_result:`, `=== tool_error:`, or `[runtime:correction]` are stripped during restore
- if such a stripped tool exchange was preceded by a pure assistant tool call that starts with `[`, that assistant message is stripped too
- full stored history remains in SQLite even when it is excluded from restored context

Live trimming is limited today:

- there is no token-aware budgeting or message trimming before generation
- every generation request sends the full in-memory conversation snapshot
- `read_file` truncates file reads to the first `200` lines
- `search_code` truncates at `50` matches
- if the live prompt still exceeds the configured llama.cpp context window, generation fails instead of auto-trimming

Current persistence behavior:

- `AppContext` auto-saves after `Submit`, `Approve`, and `Reject`
- `pending_action` is still in-memory only, so restarting the app clears any in-flight approval

One current UI/runtime mismatch also matters: restored history is loaded into the runtime, but it is not replayed into `AppState`, so the visible transcript starts fresh each launch even when the model already has restored context.

---

## Runtime Guarantees & Invariants

- At most one `pending_action` exists at a time.
- New user submissions are rejected while an approval is pending.
- The runtime owns conversation mutation, tool result injection, and approval state.
- Raw assistant tool syntax is parsed only in `tool_codec`.
- Tools return typed data; tools do not append conversation text themselves.
- Mutating tools do not write during `run()`; writes happen only in `execute_approved()`.
- `search_code` executes literal substring searches, and repeated search behavior is bounded per user turn by runtime state.
- rejected mutations are answered by the runtime without model synthesis, so the assistant cannot claim a rejected write/edit happened
- failed `read_file` calls can terminate with a runtime-owned answer, so missing-file reads do not loop
- Malformed `edit_file` repair attempts after edit errors are surfaced back to the model through runtime correction rather than silently ending the turn.
- The runtime communicates through `RuntimeRequest` and `RuntimeEvent`; it does not depend on the TUI or SQLite.
- Logging is advisory and does not participate in control flow.
- Failure paths are explicit: backend failures become `RuntimeEvent::Failed`, tool dispatch/approval failures become runtime-owned `=== tool_error: name ===` messages or failed events, and the runtime returns to a defined state instead of silently continuing.
- There is no global mutable singleton state; root path, config, backend, tools, and session handles are all passed in through construction.

---

## Known Limitations / Deferred Work

- Live context management is incomplete. Restore trimming exists, but there is no proactive token-based budgeting or live conversation trimming before generation.
- Tool-loop safety still includes a hard limit of `10` tool rounds per turn; search has narrower per-turn runtime enforcement, but broader planning quality is still model-dependent.
- `AnswerSource` currently reports runtime-owned terminal answers as `ToolAssisted`; it does not yet distinguish deterministic runtime fallback from model synthesis.
- `edit_file` can still be noisy before a valid exact edit block appears; this is a model-output quality issue, not a correctness issue once a valid tool call is parsed.
- Advanced memory is not implemented. There is no embeddings layer, structured memory, or long-term recall.
- LSP integration is not implemented.
- The tool surface is still small: `read_file`, `list_dir`, `search_code`, `edit_file`, and `write_file` only.
- Tool UX is still compact-first. There is no file preview mode, diff visualization, or expandable tool-output UI.
- Restore UX is incomplete because restored context is not replayed into the visible TUI transcript.
