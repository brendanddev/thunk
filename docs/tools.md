# Tools

Describes the current tool layer: the common tool contract, how tools are registered and dispatched, and what each built-in tool does at a high level.

---

## Purpose

The tool layer gives the runtime a typed interface for project-local actions.

Today that tool surface is intentionally small:

- `read_file`
- `list_dir`
- `search_code`
- `edit_file`
- `write_file`

The layer is built around explicit types rather than text parsing. Raw assistant text is parsed in `runtime/tool_codec.rs` before any tool is called.

---

## Core Contract

### `Tool`

Every tool implements the `Tool` trait in `src/tools/mod.rs`.

It exposes:

- `spec()` for static tool metadata
- `run()` for phase-one execution
- `execute_approved()` for phase-two execution when approval is required

Read-only tools only use `run()`. Mutating tools use both phases.

### `ToolInput`

The runtime never passes raw strings into tools. It passes one typed `ToolInput` variant:

- `ReadFile`
- `ListDir`
- `SearchCode`
- `EditFile`
- `WriteFile`

### `ToolOutput`

Completed tools return typed outputs:

- file contents
- directory listings
- search results
- edit confirmations
- write confirmations

The runtime later decides how those outputs are rendered for the model and for the TUI.

### `ToolRunResult`

`run()` returns one of two outcomes:

- `Immediate(ToolOutput)` for read-only work
- `Approval(PendingAction)` for proposed mutations

### `PendingAction`

`PendingAction` is the handoff object between the tool layer and the runtime approval flow.

It contains:

- `tool_name`
- `summary`
- `risk`
- `payload`

The runtime owns the approval lifecycle, but `payload` is opaque tool-owned data that is passed back into `execute_approved()`.

---

## Registry And Dispatch

`ToolRegistry` owns tool registration and lookup.

It is responsible for:

- registering tools by their `spec().name`
- dispatching typed `ToolInput` to the right tool
- delegating approved mutations back to the correct tool
- exposing sorted tool specs for the system prompt

The default registry is built in `src/tools/mod.rs` and is rooted at the discovered project root.

---

## Path Resolution

`ToolContext` carries the discovered project root into each tool.

Relative paths:

- are resolved against the project root, not the process working directory

Absolute paths:

- pass through unchanged for read-only tools
- are allowed for mutating tools only if they stay within the project root

Mutating tools also reject `..` path traversal.

---

## Built-In Tools

### `read_file`

Reads a file and returns its contents immediately.

Current behavior:

- resolves paths against the project root
- reads raw bytes, then converts to UTF-8 lossily
- truncates at `100_000` bytes
- backs off to a valid UTF-8 boundary before truncating
- reports line count and truncation status

### `list_dir`

Lists the immediate contents of one directory.

Current behavior:

- does not recurse
- returns entry name, kind, and file size when available
- sorts directories before files
- sorts alphabetically within each group

### `search_code`

Searches recursively for lines containing a query string.

Current behavior:

- rejects empty queries
- walks the project tree recursively
- skips hidden directories and common build/output directories such as `target`, `node_modules`, `.git`, `dist`, and `build`
- searches only a fixed set of text-like extensions
- returns matching file path, line number, and line text
- truncates at `50` matches

The typed input supports an optional scoped path, but the current model-facing wire format does not expose that scoped form yet.

### `edit_file`

Proposes an exact text replacement in an existing file.

Current behavior:

- requires a non-empty path and non-empty search text
- requires the search text to exist before approval is requested
- returns `Approval(PendingAction)` instead of mutating immediately
- uses `RiskLevel::Medium`
- re-checks that the search text still exists when approved
- replaces only the first occurrence

This tool is intentionally exact. It does not do fuzzy patching or diff application.

### `write_file`

Proposes creating or overwriting a file with full content.

Current behavior:

- requires a non-empty path
- returns `Approval(PendingAction)` instead of writing immediately
- uses `RiskLevel::Medium` for creates and `RiskLevel::High` for overwrites
- checks actual file existence again at execution time
- does not create missing parent directories

This is a full-file write tool, not an append or patch tool.

---

## What Tools Must Not Do

Tools must not:

- parse raw assistant output
- manage approval state after returning `PendingAction`
- write UI messages
- persist sessions
- depend on terminal state

Those responsibilities belong to the runtime, app/storage, and TUI layers.

---

## Current Limitations

- There are only five built-in tools.
- There is no shell, git, web, or external integration tool yet.
- `search_code` uses a simple line substring search, not regex or semantic search.
- `edit_file` only replaces the first exact match.
- `write_file` does not create parent directories.
- Tool result rendering is optimized for the runtime and TUI, not for rich previews or diffs.
