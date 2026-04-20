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
- truncates to the first `200` lines
- reports line count and truncation status

Runtime behavior adds one guardrail around failed reads: if `read_file` cannot read the requested file, the runtime injects the tool error and emits a runtime-owned terminal answer instead of asking the model to retry repeatedly.

### `list_dir`

Lists the immediate contents of one directory.

Current behavior:

- does not recurse
- returns entry name, kind, and file size when available
- sorts directories before files
- sorts alphabetically within each group

### `search_code`

Searches recursively for lines containing a literal substring.

Current behavior:

- rejects empty queries
- walks the project tree recursively
- skips hidden directories and common build/output directories such as `target`, `node_modules`, `.git`, `dist`, and `build`
- searches only a fixed set of text-like extensions
- returns matching file path, line number, and line text
- collects up to `50` matches internally
- orders collected matches by file class before truncation: source files, then config/data files, then docs/text/unknown files
- preserves deterministic ordering within each file class
- shows up to `15` matches in the conversation output
- does not interpret the query as regex or semantic search

Rendered search output is grouped by file for model readability:

- each file group shows the file path and match count
- each file group shows up to `MAX_LINES_PER_FILE = 3` representative matching lines
- files with more shown matches include a per-file "showing" count

This grouping is presentation-only in `tool_codec`. The underlying typed data remains the same:
`SearchMatch` and `SearchResultsOutput` are unchanged, and runtime behavior does not depend on
parsing the grouped text.

The typed input supports an optional scoped path, but the current model-facing wire format does not expose that scoped form yet.

Runtime behavior adds guardrails around the tool because prompt-only guidance was not enough for live local-model behavior:

- model-facing instructions ask for one plain literal keyword or identifier
- runtime simplifies phrase-like or method-like model queries to one literal token before dispatch
- each user turn gets one search, plus one retry only if the first search returned no matches
- after search is closed, additional `search_code` attempts are blocked by runtime correction

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

If the model tries to repair a malformed `edit_file` call after an edit tool error but still omits the required structure, the runtime injects an edit-specific correction and asks for a valid block instead of silently treating the malformed retry as a final answer.

The model-facing form remains the canonical `---search---` / `---replace---` block. For observed local-model drift, `tool_codec` also accepts narrow compatibility forms such as `old content:` / `new content:` labels and generic triple-dash delimiter pairs inside `[edit_file]...[/edit_file]`. Those still become exact `EditFile` inputs and go through normal validation and approval.

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
- `search_code` uses a simple literal line substring search, not regex or semantic search.
- `edit_file` only replaces the first exact match.
- `write_file` does not create parent directories.
- Tool result rendering is optimized for the runtime and TUI, not for rich previews or diffs.
