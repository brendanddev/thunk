# params-cli

Local-first, personal AI coding assistant CLI focused on local-first workflows, modular architecture, privacy, and real coding actions.

> Version 0.8.18

---

## Overview

`params-cli` is a Rust-based personal AI coding assistant built around a small, explicit runtime:

- a terminal UI for chat and control commands
- a runtime that owns conversation state and tool dispatch
- a tool layer with typed inputs and outputs
- SQLite-backed session persistence
- swappable model backends

The project is structured to keep model generation, tool execution, persistence, and UI separate instead of folding everything into one text-driven loop.

---

## Whats Different

- Runtime owned correctness
- Structural, not semantic
- Grounded code investigation
- Controlled multi-turn continuity
- Built for local and low resource environments

---

## What It Does Today

- Runs as a local terminal app with an alternate-screen TUI.
- Supports two model backends: `mock` and `llama_cpp`.
- Builds a system prompt from the app name, project root, and registered tool specs.
- Streams assistant output into the conversation while emitting UI-facing runtime events.
- Parses tool calls centrally in `src/runtime/tool_codec.rs`.
- Executes read-only tools immediately and pauses for approval before mutating files.
- Re-enters model generation after tool results so the assistant can synthesize a grounded same-turn answer.
- Uses runtime-owned terminal answers when the runtime already knows the outcome, such as rejected mutations or failed file reads.
- Enforces bounded per-turn `search_code` behavior at runtime instead of relying only on prompt wording.
- Persists sessions in `data/sessions.db` and restores the most recent session on startup.
- Writes best-effort per-session logs under `logs/`.

Current built-in tools:

- `read_file`
- `list_dir`
- `search_code`
- `edit_file`
- `write_file`

Current control commands:

- `/help`
- `/clear`
- `/quit`
- `/approve`
- `/reject`

---

## Runtime Behavior

At a high level:

1. The user submits a prompt in the TUI.
2. The runtime sends the full in-memory conversation to the active model backend.
3. The assistant response is scanned for tool calls.
4. Tool calls are dispatched in document order.
5. Immediate tool results are injected back into the conversation as runtime-owned result blocks.
6. The runtime normally re-enters generation with those results so the model can answer from actual tool output.
7. If a mutating tool proposes a change, the runtime stores a single `PendingAction` and waits for `/approve` or `/reject`.

Some outcomes are deliberately terminal and runtime-owned: rejecting a pending mutation produces a cancellation answer without asking the model to summarize, and a failed `read_file` can end cleanly without retrying in a loop.

`search_code` is a literal substring search. The runtime now simplifies model-generated search phrases into a single literal keyword and enforces a per-turn budget: one search is allowed, a second search is allowed only when the first returned no matches, and later search attempts are blocked with a correction so the model must answer cleanly.

---

## Architecture

The codebase is split into six main layers:

- `src/app/` — startup, config, paths, session orchestration
- `src/runtime/` — conversation loop, tool parsing, approval state, runtime events
- `src/tools/` — tool contracts, registry, and implementations
- `src/storage/` — SQLite session storage
- `src/llm/` — backend abstraction and providers
- `src/tui/` — terminal input, rendering, and slash commands

Key architectural rules reflected in the code:

- parsing of raw tool syntax lives in `runtime/tool_codec.rs`
- tools operate on typed `ToolInput` / `ToolOutput`, not raw model text
- mutating tools separate `run()` from `execute_approved()`
- the runtime does not depend on the TUI or SQLite directly
- the TUI renders events but does not execute tools

---

## Current Limitations

- No shell, git, web, or external integration tools yet.
- No LSP integration or advanced memory system.
- No token-aware live context budgeting before generation.
- Pending approvals are not persisted across restarts.
- Restored session history is loaded into the runtime, but not replayed into the visible TUI transcript.
- Tool UI is compact and text-based; there is no diff view or expandable preview UI yet.

---

## Running

Requirements:

- Rust stable
- Interactive terminal (`stdout` must be a TTY and `TERM` must not be `dumb`)
- A local `.gguf` model if using `llama_cpp`

Run the app:

```bash
cargo run
```

Run tests:

```bash
cargo test
```

Configuration lives in `config.toml`.

- `llm.provider = "mock"` uses the built-in mock backend.
- `llm.provider = "llama_cpp"` uses the local llama.cpp backend.
- `llama_cpp.model_path` points to the local `.gguf` file to load.

---

## Documentation

| Section | Description |
| --- | --- |
| [Architecture](docs/architecture.md) | Code-accurate system architecture and runtime behavior |
| [Runtime](docs/runtime.md) | Focused overview of the runtime loop, events, and approval flow |
| [Tools](docs/tools.md) | Current tool contract, registry model, and built-in tool behavior |
| [Sessions](docs/sessions.md) | Session storage, restore behavior, and persistence limits |
| [Setup](docs/setup.md) | Requirements, run/test commands, and config basics |
| [Benchmarks](docs/benchmarks.md) | Performance notes and measurements |
