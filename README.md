# params-cli

Personal AI coding assistant CLI focused on local-first workflows, modular architecture, privacy, and real coding actions.

> Version 0.8.7 — Focused rebuild

---

## Overview

`params-cli` is a Rust-based AI coding agent designed to be:

- local-first  
- tool-using  
- modular and extensible  
- usable for real development workflows  

This is not a simple chatbot wrapper.

The goal is to build a **durable coding agent runtime** that can evolve over time without constant rewrites.

The rebuild focuses on:

- clear architectural boundaries  
- low coupling between subsystems  
- explicit runtime behavior  
- long-term maintainability  

---

## Current Status

The project is mid-rebuild and developed in controlled phases.

### What is implemented

- explicit runtime loop with tool dispatch  
- typed tool interface (`ToolInput` / `ToolOutput`)  
- centralized tool protocol (`runtime/tool_codec.rs`)  
- session persistence via app layer (`AppContext`, `ActiveSession`)  
- command parsing isolated in `tui/commands/`  
- project-root-aware tools via `ToolContext`  
- consistent path resolution (no reliance on CWD)  
- model-aware system prompt with project context  
- clean separation between app, runtime, tools, storage, and UI  
- test coverage across boundaries  

### What is intentionally not built yet

- write/mutating tools (edit/write)  
- approval/rejection flow  
- bash execution  
- structured tool outputs  
- logging/observability  
- memory system  
- LSP integration  
- web fetch  
- advanced session UX  

---

## Architecture

The system follows strict layer separation:

- `app/` — orchestration (runtime + sessions + config)  
- `runtime/` — execution loop, conversation, tool protocol  
- `llm/` — backend abstraction and model providers  
- `storage/` — persistence (sessions and data)  
- `tools/` — tool registry and implementations  
- `tui/` — terminal UI, rendering, input, commands  

### Design principles

- explicit over implicit  
- no hidden coupling  
- no text-as-API (long-term goal)  
- modular boundaries  
- local-first execution  
- built to evolve without rewrites  

---

## Project Structure

```
params-cli/
├── README.md
├── Cargo.toml
├── config.toml
├── data/
├── docs/
├── logs/
├── src/
│   ├── app/        # orchestration, session coordination
│   ├── llm/        # backend abstraction and providers
│   ├── runtime/    # engine, conversation, tool codec
│   ├── storage/    # persistence layer
│   ├── tools/      # tool registry and implementations
│   ├── tui/        # UI, rendering, commands
│   ├── lib.rs
│   └── main.rs
└── tests/
```

---

## Running the project

### Requirements

- Rust (stable)
- Interactive terminal

### Run

```
cargo run
```

---

## Documentation

| Section | Description |
|--------|------------|
| [Architecture](docs/architecture.md) | System design and module boundaries |
| [Setup](docs/setup.md) | Setup instructions |
