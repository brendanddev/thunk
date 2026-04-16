# params-cli

Personal AI coding assistant CLI focused on local-first workflows, modular backends, privacy, and real coding actions.

> Version 0.8.5 - Focused rebuild

---

## Overview

`params-cli` is a Rust-based AI coding agent designed to be:

- local-first
- tool-using
- modular and extensible
- usable for real development workflows

It is not a simple chatbot wrapper. The goal is to build a durable coding agent runtime that can evolve over time without constant rewrites.

The project is currently being rebuilt with a strong focus on:

- clear architectural boundaries
- low coupling between subsystems
- explicit runtime behavior
- long-term maintainability

---

## Requirements

- Rust (stable)
- An interactive terminal

---

## Project structure

```
params-cli/
├── README.md
├── Cargo.toml
├── config.toml
├── data/
├── docs/           # Documentation and design notes
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

## Documentation
| Section | Description |
|---------|-------------|
| [Architecture](docs/architecture.md) | Overview of the system's modular design and components. |
| [Setup](docs/setup.md) | Instructions for installing Rust and running the application. |