# Setup

Provides instructions for setting up the development environment, running the app, and understanding the configuration.

---

## Requirements

- Rust stable
- An interactive terminal (`stdout` must be a TTY and `TERM` must not be `dumb`)
- A local `.gguf` model only if you use the `llama_cpp` backend

`rusqlite` is built with the `bundled` feature, so SQLite does not need to be installed separately.

---

## Run

From the project root:

```bash
cargo run
```

On startup the app:

- finds the project root by walking up to `config.toml`
- creates `data/` and `logs/` if needed
- opens or restores the most recent session from `data/sessions.db`

---

## Tests

```bash
cargo test
```

Most tests live inline in the Rust modules.

---

## Config Basics

Configuration lives in `config.toml`.

- `llm.provider = "mock"` uses the built-in mock backend.
- `llm.provider = "llama_cpp"` uses the local llama.cpp backend.
- `llama_cpp.model_path` must point to a local `.gguf` file.
- Relative `model_path` values are resolved from the project root.

This repo currently defaults to:

```toml
[llm]
provider = "llama_cpp"

[llama_cpp]
model_path = "data/models/qwen2.5-3b-instruct-q4_k_m.gguf"
```

If that model is not present locally, either switch to `mock` or update `llama_cpp.model_path`.
