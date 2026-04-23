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
- builds the configured backend and the default tool registry
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

Code defaults are intentionally conservative. If `config.toml` is empty or a field is omitted, the current built-in defaults are:

```toml
[llm]
provider = "mock"

[llama_cpp]
# model_path is unset by default
gpu_layers = 0
context_tokens = 2048
batch_tokens = 256
max_tokens = 512
temperature = 0.7
show_native_logs = false
```

The checked-in repo config currently uses llama.cpp instead:

```toml
[llm]
provider = "llama_cpp"

[llama_cpp]
model_path = "data/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf"
gpu_layers = 0
context_tokens = 8192
batch_tokens = 2048
max_tokens = 512
temperature = 0.3
show_native_logs = false
```

If that model is not present locally, either switch to `mock` or update `llama_cpp.model_path`.
