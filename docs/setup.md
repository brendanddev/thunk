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

- discovers a config/storage root from the nearest `config.toml` (or the launch directory when absent)
- discovers the runtime project root from the nearest `.git` ancestor (or the launch directory as fallback)
- creates `data/` and `logs/` if needed
- builds the configured backend and tool registry
- opens or restores only the single most recently updated session from `data/sessions.db`, and restores it only when its stored `project_root` matches the current runtime project root

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
- `llm.provider = "openai"` uses the OpenAI backend and requires `OPENAI_API_KEY`.
- `llama_cpp.model_path` must point to a local `.gguf` file.
- Relative `model_path` values are resolved from the config root, not the runtime project root.

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

The checked-in repo config currently uses llama.cpp as the active provider and includes:

```toml
[app]
name = "thunk"

[ui]
show_activity = true

[llm]
provider = "llama_cpp"

[llama_cpp]
model_path = "data/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
gpu_layers = 0
context_tokens = 4096
batch_tokens = 2048
max_tokens = 512
temperature = 0.2
show_native_logs = false

[openai]
model = "gpt-4o-mini"
base_url = "https://api.openai.com/v1"
max_tokens = 512
temperature = 0.2
```

If that model is not present locally, either switch to `mock` or update `llama_cpp.model_path`.
