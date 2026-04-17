# Setup

## Requirements

- Rust stable toolchain
- An interactive terminal (`stdout` must be a TTY and `TERM` cannot be `dumb`)
- A local GGUF model only if you want to use the `llama_cpp` backend

SQLite does not need to be installed separately; `rusqlite` is built with the bundled feature.

## Run

From the project root:

```bash
cargo run
```

At startup the app will:

- discover the project root from `config.toml`
- create `data/` and `logs/` if they do not exist
- open or restore the most recent session from `data/sessions.db`

## Tests

Run the test suite with:

```bash
cargo test
```

Most tests live inline inside the Rust modules rather than under `tests/`.

## Configuration Notes

Configuration is read from `config.toml`.

Relevant keys:

- `llm.provider = "mock"` uses the built-in mock backend.
- `llm.provider = "llama_cpp"` uses the local llama.cpp backend.
- `llama_cpp.model_path` should point to a local `.gguf` model file.
- Relative model paths are resolved from the project root.

The current default config in this repo is set to `llama_cpp` and points at:

```text
data/models/qwen2.5-3b-instruct-q4_k_m.gguf
```

If you do not have that model locally, switch `llm.provider` to `mock` or update `llama_cpp.model_path` to a model that exists on your machine.
