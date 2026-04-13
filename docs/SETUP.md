# Setup

This file is the practical setup guide for running and developing `params-cli`.

---

## Prerequisites

Required:
- Rust stable
- a supported backend

Useful / sometimes required:
- `cmake` for `llama_cpp`
- `rust-analyzer` for the LSP features
- Ollama if you want the `ollama` backend

---

## Clone and Build

```bash
git clone https://github.com/brendanddev/params-cli
cd params-cli
cargo build --release
```

---

## Local Helper Scripts

The repo includes:

```bash
./scripts/setup.sh
./scripts/check.sh
```

What they do:
- `setup.sh`
  - creates `.local/` directories
  - seeds starter config files if missing
- `check.sh`
  - verifies common prerequisites
  - checks backend-specific basics
  - does not modify the repo

---

## Running the App

### TUI

```bash
cargo run --release --
```

### Fresh Session

```bash
cargo run --release -- --no-resume
```

### One-Shot Prompt

```bash
cargo run --release -- "explain what this function does"
```

---

## Installing the Binary

```bash
cargo install --path .
```

Important note:
- the CLI command name shown in help text is `params`
- the installed Cargo binary will usually be named after the package, which is currently `params-cli`

If you want a shell shortcut today, add an alias, or add an explicit Cargo binary name later.

The repo also ships tracked built-in skill guidance under `skills/`. Those files are part of the project and are separate from `.local/`, which is still reserved for machine-local config, cache, logs, and state.

---

## Backend Setup

### Option A: `llama_cpp`

Pros:
- local
- private
- offline-capable

Cons:
- slower startup
- more sensitive to context size and prompt weight

Setup:

```bash
mkdir -p .local/models
```

Place a `.gguf` model in `.local/models/`, then configure `.local/config.toml`.

### Option B: `ollama`

Pros:
- fast local serving
- simple setup

Setup:

```bash
ollama serve
ollama pull qwen2.5-coder:7b
```

Then set:

```toml
backend = "ollama"
```

### Option C: `openai_compat`

Pros:
- fast
- easiest way to compare local behavior against a stronger hosted model

Setup:
- configure `[openai_compat]` in `.local/config.toml`
- set the relevant API key in `.local/keys.env` or your environment

---

## Config Files

Current config layers:

1. `.params.toml` in the project root
2. `.local/config.toml`
3. built-in defaults

Use:
- `.local/config.toml` for your base environment
- `.params.toml` for per-project overrides

---

## First-Run Checklist

After setup, validate these basics:

1. start the TUI successfully
   - `cargo run --release --`
   - or `params-cli` if installed
2. confirm the selected backend responds
3. run `/read src/main.rs`
4. ask `What does this file do?`
5. run `/git status`
6. run `/memory status`
7. run `/sessions list`

---

## Recommended Dev Workflow

For routine development:

```bash
cargo fmt
cargo test -- --nocapture
```

For repo-navigation work specifically:
- run the focused tool-loop tests
- then do a manual benchmark pass using prompts from `docs/BENCHMARKS.md`

---

## Troubleshooting

### `params` or `params-cli` command not found

Use:

```bash
cargo run --release --
```

If you installed with Cargo, verify which binary name was installed.

### `llama_cpp` feels slow or unstable

Try:
- `eco` mode
- a smaller prompt
- fewer loaded context files
- switching to `ollama` or `openai_compat` when debugging runtime behavior

### LSP commands do not work

Check:
- `rust-analyzer` is installed
- `/lcheck` output
- the file path and line/column format you passed

### Repo answers seem wrong

Check:
- the backend in use
- whether the answer streamed or fell back strangely
- whether raw tool tags leaked
- whether tests/internal files appeared

Then compare against the benchmark cases in `docs/BENCHMARKS.md`.
