# params-cli

Personal AI coding assistant CLI focused on local-first workflows, modular backends, privacy, and real coding actions.

## Backends

| Backend | Speed | Cost | Requirements |
|---|---|---|---|
| `llama_cpp` | Slow startup, then fast | Free | `.gguf` model file |
| `ollama` | Fast (model stays loaded) | Free | Ollama installed |
| `openai_compat` | Fast | Per token | OpenAI-compatible API key |

Switch backends by editing `.local/config.toml`.

## What Works Today

- Streaming Ratatui TUI with slash commands and approval flow
- `llama_cpp`, `ollama`, and `openai_compat` backends
- Read-only tools: file read, directory listing, search, git, web fetch, Rust LSP diagnostics
- Mutating tools with approval: shell commands and whole-file writes with diff preview
- Three-level memory: session compression, project index, cross-session facts with quality filtering, deduplication, TTL pruning, and per-project cap
- Budget tracking, per-turn timing in the sidebar, reflection toggle, and eco mode
- Structured logging to `.local/params.log`
- Response caching for repeated generations: exact full-context hits, prompt-level fallback, and lightweight semantic reuse for plain chat turns, with TTL + project-change invalidation and `/clear-cache`
- Session persistence: conversation history auto-saved to `.local/sessions.db` and restored on the next startup; `/clear` starts a fresh session
- Project profiles: add `.params.toml` to any project directory to override backend, model, reflection, eco, LSP, and budget settings for that project

---

## Requirements

- Rust (stable)
- macOS Apple Silicon, Linux x86_64, or Windows (WSL)
- `cmake` (for llama.cpp backend)
- A `.gguf` model file, Ollama, or an API key depending on backend

---

## Setup

**1. Install Rust**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**2. Install cmake (macOS, required for llama.cpp backend)**
```bash
brew install cmake
```

**3. Clone and build**
```bash
git clone https://github.com/brendanddev/params-cli
cd params-cli
cargo build --release
```

**4. Configure a backend**

On first run, params creates `.local/config.toml`. Edit it to choose your backend.

**Option A — llama.cpp (local, offline)**
```bash
mkdir -p .local/models
# Download qwen2.5-coder-7b-instruct-q4_k_m.gguf from:
# https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
mv ~/Downloads/qwen2.5-coder-7b-instruct-q4_k_m.gguf .local/models/
```

**Option B — Ollama (recommended for speed)**
```bash
brew install ollama
ollama serve &
ollama pull qwen2.5-coder:7b
```
Then set `backend = "ollama"` in `.local/config.toml`.

**Option C — OpenAI-compatible API**
Set `backend = "openai_compat"` and configure `[openai_compat]` for Groq, OpenAI, OpenRouter, or xAI.

**5. Run**
```bash
cargo run --release
# or after installing:
params
```

---

## Configuration

Config resolution order (highest precedence first):
1. `.params.toml` in the current working directory (project-local profile)
2. `.local/config.toml` (global config, created on first run)
3. Compiled-in defaults

### Global config — `.local/config.toml`

```toml
# Backend options: "llama_cpp", "ollama", "openai_compat"
backend = "llama_cpp"

[llama_cpp]
# model_path = "/path/to/model.gguf"  # auto-detects from .local/models/ if unset

[ollama]
url = "http://localhost:11434"  # or a LAN IP for a remote machine
model = "qwen2.5-coder:7b"

[openai_compat]
url = "https://api.groq.com/openai/v1"
# api_key = ""  # or set GROQ_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY / XAI_API_KEY in .local/keys.env
model = "llama-3.3-70b-versatile"

[generation]
max_tokens = 512
temperature = 0.8

[cache]
ttl_seconds = 21600
# set to 0 to disable TTL expiration

[reflection]
enabled = false

[eco]
enabled = false

[memory]
fact_ttl_days = 90      # days before a cross-session fact is pruned; 0 disables TTL
max_facts_per_project = 150  # per-project cap; oldest facts removed first when exceeded
```

### Project profile — `.params.toml`

Drop a `.params.toml` in any project root to override specific settings for that project. Only the fields you set are changed; everything else comes from the global config.

```toml
# Example: use a cloud model + higher token limit for a large project
backend = "openai_compat"

[openai_compat]
model = "gpt-4o"

[generation]
max_tokens = 2048

[cache]
ttl_seconds = 3600

[memory]
fact_ttl_days = 30
max_facts_per_project = 200

[reflection]
enabled = true
```

When a profile is active, params shows `● ✓ profile: .params.toml` in the chat at startup.
You can commit `.params.toml` to share project settings with collaborators, or add it to `.gitignore` for personal overrides.

---

## Usage

```bash
# Open the TUI
params

# One-shot prompt
params "explain what this function does"
```

**TUI keybindings:**
- `Enter` — send message
- `↑ ↓` — scroll one line
- `PageUp / PageDown` — scroll ten lines
- `Ctrl+Q` — quit

**Useful slash commands:**
- `/read <path>`
- `/ls [path]`
- `/search <query>`
- `/git [status|diff|log]`
- `/fetch <url>`
- `/run <command>`
- `/reflect on|off|status`
- `/eco on|off|status`
- `/clear-cache`

---

## Project structure

```
src/
  main.rs            — CLI entry point, argument routing
  cache/             — exact response cache
  config.rs          — config loading, .local/config.toml
  error.rs           — unified error type
  events.rs          — shared channel event types
  inference/
    mod.rs           — public API, persistent model thread
    backend.rs       — InferenceBackend trait
    llama_cpp.rs     — llama.cpp implementation
    ollama.rs        — Ollama HTTP implementation
    openai_compat.rs — OpenAI-compatible API implementation
  tools/             — tool registry and built-in tools
  memory/            — compression, project index, cross-session facts
  tui/
    mod.rs           — Ratatui app, layout, event loop
    state.rs         — app state
```
