# params-cli

> Sounds good — I'll pick up right where we left off when you're ready. The memory module files are written (mod.rs, compression.rs, index.rs, facts.rs), but the wiring into inference/mod.rs and main.rs still needs to happen.  

---

## Backends

| Backend | Speed | Cost | Requirements |
|---|---|---|---|
| `llama_cpp` | Slow startup, then fast | Free | `.gguf` model file |
| `ollama` | Fast (model stays loaded) | Free | Ollama installed |
| `anthropic` | Fast | Per token | API key |
| `openrouter` | Fast | Per token | API key |

Switch backends by editing `~/.params/config.toml`.

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

On first run, params creates `~/.params/config.toml`. Edit it to choose your backend.

**Option A — llama.cpp (local, offline)**
```bash
mkdir -p ~/.params/models
# Download qwen2.5-coder-7b-instruct-q4_k_m.gguf from:
# https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
mv ~/Downloads/qwen2.5-coder-7b-instruct-q4_k_m.gguf ~/.params/models/
```

**Option B — Ollama (recommended for speed)**
```bash
brew install ollama
ollama serve &
ollama pull qwen2.5-coder:7b
```
Then set `backend = "ollama"` in `~/.params/config.toml`.

**Option C — Anthropic API**
Set `backend = "anthropic"` and add your API key to config or set `ANTHROPIC_API_KEY` env var.

**5. Run**
```bash
cargo run --release
# or after installing:
params
```

---

## Configuration

`~/.params/config.toml` is created automatically on first run:

```toml
# Backend options: "llama_cpp", "ollama", "anthropic", "openrouter"
backend = "llama_cpp"

[llama_cpp]
# model_path = "/path/to/model.gguf"  # auto-detects from ~/.params/models/ if unset

[ollama]
url = "http://localhost:11434"  # or a LAN IP for a remote machine
model = "qwen2.5-coder:7b"

[anthropic]
# api_key = "sk-ant-..."  # or set ANTHROPIC_API_KEY env var
model = "claude-sonnet-4-5"

[generation]
max_tokens = 512
temperature = 0.8
```

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

---

## Project structure

```
src/
  main.rs            — CLI entry point, argument routing
  config.rs          — config loading, ~/.params/config.toml
  error.rs           — unified error type
  events.rs          — shared channel event types
  inference/
    mod.rs           — public API, persistent model thread
    backend.rs       — InferenceBackend trait
    llama_cpp.rs     — llama.cpp implementation
    ollama.rs        — Ollama HTTP implementation
    anthropic.rs     — Anthropic API (planned)
  tui/
    mod.rs           — Ratatui app, layout, event loop
    state.rs         — app state
```