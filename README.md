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

- Streaming custom-rendered TUI with a framebuffer/diff renderer, calmer terminal-native layout, a single-row runtime status bar with no divider chrome, a transient activity trace that appears near the prompt when the model is working, a shared left-margin transcript gutter that anchors conversation flow, responsive resize-aware layout, multiline input, slash commands, autocomplete, collapsible tool/context transcript rows, reverse history search, and inline prompt-adjacent approvals
- `llama_cpp`, `ollama`, and `openai_compat` backends
- Read-only tools: file read, directory listing, search, git, web fetch, Rust LSP diagnostics
- Tool-first repo/code navigation: repo-understanding and code-navigation questions now enter a bounded read-only search/read/list/git/LSP loop with visible `Thinking:` and step traces, and the final answer is model-written from observed tool output
- Mutating tools with approval: shell commands, targeted file edits, and whole-file writes with diff preview
- Policy sandbox and inspection: project-only read scope, richer approval previews, destructive shell blocking, and private-network fetch blocking
- Three-level memory: session compression, incremental project index maintenance, prompt-aware fact retrieval, selective prior-session recall, and cross-session facts with verified per-turn promotion, provenance tags, observability, deduplication, TTL pruning, and per-project cap
- Budget tracking, per-turn timing in the sidebar, reflection toggle, and eco mode
- Structured logging to `.local/params.log`
- Response caching for repeated generations: exact full-context hits, prompt-level fallback, and lightweight semantic reuse for plain chat turns, with TTL + project-change invalidation and `/clear-cache`
- Session continuity: conversation history auto-saved to `.local/sessions.db`, the most recently opened session auto-restores on startup, `--no-resume` starts fresh, and `/sessions list|new|rename|resume|delete|export` manages project-scoped saved sessions
- Project profiles: add `.params.toml` to any project directory to override backend, model, reflection, eco, LSP, and budget settings for that project
- Custom slash commands: load repo-local commands from `.local/commands.toml`

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

# start fresh without restoring the most recent saved session
params --no-resume
```

**Optional bootstrap helpers**
```bash
./scripts/setup.sh
./scripts/check.sh
```
`setup.sh` creates `.local/` subdirectories plus starter `.local/config.toml` and `.local/keys.env` files if they do not exist. `check.sh` verifies the common prerequisites and backend-specific basics without modifying anything.

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

[safety]
enabled = true
read_scope = "project_only"
block_private_network = true
inspect_network = true
shell_mode = "approve_inspect"
block_destructive_shell = true
shell_allowlist = []
shell_denylist = []
network_allowlist = []
inspect_cloud_requests = true
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

[safety]
block_private_network = false

[reflection]
enabled = true
```

When a profile is active, params shows `● ✓ profile: .params.toml` in the chat at startup.
You can commit `.params.toml` to share project settings with collaborators, or add it to `.gitignore` for personal overrides.

### Project indexing

`params index .` now works incrementally:
- indexes only changed or new indexable files
- removes stale rows for deleted files
- skips oversized files

During normal TUI use, params also maintains the project index in the background while idle for non-`llama_cpp` backends, refreshing one stale file at a time with the active backend instead of forcing a full re-index on startup.

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
- `Shift+Enter` — insert newline when supported by your terminal
- `Ctrl+J` — guaranteed newline fallback
- `Alt+Up / Alt+Down` — recall and edit previous submitted prompts/commands
- `Ctrl+R` — reverse-search submitted prompts/commands in the composer
- `Ctrl+K` — open the command launcher
- `Ctrl+O` — expand/collapse focused transcript context block
- `[` / `]` — move transcript focus when the input is empty
- `↑ ↓` — scroll one line
- `PageUp / PageDown` — scroll ten lines
- `Ctrl+Q` — quit
- multiline paste is preserved

**Useful slash commands:**
- `/read <path>`
- `/ls [path]`
- `/search <query>`
- `/git [status|diff|log]`
- `/fetch <url>`
- `/run <command>`
- `/write <path> <content>` with `\n` escapes for line breaks
- `/edit <path>` followed by a multiline `params-edit` fenced block
- `/reflect on|off|status`
- `/eco on|off|status`
- `/commands list`
- `/commands reload`
- `/sessions list`
- `/sessions new [name]`
- `/sessions rename <name>`
- `/sessions resume <name-or-id>`
- `/sessions delete <name-or-id>`
- `/sessions export <name-or-id> [markdown|json]`
- `/memory [status|facts|last|recall <query>|prune]`
- `/display [status|tokens <on|off>|time <on|off>]`
- `/transcript [status|collapse|expand|toggle]`
- `/clear-cache`

Safety behavior:
- `/read`, `/ls`, `/search`, and Rust LSP file lookups are restricted to the current project
- `/fetch` only allows explicit public `http://` and `https://` URLs and blocks localhost/private-network targets
- `/run`, `/write`, `/edit`, and model tool approvals render inline above the prompt as tighter terminal-native interrupts with one concise summary/reason line, clipped preview, and caret-style approve/reject shortcuts
- `/run` and model `[bash: ...]` calls remain approval-driven, show a policy summary, block clearly destructive commands, and can be further tightened with `shell_allowlist` / `shell_denylist`
- `/edit` and model `[edit_file: ...]` calls use exact `SEARCH/REPLACE` blocks and reject stale approvals if the file changed after proposal
- `network_allowlist` restricts `/fetch` and `openai_compat` provider destinations to exact hosts or subdomains you explicitly trust
- `inspect_cloud_requests = true` adds endpoint/payload inspection before `openai_compat` requests leave the machine

Session behavior:
- params resumes the most recently opened session for the current project by default
- `params --no-resume` starts a fresh unnamed session without deleting saved sessions
- session selectors accept either an exact real session name or a unique id prefix from `/sessions list`
- `/sessions list` now marks the current session clearly and shows compact selector-friendly ids inline
- exported session transcripts are written under `.local/exports/sessions/`

Memory behavior:
- repo/code-navigation questions now start with live tool exploration first; memory/index retrieval is secondary supporting context instead of the first move
- durable facts are promoted per turn from strict evidence instead of raw end-of-session transcript extraction
- verified facts now need a concrete project/workspace anchor (files, symbols, config values, commands, URLs/hosts, or approved tool evidence), and generic educational answer content, proposal-style lines, trivial code snippets, summary-style document boilerplate, or boilerplate file-description text are filtered out instead of being stored as durable memory
- each user turn now builds a retrieval bundle from indexed file summaries, prompt-relevant durable facts, and selective prior-session excerpts from saved sessions in the current project
- `/memory status` shows loaded fact count plus the most recent retrieval query and the last selected summaries, fact matches, and session excerpts
- `/memory facts` lists the currently loaded durable facts with `legacy` vs `verified` tags
- `/memory last` shows the latest accepted/skipped memory update, retrieval summary, and the most recent consolidation stats
- `/memory recall <query>` runs an explicit retrieval-only lookup across summaries, facts, and prior sessions without mutating the conversation
- `/memory prune` removes currently stored facts that no longer pass the project-specific durability filter, including stale generic explanations and low-value snippet/summary/boilerplate facts
- routine memory activity stays in runtime/status telemetry instead of injecting extra transcript breaks into assistant replies

Transcript behavior:
- obvious repo/directory summary prompts plus broader code-navigation asks such as `where is X implemented`, `trace how X works`, and `where is X configured` now trigger a generic read-only tool loop instead of a precomputed workflow answer
- the loop shows one concise `Thinking:` note plus short step traces, lets the model iteratively choose read-only tools like `search`, `read_file`, `list_dir`, `git`, and Rust LSP lookups, and answers from the observed tool output
- repo-local context files like `README.md` and `docs/context/CLAUDE.md` are available as support context for the loop, but they do not replace live filesystem inspection
- injected tool results and slash-loaded context blocks auto-collapse into compact transcript cards
- `Ctrl+O` toggles the focused collapsed/expanded context block
- `Alt+Up` recalls the previous submitted prompt or slash command into the composer for editing, and `Alt+Down` moves forward through recall history back to your unsent draft
- `Ctrl+K` opens a quiet command palette with built-in and custom slash commands, ranked search, aliases, and a usage/source detail block for the selected entry
- `/transcript expand` and `/transcript collapse` control all collapsible transcript blocks at once
- pending approvals render inline above the prompt; the approval kind label (`shell`, `write`, `edit`) is color-coded by risk level without redundant bracketed text, uses a single concise summary/reason line, and caps preview height more aggressively on short terminals
- the composer is bare and prompt-native: a mode-sensitive marker (`›`, `?`, `:`, `!`) with no idle placeholder or tutorial footer, so the transcript stays primary
- the runtime strip is a single status line; a transient activity trace appears just above the prompt (near the composer) while the model is working, so live status stays near where your eyes are
- transcript spacing is less uniform now: same-kind message runs stay tighter, while conversation shifts still breathe so the terminal reads more like a flowing transcript than stacked cards
- the transcript, system lines, activity trace, approvals, and prompt now share a left-margin gutter language (`│`, `·`, `!`, `›`) so the whole screen reads like one terminal conversation spine instead of separate rendered regions

Rendering behavior:
- the visible UI is now painted by a custom framebuffer renderer on top of Crossterm rather than Ratatui widgets
- each frame renders into an off-screen cell buffer, diffs against the previous frame, and writes only changed runs back to the terminal
- packed styles, symbol interning, transcript fragment caching, paced redraw scheduling, and explicit resize invalidation keep streaming smoother and reduce flicker
- status and telemetry are folded into a single-row top strip with no horizontal divider; identity, runtime state, and optional compact token/time readouts live together in the header, while transient activity appears near the prompt when active rather than in the header
- the renderer now uses a single layout model for the terminal surface; the old unused wide/compact distinction has been removed until a real visual split exists
- the terminal title and cursor shape now reflect the current mode more directly: normal compose, reverse search, command launcher, pending approval, and active generation each use distinct native terminal affordances

## Custom Slash Commands

Custom commands are loaded from:
- `.local/commands.toml` — repo-local, gitignored command definitions for this checkout

Resolution rules:
- built-in slash commands are reserved and cannot be overridden
- valid custom commands appear in slash autocomplete and `/commands list`

Schema:

```toml
[commands.review_auth]
description = "Load auth files, then ask for a focused review"
usage = "/review_auth <focus>"
steps = [
  { slash = "/read src/auth.rs" },
  { slash = "/search auth middleware" },
  { prompt = "Review the loaded auth context with focus on $@. Call out bugs first." }
]
```

Prompt-template example:

```toml
[commands.commit_msg]
description = "Draft a concise commit message"
usage = "/commit_msg <change summary>"
prompt = "Write a Conventional Commit message for this change: $@"
```

Workflow example:

```toml
[commands.rust_check]
description = "Run the local Rust verification flow"
usage = "/rust_check"
steps = [
  { slash = "/read Cargo.toml" },
  { slash = "/run cargo check" }
]
```

Notes:
- positional placeholders are `$1`, `$2`, and `$@`
- workflow steps can use built-in context commands, and may end with a single final `/run`, `/write`, or `/edit`
- nested custom commands are intentionally not supported in v1

## Current Stubs

These CLI surfaces exist but are still scaffolding, not finished features:
- `params compare`
- `params bench`
- `params train`

---

## Project structure

```
src/
  main.rs            — CLI entry point, argument routing
  commands.rs        — built-in slash metadata and custom command registry
  cache/             — exact response cache
  config.rs          — config facade and core config types
  config/            — config loading, .local helpers, project profile merge logic
  error.rs           — unified error type
  events.rs          — shared channel event types
  safety.rs          — policy sandbox and typed request inspection
  inference/
    mod.rs           — public facade, backend loading, session command API
    backend.rs       — InferenceBackend trait
    session.rs       — thin facade for the session runtime submodules
    session/
      runtime.rs     — model thread + live session lifecycle orchestration
      memory.rs      — session-memory retrieval/update helpers
      support.rs     — session persistence/list/export/reset helpers
      auto_inspect.rs — legacy hidden auto-inspection helpers and tests
    budget.rs        — budget/cache accounting helpers
    cache.rs         — exact/prompt/semantic cache helpers
    approval.rs      — approval flow helpers
    indexing.rs      — idle incremental indexing helpers
    reflection.rs    — hidden reflection helpers
    runtime.rs       — traces and generation/runtime helpers
    llama_cpp.rs     — llama.cpp implementation
    ollama.rs        — Ollama HTTP implementation
    openai_compat.rs — OpenAI-compatible API implementation
  tools/             — tool registry and built-in tools
  tools/lsp/         — rust-analyzer probing, transport, parsing, and formatting helpers
  memory/
    compression.rs   — session compression for long histories
    retrieval.rs     — shared retrieval scoring/normalization helpers
    index.rs         — project file summary index
    facts.rs         — cross-session fact store and consolidation
  tui/
    mod.rs           — terminal setup/cleanup and public TUI entrypoints
    app.rs           — event loop and keyboard handling
    commands.rs      — built-in/custom slash command dispatch
    renderer/        — custom framebuffer, layout, paint, and diff renderer
    format.rs        — display sanitization helpers
    state.rs         — thin facade for AppState submodules
    state/
      input.rs       — input editing, history, reverse search, autocomplete
      runtime.rs     — transcript, status, pending-action, and timer state updates
      helpers.rs     — transcript formatting and state support helpers
      tests.rs       — AppState regression coverage
```
