# Architecture

`params-cli` is a local-first coding assistant CLI with a terminal UI, multiple inference backends, read-only and mutating tools, session persistence, and layered memory.

This document is a snapshot of the current architecture. It is meant to explain how the app is wired today, not to promise a finished long-term design.

## Goals

- Keep day-to-day coding workflows local-first when possible
- Support multiple backends behind one runtime
- Use real tools for repo understanding instead of relying only on prompt context
- Preserve session continuity and memory without letting long threads collapse into noise
- Keep risky actions approval-gated and observable

## Top-Level Runtime

At a high level, a normal interactive turn looks like this:

1. `src/main.rs` parses CLI arguments and launches either one-shot mode or the TUI.
2. `src/tui/` owns input handling, transcript state, rendering, command launcher behavior, approvals, and activity traces.
3. `src/inference/session/runtime.rs` runs the long-lived session loop:
   - loads config and backend
   - manages session restore/save
   - routes user prompts
   - triggers tool-first repo investigation for technical questions
   - falls back to normal generation for non-technical chat
4. `src/inference/tool_loop.rs` runs bounded read-only repo investigations.
5. `src/tools/` executes file, search, git, web, LSP, shell, write, and edit tools.
6. `src/memory/` and `src/session/` provide compression, index, durable facts, and saved conversation persistence.

## Main Subsystems

### CLI Entry

- `src/main.rs`
- `src/config.rs`

Responsibilities:
- parse arguments
- load config and profile overrides
- choose TUI vs one-shot mode
- initialize logging

Important note:
- the clap command name is `params`, but the Cargo package name is currently `params-cli`
- unless an explicit Cargo binary name is added, `cargo install --path .` will usually install `params-cli`

### Session Runtime

- `src/inference/session/runtime.rs`
- `src/inference/session/investigation.rs`
- `src/inference/session/memory.rs`

Responsibilities:
- own the main interactive session loop
- classify turns into technical repo investigation vs normal chat
- maintain recent investigation anchors like loaded files and search queries
- compress long history
- emit UI events and activity traces

The current design goal is:
- technical/repo/file/code-understanding prompts should prefer the read-only tool loop
- clearly non-technical turns can use the normal chat path

### Read-Only Tool Loop

- `src/inference/tool_loop.rs`
- `src/inference/tool_loop/evidence.rs`
- `src/inference/tool_loop/intent.rs`
- `src/inference/tool_loop/prompting.rs`
- `src/inference/tool_loop/parse.rs`

Responsibilities:
- detect repo-navigation intent
- bootstrap searches/reads/listings
- run bounded read-only tool iterations
- extract structured evidence from tool results
- stop when enough evidence exists
- produce a grounded final answer

Current intent families:
- repo overview
- directory overview
- implementation lookup
- config lookup
- call-site lookup
- usage lookup
- flow trace

This subsystem is where most answer-quality work is happening right now.

### Inference Backends

- `src/inference/llama_cpp.rs`
- `src/inference/ollama.rs`
- `src/inference/openai_compat.rs`
- `src/inference/cache.rs`
- `src/inference/reflection.rs`

Responsibilities:
- provide a unified generation interface
- support streaming tokens
- manage exact/prompt-level cache reuse
- optionally run reflection on non-eco turns

Backends currently supported:
- `llama_cpp`
- `ollama`
- `openai_compat`

### Terminal UI

- `src/tui/app.rs`
- `src/tui/state/`
- `src/tui/renderer/`
- `src/tui/commands.rs`

Responsibilities:
- terminal event loop
- transcript model and collapse state
- prompt editing, multiline input, history recall, reverse search
- command launcher and autocomplete
- inline approval cards
- single-row top bar and transient activity trace

### Tools

- `src/tools/fs.rs`
- `src/tools/search.rs`
- `src/tools/lsp/`
- `src/tools/web.rs`
- `src/tools/write.rs`
- `src/tools/edit.rs`
- `src/tools/mod.rs`

Tool classes:
- read-only context tools
- mutating tools with approval
- diagnostics / LSP helpers

Important safety boundary:
- technical repo investigation uses only read-only tools
- mutating tools remain approval-driven

### Memory and Persistence

- `src/memory/compression.rs`
- `src/memory/index.rs`
- `src/memory/facts.rs`
- `src/session/mod.rs`

Current layers:
- compressed recent session context
- project file summary index
- durable cross-session facts
- saved session history in SQLite

Stored state currently lives under `.local/`.

## Technical Turn Flow

For a repo-understanding prompt like `What calls load_most_recent`:

1. Session runtime receives the user turn.
2. Investigation routing classifies it as a technical turn.
3. The read-only tool loop chooses an intent such as `CallSiteLookup`.
4. The loop bootstraps a search or anchored read.
5. Tool results are parsed into structured evidence candidates.
6. If evidence is insufficient, the loop asks for one more targeted read.
7. If evidence is sufficient, the loop returns a grounded answer.
8. The session runtime records the tool results, updates investigation state, and saves the session.

## Safety Model

Read-only investigation:
- file reads are project-scoped
- directory listing is project-scoped
- fetch blocks localhost/private targets by default
- tool loop is read-only only

Mutating actions:
- shell, write, and edit actions are approval-gated
- destructive shell commands are blocked by policy
- stale edit approvals are rejected if the file changes

## Current Architectural Strengths

- Clear separation between TUI, inference runtime, tools, memory, and session persistence
- Good test coverage around tool-loop regressions
- Local-first workflows supported across multiple backends
- Structured session compression and investigation-state tracking

## Current Architectural Pressure Points

- `src/inference/tool_loop/evidence.rs` is still dense and central to answer quality
- the tool loop is the hardest subsystem to keep both fast and accurate
- some older hidden auto-inspection code still exists for reference/legacy coverage in `src/inference/session/auto_inspect.rs`
- benchmark coverage and live eval tooling still need to mature

## Key Files To Read First

If you are new to the codebase, start here:

1. `src/main.rs`
2. `src/inference/session/runtime.rs`
3. `src/inference/session/investigation.rs`
4. `src/inference/tool_loop.rs`
5. `src/inference/tool_loop/evidence.rs`
6. `src/tui/app.rs`
7. `src/tui/state/`
8. `src/tools/mod.rs`

## Near-Term Direction

The near-term quality focus is:
- reliable repo navigation
- stronger grounded answers
- better benchmark coverage
- tighter latency and stop behavior for harder multi-file questions
