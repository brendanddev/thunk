# Commands

This file is a practical reference for both the top-level CLI and the built-in slash commands inside the TUI.

---

## Top-Level CLI

### Interactive TUI

```bash
cargo run --release --
```

If you install the project locally:

```bash
cargo install --path .
```

Important note:
- the clap command name is `params`
- the Cargo package name is currently `params-cli`
- unless an explicit Cargo binary name is added, your installed executable will usually be `params-cli`
- the examples below use `cargo run --release -- ...` for consistency; if you install locally, substitute `params-cli ...`

### One-Shot Prompt

```bash
cargo run --release -- "explain what this function does"
```

### Start Fresh Without Resume

```bash
cargo run --release -- --no-resume
```

### Index Command

```bash
cargo run --release -- index .
```

### LSP Check

```bash
cargo run --release -- lsp-check
```

### Current Stubs

These exist in the CLI surface but are not implemented yet:

- `pull`
- `compare`
- `bench`
- `train`

---

## Built-In Slash Commands

### Context / Read-Only

- `/read <path>`: load a file into context
- `/ls [path]`: list a directory inside the project
- `/search <query>`: search source files inside the project
- `/git [status|diff|log]`: load read-only git context
- `/diag <file>`: Rust LSP diagnostics
- `/hover <file>:<line>:<col>`: Rust LSP hover
- `/def <file>:<line>:<col>`: Rust LSP definition
- `/lcheck`: check local rust-analyzer setup
- `/fetch <url>`: fetch a public webpage into context

### Mutating / Approval-Gated

- `/run <command>`: propose a shell command for approval
- `/write <path> <content>`: propose a whole-file write for approval
- `/edit <path>` followed by a `params-edit` block: propose a targeted edit

### Session / Runtime Control

- `/reflect <on|off|status>`: toggle reflection mode
- `/eco <on|off|status>`: toggle eco mode
- `/debug-log <on|off|status>`: toggle separate content debug logging
- `/approve`: approve the pending action
- `/reject`: reject the pending action
- `/clear`: clear the current conversation and active saved session
- `/clear-cache`: clear the exact response cache
- `/clear-debug-log`: clear the separate content debug log

### Discovery

- `/help`: show built-in command help
- `/commands [list|reload]`: inspect or reload custom slash commands

### Sessions

- `/sessions list`
- `/sessions new [name]`
- `/sessions rename <name>`
- `/sessions resume <name-or-id>`
- `/sessions delete <name-or-id>`
- `/sessions export <name-or-id> [markdown|json]`

### Memory

- `/memory status`
- `/memory facts`
- `/memory last`
- `/memory recall <query>`
- `/memory prune`

### Display / Transcript

- `/display [status|tokens <on|off>|time <on|off>]`
- `/transcript [status|collapse|expand|toggle]`

## Common Usage Patterns

### Inspect a file

```text
/read src/main.rs
What does this file do?
```

### Search the repo

```text
/search load_most_recent
What calls load_most_recent?
```

### Inspect git state

```text
/git status
/git diff
```

### Run a command with approval

```text
/run cargo test
```

### Edit a file with approval

````text
/edit src/main.rs
```params-edit
<<<<<<< SEARCH
old text
=======
new text
>>>>>>> REPLACE
```
````

---

## Implementation Map

The command system is currently split like this:

- `src/commands.rs`: built-in command metadata and custom command registry/loading
- `src/tui/commands.rs`: slash-command dispatch facade
- `src/tui/commands/parse.rs`: argument/body parsing helpers
- `src/tui/commands/display.rs`: status/help/memory formatting helpers

Custom slash commands still live in `.local/commands.toml`.

---

## Keyboard Shortcuts

Current high-value shortcuts:

- `Enter`: send message
- `Shift+Enter`: insert newline when the terminal supports it
- `Ctrl+J`: guaranteed newline fallback
- `Ctrl+R`: reverse-search previous submissions
- `Ctrl+K`: open the command launcher
- `Ctrl+O`: expand/collapse focused transcript context block
- `Ctrl+Q`: quit

---

## Notes

- Technical repo/code/file questions increasingly prefer the read-only tool loop over plain chat.
- Mutating actions should always be expected to go through approval.
- Custom slash commands live in `.local/commands.toml`.
