# Repo Navigation

## Purpose

Use this skill to answer repository and code-navigation questions accurately and quickly.

This skill is for prompts like:
- What does this file do?
- What calls X?
- What uses X?
- Explain how X works
- Where is X configured?
- Can you see my project?

The goal is to behave like a useful coding agent:
- inspect the repo
- gather source evidence
- answer from real files and lines
- avoid guessing

## Use When

Use this skill when the user is asking about:
- repository structure
- file purpose
- symbol definitions
- call-sites
- usages
- config locations
- execution flow

Do not use this skill for:
- general chat
- creative writing
- non-technical conversation
- generic life advice

## Priorities

1. Prefer source files over tests, docs, examples, and generated files.
2. Prefer direct evidence over summaries.
3. Prefer a short correct answer over a long weak answer.
4. If evidence is insufficient, say so clearly instead of guessing.

## Relevant Files To Inspect First

For Rust CLI/TUI projects, start with:
- `Cargo.toml`
- `README.md`
- `src/main.rs`
- `src/lib.rs` if present

For deeper code questions, inspect likely runtime areas next:
- `src/inference/`
- `src/session/`
- `src/config/`
- `src/tools/`
- `src/tui/`

## Intent Rules

### Repo Overview
Answer:
- what the project is
- where startup happens
- what the main subsystems are

Do:
- cite real files
- keep the answer short and structural

Do not:
- dump dependency versions unless asked
- give generic “next steps” advice

### File Summary
Answer:
- what the file is responsible for
- the main top-level items in the file
- how it fits into the project

Do:
- stay anchored to the loaded file
- mention key structs, enums, functions, or modules

Do not:
- repeat only the first line of the file
- pivot into unrelated files unless the file clearly delegates elsewhere

### Caller Lookup
Answer:
- real non-definition source call-sites

Do:
- prefer true invocations
- include file:line references

Do not treat these as callers:
- string literals
- symbol-name helpers
- tests unless the user asked about tests
- docs or fixtures
- internal runtime helper files unless they are true production callers

### Usage Lookup
Answer:
- real non-definition source imports/usages/references

Do:
- include import lines if they are meaningful usages
- prefer production source files

Do not:
- return the symbol definition itself as the answer
- include tests unless asked

### Flow Trace
Answer:
- a short ordered explanation of what happens
- grounded in source files and source lines

Do:
- preserve branch behavior correctly
- connect the local function to the broader runtime path when source evidence exists

Do not:
- dump raw numbered code lines
- use tests or internal tool-loop files unless explicitly requested

### Config Lookup
Answer:
- where the setting is defined, merged, parsed, or checked

Do:
- find the actual field/access/use site
- mention the real config path or merge point

Do not:
- answer with nearby but irrelevant struct fields

## Workflow

1. Identify the question type.
2. Search for the relevant symbol, file, or config key.
3. Read the best source candidate files.
4. Filter out weak evidence:
   - tests
   - docs
   - fixtures
   - string-only matches
   - internal helper artifacts
5. Stop once enough source evidence exists.
6. Answer in concise grounded prose.

## Output Style

- Use short natural-language prose.
- Include `file:line` references for important facts.
- Be direct.
- Avoid hedging when the source evidence is clear.
- If the evidence is weak, say that briefly.

## Avoid

- raw tool tags
- code fences unless the user asked for code
- “let’s inspect” or “next we should”
- repeating the same answer on follow-up
- summarizing from tests when production source exists

## Good Answer Shape

Example:
`load_most_recent` is called from `src/inference/session/runtime.rs:166`, where the runtime checks whether it should resume the most recent saved session before continuing startup.

## Bad Answer Shape

Example:
- raw tool output
- numbered raw source lines with no explanation
- “I think this probably...”
- repeating the same one-line file summary twice
