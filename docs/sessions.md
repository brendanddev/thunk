# Sessions

Describes the current session and persistence model: what is stored, how restore works, and where the key boundaries between runtime and storage live.

---

## Purpose

Sessions make the app durable across launches without coupling the runtime directly to SQLite.

The current design splits that work across two layers:

- `app/session.rs` owns the bridge between runtime messages and stored messages
- `storage/session/` owns SQLite schema and CRUD

`AppContext` uses those pieces to inspect the single most recently updated saved session at startup, restore it only when its stored `project_root` matches the current runtime project root, and save conversation state after completed submit, approve, and reject requests.

---

## Main Components

### `ActiveSession`

`ActiveSession` in `src/app/session.rs` owns:

- the active `SessionStore`
- the current session ID
- conversion between runtime `Message` values and stored message rows

This is the only place that sees both runtime message types and storage message types.

### `SessionStore`

`SessionStore` in `src/storage/session/store.rs` owns the SQLite database handle and the basic session CRUD operations.

### Stored Types

The storage layer defines:

- `SessionId`
- `SessionMeta`
- `StoredMessage`
- `SavedSession`

Storage uses plain strings for roles so it stays decoupled from the runtime’s typed `Role` enum.

---

## Database Layout

The session database lives at `data/sessions.db`.

Current schema:

- `sessions`
  - `id`
  - `project_root`
  - `created_at`
  - `updated_at`
  - `msg_count`
- `session_messages`
  - `session_id`
  - `seq`
  - `role`
  - `content`

Messages are stored in order by `(session_id, seq)`.

`SessionStore::save()` replaces the stored message set for a session instead of appending deltas.

---

## What Gets Stored

The runtime provides the full in-memory conversation to `ActiveSession::save()`, but `ActiveSession` filters it before writing.

Stored:

- user messages
- assistant messages
- runtime-injected tool result and tool error blocks, because they are stored as user messages

Not stored:

- system messages
- TUI-only status/system lines
- pending approval state

The system prompt is intentionally not persisted. It is rebuilt from current config and tool specs on each startup.

---

## Startup And Restore

At startup:

1. `app::run()` opens the session DB
2. `ActiveSession::open_or_restore()` asks `SessionStore` for the single most recently updated session overall
3. if that session's stored `project_root` exactly matches the current canonical project root, stored messages are converted back into runtime messages
4. if that single most recent session has a missing or different `project_root`, restore does not continue scanning older sessions; a new empty session is created instead
5. if no prior session exists, a new empty session is created
6. `AppContext::build()` loads the restored history into the runtime after creating a fresh system prompt

Restore is intentionally narrower than storage.

### Restore Window

Only the most recent `10` stored messages are injected back into the runtime.

Older messages stay in SQLite, but they are not reloaded into the live model context.

### Tool Exchange Stripping

Restore also strips runtime-generated tool exchanges:

- user messages starting with `=== tool_result:`, `=== tool_error:`, or `[runtime:correction]` are dropped
- if one of those was immediately preceded by a pure assistant tool call that starts with `[`, that assistant message is dropped too

This keeps raw file contents, directory listings, and other tool outputs out of restored context while preserving the full transcript on disk.

### Current UI Gap

Restored history is loaded into the runtime, but it is not replayed into `AppState`.

That means:

- the model has restored context after startup
- the visible TUI transcript still starts from the fresh welcome message

---

## Saving Behavior

`AppContext::handle()` auto-saves after:

- `RuntimeRequest::Submit`
- `RuntimeRequest::Approve`
- `RuntimeRequest::Reject`

This means normal prompt turns and approval decisions are persisted after the runtime finishes handling them. Pending approvals are still in-memory only until the user approves or rejects.

---

## Reset Behavior

`/clear` eventually calls `AppContext::reset()`.

That does two things:

- resets the runtime conversation back to the system prompt
- creates a brand new session ID in storage

The old session remains in SQLite; reset does not delete prior sessions.

---

## IDs And Ordering

Session IDs are generated as 16-character lowercase hex strings.

Sessions are considered for restore by `updated_at` descending, and the app only resumes the most recently updated saved session when its stored `project_root` exactly matches the current canonical project root.
The docs intentionally treat those timestamp fields as opaque stored ordering values rather than promising a specific unit.

Messages within a session are stored and loaded in ascending `seq` order.

---

## Current Limitations

- Only the single most recently updated session is considered for automatic restore, and it is restored only when its stored `project_root` matches the current runtime project root.
- Pending approvals are not persisted.
- Restore uses a fixed message window rather than token-aware budgeting.
- The full stored transcript can be larger than the context reloaded into the runtime.
- The TUI does not yet show restored transcript history on startup.
