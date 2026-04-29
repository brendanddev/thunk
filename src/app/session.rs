use std::path::{Path, PathBuf};

use crate::llm::backend::{Message, Role};
use crate::runtime::ProjectRoot;
use crate::storage::session::{SavedSession, SessionId, SessionStore, StoredMessage};

use super::Result;

/// Owns the active database handle and current session ID.
/// Responsible for: opening/creating sessions, auto-saving, and the
/// explicit conversion between runtime Message types and stored records.
/// This is the only layer permitted to see both types simultaneously.
pub struct ActiveSession {
    store: SessionStore,
    session_id: SessionId,
    project_root: PathBuf,
}

impl ActiveSession {
    /// Opens the session database and returns the active session plus any
    /// previously stored messages to restore into the runtime. Returns an
    /// empty vec if no prior session exists.
    pub fn open_or_restore(
        db_path: &Path,
        project_root: &ProjectRoot,
    ) -> Result<(Self, Vec<Message>)> {
        let store = SessionStore::open(db_path)?;
        let current_root = project_root.path();
        let current_root_str = current_root.to_string_lossy();

        match store.load_most_recent()? {
            Some(saved)
                if saved.meta.project_root.as_deref() == Some(current_root_str.as_ref()) =>
            {
                let messages = from_stored(&saved);
                let session_id = saved.meta.id;
                Ok((
                    Self {
                        store,
                        session_id,
                        project_root: current_root.to_path_buf(),
                    },
                    messages,
                ))
            }
            Some(_) | None => {
                let meta = store.create(current_root)?;
                Ok((
                    Self {
                        store,
                        session_id: meta.id,
                        project_root: current_root.to_path_buf(),
                    },
                    vec![],
                ))
            }
        }
    }

    /// Persists the current conversation state. The caller provides the full
    /// runtime message list; system messages are stripped before storage.
    pub fn save(&self, runtime_messages: &[Message]) -> Result<()> {
        let stored = to_stored(runtime_messages);
        self.store.save(&self.session_id, &stored)?;
        Ok(())
    }

    /// Creates a new session and makes it the active one.
    /// Called when the user explicitly starts a fresh conversation.
    pub fn begin_new(&mut self) -> Result<()> {
        let meta = self.store.create(&self.project_root)?;
        self.session_id = meta.id;
        Ok(())
    }
}

// Conversion: runtime <--> storage
//
// System messages are excluded. The system prompt is reconstructed at runtime
// from config; storing it would create a stale copy that could diverge.

/// Maximum number of messages to inject into a fresh conversation on restore.
/// Prevents large accumulated histories from overflowing the model's context window.
const RESTORE_WINDOW: usize = 10;

/// Converts runtime messages to storable form, excluding system messages.
fn to_stored(messages: &[Message]) -> Vec<StoredMessage> {
    messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(|m| StoredMessage {
            role: m.role.as_str().to_string(),
            content: m.content.clone(),
        })
        .collect()
}

/// Converts stored messages back to runtime form, applying two rules:
///
/// 1. Window trim — only the most recent RESTORE_WINDOW messages are loaded.
///    Older history stays in the DB but is not injected into context.
///
/// 2. Tool exchange stripping — user messages that are runtime tool results or errors
///    are dropped entirely, along with the immediately preceding assistant message if
///    it was a pure tool call (starts with `[`). Raw file contents and directory
///    listings are never re-injected into the context window on restore. Full content
///    is preserved in storage; only context injection is affected.
///
///    Placeholders are intentionally not used: a placeholder that looks like a real
///    tool result causes the model to believe the exchange already completed, suppressing
///    fresh tool use when the user re-requests the same operation.
fn from_stored(session: &SavedSession) -> Vec<Message> {
    let total = session.messages.len();
    let start = total.saturating_sub(RESTORE_WINDOW);
    let slice = &session.messages[start..];
    let n = slice.len();

    let mut exclude = vec![false; n];
    for (i, m) in slice.iter().enumerate() {
        if m.role == "user" && is_tool_exchange(&m.content) {
            exclude[i] = true;
            // Drop the preceding assistant message too if it contains no conversational
            // text — only a bare tool call or fabricated result block. Without the result
            // it has no value and would leave an orphaned exchange in context.
            if i > 0 && slice[i - 1].role == "assistant" {
                let prev = slice[i - 1].content.trim_start();
                let is_bare_action = prev.starts_with('[')
                    || prev.starts_with("=== tool_result:")
                    || prev.starts_with("=== tool_error:");
                if is_bare_action {
                    exclude[i - 1] = true;
                }
            }
        }
    }

    slice
        .iter()
        .zip(exclude.iter())
        .filter(|(_, &ex)| !ex)
        .filter_map(|(m, _)| match m.role.as_str() {
            "user" => Some(Message::user(m.content.clone())),
            "assistant" => Some(Message::assistant(m.content.clone())),
            _ => None,
        })
        .collect()
}

/// Returns true when a user message is a tool result, tool error, or runtime correction
/// injected by the engine — none of which should be re-injected into a restored context.
fn is_tool_exchange(content: &str) -> bool {
    content.starts_with("=== tool_result:")
        || content.starts_with("=== tool_error:")
        || content.starts_with("[runtime:correction]")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::backend::Role;

    fn make_messages() -> Vec<Message> {
        vec![
            Message::system("system prompt"),
            Message::user("hello"),
            Message::assistant("hi there"),
        ]
    }

    #[test]
    fn to_stored_excludes_system_messages() {
        let stored = to_stored(&make_messages());
        assert_eq!(stored.len(), 2);
        assert!(stored.iter().all(|m| m.role != "system"));
    }

    #[test]
    fn to_stored_preserves_role_strings() {
        let stored = to_stored(&make_messages());
        assert_eq!(stored[0].role, "user");
        assert_eq!(stored[1].role, "assistant");
    }

    #[test]
    fn roundtrip_through_stored_messages() {
        use crate::storage::session::{SavedSession, SessionMeta};

        let original = make_messages();
        let stored = to_stored(&original);

        let saved = SavedSession {
            meta: SessionMeta {
                id: "test".into(),
                project_root: Some("/tmp/project".into()),
                created_at: 0,
                updated_at: 0,
                message_count: stored.len(),
            },
            messages: stored,
        };

        let restored = from_stored(&saved);
        assert_eq!(restored.len(), 2);
        assert_eq!(restored[0].role, Role::User);
        assert_eq!(restored[0].content, "hello");
        assert_eq!(restored[1].role, Role::Assistant);
        assert_eq!(restored[1].content, "hi there");
    }

    #[test]
    fn from_stored_trims_to_restore_window() {
        use crate::storage::session::{SavedSession, SessionMeta, StoredMessage};

        // Create 14 messages — more than RESTORE_WINDOW (10)
        let messages: Vec<StoredMessage> = (0..14)
            .map(|i| StoredMessage {
                role: if i % 2 == 0 { "user" } else { "assistant" }.into(),
                content: format!("msg {i}"),
            })
            .collect();

        let saved = SavedSession {
            meta: SessionMeta {
                id: "t".into(),
                project_root: Some("/tmp/project".into()),
                created_at: 0,
                updated_at: 0,
                message_count: 14,
            },
            messages,
        };

        let restored = from_stored(&saved);
        assert_eq!(restored.len(), RESTORE_WINDOW);
        // Should be the last 10 messages (indices 4–13)
        assert_eq!(restored[0].content, "msg 4");
        assert_eq!(restored[9].content, "msg 13");
    }

    #[test]
    fn from_stored_strips_tool_exchange_user_messages() {
        use crate::storage::session::{SavedSession, SessionMeta, StoredMessage};

        let tool_result =
            "=== tool_result: read_file ===\nsome file content\n=== /tool_result ===\n\n"
                .to_string();

        let saved = SavedSession {
            meta: SessionMeta {
                id: "t".into(),
                project_root: Some("/tmp/project".into()),
                created_at: 0,
                updated_at: 0,
                message_count: 1,
            },
            messages: vec![StoredMessage {
                role: "user".into(),
                content: tool_result,
            }],
        };

        let restored = from_stored(&saved);
        assert!(
            restored.is_empty(),
            "tool exchange messages must not be injected on restore"
        );
    }

    #[test]
    fn from_stored_strips_adjacent_pure_tool_call_assistant_message() {
        use crate::storage::session::{SavedSession, SessionMeta, StoredMessage};

        // A tool-assisted turn: user prompt → assistant tool call → user tool result
        let saved = SavedSession {
            meta: SessionMeta {
                id: "t".into(),
                project_root: Some("/tmp/project".into()),
                created_at: 0,
                updated_at: 0,
                message_count: 3,
            },
            messages: vec![
                StoredMessage {
                    role: "user".into(),
                    content: "read README.md".into(),
                },
                StoredMessage {
                    role: "assistant".into(),
                    content: "[read_file: README.md]".into(),
                },
                StoredMessage {
                    role: "user".into(),
                    content: "=== tool_result: read_file ===\ncontent\n=== /tool_result ===\n\n"
                        .into(),
                },
            ],
        };

        let restored = from_stored(&saved);
        // Only the original user prompt survives; the tool-call assistant and tool result are stripped
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].content, "read README.md");
    }

    #[test]
    fn from_stored_keeps_conversational_assistant_messages() {
        use crate::storage::session::{SavedSession, SessionMeta, StoredMessage};

        // An assistant message that starts with natural language is kept
        let saved = SavedSession {
            meta: SessionMeta {
                id: "t".into(),
                project_root: Some("/tmp/project".into()),
                created_at: 0,
                updated_at: 0,
                message_count: 2,
            },
            messages: vec![
                StoredMessage {
                    role: "user".into(),
                    content: "hello".into(),
                },
                StoredMessage {
                    role: "assistant".into(),
                    content: "Hi there! How can I help?".into(),
                },
            ],
        };

        let restored = from_stored(&saved);
        assert_eq!(restored.len(), 2);
        assert_eq!(restored[1].content, "Hi there! How can I help?");
    }

    #[test]
    fn from_stored_strips_fabrication_correction_messages() {
        use crate::storage::session::{SavedSession, SessionMeta, StoredMessage};

        let correction = "[runtime:correction] Your response contained a result block which is forbidden. \
            You must emit ONLY a tool call tag (e.g. [read_file: path]) or answer directly in plain text. \
            Output the tool call tag now, with no other text.".to_string();

        let saved = SavedSession {
            meta: SessionMeta {
                id: "t".into(),
                project_root: Some("/tmp/project".into()),
                created_at: 0,
                updated_at: 0,
                message_count: 3,
            },
            messages: vec![
                StoredMessage {
                    role: "user".into(),
                    content: "list the files".into(),
                },
                StoredMessage {
                    role: "assistant".into(),
                    content: "=== tool_result: list_dir ===\nfoo\n=== /tool_result ===".into(),
                },
                StoredMessage {
                    role: "user".into(),
                    content: correction,
                },
            ],
        };

        let restored = from_stored(&saved);
        // Original user message survives; correction and fabricated assistant message are stripped
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].content, "list the files");
    }

    #[test]
    fn from_stored_skips_unknown_roles() {
        use crate::storage::session::{SavedSession, SessionMeta, StoredMessage};

        let saved = SavedSession {
            meta: SessionMeta {
                id: "test".into(),
                project_root: Some("/tmp/project".into()),
                created_at: 0,
                updated_at: 0,
                message_count: 1,
            },
            messages: vec![StoredMessage {
                role: "unknown_role".into(),
                content: "some content".into(),
            }],
        };

        let restored = from_stored(&saved);
        assert!(restored.is_empty());
    }

    fn temp_project_root() -> tempfile::TempDir {
        tempfile::TempDir::new().unwrap()
    }

    fn canonical_project_root(dir: &tempfile::TempDir) -> ProjectRoot {
        ProjectRoot::new(dir.path().to_path_buf()).unwrap()
    }

    fn session_db_path(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("sessions.db")
    }

    #[test]
    fn open_or_restore_restores_session_when_project_root_matches() {
        let db_dir = tempfile::TempDir::new().unwrap();
        let root_dir = temp_project_root();
        let root = canonical_project_root(&root_dir);
        let db_path = session_db_path(&db_dir);

        let store = SessionStore::open(&db_path).unwrap();
        let meta = store.create(root.path()).unwrap();
        store
            .save(
                &meta.id,
                &[
                    StoredMessage {
                        role: "user".into(),
                        content: "hello".into(),
                    },
                    StoredMessage {
                        role: "assistant".into(),
                        content: "hi there".into(),
                    },
                ],
            )
            .unwrap();

        let (_session, history) = ActiveSession::open_or_restore(&db_path, &root).unwrap();

        assert_eq!(history.len(), 2);
        assert_eq!(history[0].content, "hello");
        assert_eq!(history[1].content, "hi there");
        assert_eq!(
            SessionStore::open(&db_path).unwrap().list().unwrap().len(),
            1
        );
    }

    #[test]
    fn open_or_restore_creates_new_session_when_project_root_differs() {
        let db_dir = tempfile::TempDir::new().unwrap();
        let original_root_dir = temp_project_root();
        let current_root_dir = temp_project_root();
        let original_root = canonical_project_root(&original_root_dir);
        let current_root = canonical_project_root(&current_root_dir);
        let db_path = session_db_path(&db_dir);

        let store = SessionStore::open(&db_path).unwrap();
        let original = store.create(original_root.path()).unwrap();
        store
            .save(
                &original.id,
                &[StoredMessage {
                    role: "user".into(),
                    content: "stale history".into(),
                }],
            )
            .unwrap();

        let (_session, history) = ActiveSession::open_or_restore(&db_path, &current_root).unwrap();

        assert!(history.is_empty());

        let store = SessionStore::open(&db_path).unwrap();
        let sessions = store.list().unwrap();
        assert_eq!(sessions.len(), 2);
        assert_ne!(sessions[0].id, original.id);
        assert_eq!(
            sessions[0].project_root.as_deref(),
            Some(current_root.path().to_string_lossy().as_ref())
        );
        assert_eq!(sessions[0].message_count, 0);
    }

    #[test]
    fn open_or_restore_creates_new_session_when_project_root_is_missing() {
        use rusqlite::Connection;

        let db_dir = tempfile::TempDir::new().unwrap();
        let root_dir = temp_project_root();
        let root = canonical_project_root(&root_dir);
        let db_path = session_db_path(&db_dir);

        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "
            CREATE TABLE sessions (
                id          TEXT PRIMARY KEY,
                created_at  INTEGER NOT NULL,
                updated_at  INTEGER NOT NULL,
                msg_count   INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE session_messages (
                session_id  TEXT NOT NULL,
                seq         INTEGER NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                PRIMARY KEY (session_id, seq)
            );

            CREATE INDEX idx_sessions_updated
                ON sessions(updated_at DESC);

            CREATE INDEX idx_session_messages_lookup
                ON session_messages(session_id, seq);

            PRAGMA user_version = 1;
            ",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO sessions (id, created_at, updated_at, msg_count)
             VALUES (?1, ?2, ?2, 1)",
            ("legacy", 1_i64),
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session_messages (session_id, seq, role, content)
             VALUES (?1, 0, ?2, ?3)",
            ("legacy", "user", "legacy history"),
        )
        .unwrap();
        drop(conn);

        let (_session, history) = ActiveSession::open_or_restore(&db_path, &root).unwrap();
        assert!(history.is_empty());

        let store = SessionStore::open(&db_path).unwrap();
        let legacy = store.load("legacy").unwrap().unwrap();
        assert_eq!(legacy.meta.project_root, None);

        let sessions = store.list().unwrap();
        assert_eq!(sessions.len(), 2);
        assert_eq!(
            sessions[0].project_root.as_deref(),
            Some(root.path().to_string_lossy().as_ref())
        );
        assert_eq!(sessions[0].message_count, 0);
    }
}
