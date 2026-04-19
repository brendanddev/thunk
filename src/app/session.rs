use std::path::Path;

use crate::llm::backend::{Message, Role};
use crate::storage::session::{SavedSession, SessionId, SessionStore, StoredMessage};

use super::Result;

/// Owns the active database handle and current session ID.
/// Responsible for: opening/creating sessions, auto-saving, and the
/// explicit conversion between runtime Message types and stored records.
/// This is the only layer permitted to see both types simultaneously.
pub struct ActiveSession {
    store: SessionStore,
    session_id: SessionId,
}

impl ActiveSession {
    /// Opens the session database and returns the active session plus any
    /// previously stored messages to restore into the runtime. Returns an
    /// empty vec if no prior session exists.
    pub fn open_or_restore(db_path: &Path) -> Result<(Self, Vec<Message>)> {
        let store = SessionStore::open(db_path)?;

        match store.load_most_recent()? {
            Some(saved) => {
                let messages = from_stored(&saved);
                let session_id = saved.meta.id;
                Ok((Self { store, session_id }, messages))
            }
            None => {
                let meta = store.create()?;
                Ok((
                    Self {
                        store,
                        session_id: meta.id,
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
        let meta = self.store.create()?;
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
}
