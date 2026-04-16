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
                Ok((Self { store, session_id: meta.id }, vec![]))
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

// ── Conversion: runtime ↔ storage ────────────────────────────────────────────
//
// System messages are excluded. The system prompt is reconstructed at runtime
// from config; storing it would create a stale copy that could diverge.

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

/// Converts stored messages back to runtime form.
/// Unknown role strings are skipped rather than crashing.
fn from_stored(session: &SavedSession) -> Vec<Message> {
    session
        .messages
        .iter()
        .filter_map(|m| match m.role.as_str() {
            "user" => Some(Message::user(m.content.clone())),
            "assistant" => Some(Message::assistant(m.content.clone())),
            _ => None,
        })
        .collect()
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
