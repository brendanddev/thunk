use std::path::Path;

use rusqlite::{params, Connection, OptionalExtension};

use crate::app::{AppError, Result};

use super::schema;
use super::types::{generate_session_id, now_ms, SavedSession, SessionMeta, StoredMessage};

pub struct SessionStore {
    conn: Connection,
}

impl SessionStore {
    /// Opens (or creates) a session database at the given path.
    /// The parent directory must already exist — callers should use AppPaths::ensure_runtime_dirs first.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).map_err(|e| AppError::Storage(e.to_string()))?;
        schema::initialize(&conn)?;
        Ok(Self { conn })
    }

    /// Creates a new empty session and returns its metadata.
    pub fn create(&self, project_root: &Path) -> Result<SessionMeta> {
        let id = generate_session_id();
        let now = now_ms();
        let project_root = project_root.to_string_lossy().into_owned();
        self.conn
            .execute(
                "INSERT INTO sessions (id, project_root, created_at, updated_at, msg_count)
                 VALUES (?1, ?2, ?3, ?3, 0)",
                params![id, project_root, now as i64],
            )
            .map_err(|e| AppError::Storage(e.to_string()))?;
        self.require_meta(&id)
    }

    /// Persists messages for an existing session. Replaces any previously saved messages.
    /// Returns updated metadata with the new message count and timestamp.
    pub fn save(&self, id: &str, messages: &[StoredMessage]) -> Result<SessionMeta> {
        let now = now_ms();
        let count = messages.len();

        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| AppError::Storage(e.to_string()))?;

        tx.execute(
            "UPDATE sessions SET updated_at = ?2, msg_count = ?3 WHERE id = ?1",
            params![id, now as i64, count as i64],
        )
        .map_err(|e| AppError::Storage(e.to_string()))?;

        tx.execute(
            "DELETE FROM session_messages WHERE session_id = ?1",
            params![id],
        )
        .map_err(|e| AppError::Storage(e.to_string()))?;

        for (seq, msg) in messages.iter().enumerate() {
            tx.execute(
                "INSERT INTO session_messages (session_id, seq, role, content)
                 VALUES (?1, ?2, ?3, ?4)",
                params![id, seq as i64, msg.role, msg.content],
            )
            .map_err(|e| AppError::Storage(e.to_string()))?;
        }

        tx.commit().map_err(|e| AppError::Storage(e.to_string()))?;

        self.require_meta(id)
    }

    /// Loads a session by ID. Returns None if the ID does not exist.
    pub fn load(&self, id: &str) -> Result<Option<SavedSession>> {
        let Some(meta) = self.load_meta(id)? else {
            return Ok(None);
        };

        let messages = self
            .conn
            .prepare(
                "SELECT role, content
                 FROM session_messages
                 WHERE session_id = ?1
                 ORDER BY seq ASC",
            )
            .map_err(|e| AppError::Storage(e.to_string()))?
            .query_map(params![id], |row| {
                Ok(StoredMessage {
                    role: row.get(0)?,
                    content: row.get(1)?,
                })
            })
            .map_err(|e| AppError::Storage(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| AppError::Storage(e.to_string()))?;

        Ok(Some(SavedSession { meta, messages }))
    }

    /// Loads the most recently updated session. Returns None if there are no sessions.
    pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {
        let id = self
            .conn
            .query_row(
                "SELECT id FROM sessions ORDER BY updated_at DESC LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|e| AppError::Storage(e.to_string()))?;

        match id {
            Some(id) => self.load(&id),
            None => Ok(None),
        }
    }

    /// Lists all sessions ordered by most recently updated.
    pub fn list(&self) -> Result<Vec<SessionMeta>> {
        self.conn
            .prepare(
                "SELECT id, project_root, created_at, updated_at, msg_count
                 FROM sessions
                 ORDER BY updated_at DESC",
            )
            .map_err(|e| AppError::Storage(e.to_string()))?
            .query_map([], |row| {
                Ok(SessionMeta {
                    id: row.get(0)?,
                    project_root: row.get(1)?,
                    created_at: row.get::<_, i64>(2)? as u64,
                    updated_at: row.get::<_, i64>(3)? as u64,
                    message_count: row.get::<_, i64>(4)? as usize,
                })
            })
            .map_err(|e| AppError::Storage(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| AppError::Storage(e.to_string()))
    }

    /// Deletes a session and all its messages.
    pub fn delete(&self, id: &str) -> Result<()> {
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| AppError::Storage(e.to_string()))?;

        tx.execute(
            "DELETE FROM session_messages WHERE session_id = ?1",
            params![id],
        )
        .map_err(|e| AppError::Storage(e.to_string()))?;

        tx.execute("DELETE FROM sessions WHERE id = ?1", params![id])
            .map_err(|e| AppError::Storage(e.to_string()))?;

        tx.commit().map_err(|e| AppError::Storage(e.to_string()))
    }

    fn load_meta(&self, id: &str) -> Result<Option<SessionMeta>> {
        self.conn
            .query_row(
                "SELECT id, project_root, created_at, updated_at, msg_count
                 FROM sessions WHERE id = ?1",
                params![id],
                |row| {
                    Ok(SessionMeta {
                        id: row.get(0)?,
                        project_root: row.get(1)?,
                        created_at: row.get::<_, i64>(2)? as u64,
                        updated_at: row.get::<_, i64>(3)? as u64,
                        message_count: row.get::<_, i64>(4)? as usize,
                    })
                },
            )
            .optional()
            .map_err(|e| AppError::Storage(e.to_string()))
    }

    fn require_meta(&self, id: &str) -> Result<SessionMeta> {
        self.load_meta(id)?
            .ok_or_else(|| AppError::Storage(format!("session not found: {id}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn in_memory() -> SessionStore {
        let conn = Connection::open_in_memory().unwrap();
        schema::initialize(&conn).unwrap();
        SessionStore { conn }
    }

    #[test]
    fn create_and_list() {
        let store = in_memory();
        let a = store.create(Path::new("/tmp/project-a")).unwrap();
        let b = store.create(Path::new("/tmp/project-b")).unwrap();
        let sessions = store.list().unwrap();
        assert_eq!(sessions.len(), 2);
        assert!(sessions.iter().any(|s| s.id == a.id));
        assert!(sessions.iter().any(|s| s.id == b.id));
        assert_eq!(a.project_root.as_deref(), Some("/tmp/project-a"));
        assert_eq!(b.project_root.as_deref(), Some("/tmp/project-b"));
    }

    #[test]
    fn save_and_load_roundtrip() {
        let store = in_memory();
        let meta = store.create(Path::new("/tmp/project")).unwrap();

        let messages = vec![
            StoredMessage {
                role: "user".into(),
                content: "hello".into(),
            },
            StoredMessage {
                role: "assistant".into(),
                content: "hi there".into(),
            },
        ];
        let saved = store.save(&meta.id, &messages).unwrap();
        assert_eq!(saved.message_count, 2);
        assert_eq!(saved.project_root.as_deref(), Some("/tmp/project"));

        let loaded = store.load(&meta.id).unwrap().unwrap();
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.messages[0].role, "user");
        assert_eq!(loaded.messages[1].content, "hi there");
        assert_eq!(loaded.meta.project_root.as_deref(), Some("/tmp/project"));
    }

    #[test]
    fn save_replaces_existing_messages() {
        let store = in_memory();
        let meta = store.create(Path::new("/tmp/project")).unwrap();

        store
            .save(
                &meta.id,
                &[StoredMessage {
                    role: "user".into(),
                    content: "first".into(),
                }],
            )
            .unwrap();

        store
            .save(
                &meta.id,
                &[StoredMessage {
                    role: "user".into(),
                    content: "replaced".into(),
                }],
            )
            .unwrap();

        let loaded = store.load(&meta.id).unwrap().unwrap();
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0].content, "replaced");
    }

    #[test]
    fn load_most_recent_returns_latest() {
        let store = in_memory();
        let a = store.create(Path::new("/tmp/project-a")).unwrap();
        let b = store.create(Path::new("/tmp/project-b")).unwrap();

        // Save to b last so it is most recent
        store
            .save(
                &a.id,
                &[StoredMessage {
                    role: "user".into(),
                    content: "a".into(),
                }],
            )
            .unwrap();
        store
            .save(
                &b.id,
                &[StoredMessage {
                    role: "user".into(),
                    content: "b".into(),
                }],
            )
            .unwrap();

        let recent = store.load_most_recent().unwrap().unwrap();
        assert_eq!(recent.meta.id, b.id);
        assert_eq!(recent.meta.project_root.as_deref(), Some("/tmp/project-b"));
    }

    #[test]
    fn delete_removes_session_and_messages() {
        let store = in_memory();
        let meta = store.create(Path::new("/tmp/project")).unwrap();
        store
            .save(
                &meta.id,
                &[StoredMessage {
                    role: "user".into(),
                    content: "gone".into(),
                }],
            )
            .unwrap();

        store.delete(&meta.id).unwrap();

        assert!(store.load(&meta.id).unwrap().is_none());
        assert!(store.list().unwrap().is_empty());
    }

    #[test]
    fn load_unknown_id_returns_none() {
        let store = in_memory();
        assert!(store.load("does-not-exist").unwrap().is_none());
    }
}
