// src/session/mod.rs
//
// Session persistence — saves and restores conversation history across app restarts.
//
// Design:
//  - Sessions are scoped to the current project directory and stored in .local/sessions.db.
//  - Non-system messages are saved per active session; the system prompt is regenerated fresh.
//  - The most recently opened session is restored by default unless --no-resume is used.
//  - /clear deletes the currently active saved session and starts a fresh unnamed session.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::info;

use crate::config;
use crate::error::{ParamsError, Result};
use crate::inference::Message;

const SCHEMA_VERSION: i64 = 2;

#[derive(Serialize, Deserialize)]
struct StoredMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub id: String,
    pub project_root: String,
    pub name: Option<String>,
    pub backend: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub last_opened_at: u64,
    pub message_count: usize,
}

#[derive(Debug, Clone)]
pub struct SavedSession {
    pub summary: SessionSummary,
    pub messages: Vec<Message>,
    pub saved_at: u64,
}

pub struct SessionStore {
    conn: Connection,
    project_root: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionExportFormat {
    Markdown,
    Json,
}

impl SessionExportFormat {
    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "markdown" | "md" => Some(Self::Markdown),
            "json" => Some(Self::Json),
            _ => None,
        }
    }

    fn extension(self) -> &'static str {
        match self {
            Self::Markdown => "md",
            Self::Json => "json",
        }
    }
}

impl SessionStore {
    pub fn open() -> Result<Self> {
        let project_root = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .to_string_lossy()
            .to_string();
        let path = config::local_dir()?.join("sessions.db");
        Self::open_at(&path, &project_root)
    }

    pub fn open_at(path: &Path, project_root: &str) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        let store = Self {
            conn,
            project_root: project_root.to_string(),
        };
        store.initialize_schema()?;
        info!(db = %path.display(), project_root, "session store opened");
        Ok(store)
    }

    fn initialize_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY,
                backend     TEXT NOT NULL DEFAULT '',
                messages    TEXT NOT NULL,
                saved_at    INTEGER NOT NULL,
                msg_count   INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_records (
                id              TEXT PRIMARY KEY,
                project_root    TEXT NOT NULL,
                name            TEXT,
                backend         TEXT NOT NULL,
                created_at      INTEGER NOT NULL,
                updated_at      INTEGER NOT NULL,
                last_opened_at  INTEGER NOT NULL,
                message_count   INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_messages (
                session_id   TEXT NOT NULL,
                seq          INTEGER NOT NULL,
                role         TEXT NOT NULL,
                content      TEXT NOT NULL,
                PRIMARY KEY (session_id, seq)
            );
            CREATE INDEX IF NOT EXISTS idx_session_records_project_last_opened
                ON session_records(project_root, last_opened_at DESC);
            CREATE INDEX IF NOT EXISTS idx_session_messages_session
                ON session_messages(session_id, seq);",
        )?;

        let user_version: i64 = self
            .conn
            .pragma_query_value(None, "user_version", |row| row.get(0))?;
        if user_version < SCHEMA_VERSION {
            self.migrate_legacy_sessions()?;
            self.conn
                .pragma_update(None, "user_version", SCHEMA_VERSION)?;
        }
        Ok(())
    }

    fn migrate_legacy_sessions(&self) -> Result<()> {
        let record_count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM session_records", [], |row| row.get(0))?;
        if record_count > 0 {
            return Ok(());
        }

        let legacy = self
            .conn
            .query_row(
                "SELECT backend, messages, saved_at, msg_count FROM sessions ORDER BY id DESC LIMIT 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                },
            )
            .optional()?;

        let Some((backend, json, saved_at, msg_count)) = legacy else {
            return Ok(());
        };

        let stored: Vec<StoredMessage> = serde_json::from_str(&json)
            .map_err(|e| ParamsError::Config(format!("session deserialize error: {e}")))?;
        let session_id = self.generate_session_id();
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO session_records (
                id, project_root, name, backend, created_at, updated_at, last_opened_at, message_count
            ) VALUES (?1, ?2, NULL, ?3, ?4, ?4, ?4, ?5)",
            params![
                session_id,
                self.project_root,
                backend,
                saved_at,
                msg_count.max(0)
            ],
        )?;
        for (seq, message) in stored.into_iter().enumerate() {
            tx.execute(
                "INSERT INTO session_messages (session_id, seq, role, content)
                 VALUES (?1, ?2, ?3, ?4)",
                params![session_id, seq as i64, message.role, message.content],
            )?;
        }
        tx.commit()?;
        info!("legacy single-session data migrated");
        Ok(())
    }

    pub fn create_session(&self, name: Option<&str>, backend: &str) -> Result<SessionSummary> {
        let id = self.generate_session_id();
        let now = now_secs();
        self.conn.execute(
            "INSERT INTO session_records (
                id, project_root, name, backend, created_at, updated_at, last_opened_at, message_count
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?5, ?5, 0)",
            params![id, self.project_root, clean_name(name), backend, now as i64],
        )?;
        self.get_session_by_id(&id)?
            .ok_or_else(|| ParamsError::Config("created session could not be reloaded".to_string()))
    }

    pub fn save_messages(
        &self,
        session_id: &str,
        messages: &[Message],
        backend: &str,
    ) -> Result<SessionSummary> {
        let non_system: Vec<StoredMessage> = messages
            .iter()
            .filter(|m| m.role != "system")
            .map(|m| StoredMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();

        let now = now_secs();
        let count = non_system.len();
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "UPDATE session_records
             SET backend = ?2, updated_at = ?3, last_opened_at = ?3, message_count = ?4
             WHERE id = ?1 AND project_root = ?5",
            params![
                session_id,
                backend,
                now as i64,
                count as i64,
                self.project_root
            ],
        )?;
        tx.execute(
            "DELETE FROM session_messages WHERE session_id = ?1",
            params![session_id],
        )?;
        for (seq, message) in non_system.into_iter().enumerate() {
            tx.execute(
                "INSERT INTO session_messages (session_id, seq, role, content)
                 VALUES (?1, ?2, ?3, ?4)",
                params![session_id, seq as i64, message.role, message.content],
            )?;
        }
        tx.commit()?;

        info!(session_id, msg_count = count, backend, "session saved");
        self.get_session_by_id(session_id)?
            .ok_or_else(|| ParamsError::Config("saved session could not be reloaded".to_string()))
    }

    pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {
        let Some(summary) = self.list_sessions()?.into_iter().next() else {
            return Ok(None);
        };
        self.load_session_by_id(&summary.id)
    }

    pub fn load_session(&self, selector: &str) -> Result<SavedSession> {
        let summary = self.resolve_session(selector)?;
        self.load_session_by_id(&summary.id)?
            .ok_or_else(|| ParamsError::Config("session not found".to_string()))
    }

    fn load_session_by_id(&self, session_id: &str) -> Result<Option<SavedSession>> {
        let Some(mut summary) = self.get_session_by_id(session_id)? else {
            return Ok(None);
        };

        let messages = self
            .conn
            .prepare(
                "SELECT role, content
                 FROM session_messages
                 WHERE session_id = ?1
                 ORDER BY seq ASC",
            )?
            .query_map(params![session_id], |row| {
                Ok(Message {
                    role: row.get(0)?,
                    content: row.get(1)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let now = now_secs();
        self.conn.execute(
            "UPDATE session_records SET last_opened_at = ?2 WHERE id = ?1",
            params![session_id, now as i64],
        )?;
        summary.last_opened_at = now;

        info!(session_id, msg_count = messages.len(), "session loaded");
        Ok(Some(SavedSession {
            saved_at: summary.updated_at,
            summary,
            messages,
        }))
    }

    pub fn list_sessions(&self) -> Result<Vec<SessionSummary>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, project_root, name, backend, created_at, updated_at, last_opened_at, message_count
             FROM session_records
             WHERE project_root = ?1
             ORDER BY last_opened_at DESC, updated_at DESC, created_at DESC",
        )?;

        let rows = stmt.query_map(params![self.project_root.as_str()], map_session_summary)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn rename_session(&self, session_id: &str, name: &str) -> Result<SessionSummary> {
        let clean = clean_name(Some(name))
            .ok_or_else(|| ParamsError::Config("Session name cannot be empty".to_string()))?;
        let now = now_secs();
        self.conn.execute(
            "UPDATE session_records
             SET name = ?2, updated_at = ?3, last_opened_at = ?3
             WHERE id = ?1 AND project_root = ?4",
            params![session_id, clean, now as i64, self.project_root],
        )?;
        self.get_session_by_id(session_id)?
            .ok_or_else(|| ParamsError::Config("session not found".to_string()))
    }

    pub fn delete_session(&self, session_id: &str) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "DELETE FROM session_messages WHERE session_id = ?1",
            params![session_id],
        )?;
        tx.execute(
            "DELETE FROM session_records WHERE id = ?1 AND project_root = ?2",
            params![session_id, self.project_root],
        )?;
        tx.commit()?;
        info!(session_id, "saved session cleared");
        Ok(())
    }

    pub fn delete_if_empty_unnamed(&self, session_id: &str) -> Result<()> {
        let maybe = self.get_session_by_id(session_id)?;
        if let Some(summary) = maybe {
            if summary.name.is_none() && summary.message_count == 0 {
                self.delete_session(session_id)?;
            }
        }
        Ok(())
    }

    pub fn export_session(
        &self,
        selector: &str,
        format: SessionExportFormat,
    ) -> Result<(SessionSummary, PathBuf)> {
        let saved = self.load_session(selector)?;
        let export_dir = config::local_dir()?.join("exports").join("sessions");
        fs::create_dir_all(&export_dir)?;
        let path = export_dir.join(export_file_name(&saved.summary, format));
        let body = match format {
            SessionExportFormat::Markdown => render_markdown_export(&saved),
            SessionExportFormat::Json => render_json_export(&saved)?,
        };
        fs::write(&path, body)?;
        info!(
            session_id = saved.summary.id.as_str(),
            format = format.extension(),
            path = %path.display(),
            "session exported"
        );
        Ok((saved.summary, path))
    }

    pub fn resolve_session(&self, selector: &str) -> Result<SessionSummary> {
        let selector = selector.trim();
        if selector.is_empty() {
            return Err(ParamsError::Config(
                "Session selector cannot be empty".to_string(),
            ));
        }

        if let Some(summary) = self.get_session_by_id(selector)? {
            return Ok(summary);
        }

        let id_matches = self
            .list_sessions()?
            .into_iter()
            .filter(|session| session.id.starts_with(selector))
            .collect::<Vec<_>>();

        match id_matches.len() {
            1 => return Ok(id_matches.into_iter().next().unwrap()),
            n if n > 1 => {
                let options = id_matches
                    .into_iter()
                    .map(|session| {
                        format!(
                            "{} ({})",
                            short_id(&session.id),
                            describe_session_age(session.updated_at)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(ParamsError::Config(format!(
                    "Multiple sessions matched id prefix `{selector}`: {options}. Use a longer id prefix from /sessions list."
                )));
            }
            _ => {}
        }

        let matches = self
            .list_sessions()?
            .into_iter()
            .filter(|session| session.name.as_deref() == Some(selector))
            .collect::<Vec<_>>();

        match matches.len() {
            0 => Err(ParamsError::Config(format!(
                "No session matched `{selector}`. Use /sessions list and pass an exact name or unique id prefix."
            ))),
            1 => Ok(matches.into_iter().next().unwrap()),
            _ => {
                let options = matches
                    .into_iter()
                    .map(|session| {
                        format!(
                            "{} ({})",
                            session.id,
                            describe_session_age(session.updated_at)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                Err(ParamsError::Config(format!(
                    "Multiple sessions matched `{selector}`: {options}. Use a longer id prefix from /sessions list."
                )))
            }
        }
    }

    fn get_session_by_id(&self, session_id: &str) -> Result<Option<SessionSummary>> {
        self.conn
            .query_row(
                "SELECT id, project_root, name, backend, created_at, updated_at, last_opened_at, message_count
                 FROM session_records
                 WHERE id = ?1 AND project_root = ?2",
                params![session_id, self.project_root],
                map_session_summary,
            )
            .optional()
            .map_err(Into::into)
    }

    fn generate_session_id(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.project_root.as_bytes());
        hasher.update(now_nanos().to_le_bytes());
        hasher.update(std::process::id().to_le_bytes());
        let digest = hasher.finalize();
        format!("{:x}", digest)[..12].to_string()
    }
}

fn map_session_summary(row: &rusqlite::Row<'_>) -> rusqlite::Result<SessionSummary> {
    Ok(SessionSummary {
        id: row.get(0)?,
        project_root: row.get(1)?,
        name: row.get(2)?,
        backend: row.get(3)?,
        created_at: row.get::<_, i64>(4)? as u64,
        updated_at: row.get::<_, i64>(5)? as u64,
        last_opened_at: row.get::<_, i64>(6)? as u64,
        message_count: row.get::<_, i64>(7)? as usize,
    })
}

fn clean_name(name: Option<&str>) -> Option<String> {
    name.map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string())
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

fn export_file_name(summary: &SessionSummary, format: SessionExportFormat) -> String {
    let base = summary
        .name
        .as_deref()
        .map(slugify)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| summary.id.clone());
    format!("{base}.{}", format.extension())
}

fn slugify(value: &str) -> String {
    let mut output = String::new();
    let mut last_dash = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            output.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            output.push('-');
            last_dash = true;
        }
    }
    output.trim_matches('-').to_string()
}

fn render_markdown_export(saved: &SavedSession) -> String {
    let title = saved.summary.name.as_deref().unwrap_or("unnamed session");
    let mut out = String::new();
    out.push_str("# ");
    out.push_str(title);
    out.push_str("\n\n");
    out.push_str(&format!("- Session ID: `{}`\n", saved.summary.id));
    out.push_str(&format!("- Backend: `{}`\n", saved.summary.backend));
    out.push_str(&format!("- Updated: `{}`\n", saved.summary.updated_at));
    out.push_str(&format!(
        "- Messages: `{}`\n\n",
        saved.summary.message_count
    ));

    for message in &saved.messages {
        let heading = match message.role.as_str() {
            "user" => "User",
            "assistant" => "Assistant",
            other => other,
        };
        out.push_str("## ");
        out.push_str(heading);
        out.push_str("\n\n");
        out.push_str(&message.content);
        out.push_str("\n\n");
    }

    out
}

fn render_json_export(saved: &SavedSession) -> Result<String> {
    #[derive(Serialize)]
    struct Export<'a> {
        id: &'a str,
        name: &'a Option<String>,
        backend: &'a str,
        project_root: &'a str,
        created_at: u64,
        updated_at: u64,
        last_opened_at: u64,
        message_count: usize,
        messages: Vec<StoredMessage>,
    }

    let messages = saved
        .messages
        .iter()
        .map(|message| StoredMessage {
            role: message.role.clone(),
            content: message.content.clone(),
        })
        .collect::<Vec<_>>();

    serde_json::to_string_pretty(&Export {
        id: &saved.summary.id,
        name: &saved.summary.name,
        backend: &saved.summary.backend,
        project_root: &saved.summary.project_root,
        created_at: saved.summary.created_at,
        updated_at: saved.summary.updated_at,
        last_opened_at: saved.summary.last_opened_at,
        message_count: saved.summary.message_count,
        messages,
    })
    .map_err(|e| ParamsError::Config(format!("session export serialize error: {e}")))
}

pub fn describe_session_age(saved_at: u64) -> String {
    let now = now_secs();
    let delta = now.saturating_sub(saved_at);
    if delta < 60 {
        "just now".to_string()
    } else if delta < 3_600 {
        format!("{}m ago", delta / 60)
    } else if delta < 86_400 {
        format!("{}h ago", delta / 3_600)
    } else {
        format!("{}d ago", delta / 86_400)
    }
}

pub fn display_name(summary: &SessionSummary) -> String {
    summary
        .name
        .clone()
        .unwrap_or_else(|| format!("unnamed · {}", describe_session_age(summary.updated_at)))
}

pub fn list_label(summary: &SessionSummary) -> String {
    summary
        .name
        .clone()
        .unwrap_or_else(|| "unnamed".to_string())
}

pub fn short_id(session_id: &str) -> String {
    session_id.chars().take(8).collect()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::*;

    fn temp_db_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("params-cli-test-{}-{}.db", name, now_nanos()));
        let _ = fs::remove_file(&path);
        path
    }

    fn open_store(name: &str) -> SessionStore {
        let path = temp_db_path(name);
        SessionStore::open_at(&path, "/tmp/project").expect("open store")
    }

    #[test]
    fn saves_and_loads_multiple_sessions() {
        let store = open_store("multi");
        let alpha = store.create_session(Some("alpha"), "llama.cpp").unwrap();
        let beta = store.create_session(Some("beta"), "llama.cpp").unwrap();

        let _ = store
            .save_messages(
                &alpha.id,
                &[Message::user("hello"), Message::assistant("world")],
                "llama.cpp",
            )
            .unwrap();
        let _ = store
            .save_messages(&beta.id, &[Message::user("second")], "llama.cpp")
            .unwrap();

        let sessions = store.list_sessions().unwrap();
        assert_eq!(sessions.len(), 2);
        assert!(sessions.iter().any(|session| session.id == alpha.id));
        assert!(sessions.iter().any(|session| session.id == beta.id));

        let loaded = store.load_session("alpha").unwrap();
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.summary.id, alpha.id);
    }

    #[test]
    fn duplicate_name_resolution_errors() {
        let store = open_store("dupe");
        let first = store.create_session(Some("same"), "llama.cpp").unwrap();
        let second = store.create_session(Some("same"), "llama.cpp").unwrap();
        assert_ne!(first.id, second.id);

        let err = store.load_session("same").unwrap_err().to_string();
        assert!(err.contains("Multiple sessions matched"));
    }

    #[test]
    fn deletes_saved_session_by_id() {
        let store = open_store("delete");
        let session = store.create_session(Some("trash"), "llama.cpp").unwrap();
        let _ = store
            .save_messages(&session.id, &[Message::user("hello")], "llama.cpp")
            .unwrap();

        store.delete_session(&session.id).unwrap();

        let sessions = store.list_sessions().unwrap();
        assert!(sessions.iter().all(|saved| saved.id != session.id));
    }

    #[test]
    fn resolves_session_by_short_id_prefix() {
        let store = open_store("prefix");
        let session = store.create_session(Some("review"), "llama.cpp").unwrap();

        let resolved = store.resolve_session(&short_id(&session.id)).unwrap();

        assert_eq!(resolved.id, session.id);
    }

    #[test]
    fn missing_selector_error_points_back_to_sessions_list() {
        let store = open_store("missing-selector");
        let err = store.resolve_session("unknown").unwrap_err().to_string();

        assert!(err.contains("/sessions list"));
        assert!(err.contains("unique id prefix"));
    }

    #[test]
    fn exports_markdown_and_json() {
        let store = open_store("export");
        let session = store.create_session(Some("review"), "llama.cpp").unwrap();
        let _ = store
            .save_messages(
                &session.id,
                &[Message::user("hello"), Message::assistant("world")],
                "llama.cpp",
            )
            .unwrap();

        let (_, md_path) = store
            .export_session("review", SessionExportFormat::Markdown)
            .unwrap();
        let (_, json_path) = store
            .export_session("review", SessionExportFormat::Json)
            .unwrap();

        assert!(fs::read_to_string(md_path).unwrap().contains("# review"));
        assert!(fs::read_to_string(json_path)
            .unwrap()
            .contains("\"message_count\": 2"));
    }

    #[test]
    fn migrates_legacy_single_session() {
        let path = temp_db_path("legacy");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE sessions (
                id INTEGER PRIMARY KEY,
                backend TEXT NOT NULL DEFAULT '',
                messages TEXT NOT NULL,
                saved_at INTEGER NOT NULL,
                msg_count INTEGER NOT NULL
            );
            INSERT INTO sessions (backend, messages, saved_at, msg_count)
            VALUES ('llama.cpp', '[{\"role\":\"user\",\"content\":\"hello\"}]', 123, 1);",
        )
        .unwrap();
        drop(conn);

        let store = SessionStore::open_at(&path, "/tmp/project").unwrap();
        let loaded = store.load_most_recent().unwrap().unwrap();
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.summary.message_count, 1);
    }
}
