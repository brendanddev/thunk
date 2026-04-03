// src/session/mod.rs
//
// Session persistence — saves and restores conversation history across app restarts.
//
// Design:
//  - One saved session per project directory, stored in .local/sessions.db.
//  - Saves all non-system messages after each completed generation.
//  - Auto-restores on startup if a session exists.
//  - /clear wipes the saved session.
//  - The system prompt is never saved — it's regenerated fresh each startup
//    with current facts, tools, and project index matches.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

use crate::config;
use crate::error::{ParamsError, Result};
use crate::inference::Message;

#[derive(Serialize, Deserialize)]
struct StoredMessage {
    role: String,
    content: String,
}

pub struct SavedSession {
    pub messages: Vec<Message>,
    pub saved_at: u64,
}

pub struct SessionStore {
    conn: Connection,
}

impl SessionStore {
    pub fn open() -> Result<Self> {
        let path = config::local_dir()?.join("sessions.db");
        let conn = Connection::open(&path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY,
                backend     TEXT NOT NULL DEFAULT '',
                messages    TEXT NOT NULL,
                saved_at    INTEGER NOT NULL,
                msg_count   INTEGER NOT NULL
            );",
        )?;
        info!(db = %path.display(), "session store opened");
        Ok(Self { conn })
    }

    /// Save the current session. Non-system messages only; the system prompt is
    /// regenerated at startup. Overwrites any previously saved session.
    pub fn save(&self, messages: &[Message], backend: &str) -> Result<()> {
        let non_system: Vec<StoredMessage> = messages
            .iter()
            .filter(|m| m.role != "system")
            .map(|m| StoredMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();

        if non_system.is_empty() {
            return Ok(());
        }

        let json = serde_json::to_string(&non_system)
            .map_err(|e| ParamsError::Config(format!("session serialize error: {e}")))?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let count = non_system.len();
        self.conn.execute("DELETE FROM sessions", [])?;
        self.conn.execute(
            "INSERT INTO sessions (backend, messages, saved_at, msg_count) VALUES (?1, ?2, ?3, ?4)",
            params![backend, json, now as i64, count as i64],
        )?;

        info!(msg_count = count, backend, "session saved");
        Ok(())
    }

    /// Load the most recent saved session, if any.
    pub fn load(&self) -> Result<Option<SavedSession>> {
        let result = self.conn.query_row(
            "SELECT messages, saved_at FROM sessions ORDER BY id DESC LIMIT 1",
            [],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
        );

        match result {
            Ok((json, saved_at)) => {
                let stored: Vec<StoredMessage> = serde_json::from_str(&json)
                    .map_err(|e| ParamsError::Config(format!("session deserialize error: {e}")))?;

                let messages: Vec<Message> = stored
                    .into_iter()
                    .map(|m| Message { role: m.role, content: m.content })
                    .collect();

                info!(msg_count = messages.len(), "session loaded");
                Ok(Some(SavedSession {
                    messages,
                    saved_at: saved_at as u64,
                }))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Clear the saved session. Called when the user runs /clear.
    pub fn clear(&self) -> Result<()> {
        self.conn.execute("DELETE FROM sessions", [])?;
        info!("saved session cleared");
        Ok(())
    }
}
