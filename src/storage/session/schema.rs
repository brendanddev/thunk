use rusqlite::Connection;

use crate::app::{AppError, Result};

const CURRENT_VERSION: i32 = 2;

const SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS sessions (
        id          TEXT PRIMARY KEY,
        project_root TEXT,
        created_at  INTEGER NOT NULL,
        updated_at  INTEGER NOT NULL,
        msg_count   INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS session_messages (
        session_id  TEXT NOT NULL,
        seq         INTEGER NOT NULL,
        role        TEXT NOT NULL,
        content     TEXT NOT NULL,
        PRIMARY KEY (session_id, seq)
    );

    CREATE INDEX IF NOT EXISTS idx_sessions_updated
        ON sessions(updated_at DESC);

    CREATE INDEX IF NOT EXISTS idx_session_messages_lookup
        ON session_messages(session_id, seq);
";

pub(super) fn initialize(conn: &Connection) -> Result<()> {
    conn.execute_batch(SCHEMA)
        .map_err(|e| AppError::Storage(e.to_string()))?;

    let version: i32 = conn
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(|e| AppError::Storage(e.to_string()))?;

    if version < 2 && !has_column(conn, "sessions", "project_root")? {
        conn.execute("ALTER TABLE sessions ADD COLUMN project_root TEXT", [])
            .map_err(|e| AppError::Storage(e.to_string()))?;
    }

    if version < CURRENT_VERSION {
        conn.pragma_update(None, "user_version", CURRENT_VERSION)
            .map_err(|e| AppError::Storage(e.to_string()))?;
    }

    Ok(())
}

fn has_column(conn: &Connection, table: &str, column: &str) -> Result<bool> {
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(|e| AppError::Storage(e.to_string()))?;

    let mut rows = stmt
        .query([])
        .map_err(|e| AppError::Storage(e.to_string()))?;

    while let Some(row) = rows.next().map_err(|e| AppError::Storage(e.to_string()))? {
        let name: String = row.get(1).map_err(|e| AppError::Storage(e.to_string()))?;
        if name == column {
            return Ok(true);
        }
    }

    Ok(false)
}
