// src/memory/index.rs
//
// Level 2: project file index.
//
// Stores a 2-3 sentence summary for each source file in a SQLite database at
// ~/.params/memory/{project_hash}.db. The hash is derived from the current
// working directory path so each project gets its own database.
//
// Summary generation calls the active backend, so indexing runs on the model
// thread (or in the `params index .` command path) — never on the UI thread.
//
// find_relevant() does simple keyword scoring for now. Semantic search via
// embeddings can be layered on top later without changing the schema.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use rusqlite::{params, Connection};
use tracing::{debug, info, warn};

use crate::error::{ParamsError, Result};
use crate::inference::{InferenceBackend, Message};
use super::run_prompt_sync;

pub struct ProjectIndex {
    conn: Connection,
}

impl ProjectIndex {
    /// Open or create the index database for the current working directory.
    pub fn open() -> Result<Self> {
        let cwd = std::env::current_dir()?;
        let db_path = db_path_for(&cwd)?;

        let conn = Connection::open(&db_path)?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS files (
                path          TEXT    PRIMARY KEY,
                summary       TEXT    NOT NULL,
                embedding_json TEXT,
                last_modified INTEGER NOT NULL
            );",
        )?;

        info!(db = %db_path.display(), "project index opened");

        Ok(Self { conn })
    }

    /// Summarize `content` via the backend and store it against `path`.
    ///
    /// Replaces any existing entry for this path.
    pub fn index_file(
        &self,
        path: &Path,
        content: &str,
        backend: &dyn InferenceBackend,
    ) -> Result<()> {
        let path_str = path.to_string_lossy().to_string();
        let mtime = file_mtime(path).unwrap_or(0) as i64;

        let prompt = vec![
            Message::system(
                "You are a helpful assistant that writes concise code file summaries.",
            ),
            Message::user(&format!(
                "Summarize this file in 2-3 sentences. \
                 State what it does and name its key types or functions.\n\
                 File: {path_str}\n\n```\n{content}\n```"
            )),
        ];

        let summary = run_prompt_sync(backend, &prompt)?;
        let summary = summary.trim().to_string();

        if summary.is_empty() {
            warn!(path = %path_str, "summary was empty, skipping");
            return Ok(());
        }

        self.conn.execute(
            "INSERT OR REPLACE INTO files \
             (path, summary, embedding_json, last_modified) \
             VALUES (?1, ?2, NULL, ?3)",
            params![path_str, summary, mtime],
        )?;

        debug!(path = %path_str, "file indexed");

        Ok(())
    }

    /// Return up to `limit` (path, summary) pairs whose summaries match the
    /// query by keyword overlap. Unmatched entries are excluded; ties are
    /// broken by match count descending.
    pub fn find_relevant(&self, query: &str, limit: usize) -> Result<Vec<(String, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, summary FROM files")?;

        let query_lower = query.to_lowercase();
        let keywords: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored: Vec<(usize, String, String)> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .flatten()
            .filter_map(|(path, summary)| {
                let summary_lower = summary.to_lowercase();
                let score = keywords
                    .iter()
                    .filter(|kw| summary_lower.contains(*kw))
                    .count();
                if score > 0 {
                    Some((score, path, summary))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored.truncate(limit);

        Ok(scored.into_iter().map(|(_, p, s)| (p, s)).collect())
    }

    /// Returns true if `path` is not yet indexed or its mtime has changed.
    pub fn needs_reindex(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy().to_string();

        let current_mtime = match file_mtime(path) {
            Some(m) => m as i64,
            None => return false, // Can't stat — skip safely
        };

        let stored: rusqlite::Result<i64> = self.conn.query_row(
            "SELECT last_modified FROM files WHERE path = ?1",
            params![path_str],
            |row| row.get(0),
        );

        match stored {
            Ok(stored_mtime) => stored_mtime != current_mtime,
            Err(_) => true, // Not in the DB yet
        }
    }
}

/// Build the path to the project index database.
/// Creates `~/.params/memory/` if it doesn't exist.
fn db_path_for(project_root: &Path) -> Result<PathBuf> {
    let home = dirs::home_dir()
        .ok_or_else(|| ParamsError::Config("Could not find home directory".into()))?;
    let memory_dir = home.join(".params").join("memory");
    std::fs::create_dir_all(&memory_dir)?;

    let hash = path_hash(&project_root.to_string_lossy());
    Ok(memory_dir.join(format!("{hash}.db")))
}

fn path_hash(s: &str) -> String {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn file_mtime(path: &Path) -> Option<u64> {
    std::fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
}
