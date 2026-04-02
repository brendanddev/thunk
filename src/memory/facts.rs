// src/memory/facts.rs
//
// Level 3: cross-session fact store.
//
// At the end of each session, key facts (files examined, decisions made,
// errors resolved) are extracted from the conversation via the backend and
// stored in a shared SQLite database at .local/memory/facts.db.
//
// Facts are scoped per project (current working directory path string).
// At the start of a new session the top relevant facts are injected into
// the system prompt so the model remembers prior context without replaying
// the full history.
//
// The fact store degrades gracefully — errors are logged and the session
// continues normally.

use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};
use tracing::{debug, info, warn};

use crate::config;
use crate::error::Result;
use crate::inference::{InferenceBackend, Message};
use super::run_prompt_sync;

pub struct FactStore {
    conn: Connection,
}

impl FactStore {
    /// Open or create the shared fact database at .local/memory/facts.db.
    pub fn open() -> Result<Self> {
        let memory_dir = config::memory_dir()?;
        let db_path = memory_dir.join("facts.db");
        let conn = Connection::open(&db_path)?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS facts (
                id         INTEGER PRIMARY KEY,
                project    TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                created_at INTEGER NOT NULL,
                last_seen  INTEGER NOT NULL
            );",
        )?;

        info!(db = %db_path.display(), "fact store opened");

        Ok(Self { conn })
    }

    /// Persist a single fact for the given project.
    pub fn store_fact(&self, project: &str, content: &str) -> Result<()> {
        let now = now_secs();
        self.conn.execute(
            "INSERT INTO facts (project, content, created_at, last_seen) \
             VALUES (?1, ?2, ?3, ?3)",
            params![project, content, now],
        )?;

        debug!(project, "fact stored");

        Ok(())
    }

    /// Return up to `limit` facts for the project, ranked by keyword overlap
    /// with `query`. When `query` is empty, returns the most recently seen facts.
    pub fn get_relevant_facts(
        &self,
        project: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT content FROM facts WHERE project = ?1 \
             ORDER BY last_seen DESC LIMIT 50",
        )?;

        let all: Vec<String> = stmt
            .query_map(params![project], |row| row.get(0))?
            .flatten()
            .collect();

        if query.is_empty() {
            return Ok(all.into_iter().take(limit).collect());
        }

        let query_lower = query.to_lowercase();
        let keywords: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored: Vec<(usize, String)> = all
            .into_iter()
            .map(|fact| {
                let fact_lower = fact.to_lowercase();
                let score = keywords
                    .iter()
                    .filter(|kw| fact_lower.contains(*kw))
                    .count();
                (score, fact)
            })
            .collect();

        scored.sort_by(|a, b| b.0.cmp(&a.0));

        Ok(scored.into_iter().map(|(_, f)| f).take(limit).collect())
    }

    /// Extract key facts from `messages` via the backend and store them.
    ///
    /// Called at the end of a session (when the prompt channel closes).
    /// Degrades silently on failure — logs warnings but never propagates errors.
    pub fn extract_and_store(
        &self,
        messages: &[Message],
        project: &str,
        backend: &dyn InferenceBackend,
    ) {
        let meaningful: Vec<&Message> = messages
            .iter()
            .filter(|m| m.role == "user" || m.role == "assistant")
            .collect();

        // Nothing useful to extract from a very short session.
        if meaningful.len() < 2 {
            return;
        }

        let conversation_text: String = meaningful
            .iter()
            .map(|m| format!("{}: {}\n", m.role, m.content))
            .collect();

        let prompt = vec![
            Message::system(
                "You are a helpful assistant that extracts factual notes from conversations.",
            ),
            Message::user(&format!(
                "Extract 3-5 key facts from this conversation. \
                 Include: files discussed by name, architectural decisions made, \
                 bugs resolved, and important conclusions. \
                 Write one fact per line. Be specific. Do not number the lines.\n\n\
                 {conversation_text}"
            )),
        ];

        let response = match run_prompt_sync(backend, &prompt) {
            Ok(r) => r,
            Err(e) => {
                warn!(error = %e, "fact extraction prompt failed");
                return;
            }
        };

        let mut stored = 0usize;
        for line in response.lines() {
            let fact = line.trim();
            // Skip empty lines and very short fragments that are unlikely to be real facts.
            if fact.is_empty() || fact.len() < 10 {
                continue;
            }
            match self.store_fact(project, fact) {
                Ok(()) => stored += 1,
                Err(e) => warn!(error = %e, "failed to store fact"),
            }
        }

        if stored > 0 {
            info!(project, count = stored, "session facts stored");
        }
    }
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
