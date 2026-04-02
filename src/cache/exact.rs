use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};
use tracing::info;

use crate::config;
use crate::error::Result;
use crate::inference::Message;

pub struct ExactCache {
    conn: Connection,
}

impl ExactCache {
    pub fn open() -> Result<Self> {
        let path = config::local_dir()?.join("cache.db");
        let conn = Connection::open(&path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS exact_cache (
                key_hash TEXT PRIMARY KEY,
                backend  TEXT NOT NULL,
                payload  TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS prompt_cache (
                key_hash TEXT PRIMARY KEY,
                backend  TEXT NOT NULL,
                payload  TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS semantic_prompt_cache (
                key_hash TEXT PRIMARY KEY,
                backend  TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                user_prompt TEXT NOT NULL,
                normalized_prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );",
        )?;
        info!(db = %path.display(), "exact cache opened");
        Ok(Self { conn })
    }

    pub fn get(&self, backend: &str, messages: &[Message]) -> Result<Option<String>> {
        let payload = canonical_payload(backend, messages)?;
        let key = hash_key(&payload);
        self.conn
            .query_row(
                "SELECT response FROM exact_cache WHERE key_hash = ?1",
                params![key],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn put(&self, backend: &str, messages: &[Message], response: &str) -> Result<()> {
        let payload = canonical_payload(backend, messages)?;
        let key = hash_key(&payload);
        self.conn.execute(
            "INSERT OR REPLACE INTO exact_cache (key_hash, backend, payload, response, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                key,
                backend,
                payload,
                response,
                now_unix() as i64,
            ],
        )?;
        Ok(())
    }

    pub fn get_prompt_level(
        &self,
        backend: &str,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<Option<String>> {
        let payload = prompt_payload(backend, system_prompt, user_prompt)?;
        let key = hash_key(&payload);
        self.conn
            .query_row(
                "SELECT response FROM prompt_cache WHERE key_hash = ?1",
                params![key],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn put_prompt_level(
        &self,
        backend: &str,
        system_prompt: &str,
        user_prompt: &str,
        response: &str,
    ) -> Result<()> {
        let payload = prompt_payload(backend, system_prompt, user_prompt)?;
        let key = hash_key(&payload);
        self.conn.execute(
            "INSERT OR REPLACE INTO prompt_cache (key_hash, backend, payload, response, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                key,
                backend,
                payload,
                response,
                now_unix() as i64,
            ],
        )?;
        let normalized_prompt = normalize_prompt(user_prompt);
        self.conn.execute(
            "INSERT OR REPLACE INTO semantic_prompt_cache
             (key_hash, backend, system_prompt, user_prompt, normalized_prompt, response, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                key,
                backend,
                system_prompt,
                user_prompt,
                normalized_prompt,
                response,
                now_unix() as i64,
            ],
        )?;
        Ok(())
    }

    pub fn get_semantic_prompt_level(
        &self,
        backend: &str,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<Option<String>> {
        let normalized = normalize_prompt(user_prompt);
        if normalized.is_empty() {
            return Ok(None);
        }

        let mut stmt = self.conn.prepare(
            "SELECT normalized_prompt, response
             FROM semantic_prompt_cache
             WHERE backend = ?1 AND system_prompt = ?2
             ORDER BY created_at DESC
             LIMIT 50",
        )?;

        let mut rows = stmt.query(params![backend, system_prompt])?;
        let mut best_score = 0.0f64;
        let mut best_response: Option<String> = None;

        while let Some(row) = rows.next()? {
            let candidate_prompt: String = row.get(0)?;
            let candidate_response: String = row.get(1)?;
            let score = semantic_similarity(&normalized, &candidate_prompt);
            if score > best_score {
                best_score = score;
                best_response = Some(candidate_response);
            }
        }

        if best_score >= 0.72 {
            Ok(best_response)
        } else {
            Ok(None)
        }
    }

    pub fn clear(&self) -> Result<usize> {
        let exact_deleted = self.conn.execute("DELETE FROM exact_cache", [])?;
        let prompt_deleted = self.conn.execute("DELETE FROM prompt_cache", [])?;
        let semantic_deleted = self.conn.execute("DELETE FROM semantic_prompt_cache", [])?;
        Ok(exact_deleted
            .saturating_add(prompt_deleted)
            .saturating_add(semantic_deleted))
    }
}

fn canonical_payload(backend: &str, messages: &[Message]) -> Result<String> {
    #[derive(serde::Serialize)]
    struct CachePayload<'a> {
        backend: &'a str,
        messages: &'a [Message],
    }

    let payload = CachePayload { backend, messages };
    serde_json::to_string(&payload)
        .map_err(|e| crate::error::ParamsError::Config(format!("Cache serialization failed: {e}")))
}

fn prompt_payload(backend: &str, system_prompt: &str, user_prompt: &str) -> Result<String> {
    #[derive(serde::Serialize)]
    struct PromptPayload<'a> {
        backend: &'a str,
        system_prompt: &'a str,
        user_prompt: &'a str,
    }

    let payload = PromptPayload {
        backend,
        system_prompt,
        user_prompt,
    };
    serde_json::to_string(&payload)
        .map_err(|e| crate::error::ParamsError::Config(format!("Cache serialization failed: {e}")))
}

fn hash_key(payload: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(payload.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn normalize_prompt(prompt: &str) -> String {
    let cleaned: String = prompt
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c.is_whitespace() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect();

    let stopwords = [
        "a", "an", "the", "what", "is", "are", "was", "were", "do", "does", "did", "can",
        "could", "would", "should", "please", "tell", "me", "about", "explain", "define",
        "in", "programming", "code", "for", "to", "of", "and", "or", "how",
    ];

    cleaned
        .split_whitespace()
        .filter_map(|token| {
            if stopwords.contains(&token) {
                return None;
            }
            let stemmed = token
                .strip_suffix("ing")
                .filter(|s| s.len() >= 4)
                .or_else(|| token.strip_suffix("ers").filter(|s| s.len() >= 4))
                .or_else(|| token.strip_suffix("er").filter(|s| s.len() >= 4))
                .or_else(|| token.strip_suffix('s').filter(|s| s.len() >= 4))
                .unwrap_or(token);
            Some(stemmed.to_string())
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn semantic_similarity(left: &str, right: &str) -> f64 {
    let left_tokens: Vec<&str> = left.split_whitespace().collect();
    let right_tokens: Vec<&str> = right.split_whitespace().collect();

    if left_tokens.is_empty() || right_tokens.is_empty() {
        return 0.0;
    }

    let overlap = left_tokens
        .iter()
        .filter(|token| right_tokens.contains(token))
        .count();

    if overlap == 0 {
        return 0.0;
    }

    let union = left_tokens.len() + right_tokens.len() - overlap;
    let jaccard = overlap as f64 / union as f64;
    let containment = overlap as f64 / left_tokens.len().min(right_tokens.len()) as f64;

    (jaccard * 0.35) + (containment * 0.65)
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_changes_when_messages_change() {
        let backend = "llama.cpp:test";
        let a = vec![Message::user("hello")];
        let b = vec![Message::user("hello there")];

        let a_payload = canonical_payload(backend, &a).unwrap();
        let b_payload = canonical_payload(backend, &b).unwrap();

        assert_ne!(hash_key(&a_payload), hash_key(&b_payload));
    }

    #[test]
    fn prompt_hash_changes_when_prompt_changes() {
        let a = prompt_payload("llama.cpp:test", "system", "what is a pointer?").unwrap();
        let b = prompt_payload("llama.cpp:test", "system", "what is ownership?").unwrap();

        assert_ne!(hash_key(&a), hash_key(&b));
    }

    #[test]
    fn normalize_prompt_strips_wrapper_words() {
        assert_eq!(normalize_prompt("What is a pointer?"), "point");
        assert_eq!(normalize_prompt("Explain pointers"), "point");
    }

    #[test]
    fn semantic_similarity_prefers_near_duplicate_prompts() {
        let pointer = normalize_prompt("What is a pointer?");
        let explain = normalize_prompt("Explain pointers");
        let ownership = normalize_prompt("What is ownership?");

        assert!(semantic_similarity(&pointer, &explain) > 0.9);
        assert!(semantic_similarity(&pointer, &ownership) < 0.2);
    }
}
