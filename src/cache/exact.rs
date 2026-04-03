use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};
use tracing::{debug, info};

use crate::config;
use crate::error::Result;
use crate::inference::Message;

pub struct ExactCache {
    conn: Connection,
}

#[derive(Debug, Clone)]
pub struct CacheScope {
    pub project_fingerprint: String,
    pub ttl_seconds: Option<u64>,
}

#[derive(Debug)]
struct CacheRow {
    key_hash: String,
    response: String,
    expires_at: Option<i64>,
    project_fingerprint: Option<String>,
}

const CACHE_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS exact_cache (
        key_hash TEXT PRIMARY KEY,
        backend  TEXT NOT NULL,
        payload  TEXT NOT NULL,
        response TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        expires_at INTEGER,
        project_fingerprint TEXT
    );
    CREATE TABLE IF NOT EXISTS prompt_cache (
        key_hash TEXT PRIMARY KEY,
        backend  TEXT NOT NULL,
        payload  TEXT NOT NULL,
        response TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        expires_at INTEGER,
        project_fingerprint TEXT
    );
    CREATE TABLE IF NOT EXISTS semantic_prompt_cache (
        key_hash TEXT PRIMARY KEY,
        backend  TEXT NOT NULL,
        system_prompt TEXT NOT NULL,
        user_prompt TEXT NOT NULL,
        normalized_prompt TEXT NOT NULL,
        response TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        expires_at INTEGER,
        project_fingerprint TEXT
    );";

const FINGERPRINT_SKIP_DIRS: &[&str] = &[".git", ".local", "target", "node_modules"];

impl ExactCache {
    pub fn open() -> Result<Self> {
        let path = config::local_dir()?.join("cache.db");
        let conn = Connection::open(&path)?;
        let cache = Self { conn };
        cache.ensure_schema()?;
        info!(db = %path.display(), "exact cache opened");
        Ok(cache)
    }

    fn ensure_schema(&self) -> Result<()> {
        self.conn.execute_batch(CACHE_SCHEMA)?;
        self.ensure_column("exact_cache", "expires_at", "INTEGER")?;
        self.ensure_column("exact_cache", "project_fingerprint", "TEXT")?;
        self.ensure_column("prompt_cache", "expires_at", "INTEGER")?;
        self.ensure_column("prompt_cache", "project_fingerprint", "TEXT")?;
        self.ensure_column("semantic_prompt_cache", "expires_at", "INTEGER")?;
        self.ensure_column("semantic_prompt_cache", "project_fingerprint", "TEXT")?;
        Ok(())
    }

    fn ensure_column(&self, table: &str, column: &str, definition: &str) -> Result<()> {
        let pragma = format!("PRAGMA table_info({table})");
        let mut stmt = self.conn.prepare(&pragma)?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let existing: String = row.get(1)?;
            if existing == column {
                return Ok(());
            }
        }

        let alter = format!("ALTER TABLE {table} ADD COLUMN {column} {definition}");
        self.conn.execute(&alter, [])?;
        Ok(())
    }

    pub fn get(&self, backend: &str, messages: &[Message], scope: &CacheScope) -> Result<Option<String>> {
        let payload = canonical_payload(backend, messages)?;
        let key = hash_key(&payload);
        let row = self
            .conn
            .query_row(
                "SELECT key_hash, response, expires_at, project_fingerprint
                 FROM exact_cache
                 WHERE key_hash = ?1",
                params![key],
                |row| {
                    Ok(CacheRow {
                        key_hash: row.get(0)?,
                        response: row.get(1)?,
                        expires_at: row.get(2)?,
                        project_fingerprint: row.get(3)?,
                    })
                },
            )
            .optional()?;

        self.resolve_row("exact_cache", row, "exact", scope)
    }

    pub fn put(
        &self,
        backend: &str,
        messages: &[Message],
        response: &str,
        scope: &CacheScope,
    ) -> Result<()> {
        let payload = canonical_payload(backend, messages)?;
        let key = hash_key(&payload);
        self.conn.execute(
            "INSERT OR REPLACE INTO exact_cache
             (key_hash, backend, payload, response, created_at, expires_at, project_fingerprint)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                key,
                backend,
                payload,
                response,
                now_unix() as i64,
                expires_at(scope),
                scope.project_fingerprint,
            ],
        )?;
        Ok(())
    }

    pub fn get_prompt_level(
        &self,
        backend: &str,
        system_prompt: &str,
        user_prompt: &str,
        scope: &CacheScope,
    ) -> Result<Option<String>> {
        let payload = prompt_payload(backend, system_prompt, user_prompt)?;
        let key = hash_key(&payload);
        let row = self
            .conn
            .query_row(
                "SELECT key_hash, response, expires_at, project_fingerprint
                 FROM prompt_cache
                 WHERE key_hash = ?1",
                params![key],
                |row| {
                    Ok(CacheRow {
                        key_hash: row.get(0)?,
                        response: row.get(1)?,
                        expires_at: row.get(2)?,
                        project_fingerprint: row.get(3)?,
                    })
                },
            )
            .optional()?;

        self.resolve_row("prompt_cache", row, "prompt-level", scope)
    }

    pub fn put_prompt_level(
        &self,
        backend: &str,
        system_prompt: &str,
        user_prompt: &str,
        response: &str,
        scope: &CacheScope,
    ) -> Result<()> {
        let payload = prompt_payload(backend, system_prompt, user_prompt)?;
        let key = hash_key(&payload);
        let expires_at = expires_at(scope);
        self.conn.execute(
            "INSERT OR REPLACE INTO prompt_cache
             (key_hash, backend, payload, response, created_at, expires_at, project_fingerprint)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                key,
                backend,
                payload,
                response,
                now_unix() as i64,
                expires_at,
                scope.project_fingerprint,
            ],
        )?;
        let normalized_prompt = normalize_prompt(user_prompt);
        self.conn.execute(
            "INSERT OR REPLACE INTO semantic_prompt_cache
             (key_hash, backend, system_prompt, user_prompt, normalized_prompt, response, created_at, expires_at, project_fingerprint)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                key,
                backend,
                system_prompt,
                user_prompt,
                normalized_prompt,
                response,
                now_unix() as i64,
                expires_at,
                scope.project_fingerprint,
            ],
        )?;
        Ok(())
    }

    pub fn get_semantic_prompt_level(
        &self,
        backend: &str,
        system_prompt: &str,
        user_prompt: &str,
        scope: &CacheScope,
    ) -> Result<Option<String>> {
        let normalized = normalize_prompt(user_prompt);
        if normalized.is_empty() {
            return Ok(None);
        }

        let mut stmt = self.conn.prepare(
            "SELECT key_hash, normalized_prompt, response, expires_at, project_fingerprint
             FROM semantic_prompt_cache
             WHERE backend = ?1 AND system_prompt = ?2
             ORDER BY created_at DESC
             LIMIT 50",
        )?;

        let mut rows = stmt.query(params![backend, system_prompt])?;
        let mut best_score = 0.0f64;
        let mut best_response: Option<String> = None;
        let mut stale_keys = Vec::new();

        while let Some(row) = rows.next()? {
            let key_hash: String = row.get(0)?;
            let candidate_prompt: String = row.get(1)?;
            let candidate_response: String = row.get(2)?;
            let expires_at: Option<i64> = row.get(3)?;
            let project_fingerprint: Option<String> = row.get(4)?;

            if let Some(reason) = stale_reason(expires_at, project_fingerprint.as_deref(), scope) {
                debug!(reason, key_hash, "semantic prompt cache entry invalidated");
                stale_keys.push(key_hash);
                continue;
            }

            let score = semantic_similarity(&normalized, &candidate_prompt);
            if score > best_score {
                best_score = score;
                best_response = Some(candidate_response);
            }
        }

        drop(rows);
        drop(stmt);

        for key in stale_keys {
            self.delete_key("semantic_prompt_cache", &key)?;
        }

        if best_score >= 0.72 {
            Ok(best_response)
        } else {
            Ok(None)
        }
    }

    fn resolve_row(
        &self,
        table: &str,
        row: Option<CacheRow>,
        cache_name: &str,
        scope: &CacheScope,
    ) -> Result<Option<String>> {
        let Some(row) = row else {
            return Ok(None);
        };

        if let Some(reason) = stale_reason(row.expires_at, row.project_fingerprint.as_deref(), scope) {
            debug!(table, cache_name, reason, key = row.key_hash, "cache entry invalidated");
            self.delete_key(table, &row.key_hash)?;
            return Ok(None);
        }

        Ok(Some(row.response))
    }

    fn delete_key(&self, table: &str, key_hash: &str) -> Result<()> {
        let sql = format!("DELETE FROM {table} WHERE key_hash = ?1");
        self.conn.execute(&sql, params![key_hash])?;
        Ok(())
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

pub fn build_cache_scope(project_root: &Path, ttl_seconds: u64) -> Result<CacheScope> {
    Ok(CacheScope {
        project_fingerprint: fingerprint_project(project_root)?,
        ttl_seconds: if ttl_seconds == 0 { None } else { Some(ttl_seconds) },
    })
}

fn fingerprint_project(project_root: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"params-cache-scope-v1");
    hash_directory(project_root, project_root, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_directory(root: &Path, dir: &Path, hasher: &mut Sha256) -> Result<()> {
    let mut entries = fs::read_dir(dir)?
        .flatten()
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    entries.sort();

    for path in entries {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };

        let Ok(file_type) = fs::symlink_metadata(&path).map(|meta| meta.file_type()) else {
            continue;
        };

        if file_type.is_dir() {
            if FINGERPRINT_SKIP_DIRS.contains(&name) {
                continue;
            }
            hash_directory(root, &path, hasher)?;
            continue;
        }

        if !file_type.is_file() {
            continue;
        }

        if name == ".DS_Store" {
            continue;
        }

        let relative = match path.strip_prefix(root) {
            Ok(relative) => relative,
            Err(_) => continue,
        };

        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(_) => continue,
        };
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_secs())
            .unwrap_or(0);

        hasher.update(relative.to_string_lossy().as_bytes());
        hasher.update(b"\0");
        hasher.update(metadata.len().to_le_bytes());
        hasher.update(modified.to_le_bytes());
        hasher.update(b"\0");
    }

    Ok(())
}

fn stale_reason(
    expires_at: Option<i64>,
    project_fingerprint: Option<&str>,
    scope: &CacheScope,
) -> Option<&'static str> {
    if let Some(expires_at) = expires_at {
        if (now_unix() as i64) >= expires_at {
            return Some("expired");
        }
    } else {
        return Some("legacy_missing_expiry");
    }

    match project_fingerprint {
        Some(fingerprint) if fingerprint == scope.project_fingerprint => None,
        Some(_) => Some("project_changed"),
        None => Some("legacy_missing_project_fingerprint"),
    }
}

fn expires_at(scope: &CacheScope) -> Option<i64> {
    scope.ttl_seconds
        .map(|ttl| now_unix().saturating_add(ttl) as i64)
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
        "in", "programming", "code", "for", "to", "of", "and", "or", "how", "whats", "s",
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

    fn test_cache() -> ExactCache {
        let conn = Connection::open_in_memory().unwrap();
        let cache = ExactCache { conn };
        cache.ensure_schema().unwrap();
        cache
    }

    fn scope(name: &str) -> CacheScope {
        CacheScope {
            project_fingerprint: name.to_string(),
            ttl_seconds: Some(60),
        }
    }

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
        assert_eq!(normalize_prompt("Whats a pointer"), "point");
    }

    #[test]
    fn semantic_similarity_prefers_near_duplicate_prompts() {
        let pointer = normalize_prompt("What is a pointer?");
        let explain = normalize_prompt("Explain pointers");
        let ownership = normalize_prompt("What is ownership?");

        assert!(semantic_similarity(&pointer, &explain) > 0.9);
        assert!(semantic_similarity(&pointer, &ownership) < 0.2);
    }

    #[test]
    fn exact_cache_invalidates_when_project_changes() {
        let cache = test_cache();
        let messages = vec![Message::user("hello")];
        cache.put("llama.cpp:test", &messages, "world", &scope("a")).unwrap();

        let hit = cache.get("llama.cpp:test", &messages, &scope("b")).unwrap();
        assert!(hit.is_none());
    }

    #[test]
    fn prompt_cache_invalidates_when_expired() {
        let cache = test_cache();
        let expired_scope = CacheScope {
            project_fingerprint: "a".to_string(),
            ttl_seconds: Some(0),
        };
        cache.put_prompt_level("llama.cpp:test", "system", "hello", "world", &expired_scope).unwrap();

        let hit = cache
            .get_prompt_level("llama.cpp:test", "system", "hello", &scope("a"))
            .unwrap();
        assert!(hit.is_none());
    }

    #[test]
    fn project_fingerprint_changes_when_file_metadata_changes() {
        let base = std::env::temp_dir().join(format!("params-cache-fp-{}", now_unix()));
        fs::create_dir_all(&base).unwrap();
        let file = base.join("main.rs");
        fs::write(&file, "fn main() {}\n").unwrap();
        let first = fingerprint_project(&base).unwrap();

        std::thread::sleep(std::time::Duration::from_secs(1));
        fs::write(&file, "fn main() { println!(\"hi\"); }\n").unwrap();
        let second = fingerprint_project(&base).unwrap();

        assert_ne!(first, second);
        let _ = fs::remove_file(file);
        let _ = fs::remove_dir(base);
    }

    #[test]
    fn fingerprint_skips_local_runtime_dir() {
        let base = std::env::temp_dir().join(format!("params-cache-local-{}", now_unix()));
        fs::create_dir_all(base.join(".local")).unwrap();
        fs::write(base.join("main.rs"), "fn main() {}\n").unwrap();
        let first = fingerprint_project(&base).unwrap();
        fs::write(base.join(".local").join("cache.db"), "runtime").unwrap();
        let second = fingerprint_project(&base).unwrap();

        assert_eq!(first, second);
        let _ = fs::remove_file(base.join(".local").join("cache.db"));
        let _ = fs::remove_file(base.join("main.rs"));
        let _ = fs::remove_dir(base.join(".local"));
        let _ = fs::remove_dir(base);
    }
}
