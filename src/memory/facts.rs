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

use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};
use tracing::{debug, info, warn};

use crate::config::{self, MemoryConfig};
use crate::error::Result;
use crate::inference::{InferenceBackend, Message};
use super::run_prompt_sync;

pub struct FactStore {
    conn: Connection,
}

pub struct ConsolidationStats {
    pub ttl_pruned: usize,
    pub dedup_removed: usize,
    pub cap_removed: usize,
}

impl FactStore {
    /// Open or create the shared fact database at .local/memory/facts.db.
    pub fn open() -> Result<Self> {
        let memory_dir = config::memory_dir()?;
        let db_path = memory_dir.join("facts.db");
        let conn = Connection::open(&db_path)?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS facts (
                id          INTEGER PRIMARY KEY,
                project     TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                created_at  INTEGER NOT NULL,
                last_seen   INTEGER NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0
            );",
        )?;

        // Migration: add access_count to existing databases (ignore if column exists).
        let _ = conn.execute(
            "ALTER TABLE facts ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0",
            [],
        );

        info!(db = %db_path.display(), "fact store opened");

        Ok(Self { conn })
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

    /// Try to store a fact, skipping near-duplicates of already-stored facts.
    ///
    /// If a near-duplicate is found, the existing fact's `last_seen` is updated.
    /// Returns true if a new fact was stored, false if deduplicated.
    fn try_store_fact_deduped(&self, project: &str, content: &str) -> Result<bool> {
        let now = now_secs();

        // Load existing facts to check for near-duplicates.
        let mut stmt = self.conn.prepare(
            "SELECT id, content FROM facts WHERE project = ?1",
        )?;

        let existing: Vec<(i64, String)> = stmt
            .query_map(params![project], |row| Ok((row.get(0)?, row.get(1)?)))?
            .flatten()
            .collect();

        for (id, existing_content) in &existing {
            if are_near_duplicate(content, existing_content) {
                // Update last_seen on the existing fact instead of inserting.
                self.conn.execute(
                    "UPDATE facts SET last_seen = ?1 WHERE id = ?2",
                    params![now, id],
                )?;
                debug!(project, "fact deduplicated (near-duplicate found)");
                return Ok(false);
            }
        }

        self.conn.execute(
            "INSERT INTO facts (project, content, created_at, last_seen, access_count) \
             VALUES (?1, ?2, ?3, ?3, 0)",
            params![project, content, now],
        )?;

        debug!(project, "fact stored");
        Ok(true)
    }

    /// Prune stale facts, remove near-duplicates, and enforce the per-project cap.
    ///
    /// Called at end of session (non-blocking — runs on the model thread after
    /// the user has already quit). Degrades gracefully on failure.
    pub fn consolidate(&self, project: &str, memory_cfg: &MemoryConfig) -> Result<ConsolidationStats> {
        let now = now_secs();
        let mut stats = ConsolidationStats { ttl_pruned: 0, dedup_removed: 0, cap_removed: 0 };

        // --- Step 1: TTL pruning ---
        if memory_cfg.fact_ttl_days > 0 {
            let ttl_secs = memory_cfg.fact_ttl_days as i64 * 86_400;
            let cutoff = now - ttl_secs;
            let n = self.conn.execute(
                "DELETE FROM facts WHERE project = ?1 AND last_seen < ?2",
                params![project, cutoff],
            )?;
            stats.ttl_pruned = n;
            if n > 0 {
                debug!(project, count = n, "facts pruned by TTL");
            }
        }

        // --- Step 2: Near-duplicate removal ---
        // Load all facts ordered by most recently seen (keep the freshest).
        let mut stmt = self.conn.prepare(
            "SELECT id, content FROM facts WHERE project = ?1 ORDER BY last_seen DESC",
        )?;

        let rows: Vec<(i64, String)> = stmt
            .query_map(params![project], |row| Ok((row.get(0)?, row.get(1)?)))?
            .flatten()
            .collect();

        let mut seen_ids: Vec<i64> = Vec::new();
        let mut contents_kept: Vec<String> = Vec::new();
        let mut ids_to_delete: Vec<i64> = Vec::new();

        for (id, content) in rows {
            let is_dup = contents_kept.iter().any(|kept| are_near_duplicate(&content, kept));
            if is_dup {
                ids_to_delete.push(id);
            } else {
                seen_ids.push(id);
                contents_kept.push(content);
            }
        }

        for id in &ids_to_delete {
            self.conn.execute("DELETE FROM facts WHERE id = ?1", params![id])?;
            stats.dedup_removed += 1;
        }

        if stats.dedup_removed > 0 {
            debug!(project, count = stats.dedup_removed, "near-duplicate facts removed");
        }

        // --- Step 3: Cap enforcement (remove oldest beyond limit) ---
        let cap = memory_cfg.max_facts_per_project;
        if seen_ids.len() > cap {
            // seen_ids is already in DESC order (most recent first), so trim the tail.
            let to_remove = &seen_ids[cap..];
            for id in to_remove {
                self.conn.execute("DELETE FROM facts WHERE id = ?1", params![id])?;
                stats.cap_removed += 1;
            }
            if stats.cap_removed > 0 {
                debug!(project, count = stats.cap_removed, cap, "facts removed to enforce cap");
            }
        }

        Ok(stats)
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
                "You are a precise technical note-taker. Extract facts from coding conversations.",
            ),
            Message::user(&format!(
                "Extract 3-6 specific, concrete technical facts from this conversation.\n\
                 \n\
                 Include only:\n\
                 - Specific source files referenced (with paths)\n\
                 - Concrete architectural or design decisions made\n\
                 - Specific bugs found and how they were resolved\n\
                 - Exact function or struct names that were discussed or changed\n\
                 - Confirmed configuration values or dependencies\n\
                 \n\
                 Do NOT include:\n\
                 - General summaries or meta-commentary about the session\n\
                 - Vague statements like \"the code was improved\"\n\
                 - Questions or uncertainties\n\
                 - Anything about the user's goals or the assistant's behavior\n\
                 \n\
                 Write one fact per line. No numbering. No bullet points. Be specific and brief.\n\
                 \n\
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
        let mut skipped = 0usize;

        for line in response.lines() {
            let fact = line.trim();
            if fact.is_empty() {
                continue;
            }
            if !is_quality_fact(fact) {
                skipped += 1;
                continue;
            }
            match self.try_store_fact_deduped(project, fact) {
                Ok(true) => stored += 1,
                Ok(false) => skipped += 1,
                Err(e) => warn!(error = %e, "failed to store fact"),
            }
        }

        if stored > 0 {
            info!(project, stored, skipped, "session facts stored");
        } else if skipped > 0 {
            debug!(project, skipped, "all extracted facts were filtered or deduplicated");
        }
    }
}

// ---------------------------------------------------------------------------
// Quality filtering
// ---------------------------------------------------------------------------

/// Returns true if a candidate fact is worth storing.
///
/// Filters out:
/// - Too short (< 20 chars) or too long (> 300 chars)
/// - Questions
/// - Lines that start with a digit (numbered lists leaked from the model)
/// - Meta-commentary about the conversation or the user
fn is_quality_fact(fact: &str) -> bool {
    if fact.len() < 20 || fact.len() > 300 {
        return false;
    }
    if fact.ends_with('?') {
        return false;
    }
    // Numbered list lines from the model (e.g. "1. something", "3) something")
    let first = fact.chars().next().unwrap_or('x');
    if first.is_ascii_digit() {
        return false;
    }
    // Meta-commentary prefixes — these describe the session rather than facts from it.
    let lower = fact.to_lowercase();
    let meta_prefixes = [
        "the user ",
        "the assistant ",
        "the conversation ",
        "the session ",
        "in this conversation",
        "in this session",
        "this conversation",
        "this session",
        "i was asked",
        "we discussed",
        "the code was",
        "the developer ",
    ];
    for prefix in meta_prefixes {
        if lower.starts_with(prefix) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Near-duplicate detection
// ---------------------------------------------------------------------------

/// Returns true if two facts are near-duplicates of each other.
///
/// Uses Jaccard similarity on the set of lowercase word tokens.
/// Threshold: ≥ 0.70 overlap = considered a duplicate.
fn are_near_duplicate(a: &str, b: &str) -> bool {
    let tokens_a: HashSet<&str> = a.split_whitespace().collect();
    let tokens_b: HashSet<&str> = b.split_whitespace().collect();

    if tokens_a.is_empty() || tokens_b.is_empty() {
        return false;
    }

    let intersection = tokens_a.intersection(&tokens_b).count();
    let union = tokens_a.union(&tokens_b).count();

    if union == 0 {
        return false;
    }

    let jaccard = intersection as f64 / union as f64;
    jaccard >= 0.70
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_fact_rejects_short() {
        assert!(!is_quality_fact("too short"));
    }

    #[test]
    fn quality_fact_rejects_question() {
        assert!(!is_quality_fact("Is this the right approach for the cache?"));
    }

    #[test]
    fn quality_fact_rejects_numbered_line() {
        assert!(!is_quality_fact("1. The cache was refactored to use SQLite"));
    }

    #[test]
    fn quality_fact_rejects_meta_commentary() {
        assert!(!is_quality_fact("The user asked about fixing the inference loop"));
        assert!(!is_quality_fact("the session covered memory hardening topics"));
    }

    #[test]
    fn quality_fact_accepts_concrete_fact() {
        assert!(is_quality_fact(
            "src/memory/facts.rs uses Jaccard similarity for near-duplicate detection"
        ));
        assert!(is_quality_fact(
            "ExactCache stores entries in .local/cache.db with TTL and project fingerprint"
        ));
    }

    #[test]
    fn near_duplicate_detects_high_overlap() {
        let a = "src/inference/mod.rs owns the session messages and tool call loop";
        let b = "src/inference/mod.rs owns the session messages and the tool call loop";
        assert!(are_near_duplicate(a, b));
    }

    #[test]
    fn near_duplicate_allows_distinct_facts() {
        let a = "src/inference/mod.rs owns the session messages and tool call loop";
        let b = "src/cache/exact.rs stores cache entries with TTL and project fingerprint";
        assert!(!are_near_duplicate(a, b));
    }

    #[test]
    fn near_duplicate_edge_case_empty() {
        assert!(!are_near_duplicate("", "something"));
        assert!(!are_near_duplicate("something", ""));
    }
}
