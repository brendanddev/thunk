// src/memory/facts.rs
//
// Level 3: cross-session fact store.
//
// Facts are scoped per project and stored in .local/memory/facts.db.
// The verified-updates slice keeps durable memory strict:
// - facts are extracted per turn from an explicit evidence pack
// - facts must pass quality filters and evidence-anchor checks
// - stored facts carry provenance metadata (legacy vs verified)
//
// Consolidation still happens at session end and degrades gracefully.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};
use tracing::{debug, info, warn};

use super::retrieval::{query_terms, score_text};
use super::run_prompt_sync;
use crate::config::{self, MemoryConfig};
use crate::error::Result;
use crate::events::{FactProvenance, MemoryFactView, MemorySkippedReasonCount, MemoryUpdateReport};
use crate::inference::{InferenceBackend, Message};

const NO_FACTS_SENTINEL: &str = "NOTHING";
const MAX_FACT_LEN: usize = 300;
const MIN_FACT_LEN: usize = 20;
const MAX_TOOL_EVIDENCE_CHARS: usize = 600;
const MAX_REPLY_EVIDENCE_CHARS: usize = 1200;

pub struct FactStore {
    conn: Connection,
}

#[derive(Debug, Clone)]
pub struct StoredFact {
    pub content: String,
    pub provenance: FactProvenance,
}

#[derive(Debug, Clone)]
pub struct ToolEvidence {
    pub tool_name: String,
    pub argument: String,
    pub output: String,
    pub approved: bool,
}

#[derive(Debug, Clone, Default)]
pub struct TurnMemoryEvidence {
    pub user_prompt: String,
    pub summaries: Vec<(String, String)>,
    pub tool_results: Vec<ToolEvidence>,
    pub final_response: Option<String>,
}

pub struct ConsolidationStats {
    pub ttl_pruned: usize,
    pub dedup_removed: usize,
    pub cap_removed: usize,
}

enum StoreFactOutcome {
    Stored(StoredFact),
    Duplicate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum SkippedFactReason {
    Quality,
    Hedged,
    Unresolved,
    Unanchored,
    Duplicate,
}

impl SkippedFactReason {
    fn label(self) -> &'static str {
        match self {
            Self::Quality => "filtered quality",
            Self::Hedged => "hedged or uncertain",
            Self::Unresolved => "unresolved or intent",
            Self::Unanchored => "missing evidence anchor",
            Self::Duplicate => "duplicate",
        }
    }
}

impl TurnMemoryEvidence {
    pub fn new(prompt: String, summaries: Vec<(String, String)>) -> Self {
        Self {
            user_prompt: prompt,
            summaries,
            tool_results: Vec::new(),
            final_response: None,
        }
    }

    pub fn record_tool_result(
        &mut self,
        tool_name: impl Into<String>,
        argument: impl Into<String>,
        output: impl Into<String>,
        approved: bool,
    ) {
        self.tool_results.push(ToolEvidence {
            tool_name: tool_name.into(),
            argument: argument.into(),
            output: output.into(),
            approved,
        });
    }

    pub fn set_final_response(&mut self, response: impl Into<String>) {
        self.final_response = Some(response.into());
    }

    fn has_material_evidence(&self) -> bool {
        !self.user_prompt.trim().is_empty()
            || !self.summaries.is_empty()
            || !self.tool_results.is_empty()
            || self
                .final_response
                .as_ref()
                .map(|response| !response.trim().is_empty())
                .unwrap_or(false)
    }
}

impl FactStore {
    /// Open or create the shared fact database at .local/memory/facts.db.
    pub fn open() -> Result<Self> {
        let memory_dir = config::memory_dir()?;
        let db_path = memory_dir.join("facts.db");
        Self::open_at(&db_path)
    }

    fn open_at(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS facts (
                id           INTEGER PRIMARY KEY,
                project      TEXT    NOT NULL,
                content      TEXT    NOT NULL,
                provenance   TEXT    NOT NULL DEFAULT 'legacy',
                created_at   INTEGER NOT NULL,
                last_seen    INTEGER NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0
            );",
        )?;

        let _ = conn.execute(
            "ALTER TABLE facts ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0",
            [],
        );
        let _ = conn.execute(
            "ALTER TABLE facts ADD COLUMN provenance TEXT NOT NULL DEFAULT 'legacy'",
            [],
        );

        info!(db = %path.display(), "fact store opened");

        Ok(Self { conn })
    }

    /// Return up to `limit` facts for the project, ranked by keyword overlap
    /// with `query`. When `query` is empty, returns the most recently seen facts.
    pub fn get_relevant_facts(
        &self,
        project: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<StoredFact>> {
        let mut stmt = self.conn.prepare(
            "SELECT content, provenance FROM facts WHERE project = ?1 \
             ORDER BY last_seen DESC LIMIT 50",
        )?;

        let all: Vec<StoredFact> = stmt
            .query_map(params![project], |row| {
                Ok(StoredFact {
                    content: row.get(0)?,
                    provenance: parse_provenance(row.get::<_, String>(1)?.as_str()),
                })
            })?
            .flatten()
            .collect();

        if query.is_empty() {
            return Ok(all.into_iter().take(limit).collect());
        }

        let query_terms = query_terms(query);

        let mut scored: Vec<(usize, StoredFact)> = all
            .into_iter()
            .map(|fact| {
                let score = score_text(&query_terms, &fact.content);
                (score, fact)
            })
            .filter(|(score, _)| *score > 0)
            .collect();

        scored.sort_by(|a, b| b.0.cmp(&a.0));

        Ok(scored
            .into_iter()
            .map(|(_, fact)| fact)
            .take(limit)
            .collect())
    }

    pub fn verify_and_store_turn(
        &self,
        project: &str,
        evidence: &TurnMemoryEvidence,
        backend: &dyn InferenceBackend,
    ) -> MemoryUpdateReport {
        if !evidence.has_material_evidence() {
            return MemoryUpdateReport {
                accepted_facts: Vec::new(),
                skipped_reasons: Vec::new(),
                duplicate_count: 0,
            };
        }

        let response = match run_prompt_sync(backend, &build_fact_extraction_prompt(evidence)) {
            Ok(response) => response,
            Err(e) => {
                warn!(error = %e, "verified fact extraction prompt failed");
                return MemoryUpdateReport {
                    accepted_facts: Vec::new(),
                    skipped_reasons: vec![MemorySkippedReasonCount {
                        reason: SkippedFactReason::Quality.label().to_string(),
                        count: 1,
                    }],
                    duplicate_count: 0,
                };
            }
        };

        let anchors = evidence_anchors(evidence);
        let mut accepted = Vec::new();
        let mut skipped = BTreeMap::new();
        let mut duplicate_count = 0usize;

        for line in response.lines() {
            let fact = line.trim();
            if fact.is_empty() || fact.eq_ignore_ascii_case(NO_FACTS_SENTINEL) {
                continue;
            }

            match validate_candidate_fact(fact, &anchors) {
                Ok(()) => {
                    match self.try_store_fact_deduped(project, fact, FactProvenance::Verified) {
                        Ok(StoreFactOutcome::Stored(fact)) => {
                            accepted.push(MemoryFactView {
                                content: fact.content,
                                provenance: fact.provenance,
                            });
                        }
                        Ok(StoreFactOutcome::Duplicate) => {
                            duplicate_count += 1;
                            *skipped.entry(SkippedFactReason::Duplicate).or_insert(0) += 1;
                        }
                        Err(e) => warn!(error = %e, "failed to store verified fact"),
                    }
                }
                Err(reason) => {
                    *skipped.entry(reason).or_insert(0) += 1;
                }
            }
        }

        if !accepted.is_empty() {
            info!(
                project,
                accepted = accepted.len(),
                duplicate_count,
                "verified memory facts stored"
            );
        }

        MemoryUpdateReport {
            accepted_facts: accepted,
            skipped_reasons: skipped
                .into_iter()
                .map(|(reason, count)| MemorySkippedReasonCount {
                    reason: reason.label().to_string(),
                    count,
                })
                .collect(),
            duplicate_count,
        }
    }

    /// Prune stale facts, remove near-duplicates, and enforce the per-project cap.
    pub fn consolidate(
        &self,
        project: &str,
        memory_cfg: &MemoryConfig,
    ) -> Result<ConsolidationStats> {
        let now = now_secs();
        let mut stats = ConsolidationStats {
            ttl_pruned: 0,
            dedup_removed: 0,
            cap_removed: 0,
        };

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

        let mut stmt = self
            .conn
            .prepare("SELECT id, content FROM facts WHERE project = ?1 ORDER BY last_seen DESC")?;

        let rows: Vec<(i64, String)> = stmt
            .query_map(params![project], |row| Ok((row.get(0)?, row.get(1)?)))?
            .flatten()
            .collect();

        let mut seen_ids: Vec<i64> = Vec::new();
        let mut contents_kept: Vec<String> = Vec::new();
        let mut ids_to_delete: Vec<i64> = Vec::new();

        for (id, content) in rows {
            let is_dup = contents_kept
                .iter()
                .any(|kept| are_near_duplicate(&content, kept));
            if is_dup {
                ids_to_delete.push(id);
            } else {
                seen_ids.push(id);
                contents_kept.push(content);
            }
        }

        for id in &ids_to_delete {
            self.conn
                .execute("DELETE FROM facts WHERE id = ?1", params![id])?;
            stats.dedup_removed += 1;
        }

        if stats.dedup_removed > 0 {
            debug!(
                project,
                count = stats.dedup_removed,
                "near-duplicate facts removed"
            );
        }

        let cap = memory_cfg.max_facts_per_project;
        if seen_ids.len() > cap {
            let to_remove = &seen_ids[cap..];
            for id in to_remove {
                self.conn
                    .execute("DELETE FROM facts WHERE id = ?1", params![id])?;
                stats.cap_removed += 1;
            }
            if stats.cap_removed > 0 {
                debug!(
                    project,
                    count = stats.cap_removed,
                    cap,
                    "facts removed to enforce cap"
                );
            }
        }

        Ok(stats)
    }

    fn try_store_fact_deduped(
        &self,
        project: &str,
        content: &str,
        provenance: FactProvenance,
    ) -> Result<StoreFactOutcome> {
        let now = now_secs();
        let mut stmt = self
            .conn
            .prepare("SELECT id, content, provenance FROM facts WHERE project = ?1")?;

        let existing: Vec<(i64, String, FactProvenance)> = stmt
            .query_map(params![project], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    parse_provenance(row.get::<_, String>(2)?.as_str()),
                ))
            })?
            .flatten()
            .collect();

        for (id, existing_content, existing_provenance) in &existing {
            if are_near_duplicate(content, existing_content) {
                self.conn.execute(
                    "UPDATE facts SET last_seen = ?1, provenance = ?2 WHERE id = ?3",
                    params![
                        now,
                        merged_provenance(*existing_provenance, provenance).as_db_str(),
                        id
                    ],
                )?;
                debug!(project, "fact deduplicated (near-duplicate found)");
                return Ok(StoreFactOutcome::Duplicate);
            }
        }

        self.conn.execute(
            "INSERT INTO facts (project, content, provenance, created_at, last_seen, access_count) \
             VALUES (?1, ?2, ?3, ?4, ?4, 0)",
            params![project, content, provenance.as_db_str(), now],
        )?;

        debug!(project, provenance = provenance.as_label(), "fact stored");
        Ok(StoreFactOutcome::Stored(StoredFact {
            content: content.to_string(),
            provenance,
        }))
    }
}

trait FactProvenanceExt {
    fn as_db_str(self) -> &'static str;
    fn as_label(self) -> &'static str;
}

impl FactProvenanceExt for FactProvenance {
    fn as_db_str(self) -> &'static str {
        match self {
            FactProvenance::Legacy => "legacy",
            FactProvenance::Verified => "verified",
        }
    }

    fn as_label(self) -> &'static str {
        self.as_db_str()
    }
}

fn parse_provenance(raw: &str) -> FactProvenance {
    match raw {
        "verified" => FactProvenance::Verified,
        _ => FactProvenance::Legacy,
    }
}

fn merged_provenance(existing: FactProvenance, incoming: FactProvenance) -> FactProvenance {
    match (existing, incoming) {
        (FactProvenance::Verified, _) | (_, FactProvenance::Verified) => FactProvenance::Verified,
        _ => FactProvenance::Legacy,
    }
}

fn build_fact_extraction_prompt(evidence: &TurnMemoryEvidence) -> Vec<Message> {
    let mut body = String::new();
    body.push_str(
        "Extract 0-4 verified technical facts from this evidence pack.\n\
         Only emit a fact when it is directly supported by the evidence.\n\
         Facts must be concrete, resolved outcomes about files, symbols, config values, URLs/hosts, or approved tool results.\n\
         Do not include hedges, TODOs, plans, user intent, or meta commentary.\n\
         If no verified facts exist, reply with NOTHING.\n\
         Write one fact per line with no bullets or numbering.\n\n",
    );

    if !evidence.user_prompt.trim().is_empty() {
        body.push_str("User prompt:\n");
        body.push_str(&evidence.user_prompt);
        body.push_str("\n\n");
    }

    if !evidence.summaries.is_empty() {
        body.push_str("Indexed summaries:\n");
        for (path, summary) in &evidence.summaries {
            body.push_str("- ");
            body.push_str(path);
            body.push_str(": ");
            body.push_str(summary);
            body.push('\n');
        }
        body.push('\n');
    }

    if !evidence.tool_results.is_empty() {
        body.push_str("Tool outcomes:\n");
        for tool in &evidence.tool_results {
            body.push_str("- ");
            body.push_str(if tool.approved {
                "approved "
            } else {
                "read-only "
            });
            body.push_str(&tool.tool_name);
            body.push('(');
            body.push_str(&tool.argument);
            body.push_str("): ");
            body.push_str(&clip_text(&tool.output, MAX_TOOL_EVIDENCE_CHARS));
            body.push('\n');
        }
        body.push('\n');
    }

    if let Some(final_response) = &evidence.final_response {
        if !final_response.trim().is_empty() {
            body.push_str("Assistant final reply:\n");
            body.push_str(&clip_text(final_response, MAX_REPLY_EVIDENCE_CHARS));
        }
    }

    vec![
        Message::system(
            "You are a strict technical verifier. Emit only durable facts that are explicitly grounded in the provided evidence.",
        ),
        Message::user(&body),
    ]
}

fn validate_candidate_fact(
    fact: &str,
    anchors: &HashSet<String>,
) -> std::result::Result<(), SkippedFactReason> {
    if !is_quality_fact(fact) {
        return Err(SkippedFactReason::Quality);
    }
    if contains_hedged_language(fact) {
        return Err(SkippedFactReason::Hedged);
    }
    if contains_unresolved_language(fact) {
        return Err(SkippedFactReason::Unresolved);
    }
    if !contains_anchor(fact, anchors) {
        return Err(SkippedFactReason::Unanchored);
    }
    Ok(())
}

fn evidence_anchors(evidence: &TurnMemoryEvidence) -> HashSet<String> {
    let mut anchors = HashSet::new();
    collect_anchor_tokens(&evidence.user_prompt, &mut anchors);
    for (path, summary) in &evidence.summaries {
        anchors.insert(path.to_lowercase());
        collect_anchor_tokens(path, &mut anchors);
        collect_anchor_tokens(summary, &mut anchors);
    }
    for tool in &evidence.tool_results {
        collect_anchor_tokens(&tool.tool_name, &mut anchors);
        collect_anchor_tokens(&tool.argument, &mut anchors);
        collect_anchor_tokens(
            &clip_text(&tool.output, MAX_TOOL_EVIDENCE_CHARS),
            &mut anchors,
        );
    }
    if let Some(final_response) = &evidence.final_response {
        collect_anchor_tokens(final_response, &mut anchors);
    }
    anchors
}

fn collect_anchor_tokens(text: &str, anchors: &mut HashSet<String>) {
    for token in tokenize(text) {
        if token.len() < 4 {
            continue;
        }
        if token.contains('/')
            || token.contains('.')
            || token.contains("::")
            || token.contains('_')
            || token.contains("://")
            || token.chars().any(|c| c.is_ascii_uppercase())
        {
            anchors.insert(token.to_lowercase());
        }
    }
}

fn contains_anchor(fact: &str, anchors: &HashSet<String>) -> bool {
    let fact_lower = fact.to_lowercase();
    tokenize(fact).into_iter().any(|token| {
        let token = token.to_lowercase();
        anchors.contains(&token) || fact_lower.contains(&token) && anchors.contains(&token)
    })
}

fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|token| {
            token
                .trim_matches(|c: char| {
                    matches!(
                        c,
                        ',' | '.'
                            | ':'
                            | ';'
                            | '('
                            | ')'
                            | '['
                            | ']'
                            | '{'
                            | '}'
                            | '"'
                            | '\''
                            | '`'
                    )
                })
                .to_string()
        })
        .filter(|token| !token.is_empty())
        .collect()
}

fn clip_text(text: &str, max_chars: usize) -> String {
    let total = text.chars().count();
    if total <= max_chars {
        return text.to_string();
    }
    let clipped: String = text.chars().take(max_chars).collect();
    format!("{clipped}…")
}

fn is_quality_fact(fact: &str) -> bool {
    if fact.len() < MIN_FACT_LEN || fact.len() > MAX_FACT_LEN {
        return false;
    }
    if fact.ends_with('?') {
        return false;
    }
    let first = fact.chars().next().unwrap_or('x');
    if first.is_ascii_digit() {
        return false;
    }
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

fn contains_hedged_language(fact: &str) -> bool {
    let lower = fact.to_lowercase();
    ["maybe", "seems", "likely", "probably", "might", "could"]
        .iter()
        .any(|word| lower.contains(word))
}

fn contains_unresolved_language(fact: &str) -> bool {
    let lower = fact.to_lowercase();
    [
        "todo",
        "fixme",
        "should ",
        "needs ",
        "need to",
        "plan to",
        "consider",
        "want to",
        "later",
        "next step",
    ]
    .iter()
    .any(|word| lower.contains(word))
}

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
    use rusqlite::Connection;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_db_path(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!("params-facts-test-{label}-{nonce}.db"))
    }

    #[test]
    fn quality_fact_rejects_short() {
        assert!(!is_quality_fact("too short"));
    }

    #[test]
    fn quality_fact_rejects_question() {
        assert!(!is_quality_fact(
            "Is this the right approach for the cache?"
        ));
    }

    #[test]
    fn quality_fact_rejects_meta_commentary() {
        assert!(!is_quality_fact(
            "The user asked about fixing the inference loop"
        ));
    }

    #[test]
    fn quality_fact_accepts_concrete_fact() {
        assert!(is_quality_fact(
            "src/memory/facts.rs uses Jaccard similarity for near-duplicate detection"
        ));
    }

    #[test]
    fn validation_rejects_hedged_fact() {
        let mut anchors = HashSet::new();
        anchors.insert("src/main.rs".to_string());
        assert_eq!(
            validate_candidate_fact("src/main.rs probably updates the cache logic", &anchors),
            Err(SkippedFactReason::Hedged)
        );
    }

    #[test]
    fn validation_rejects_unanchored_fact() {
        let anchors = HashSet::new();
        assert_eq!(
            validate_candidate_fact("src/main.rs updates the cache logic", &anchors),
            Err(SkippedFactReason::Unanchored)
        );
    }

    #[test]
    fn validation_accepts_anchored_fact() {
        let mut anchors = HashSet::new();
        anchors.insert("src/main.rs".to_string());
        assert_eq!(
            validate_candidate_fact("src/main.rs updates the cache logic", &anchors),
            Ok(())
        );
    }

    #[test]
    fn near_duplicate_detects_high_overlap() {
        let a = "src/inference/mod.rs owns the session messages and tool call loop";
        let b = "src/inference/mod.rs owns the session messages and the tool call loop";
        assert!(are_near_duplicate(a, b));
    }

    #[test]
    fn turn_anchors_include_paths_and_symbols() {
        let evidence = TurnMemoryEvidence {
            user_prompt: "Review src/main.rs and FactStore".to_string(),
            summaries: vec![(
                "src/memory/facts.rs".to_string(),
                "FactStore extracts verified facts".to_string(),
            )],
            tool_results: Vec::new(),
            final_response: Some("FactStore now records verified facts".to_string()),
        };
        let anchors = evidence_anchors(&evidence);
        assert!(anchors.contains("src/main.rs"));
        assert!(anchors.contains("factstore"));
        assert!(anchors.contains("src/memory/facts.rs"));
    }

    #[test]
    fn store_promotes_legacy_duplicate_to_verified() {
        let path = temp_db_path("promote");
        let store = FactStore::open_at(&path).expect("open fact store");
        store
            .try_store_fact_deduped(
                "project",
                "src/main.rs updates cache stats in the sidebar",
                FactProvenance::Legacy,
            )
            .expect("store legacy fact");

        let outcome = store
            .try_store_fact_deduped(
                "project",
                "src/main.rs updates cache stats in the sidebar",
                FactProvenance::Verified,
            )
            .expect("store verified duplicate");
        assert!(matches!(outcome, StoreFactOutcome::Duplicate));

        let facts = store
            .get_relevant_facts("project", "", 10)
            .expect("load facts");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].provenance, FactProvenance::Verified);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn provenance_migration_defaults_existing_rows_to_legacy() {
        let path = temp_db_path("migration");
        let conn = Connection::open(&path).expect("open sqlite");
        conn.execute_batch(
            "CREATE TABLE facts (
                id INTEGER PRIMARY KEY,
                project TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                last_seen INTEGER NOT NULL
            );
            INSERT INTO facts (project, content, created_at, last_seen)
            VALUES ('project', 'src/main.rs updates cache stats in the sidebar', 1, 1);",
        )
        .expect("seed legacy table");
        drop(conn);

        let store = FactStore::open_at(&path).expect("migrate store");
        let facts = store
            .get_relevant_facts("project", "", 10)
            .expect("load facts");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].provenance, FactProvenance::Legacy);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn relevant_facts_are_ranked_by_prompt_match() {
        let path = temp_db_path("ranking");
        let store = FactStore::open_at(&path).expect("open fact store");
        store
            .try_store_fact_deduped(
                "project",
                "src/session/mod.rs resolves session selectors by unique id prefix",
                FactProvenance::Verified,
            )
            .expect("store first fact");
        store
            .try_store_fact_deduped(
                "project",
                "Cache stats are shown in the runtime header",
                FactProvenance::Verified,
            )
            .expect("store second fact");

        let facts = store
            .get_relevant_facts("project", "src/session/mod.rs selector", 5)
            .expect("load ranked facts");

        assert_eq!(
            facts.first().map(|fact| fact.content.as_str()),
            Some("src/session/mod.rs resolves session selectors by unique id prefix")
        );

        let _ = fs::remove_file(path);
    }
}
