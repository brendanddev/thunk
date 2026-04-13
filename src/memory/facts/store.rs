use std::collections::BTreeMap;
use std::path::Path;

use rusqlite::{params, Connection};
use tracing::{debug, info, warn};

use crate::config::{self, MemoryConfig};
use crate::error::Result;
use crate::events::{FactProvenance, MemoryFactView, MemorySkippedReasonCount, MemoryUpdateReport};
use crate::inference::InferenceBackend;
use crate::memory::run_prompt_sync;

use super::prompting::{
    build_fact_extraction_prompt, merged_provenance, now_secs, parse_provenance, FactProvenanceExt,
};
use super::quality::{
    are_near_duplicate, evidence_anchors, is_retrievable_project_fact, validate_candidate_fact,
};
use super::{
    ConsolidationStats, FactStore, SkippedFactReason, StoreFactOutcome, StoredFact,
    TurnMemoryEvidence,
};

impl FactStore {
    /// Open or create the shared fact database at .local/memory/facts.db.
    pub fn open() -> Result<Self> {
        let memory_dir = config::memory_dir()?;
        let db_path = memory_dir.join("facts.db");
        Self::open_at(&db_path)
    }

    pub(crate) fn open_at(path: &Path) -> Result<Self> {
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

        let all: Vec<StoredFact> = all
            .into_iter()
            .filter(|fact| is_retrievable_project_fact(&fact.content))
            .collect();

        if query.is_empty() {
            return Ok(all.into_iter().take(limit).collect());
        }

        let query_terms = crate::memory::retrieval::query_terms(query);

        let mut scored: Vec<(usize, StoredFact)> = all
            .into_iter()
            .map(|fact| {
                let score = crate::memory::retrieval::score_text(&query_terms, &fact.content);
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

    pub fn prune_irrelevant_facts(&self, project: &str) -> Result<usize> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, content FROM facts WHERE project = ?1 ORDER BY last_seen DESC")?;

        let rows: Vec<(i64, String)> = stmt
            .query_map(params![project], |row| Ok((row.get(0)?, row.get(1)?)))?
            .flatten()
            .collect();

        let remove_ids: Vec<i64> = rows
            .into_iter()
            .filter_map(|(id, content)| {
                if is_retrievable_project_fact(&content) {
                    None
                } else {
                    Some(id)
                }
            })
            .collect();

        for id in &remove_ids {
            self.conn
                .execute("DELETE FROM facts WHERE id = ?1", params![id])?;
        }

        if !remove_ids.is_empty() {
            debug!(
                project,
                removed = remove_ids.len(),
                "irrelevant facts pruned"
            );
        }

        Ok(remove_ids.len())
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
            if fact.is_empty() || fact.eq_ignore_ascii_case(super::NO_FACTS_SENTINEL) {
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

#[cfg(test)]
pub(crate) trait StoreFactTestExt {
    fn try_store_fact_deduped(
        &self,
        project: &str,
        content: &str,
        provenance: FactProvenance,
    ) -> Result<StoreFactOutcome>;
    fn open_at(path: &Path) -> Result<Self>
    where
        Self: Sized;
}

#[cfg(test)]
impl StoreFactTestExt for FactStore {
    fn try_store_fact_deduped(
        &self,
        project: &str,
        content: &str,
        provenance: FactProvenance,
    ) -> Result<StoreFactOutcome> {
        FactStore::try_store_fact_deduped(self, project, content, provenance)
    }

    fn open_at(path: &Path) -> Result<Self> {
        FactStore::open_at(path)
    }
}
