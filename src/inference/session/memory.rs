use std::sync::mpsc::Sender;

use crate::events::{
    InferenceEvent, MemoryConsolidationView, MemoryFactView, MemorySessionExcerptView,
    MemorySnapshot, MemoryUpdateReport, ProgressStatus,
};
use crate::hooks::{HookEvent, Hooks};
use crate::memory::{facts::FactStore, index::ProjectIndex};
use crate::session::{SessionExcerptMatch, SessionStore};

use super::super::runtime::{emit_trace, summary_limit};
use super::auto_inspect::AutoInspectIntent;

#[derive(Default)]
pub(super) struct RuntimeMemoryState {
    pub(super) loaded_facts: Vec<MemoryFactView>,
    pub(super) last_summary_paths: Vec<String>,
    pub(super) last_retrieval_query: Option<String>,
    pub(super) last_selected_facts: Vec<MemoryFactView>,
    pub(super) last_selected_session_excerpts: Vec<MemorySessionExcerptView>,
    pub(super) last_update: Option<MemoryUpdateReport>,
    pub(super) last_consolidation: Option<MemoryConsolidationView>,
}

impl RuntimeMemoryState {
    fn snapshot(&self) -> MemorySnapshot {
        MemorySnapshot {
            loaded_facts: self.loaded_facts.clone(),
            last_summary_paths: self.last_summary_paths.clone(),
            last_retrieval_query: self.last_retrieval_query.clone(),
            last_selected_facts: self.last_selected_facts.clone(),
            last_selected_session_excerpts: self.last_selected_session_excerpts.clone(),
            last_update: self.last_update.clone(),
            last_consolidation: self.last_consolidation.clone(),
        }
    }
}

pub(super) fn emit_memory_state(
    token_tx: &Sender<InferenceEvent>,
    memory_state: &RuntimeMemoryState,
) {
    let _ = token_tx.send(InferenceEvent::MemoryState(memory_state.snapshot()));
}

pub(super) fn refresh_loaded_facts(
    memory_state: &mut RuntimeMemoryState,
    fact_store: Option<&FactStore>,
    project_name: &str,
) {
    memory_state.loaded_facts = fact_store
        .and_then(|store| store.get_relevant_facts(project_name, "", 5).ok())
        .unwrap_or_default()
        .into_iter()
        .map(|fact| MemoryFactView {
            content: fact.content,
            provenance: fact.provenance,
        })
        .collect();
}

#[derive(Default)]
pub(super) struct RetrievalBundle {
    pub(super) summaries: Vec<(String, String)>,
    pub(super) facts: Vec<MemoryFactView>,
    pub(super) session_excerpts: Vec<MemorySessionExcerptView>,
}

pub(super) fn memory_fact_lines(facts: &[MemoryFactView]) -> Vec<String> {
    facts.iter().map(|fact| fact.content.clone()).collect()
}

fn map_session_excerpt(match_: SessionExcerptMatch) -> MemorySessionExcerptView {
    MemorySessionExcerptView {
        session_label: match_.session_label,
        role: match_.role,
        excerpt: match_.excerpt,
    }
}

pub(super) fn clear_memory_retrieval(memory_state: &mut RuntimeMemoryState) {
    memory_state.last_summary_paths.clear();
    memory_state.last_retrieval_query = None;
    memory_state.last_selected_facts.clear();
    memory_state.last_selected_session_excerpts.clear();
}

pub(super) fn set_memory_retrieval(
    memory_state: &mut RuntimeMemoryState,
    query: &str,
    bundle: &RetrievalBundle,
) {
    memory_state.last_summary_paths = bundle
        .summaries
        .iter()
        .map(|(path, _)| path.clone())
        .collect();
    memory_state.last_retrieval_query = Some(query.to_string());
    memory_state.last_selected_facts = bundle.facts.clone();
    memory_state.last_selected_session_excerpts = bundle.session_excerpts.clone();
}

pub(super) fn suppress_retrieval_for_auto_inspection(intent: AutoInspectIntent) -> bool {
    matches!(
        intent,
        AutoInspectIntent::WhereIsImplementation
            | AutoInspectIntent::FeatureTrace
            | AutoInspectIntent::ConfigLocate
    )
}

pub(super) fn collect_retrieval_bundle(
    prompt: &str,
    eco_enabled: bool,
    project_name: &str,
    project_index: Option<&ProjectIndex>,
    fact_store: Option<&FactStore>,
    session_store: Option<&SessionStore>,
    active_session_id: Option<&str>,
    loaded_facts: &[MemoryFactView],
) -> RetrievalBundle {
    let fact_limit = if eco_enabled { 2 } else { 4 };
    let session_limit = if eco_enabled { 1 } else { 2 };

    let summaries = project_index
        .and_then(|index| index.find_relevant(prompt, summary_limit(eco_enabled)).ok())
        .unwrap_or_default();

    let facts = if let Some(store) = fact_store {
        store
            .get_relevant_facts(project_name, prompt, fact_limit)
            .map(|facts| {
                facts
                    .into_iter()
                    .map(|fact| MemoryFactView {
                        content: fact.content,
                        provenance: fact.provenance,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    } else {
        let query = crate::memory::retrieval::query_terms(prompt);
        let mut scored = loaded_facts
            .iter()
            .cloned()
            .filter_map(|fact| {
                let score = crate::memory::retrieval::score_text(&query, &fact.content);
                if score == 0 {
                    None
                } else {
                    Some((score, fact))
                }
            })
            .collect::<Vec<_>>();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored.truncate(fact_limit);
        scored.into_iter().map(|(_, fact)| fact).collect()
    };

    let session_excerpts = session_store
        .and_then(|store| {
            store
                .search_session_excerpts(prompt, active_session_id, session_limit)
                .ok()
        })
        .unwrap_or_default()
        .into_iter()
        .map(map_session_excerpt)
        .collect::<Vec<_>>();

    RetrievalBundle {
        summaries,
        facts,
        session_excerpts,
    }
}

pub(super) fn retrieval_trace_label(bundle: &RetrievalBundle) -> Option<String> {
    let mut parts = Vec::new();
    if !bundle.summaries.is_empty() {
        parts.push(format!(
            "{} summar{}",
            bundle.summaries.len(),
            if bundle.summaries.len() == 1 {
                "y"
            } else {
                "ies"
            }
        ));
    }
    if !bundle.facts.is_empty() {
        parts.push(format!(
            "{} fact{}",
            bundle.facts.len(),
            if bundle.facts.len() == 1 { "" } else { "s" }
        ));
    }
    if !bundle.session_excerpts.is_empty() {
        parts.push(format!(
            "{} session excerpt{}",
            bundle.session_excerpts.len(),
            if bundle.session_excerpts.len() == 1 {
                ""
            } else {
                "s"
            }
        ));
    }

    if parts.is_empty() {
        None
    } else {
        Some(format!("memory: selected {}", parts.join(", ")))
    }
}

pub(super) fn format_memory_recall(query: &str, bundle: &RetrievalBundle) -> String {
    let mut lines = vec![format!("memory recall for `{query}`:")];

    if bundle.summaries.is_empty() {
        lines.push("  summaries: (none)".to_string());
    } else {
        lines.push("  summaries:".to_string());
        for (path, summary) in &bundle.summaries {
            lines.push(format!("    - {}: {}", path, summary));
        }
    }

    if bundle.facts.is_empty() {
        lines.push("  facts: (none)".to_string());
    } else {
        lines.push("  facts:".to_string());
        for fact in &bundle.facts {
            let label = match fact.provenance {
                crate::events::FactProvenance::Legacy => "legacy",
                crate::events::FactProvenance::Verified => "verified",
            };
            lines.push(format!("    - [{label}] {}", fact.content));
        }
    }

    if bundle.session_excerpts.is_empty() {
        lines.push("  prior sessions: (none)".to_string());
    } else {
        lines.push("  prior sessions:".to_string());
        for excerpt in &bundle.session_excerpts {
            lines.push(format!(
                "    - {} · {}: {}",
                excerpt.session_label, excerpt.role, excerpt.excerpt
            ));
        }
    }

    lines.join("\n")
}

fn skipped_fact_count(report: &MemoryUpdateReport) -> usize {
    report
        .skipped_reasons
        .iter()
        .map(|reason| reason.count)
        .sum()
}

pub(super) fn apply_memory_update(
    token_tx: &Sender<InferenceEvent>,
    hooks: &Hooks,
    memory_state: &mut RuntimeMemoryState,
    update: MemoryUpdateReport,
) {
    let accepted_count = update.accepted_facts.len();
    let skipped_count = skipped_fact_count(&update);

    for fact in &update.accepted_facts {
        if !memory_state
            .loaded_facts
            .iter()
            .any(|existing| existing.content == fact.content)
        {
            memory_state.loaded_facts.push(fact.clone());
        }
    }

    memory_state.last_update = Some(update.clone());
    emit_memory_state(token_tx, memory_state);
    hooks.dispatch(HookEvent::MemoryUpdateEvaluated {
        accepted_count,
        skipped_count,
        duplicate_count: update.duplicate_count,
    });

    if accepted_count > 0 {
        emit_trace(
            token_tx,
            ProgressStatus::Finished,
            &format!(
                "memory: stored {accepted_count} fact{}",
                if accepted_count == 1 { "" } else { "s" }
            ),
            false,
        );
    } else if skipped_count > 0 || update.duplicate_count > 0 {
        emit_trace(
            token_tx,
            ProgressStatus::Finished,
            &format!(
                "memory: skipped {} fact{}",
                skipped_count + update.duplicate_count,
                if skipped_count + update.duplicate_count == 1 {
                    ""
                } else {
                    "s"
                }
            ),
            false,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::FactProvenance;

    #[test]
    fn set_memory_retrieval_updates_snapshot_fields() {
        let mut state = RuntimeMemoryState::default();
        let bundle = RetrievalBundle {
            summaries: vec![("src/main.rs".to_string(), "main entrypoint".to_string())],
            facts: vec![MemoryFactView {
                content: "src/main.rs updates cache stats".to_string(),
                provenance: FactProvenance::Verified,
            }],
            session_excerpts: vec![MemorySessionExcerptView {
                session_label: "review".to_string(),
                role: "assistant".to_string(),
                excerpt: "cache stats are shown in the runtime bar".to_string(),
            }],
        };

        set_memory_retrieval(&mut state, "cache stats", &bundle);
        let snapshot = state.snapshot();

        assert_eq!(
            snapshot.last_retrieval_query.as_deref(),
            Some("cache stats")
        );
        assert_eq!(snapshot.last_summary_paths, vec!["src/main.rs".to_string()]);
        assert_eq!(snapshot.last_selected_facts.len(), 1);
        assert_eq!(snapshot.last_selected_session_excerpts.len(), 1);
    }

    #[test]
    fn retrieval_trace_label_summarizes_selected_sources() {
        let label = retrieval_trace_label(&RetrievalBundle {
            summaries: vec![("a.rs".to_string(), "summary".to_string())],
            facts: vec![MemoryFactView {
                content: "fact".to_string(),
                provenance: FactProvenance::Verified,
            }],
            session_excerpts: vec![MemorySessionExcerptView {
                session_label: "review".to_string(),
                role: "assistant".to_string(),
                excerpt: "excerpt".to_string(),
            }],
        });

        assert_eq!(
            label.as_deref(),
            Some("memory: selected 1 summary, 1 fact, 1 session excerpt")
        );
    }

    #[test]
    fn memory_recall_format_groups_summaries_facts_and_sessions() {
        let output = format_memory_recall(
            "cache stats",
            &RetrievalBundle {
                summaries: vec![("src/main.rs".to_string(), "entrypoint".to_string())],
                facts: vec![MemoryFactView {
                    content: "src/main.rs updates cache stats".to_string(),
                    provenance: FactProvenance::Verified,
                }],
                session_excerpts: vec![MemorySessionExcerptView {
                    session_label: "review".to_string(),
                    role: "assistant".to_string(),
                    excerpt: "cache stats are shown in the runtime bar".to_string(),
                }],
            },
        );

        assert!(output.contains("memory recall for `cache stats`:"));
        assert!(output.contains("summaries:"));
        assert!(output.contains("[verified]"));
        assert!(output.contains("prior sessions:"));
        assert!(output.contains("review · assistant"));
    }
}
