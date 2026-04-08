use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};

use tracing::{debug, info, warn};

use crate::cache::ExactCache;
use crate::config;
use crate::debug_log;
use crate::error::Result;
use crate::events::{
    InferenceEvent, MemoryConsolidationView, MemoryFactView, MemorySessionExcerptView,
    MemorySnapshot, MemoryUpdateReport, ProgressStatus, SessionInfo,
};
use crate::hooks::{HookEvent, Hooks};
use crate::memory::{
    compression,
    facts::{FactStore, TurnMemoryEvidence},
    index::ProjectIndex,
};
use crate::session::{
    display_name, list_label, short_id, SessionExcerptMatch, SessionExportFormat, SessionStore,
    SessionSummary,
};
use crate::tools::{BashTool, ListDir, ReadFile, Tool, ToolRegistry, ToolResult, ToolRunResult};

use super::approval::{handle_pending_action, ApprovalContext};
use super::budget::{
    emit_budget_update, emit_cache_update, estimate_message_tokens, record_generation_budget,
    SessionBudget, SessionCacheStats,
};
use super::cache::{generate_with_cache, store_exact_cache, store_prompt_level_cache, CacheMode};
use super::indexing::{run_idle_index_step, IncrementalIndexState, IDLE_INDEX_POLL_INTERVAL};
use super::reflection::reflect_response;
use super::runtime::{
    eco_tool_result_limit, effective_reflection, emit_generation_started, emit_trace,
    log_debug_response, summary_limit,
};
use super::{build_system_prompt, load_backend_with_fallback, Message, SessionCommand};

#[derive(Clone, Copy, Default)]
pub struct SessionRuntimeOptions {
    pub no_resume: bool,
}

fn session_info(summary: &SessionSummary) -> SessionInfo {
    SessionInfo {
        id: summary.id.clone(),
        name: summary.name.clone(),
        message_count: summary.message_count,
    }
}

#[derive(Default)]
struct RuntimeMemoryState {
    loaded_facts: Vec<MemoryFactView>,
    last_summary_paths: Vec<String>,
    last_retrieval_query: Option<String>,
    last_selected_facts: Vec<MemoryFactView>,
    last_selected_session_excerpts: Vec<MemorySessionExcerptView>,
    last_update: Option<MemoryUpdateReport>,
    last_consolidation: Option<MemoryConsolidationView>,
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

fn emit_memory_state(token_tx: &Sender<InferenceEvent>, memory_state: &RuntimeMemoryState) {
    let _ = token_tx.send(InferenceEvent::MemoryState(memory_state.snapshot()));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AutoInspectIntent {
    RepoOverview,
    DirectoryOverview,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AutoInspectStep {
    label: String,
    tool_name: &'static str,
    argument: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AutoInspectPlan {
    thinking: &'static str,
    status_label: &'static str,
    context_label: &'static str,
    steps: Vec<AutoInspectStep>,
}

struct AutoInspectOutcome {
    hidden_context: Option<String>,
    tool_results: Vec<ToolResult>,
}

fn detect_auto_inspect_intent(prompt: &str) -> Option<AutoInspectIntent> {
    let normalized = normalize_intent_text(prompt);
    if normalized.starts_with('/') {
        return None;
    }

    if [
        "whats in this repo",
        "what is in this repo",
        "whats in this project",
        "what is in this project",
        "summarize this repo",
        "summarize this project",
        "summarize this codebase",
        "what does this repo do",
        "what does this project do",
    ]
    .iter()
    .any(|pattern| normalized.contains(pattern))
    {
        return Some(AutoInspectIntent::RepoOverview);
    }

    if [
        "whats in this directory",
        "what is in this directory",
        "whats in this folder",
        "what is in this folder",
        "whats here",
        "what is here",
    ]
    .iter()
    .any(|pattern| normalized.contains(pattern))
    {
        return Some(AutoInspectIntent::DirectoryOverview);
    }

    None
}

fn normalize_intent_text(text: &str) -> String {
    let stripped = text.to_ascii_lowercase().replace(['\'', '’'], "");
    stripped
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '/' {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn plan_auto_inspection(intent: AutoInspectIntent, project_root: &Path) -> AutoInspectPlan {
    let mut steps = vec![AutoInspectStep {
        label: "List .".to_string(),
        tool_name: "list_dir",
        argument: ".".to_string(),
    }];

    let mut push_read = |rel: &str| {
        if project_root.join(rel).is_file() {
            steps.push(AutoInspectStep {
                label: format!("Read {rel}"),
                tool_name: "read_file",
                argument: rel.to_string(),
            });
        }
    };

    match intent {
        AutoInspectIntent::RepoOverview => {
            push_read("README.md");
            push_read("Cargo.toml");
            if project_root.join("src/main.rs").is_file() {
                push_read("src/main.rs");
            } else {
                push_read("src/lib.rs");
            }

            AutoInspectPlan {
                thinking: "Thinking: exploring the repo structure and key project docs.",
                status_label: "inspecting repo...",
                context_label: "this repo summary request",
                steps,
            }
        }
        AutoInspectIntent::DirectoryOverview => {
            push_read("README.md");
            for manifest in ["Cargo.toml", "package.json", "pyproject.toml", "go.mod"] {
                if project_root.join(manifest).is_file() {
                    push_read(manifest);
                    break;
                }
            }

            AutoInspectPlan {
                thinking: "Thinking: checking the current directory and its key files.",
                status_label: "inspecting directory...",
                context_label: "this directory summary request",
                steps,
            }
        }
    }
}

fn run_auto_inspection(
    plan: &AutoInspectPlan,
    token_tx: &Sender<InferenceEvent>,
    eco_enabled: bool,
) -> AutoInspectOutcome {
    let _ = token_tx.send(InferenceEvent::SystemMessage(plan.thinking.to_string()));

    let mut results = Vec::new();

    for step in &plan.steps {
        emit_trace(token_tx, ProgressStatus::Started, &step.label, false);
        let run_result = match step.tool_name {
            "list_dir" => ListDir.run(&step.argument),
            "read_file" => ReadFile.run(&step.argument),
            _ => continue,
        };

        match run_result {
            Ok(ToolRunResult::Immediate(output)) => {
                emit_trace(token_tx, ProgressStatus::Finished, &step.label, false);
                results.push(ToolResult {
                    tool_name: step.tool_name.to_string(),
                    argument: step.argument.clone(),
                    output,
                });
            }
            Ok(ToolRunResult::RequiresApproval(_)) => {
                emit_trace(token_tx, ProgressStatus::Failed, &step.label, false);
            }
            Err(error) => {
                warn!(label = step.label.as_str(), error = %error, "auto inspection step failed");
                emit_trace(token_tx, ProgressStatus::Failed, &step.label, false);
            }
        }
    }

    let hidden_context =
        ToolRegistry::format_results_with_limit(&results, eco_tool_result_limit(eco_enabled)).map(
            |formatted| {
                format!(
                    "Automatic inspection context for {}:\n\n{}",
                    plan.context_label, formatted
                )
            },
        );

    AutoInspectOutcome {
        hidden_context,
        tool_results: results,
    }
}

fn refresh_loaded_facts(
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
struct RetrievalBundle {
    summaries: Vec<(String, String)>,
    facts: Vec<MemoryFactView>,
    session_excerpts: Vec<MemorySessionExcerptView>,
}

fn memory_fact_lines(facts: &[MemoryFactView]) -> Vec<String> {
    facts.iter().map(|fact| fact.content.clone()).collect()
}

fn map_session_excerpt(match_: SessionExcerptMatch) -> MemorySessionExcerptView {
    MemorySessionExcerptView {
        session_label: match_.session_label,
        role: match_.role,
        excerpt: match_.excerpt,
    }
}

fn clear_memory_retrieval(memory_state: &mut RuntimeMemoryState) {
    memory_state.last_summary_paths.clear();
    memory_state.last_retrieval_query = None;
    memory_state.last_selected_facts.clear();
    memory_state.last_selected_session_excerpts.clear();
}

fn set_memory_retrieval(
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

fn collect_retrieval_bundle(
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

fn retrieval_trace_label(bundle: &RetrievalBundle) -> Option<String> {
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

fn format_memory_recall(query: &str, bundle: &RetrievalBundle) -> String {
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

fn save_session(
    store: Option<&SessionStore>,
    active_session: &mut Option<SessionSummary>,
    messages: &[Message],
    backend_name: &str,
    token_tx: &Sender<InferenceEvent>,
) {
    if let (Some(s), Some(current)) = (store, active_session.as_ref()) {
        match s.save_messages(&current.id, messages, backend_name) {
            Ok(updated) => {
                *active_session = Some(updated.clone());
                let _ = token_tx.send(InferenceEvent::SessionStatus(session_info(&updated)));
            }
            Err(e) => warn!(error = %e, "session save failed"),
        }
    }
}

fn reset_session_runtime(
    session_messages: &mut Vec<Message>,
    tools: &ToolRegistry,
    eco_enabled: bool,
    budget: &mut SessionBudget,
    cache_stats: &mut SessionCacheStats,
    backend_name: &str,
    token_tx: &Sender<InferenceEvent>,
) {
    session_messages.clear();
    session_messages.push(Message::system(&build_system_prompt(
        tools,
        &[],
        &[],
        &[],
        eco_enabled,
    )));
    *budget = SessionBudget {
        has_cost_estimate: backend_name == "llama_cpp" || backend_name == "ollama",
        ..SessionBudget::default()
    };
    *cache_stats = SessionCacheStats::default();
    emit_budget_update(budget, token_tx);
    emit_cache_update(cache_stats, false, token_tx);
}

fn skipped_fact_count(report: &MemoryUpdateReport) -> usize {
    report
        .skipped_reasons
        .iter()
        .map(|reason| reason.count)
        .sum()
}

fn apply_memory_update(
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

fn format_sessions_list(sessions: &[SessionSummary], active_session_id: Option<&str>) -> String {
    let mut lines = vec![format!("sessions · {}", sessions.len())];
    if sessions.is_empty() {
        lines.push("  (none saved for this project)".to_string());
        return lines.join("\n");
    }

    for session in sessions {
        let marker = if Some(session.id.as_str()) == active_session_id {
            "●"
        } else {
            "·"
        };
        let message_label = if session.message_count == 0 {
            "empty".to_string()
        } else {
            format!(
                "{} msg{}",
                session.message_count,
                if session.message_count == 1 { "" } else { "s" }
            )
        };
        lines.push(format!(
            "  {marker} {} · {} · {} · #{}",
            list_label(session),
            message_label,
            crate::session::describe_session_age(session.updated_at),
            short_id(&session.id)
        ));
    }

    lines.push("  /sessions resume|delete|export <name-or-id>".to_string());

    lines.join("\n")
}

#[cfg(test)]
mod format_tests {
    use super::format_sessions_list;
    use crate::session::SessionSummary;

    fn summary(
        id: &str,
        name: Option<&str>,
        updated_at: u64,
        message_count: usize,
    ) -> SessionSummary {
        SessionSummary {
            id: id.to_string(),
            project_root: "/tmp/project".to_string(),
            name: name.map(str::to_string),
            backend: "llama.cpp".to_string(),
            created_at: updated_at,
            updated_at,
            last_opened_at: updated_at,
            message_count,
        }
    }

    #[test]
    fn sessions_list_marks_current_and_shows_selector_hint() {
        let output = format_sessions_list(
            &[
                summary("7353e9a31234", Some("b"), 0, 2),
                summary("cbc8da921234", None, 0, 0),
            ],
            Some("7353e9a31234"),
        );

        assert!(output.contains("sessions · 2"));
        assert!(output.contains("● b · 2 msgs"));
        assert!(output.contains("· unnamed · empty"));
        // short id shown with # prefix, no "id " keyword
        assert!(output.contains("#7353e9a3"));
        assert!(!output.contains("id 7353e9a3"));
        assert!(output.contains("/sessions resume|delete|export <name-or-id>"));
    }
}

fn parse_export_format(raw: Option<String>) -> Result<SessionExportFormat> {
    match raw {
        None => Ok(SessionExportFormat::Markdown),
        Some(value) => SessionExportFormat::from_str(value.trim()).ok_or_else(|| {
            crate::error::ParamsError::Config(
                "Export format must be `markdown` or `json`".to_string(),
            )
        }),
    }
}

/// Persistent model thread — loads the backend once, handles prompts in a loop.
/// After each response it checks for tool calls and runs a follow-up if needed.
#[allow(dead_code)]
pub fn model_thread(prompt_rx: Receiver<SessionCommand>, token_tx: Sender<InferenceEvent>) {
    model_thread_with_options(prompt_rx, token_tx, SessionRuntimeOptions::default());
}

pub fn model_thread_with_options(
    prompt_rx: Receiver<SessionCommand>,
    token_tx: Sender<InferenceEvent>,
    options: SessionRuntimeOptions,
) {
    let cfg = match config::load_with_profile() {
        Ok(c) => c,
        Err(e) => {
            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
            return;
        }
    };

    let backend = match load_backend_with_fallback(&cfg, &token_tx) {
        Ok(backend) => backend,
        Err(e) => {
            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
            return;
        }
    };
    info!(backend = backend.name(), "model thread initialized");
    let hooks = Hooks::default();

    let _ = token_tx.send(InferenceEvent::Ready);
    let _ = token_tx.send(InferenceEvent::BackendName(backend.name()));
    let mut eco_enabled = cfg.eco.enabled;
    let mut reflection_requested = cfg.reflection.enabled;
    let mut reflection_enabled = effective_reflection(reflection_requested, eco_enabled);
    let mut debug_logging_enabled = cfg.debug_logging.content;
    info!(enabled = eco_enabled, "eco initial state");
    info!(enabled = reflection_enabled, "reflection initial state");
    info!(
        enabled = debug_logging_enabled,
        "debug logging initial state"
    );
    let _ = token_tx.send(InferenceEvent::EcoEnabled(eco_enabled));
    let _ = token_tx.send(InferenceEvent::ReflectionEnabled(reflection_enabled));
    let _ = token_tx.send(InferenceEvent::DebugLoggingEnabled(debug_logging_enabled));

    if let Some(ref profile_path) = cfg.active_profile {
        let name = profile_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(config::PROJECT_PROFILE_FILE);
        info!(profile = name, "project profile active");
        emit_trace(
            &token_tx,
            ProgressStatus::Finished,
            &format!("profile: {name}"),
            true,
        );
    }

    let tools = ToolRegistry::default();
    let project_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let project_name = project_root.to_string_lossy().to_string();
    let exact_cache = ExactCache::open().ok();
    let fact_store = FactStore::open().ok();
    let project_index = ProjectIndex::open_for(&project_root).ok();
    let mut index_state = if cfg.backend == "llama_cpp" {
        info!("idle incremental indexing disabled for llama_cpp backend");
        None
    } else {
        project_index.as_ref().map(|_| IncrementalIndexState::new())
    };
    let mut memory_state = RuntimeMemoryState::default();
    refresh_loaded_facts(&mut memory_state, fact_store.as_ref(), &project_name);
    hooks.dispatch(HookEvent::MemoryFactsLoaded {
        fact_count: memory_state.loaded_facts.len(),
    });
    let mut session_messages = vec![Message::system(&build_system_prompt(
        &tools,
        &[],
        &[],
        &[],
        eco_enabled,
    ))];
    let mut budget = SessionBudget {
        has_cost_estimate: cfg.backend == "llama_cpp" || cfg.backend == "ollama",
        ..SessionBudget::default()
    };
    let mut cache_stats = SessionCacheStats::default();
    emit_cache_update(&cache_stats, false, &token_tx);
    let mut next_action_id = 1u64;
    emit_memory_state(&token_tx, &memory_state);
    if !memory_state.loaded_facts.is_empty() {
        emit_trace(
            &token_tx,
            ProgressStatus::Finished,
            &format!(
                "memory: loaded {} fact{}",
                memory_state.loaded_facts.len(),
                if memory_state.loaded_facts.len() == 1 {
                    ""
                } else {
                    "s"
                }
            ),
            true,
        );
    }

    let session_store = SessionStore::open().ok();
    let mut active_session = None;
    if let Some(ref store) = session_store {
        if !options.no_resume {
            match store.load_most_recent() {
                Ok(Some(saved)) => {
                    let restored_count = saved.messages.len();
                    info!(
                        session_id = saved.summary.id.as_str(),
                        msg_count = restored_count,
                        saved_at = saved.saved_at,
                        "restoring previous session"
                    );
                    let display_messages: Vec<(String, String)> = saved
                        .messages
                        .iter()
                        .map(|m| (m.role.clone(), m.content.clone()))
                        .collect();
                    session_messages.extend(saved.messages);
                    let info = session_info(&saved.summary);
                    active_session = Some(saved.summary.clone());
                    let _ = token_tx.send(InferenceEvent::SessionLoaded {
                        session: info,
                        display_messages,
                        saved_at: Some(saved.saved_at),
                    });
                    hooks.dispatch(HookEvent::SessionRestored {
                        message_count: restored_count,
                        saved_at: saved.saved_at,
                    });
                    hooks.dispatch(HookEvent::SessionResumed {
                        session_id: saved.summary.id.clone(),
                        named: saved.summary.name.is_some(),
                        message_count: restored_count,
                    });
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(error = %e, "session load failed — starting fresh");
                }
            }
        }

        if active_session.is_none() {
            match store.create_session(None, &backend.name()) {
                Ok(summary) => {
                    let info = session_info(&summary);
                    hooks.dispatch(HookEvent::SessionCreated {
                        session_id: summary.id.clone(),
                        named: false,
                    });
                    active_session = Some(summary);
                    let _ = token_tx.send(InferenceEvent::SessionStatus(info));
                }
                Err(e) => warn!(error = %e, "initial session create failed"),
            }
        }
    }

    loop {
        if let (Some(index), Some(state)) = (project_index.as_ref(), index_state.as_mut()) {
            run_idle_index_step(state, index, &project_root, &*backend);
        }

        let command = match prompt_rx.recv_timeout(IDLE_INDEX_POLL_INTERVAL) {
            Ok(command) => command,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };

        match command {
            SessionCommand::ClearSession => {
                info!("session cleared");
                if let (Some(store), Some(current)) =
                    (session_store.as_ref(), active_session.as_ref())
                {
                    if let Err(e) = store.delete_session(&current.id) {
                        warn!(error = %e, "session clear failed");
                    } else {
                        hooks.dispatch(HookEvent::SessionCleared {
                            session_id: current.id.clone(),
                        });
                    }
                }
                reset_session_runtime(
                    &mut session_messages,
                    &tools,
                    eco_enabled,
                    &mut budget,
                    &mut cache_stats,
                    &cfg.backend,
                    &token_tx,
                );
                clear_memory_retrieval(&mut memory_state);
                memory_state.last_update = None;
                emit_memory_state(&token_tx, &memory_state);
                if let Some(ref store) = session_store {
                    match store.create_session(None, &backend.name()) {
                        Ok(summary) => {
                            hooks.dispatch(HookEvent::SessionCreated {
                                session_id: summary.id.clone(),
                                named: false,
                            });
                            active_session = Some(summary.clone());
                            let _ = token_tx.send(InferenceEvent::SessionLoaded {
                                session: session_info(&summary),
                                display_messages: Vec::new(),
                                saved_at: None,
                            });
                            let _ = token_tx.send(InferenceEvent::SystemMessage(
                                "conversation cleared".to_string(),
                            ));
                        }
                        Err(e) => {
                            warn!(error = %e, "replacement session create failed");
                            active_session = None;
                            let _ = token_tx.send(InferenceEvent::SystemMessage(
                                "conversation cleared".to_string(),
                            ));
                        }
                    }
                }
                continue;
            }
            SessionCommand::ListSessions => {
                match session_store.as_ref().map(|store| store.list_sessions()) {
                    Some(Ok(sessions)) => {
                        let active_id = active_session.as_ref().map(|session| session.id.as_str());
                        let _ = token_tx.send(InferenceEvent::SystemMessage(format_sessions_list(
                            &sessions, active_id,
                        )));
                    }
                    Some(Err(e)) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                    None => {
                        let _ = token_tx.send(InferenceEvent::Error(
                            "Session store is unavailable".to_string(),
                        ));
                    }
                }
                continue;
            }
            SessionCommand::NewSession(name) => {
                save_session(
                    session_store.as_ref(),
                    &mut active_session,
                    &session_messages,
                    &backend.name(),
                    &token_tx,
                );
                reset_session_runtime(
                    &mut session_messages,
                    &tools,
                    eco_enabled,
                    &mut budget,
                    &mut cache_stats,
                    &cfg.backend,
                    &token_tx,
                );
                clear_memory_retrieval(&mut memory_state);
                memory_state.last_update = None;
                emit_memory_state(&token_tx, &memory_state);
                match session_store
                    .as_ref()
                    .map(|store| store.create_session(name.as_deref(), &backend.name()))
                {
                    Some(Ok(summary)) => {
                        let session_label = display_name(&summary);
                        hooks.dispatch(HookEvent::SessionCreated {
                            session_id: summary.id.clone(),
                            named: summary.name.is_some(),
                        });
                        active_session = Some(summary.clone());
                        let _ = token_tx.send(InferenceEvent::SessionLoaded {
                            session: session_info(&summary),
                            display_messages: Vec::new(),
                            saved_at: None,
                        });
                        let _ = token_tx.send(InferenceEvent::SystemMessage(format!(
                            "started new session: {session_label}"
                        )));
                    }
                    Some(Err(e)) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                    None => {
                        let _ = token_tx.send(InferenceEvent::Error(
                            "Session store is unavailable".to_string(),
                        ));
                    }
                }
                continue;
            }
            SessionCommand::RenameSession(name) => {
                match (session_store.as_ref(), active_session.as_ref()) {
                    (Some(store), Some(current)) => {
                        match store.rename_session(&current.id, &name) {
                            Ok(updated) => {
                                hooks.dispatch(HookEvent::SessionRenamed {
                                    session_id: updated.id.clone(),
                                    named: updated.name.is_some(),
                                });
                                active_session = Some(updated.clone());
                                let _ = token_tx
                                    .send(InferenceEvent::SessionStatus(session_info(&updated)));
                                let _ = token_tx.send(InferenceEvent::SystemMessage(format!(
                                    "renamed session to {}",
                                    display_name(&updated)
                                )));
                            }
                            Err(e) => {
                                let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                            }
                        }
                    }
                    _ => {
                        let _ = token_tx.send(InferenceEvent::Error(
                            "No active session is available to rename".to_string(),
                        ));
                    }
                }
                continue;
            }
            SessionCommand::ResumeSession(selector) => {
                save_session(
                    session_store.as_ref(),
                    &mut active_session,
                    &session_messages,
                    &backend.name(),
                    &token_tx,
                );
                match session_store
                    .as_ref()
                    .map(|store| store.load_session(&selector))
                {
                    Some(Ok(saved)) => {
                        reset_session_runtime(
                            &mut session_messages,
                            &tools,
                            eco_enabled,
                            &mut budget,
                            &mut cache_stats,
                            &cfg.backend,
                            &token_tx,
                        );
                        clear_memory_retrieval(&mut memory_state);
                        memory_state.last_update = None;
                        emit_memory_state(&token_tx, &memory_state);
                        let display_messages = saved
                            .messages
                            .iter()
                            .map(|m| (m.role.clone(), m.content.clone()))
                            .collect::<Vec<_>>();
                        session_messages.extend(saved.messages);
                        hooks.dispatch(HookEvent::SessionResumed {
                            session_id: saved.summary.id.clone(),
                            named: saved.summary.name.is_some(),
                            message_count: saved.summary.message_count,
                        });
                        active_session = Some(saved.summary.clone());
                        let _ = token_tx.send(InferenceEvent::SessionLoaded {
                            session: session_info(&saved.summary),
                            display_messages,
                            saved_at: Some(saved.saved_at),
                        });
                    }
                    Some(Err(e)) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                    None => {
                        let _ = token_tx.send(InferenceEvent::Error(
                            "Session store is unavailable".to_string(),
                        ));
                    }
                }
                continue;
            }
            SessionCommand::DeleteSession(selector) => {
                match (session_store.as_ref(), active_session.as_ref()) {
                    (Some(store), Some(current_active)) => match store.resolve_session(&selector) {
                        Ok(summary) => {
                            let was_active = summary.id == current_active.id;
                            let deleted_label = display_name(&summary);
                            if let Err(e) = store.delete_session(&summary.id) {
                                let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                                continue;
                            }
                            hooks.dispatch(HookEvent::SessionDeleted {
                                session_id: summary.id.clone(),
                                was_active,
                            });

                            if was_active {
                                reset_session_runtime(
                                    &mut session_messages,
                                    &tools,
                                    eco_enabled,
                                    &mut budget,
                                    &mut cache_stats,
                                    &cfg.backend,
                                    &token_tx,
                                );
                                clear_memory_retrieval(&mut memory_state);
                                memory_state.last_update = None;
                                emit_memory_state(&token_tx, &memory_state);
                                match store.create_session(None, &backend.name()) {
                                    Ok(replacement) => {
                                        hooks.dispatch(HookEvent::SessionCreated {
                                            session_id: replacement.id.clone(),
                                            named: false,
                                        });
                                        active_session = Some(replacement.clone());
                                        let _ = token_tx.send(InferenceEvent::SessionLoaded {
                                            session: session_info(&replacement),
                                            display_messages: Vec::new(),
                                            saved_at: None,
                                        });
                                        let _ = token_tx.send(InferenceEvent::SystemMessage(
                                            format!(
                                                "deleted session: {deleted_label}; started fresh unnamed session"
                                            ),
                                        ));
                                    }
                                    Err(e) => {
                                        active_session = None;
                                        let _ = token_tx.send(InferenceEvent::Error(format!(
                                            "deleted session {deleted_label}, but failed to create replacement session: {e}"
                                        )));
                                    }
                                }
                            } else {
                                let _ = token_tx.send(InferenceEvent::SystemMessage(format!(
                                    "deleted session: {deleted_label}"
                                )));
                            }
                        }
                        Err(e) => {
                            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                        }
                    },
                    (Some(store), None) => match store.resolve_session(&selector) {
                        Ok(summary) => {
                            let deleted_label = display_name(&summary);
                            match store.delete_session(&summary.id) {
                                Ok(()) => {
                                    hooks.dispatch(HookEvent::SessionDeleted {
                                        session_id: summary.id.clone(),
                                        was_active: false,
                                    });
                                    let _ = token_tx.send(InferenceEvent::SystemMessage(format!(
                                        "deleted session: {deleted_label}"
                                    )));
                                }
                                Err(e) => {
                                    let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                                }
                            }
                        }
                        Err(e) => {
                            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                        }
                    },
                    (None, _) => {
                        let _ = token_tx.send(InferenceEvent::Error(
                            "Session store is unavailable".to_string(),
                        ));
                    }
                }
                continue;
            }
            SessionCommand::ExportSession { selector, format } => {
                match session_store.as_ref().map(|store| {
                    parse_export_format(format).and_then(|fmt| store.export_session(&selector, fmt))
                }) {
                    Some(Ok((summary, path))) => {
                        hooks.dispatch(HookEvent::SessionExported {
                            session_id: summary.id.clone(),
                            format: path
                                .extension()
                                .and_then(|ext| ext.to_str())
                                .unwrap_or("unknown")
                                .to_string(),
                        });
                        let _ = token_tx.send(InferenceEvent::SystemMessage(format!(
                            "exported session {} to {}",
                            display_name(&summary),
                            path.display()
                        )));
                    }
                    Some(Err(e)) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                    None => {
                        let _ = token_tx.send(InferenceEvent::Error(
                            "Session store is unavailable".to_string(),
                        ));
                    }
                }
                continue;
            }
            SessionCommand::InjectUserContext(content) => {
                info!(chars = content.chars().count(), "user context injected");
                session_messages.push(Message::user(&content));
                continue;
            }
            SessionCommand::RequestShellCommand(command) => {
                info!("shell command approval requested");
                match BashTool.run(&command) {
                    Ok(crate::tools::ToolRunResult::RequiresApproval(pending)) => {
                        if let Err(e) = handle_pending_action(
                            ApprovalContext {
                                prompt_rx: &prompt_rx,
                                token_tx: &token_tx,
                                backend: &*backend,
                                tools: &tools,
                                exact_cache: exact_cache.as_ref(),
                                session_messages: &mut session_messages,
                                cfg: &cfg,
                                project_root: &project_root,
                                budget: &mut budget,
                                cache_stats: &mut cache_stats,
                                debug_logging_enabled,
                                reflection_enabled,
                                eco_enabled,
                                hooks: &hooks,
                                index_state: index_state.as_mut(),
                                turn_memory: None,
                            },
                            next_action_id,
                            pending,
                            false,
                        ) {
                            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                        } else {
                            save_session(
                                session_store.as_ref(),
                                &mut active_session,
                                &session_messages,
                                &backend.name(),
                                &token_tx,
                            );
                        }
                    }
                    Ok(crate::tools::ToolRunResult::Immediate(output)) => {
                        let _ = token_tx.send(InferenceEvent::ContextMessage(output));
                    }
                    Err(e) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                }
                let _ = token_tx.send(InferenceEvent::Done);
                next_action_id = next_action_id.saturating_add(1);
                continue;
            }
            SessionCommand::RequestFileWrite { path, content } => {
                info!(path = path.as_str(), "file write approval requested");
                match crate::tools::build_pending_write_request(&path, &content) {
                    Ok(pending) => {
                        if let Err(e) = handle_pending_action(
                            ApprovalContext {
                                prompt_rx: &prompt_rx,
                                token_tx: &token_tx,
                                backend: &*backend,
                                tools: &tools,
                                exact_cache: exact_cache.as_ref(),
                                session_messages: &mut session_messages,
                                cfg: &cfg,
                                project_root: &project_root,
                                budget: &mut budget,
                                cache_stats: &mut cache_stats,
                                debug_logging_enabled,
                                reflection_enabled,
                                eco_enabled,
                                hooks: &hooks,
                                index_state: index_state.as_mut(),
                                turn_memory: None,
                            },
                            next_action_id,
                            pending,
                            false,
                        ) {
                            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                        } else {
                            save_session(
                                session_store.as_ref(),
                                &mut active_session,
                                &session_messages,
                                &backend.name(),
                                &token_tx,
                            );
                        }
                    }
                    Err(e) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                }
                let _ = token_tx.send(InferenceEvent::Done);
                next_action_id = next_action_id.saturating_add(1);
                continue;
            }
            SessionCommand::RequestFileEdit { path, edits } => {
                info!(path = path.as_str(), "file edit approval requested");
                match crate::tools::build_pending_edit_request(&path, &edits) {
                    Ok(pending) => {
                        if let Err(e) = handle_pending_action(
                            ApprovalContext {
                                prompt_rx: &prompt_rx,
                                token_tx: &token_tx,
                                backend: &*backend,
                                tools: &tools,
                                exact_cache: exact_cache.as_ref(),
                                session_messages: &mut session_messages,
                                cfg: &cfg,
                                project_root: &project_root,
                                budget: &mut budget,
                                cache_stats: &mut cache_stats,
                                debug_logging_enabled,
                                reflection_enabled,
                                eco_enabled,
                                hooks: &hooks,
                                index_state: index_state.as_mut(),
                                turn_memory: None,
                            },
                            next_action_id,
                            pending,
                            false,
                        ) {
                            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                        } else {
                            save_session(
                                session_store.as_ref(),
                                &mut active_session,
                                &session_messages,
                                &backend.name(),
                                &token_tx,
                            );
                        }
                    }
                    Err(e) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                }
                let _ = token_tx.send(InferenceEvent::Done);
                next_action_id = next_action_id.saturating_add(1);
                continue;
            }
            SessionCommand::SetReflection(enabled) => {
                reflection_requested = enabled;
                reflection_enabled = effective_reflection(reflection_requested, eco_enabled);
                info!(
                    requested = enabled,
                    effective = reflection_enabled,
                    eco_enabled,
                    "reflection state updated"
                );
                let _ = token_tx.send(InferenceEvent::ReflectionEnabled(reflection_enabled));
                continue;
            }
            SessionCommand::SetEco(enabled) => {
                eco_enabled = enabled;
                reflection_enabled = effective_reflection(reflection_requested, eco_enabled);
                info!(
                    enabled = eco_enabled,
                    reflection_enabled, "eco state updated"
                );
                if let Some(first) = session_messages.first_mut() {
                    if first.role == "system" {
                        first.content = build_system_prompt(
                            &tools,
                            &memory_fact_lines(&memory_state.last_selected_facts),
                            &[],
                            &memory_state.last_selected_session_excerpts,
                            eco_enabled,
                        );
                    }
                }
                let _ = token_tx.send(InferenceEvent::EcoEnabled(eco_enabled));
                let _ = token_tx.send(InferenceEvent::ReflectionEnabled(reflection_enabled));
                continue;
            }
            SessionCommand::SetDebugLogging(enabled) => {
                debug_logging_enabled = enabled;
                info!(enabled, "debug logging state updated");
                let _ = token_tx.send(InferenceEvent::DebugLoggingEnabled(enabled));
                continue;
            }
            SessionCommand::ClearDebugLog => {
                match debug_log::clear() {
                    Ok(()) => {
                        info!("debug content log cleared");
                    }
                    Err(e) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                }
                continue;
            }
            SessionCommand::ClearCache => {
                match exact_cache.as_ref().map(|cache| cache.clear()) {
                    Some(Ok(deleted)) => {
                        info!(deleted, "exact cache cleared");
                    }
                    Some(Err(e)) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                    None => {
                        let _ = token_tx
                            .send(InferenceEvent::Error("Cache is unavailable".to_string()));
                    }
                }
                continue;
            }
            SessionCommand::RecallMemory(query) => {
                let bundle = collect_retrieval_bundle(
                    &query,
                    eco_enabled,
                    &project_name,
                    project_index.as_ref(),
                    fact_store.as_ref(),
                    session_store.as_ref(),
                    active_session.as_ref().map(|session| session.id.as_str()),
                    &memory_state.loaded_facts,
                );
                let _ = token_tx.send(InferenceEvent::SystemMessage(format_memory_recall(
                    &query, &bundle,
                )));
                continue;
            }
            SessionCommand::PruneMemory => {
                match fact_store
                    .as_ref()
                    .map(|store| store.prune_irrelevant_facts(&project_name))
                {
                    Some(Ok(removed)) => {
                        refresh_loaded_facts(&mut memory_state, fact_store.as_ref(), &project_name);
                        clear_memory_retrieval(&mut memory_state);
                        emit_memory_state(&token_tx, &memory_state);
                        emit_trace(
                            &token_tx,
                            ProgressStatus::Finished,
                            &format!(
                                "memory: pruned {} fact{}",
                                removed,
                                if removed == 1 { "" } else { "s" }
                            ),
                            true,
                        );
                        let _ = token_tx.send(InferenceEvent::SystemMessage(format!(
                            "memory prune removed {removed} irrelevant fact{}",
                            if removed == 1 { "" } else { "s" }
                        )));
                    }
                    Some(Err(e)) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                    None => {
                        let _ = token_tx.send(InferenceEvent::Error(
                            "Memory store is unavailable".to_string(),
                        ));
                    }
                }
                continue;
            }
            SessionCommand::ApproveAction(_) | SessionCommand::RejectAction(_) => {
                warn!("approval command received with no pending action");
                let _ = token_tx.send(InferenceEvent::Error(
                    "No action is currently awaiting approval".to_string(),
                ));
                continue;
            }
            SessionCommand::SubmitUser(prompt) => {
                info!(
                    reflection_enabled,
                    existing_messages = session_messages.len(),
                    "user turn submitted"
                );
                if debug_logging_enabled {
                    if let Err(e) = debug_log::append_user_prompt(&prompt) {
                        warn!(error = %e, "debug user prompt logging failed");
                    }
                }
                session_messages.push(Message::user(&prompt));

                let auto_inspection = detect_auto_inspect_intent(&prompt)
                    .map(|intent| plan_auto_inspection(intent, &project_root));
                let auto_inspection_outcome = auto_inspection.as_ref().map(|plan| {
                    emit_generation_started(&token_tx, plan.status_label, false);
                    run_auto_inspection(plan, &token_tx, eco_enabled)
                });
                if let Some(outcome) = auto_inspection_outcome.as_ref() {
                    if let Some(hidden_context) = &outcome.hidden_context {
                        session_messages.push(Message::user(hidden_context));
                    }
                }

                let retrieval = collect_retrieval_bundle(
                    &prompt,
                    eco_enabled,
                    &project_name,
                    project_index.as_ref(),
                    fact_store.as_ref(),
                    session_store.as_ref(),
                    active_session.as_ref().map(|session| session.id.as_str()),
                    &memory_state.loaded_facts,
                );
                let mut turn_memory =
                    TurnMemoryEvidence::new(prompt.clone(), retrieval.summaries.clone());
                if let Some(outcome) = auto_inspection_outcome {
                    for result in outcome.tool_results {
                        turn_memory.record_tool_result(
                            result.tool_name,
                            result.argument,
                            result.output,
                            false,
                        );
                    }
                }
                set_memory_retrieval(&mut memory_state, &prompt, &retrieval);
                memory_state.last_update = None;
                emit_memory_state(&token_tx, &memory_state);
                hooks.dispatch(HookEvent::MemorySummariesSelected {
                    summary_count: retrieval.summaries.len(),
                });
                if let Some(label) = retrieval_trace_label(&retrieval) {
                    emit_trace(&token_tx, ProgressStatus::Finished, &label, false);
                }

                if let Some(first) = session_messages.first_mut() {
                    if first.role == "system" {
                        first.content = build_system_prompt(
                            &tools,
                            &memory_fact_lines(&retrieval.facts),
                            &retrieval.summaries,
                            &retrieval.session_excerpts,
                            eco_enabled,
                        );
                    }
                }

                compression::compress_history(&mut session_messages, &*backend, eco_enabled);

                emit_generation_started(&token_tx, "generating...", false);
                emit_trace(
                    &token_tx,
                    ProgressStatus::Started,
                    "drafting answer...",
                    false,
                );
                hooks.dispatch(HookEvent::BeforeGeneration {
                    backend: backend.name(),
                    message_count: session_messages.len(),
                    eco: eco_enabled,
                    reflection: reflection_enabled,
                });
                let response = generate_with_cache(
                    &*backend,
                    &session_messages,
                    &cfg,
                    &project_root,
                    token_tx.clone(),
                    !reflection_enabled,
                    exact_cache.as_ref(),
                    &mut cache_stats,
                    CacheMode::PreferPromptLevel,
                );
                debug!(
                    reflection_enabled,
                    message_count = session_messages.len(),
                    "generation started"
                );
                let prompt_tokens = estimate_message_tokens(&session_messages);

                match response {
                    Err(e) => {
                        emit_trace(
                            &token_tx,
                            ProgressStatus::Failed,
                            "generation failed",
                            false,
                        );
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                    Ok(response) => {
                        let response_source = response.source;
                        hooks.dispatch(HookEvent::AfterGeneration {
                            backend: backend.name(),
                            response_chars: response.text.chars().count(),
                            from_cache: response.hit,
                            elapsed_ms: response.elapsed_ms,
                        });
                        let full_response = response.text;
                        info!(
                            response_chars = full_response.chars().count(),
                            reflection_enabled, "generation completed"
                        );
                        if !response.hit {
                            record_generation_budget(
                                &cfg,
                                &mut budget,
                                &token_tx,
                                prompt_tokens,
                                &full_response,
                            );
                        }

                        emit_trace(
                            &token_tx,
                            ProgressStatus::Updated,
                            "scanning tool calls...",
                            false,
                        );
                        let tool_execution = tools.execute_tool_calls(&full_response);
                        let tool_results = tool_execution.results;
                        info!(
                            tool_results = tool_results.len(),
                            pending = tool_execution.pending.is_some(),
                            "tool scan completed"
                        );
                        for result in &tool_results {
                            hooks.dispatch(HookEvent::ToolExecuted {
                                tool_name: result.tool_name.clone(),
                                argument_chars: result.argument.chars().count(),
                                result_chars: result.output.chars().count(),
                            });
                            turn_memory.record_tool_result(
                                result.tool_name.clone(),
                                result.argument.clone(),
                                result.output.clone(),
                                false,
                            );
                        }

                        if let Some(pending) = tool_execution.pending {
                            emit_trace(
                                &token_tx,
                                ProgressStatus::Updated,
                                "waiting for approval...",
                                false,
                            );
                            session_messages.push(Message::assistant(&full_response));
                            if let Some(result_msg) = ToolRegistry::format_results_with_limit(
                                &tool_results,
                                eco_tool_result_limit(eco_enabled),
                            ) {
                                session_messages.push(Message::user(&result_msg));
                            }
                            if let Err(e) = handle_pending_action(
                                ApprovalContext {
                                    prompt_rx: &prompt_rx,
                                    token_tx: &token_tx,
                                    backend: &*backend,
                                    tools: &tools,
                                    exact_cache: exact_cache.as_ref(),
                                    session_messages: &mut session_messages,
                                    cfg: &cfg,
                                    project_root: &project_root,
                                    budget: &mut budget,
                                    cache_stats: &mut cache_stats,
                                    debug_logging_enabled,
                                    reflection_enabled,
                                    eco_enabled,
                                    hooks: &hooks,
                                    index_state: index_state.as_mut(),
                                    turn_memory: Some(&mut turn_memory),
                                },
                                next_action_id,
                                pending,
                                true,
                            ) {
                                emit_trace(
                                    &token_tx,
                                    ProgressStatus::Failed,
                                    "approval flow failed",
                                    false,
                                );
                                let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                            } else if let Some(store) = fact_store.as_ref() {
                                let update = store.verify_and_store_turn(
                                    &project_name,
                                    &turn_memory,
                                    &*backend,
                                );
                                apply_memory_update(&token_tx, &hooks, &mut memory_state, update);
                            }
                            next_action_id = next_action_id.saturating_add(1);
                        } else if let Some(result_msg) = ToolRegistry::format_results_with_limit(
                            &tool_results,
                            eco_tool_result_limit(eco_enabled),
                        ) {
                            emit_trace(
                                &token_tx,
                                ProgressStatus::Updated,
                                "running tool follow-up...",
                                false,
                            );
                            let _ = token_tx.send(InferenceEvent::ToolCall(
                                tool_results
                                    .iter()
                                    .map(|r| format!("{}({})", r.tool_name, r.argument))
                                    .collect::<Vec<_>>()
                                    .join(", "),
                            ));

                            session_messages.push(Message::assistant(&full_response));
                            session_messages.push(Message::user(&result_msg));

                            match generate_with_cache(
                                &*backend,
                                &session_messages,
                                &cfg,
                                &project_root,
                                token_tx.clone(),
                                !reflection_enabled,
                                exact_cache.as_ref(),
                                &mut cache_stats,
                                CacheMode::PreferPromptLevel,
                            ) {
                                Ok(follow_up) => {
                                    let follow_up_source = follow_up.source;
                                    let follow_up_text = follow_up.text;
                                    let prompt_tokens = estimate_message_tokens(&session_messages);
                                    if !follow_up.hit {
                                        record_generation_budget(
                                            &cfg,
                                            &mut budget,
                                            &token_tx,
                                            prompt_tokens,
                                            &follow_up_text,
                                        );
                                    }
                                    let final_response = if reflection_enabled {
                                        emit_trace(
                                            &token_tx,
                                            ProgressStatus::Updated,
                                            "reflecting final answer...",
                                            false,
                                        );
                                        reflect_response(
                                            &*backend,
                                            &cfg,
                                            &project_root,
                                            &mut budget,
                                            &token_tx,
                                            exact_cache.as_ref(),
                                            &mut cache_stats,
                                            &session_messages,
                                            &follow_up_text,
                                        )
                                    } else {
                                        Ok(follow_up_text)
                                    };

                                    match final_response {
                                        Ok(final_response) => {
                                            turn_memory.set_final_response(final_response.clone());
                                            if !final_response.trim().is_empty() {
                                                log_debug_response(
                                                    debug_logging_enabled,
                                                    &final_response,
                                                    if reflection_enabled {
                                                        debug_log::ResponseSource::Live
                                                    } else {
                                                        follow_up_source
                                                    },
                                                );
                                                store_exact_cache(
                                                    exact_cache.as_ref(),
                                                    &cfg,
                                                    &project_root,
                                                    &backend.name(),
                                                    &session_messages,
                                                    &final_response,
                                                );
                                                session_messages
                                                    .push(Message::assistant(&final_response));
                                            }
                                            emit_trace(
                                                &token_tx,
                                                ProgressStatus::Finished,
                                                "answer ready",
                                                false,
                                            );
                                            if let Some(store) = fact_store.as_ref() {
                                                let update = store.verify_and_store_turn(
                                                    &project_name,
                                                    &turn_memory,
                                                    &*backend,
                                                );
                                                apply_memory_update(
                                                    &token_tx,
                                                    &hooks,
                                                    &mut memory_state,
                                                    update,
                                                );
                                            }
                                        }
                                        Err(e) => {
                                            warn!(error = %e, "reflection after tool follow-up failed");
                                            emit_trace(
                                                &token_tx,
                                                ProgressStatus::Failed,
                                                "follow-up failed",
                                                false,
                                            );
                                            let _ =
                                                token_tx.send(InferenceEvent::Error(e.to_string()));
                                        }
                                    }
                                }
                                Err(e) => {
                                    warn!(error = %e, "tool follow-up generation failed");
                                    emit_trace(
                                        &token_tx,
                                        ProgressStatus::Failed,
                                        "tool follow-up failed",
                                        false,
                                    );
                                    let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                                }
                            }
                        } else {
                            let final_response = if reflection_enabled {
                                emit_trace(
                                    &token_tx,
                                    ProgressStatus::Updated,
                                    "reflecting final answer...",
                                    false,
                                );
                                reflect_response(
                                    &*backend,
                                    &cfg,
                                    &project_root,
                                    &mut budget,
                                    &token_tx,
                                    exact_cache.as_ref(),
                                    &mut cache_stats,
                                    &session_messages,
                                    &full_response,
                                )
                            } else {
                                Ok(full_response)
                            };

                            match final_response {
                                Ok(final_response) => {
                                    turn_memory.set_final_response(final_response.clone());
                                    if !final_response.trim().is_empty() {
                                        log_debug_response(
                                            debug_logging_enabled,
                                            &final_response,
                                            if reflection_enabled {
                                                debug_log::ResponseSource::Live
                                            } else {
                                                response_source
                                            },
                                        );
                                        store_exact_cache(
                                            exact_cache.as_ref(),
                                            &cfg,
                                            &project_root,
                                            &backend.name(),
                                            &session_messages,
                                            &final_response,
                                        );
                                        store_prompt_level_cache(
                                            exact_cache.as_ref(),
                                            &cfg,
                                            &project_root,
                                            &backend.name(),
                                            &session_messages,
                                            &final_response,
                                        );
                                        session_messages.push(Message::assistant(&final_response));
                                    }
                                    emit_trace(
                                        &token_tx,
                                        ProgressStatus::Finished,
                                        "answer ready",
                                        false,
                                    );
                                    if let Some(store) = fact_store.as_ref() {
                                        let update = store.verify_and_store_turn(
                                            &project_name,
                                            &turn_memory,
                                            &*backend,
                                        );
                                        apply_memory_update(
                                            &token_tx,
                                            &hooks,
                                            &mut memory_state,
                                            update,
                                        );
                                    }
                                }
                                Err(e) => {
                                    warn!(error = %e, "final response post-processing failed");
                                    emit_trace(
                                        &token_tx,
                                        ProgressStatus::Failed,
                                        "final answer failed",
                                        false,
                                    );
                                    let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                                }
                            }
                        }

                        save_session(
                            session_store.as_ref(),
                            &mut active_session,
                            &session_messages,
                            &backend.name(),
                            &token_tx,
                        );
                        let _ = token_tx.send(InferenceEvent::Done);
                    }
                }
            }
        }
    }

    hooks.dispatch(HookEvent::SessionEnding {
        message_count: session_messages.len(),
    });

    if let (Some(store), Some(current)) = (session_store.as_ref(), active_session.as_ref()) {
        if let Err(e) = store.delete_if_empty_unnamed(&current.id) {
            warn!(error = %e, "empty session cleanup failed");
        }
    }

    if let Some(store) = fact_store.as_ref() {
        match store.consolidate(&project_name, &cfg.memory) {
            Ok(stats) => {
                if stats.ttl_pruned + stats.dedup_removed + stats.cap_removed > 0 {
                    info!(
                        project = project_name.as_str(),
                        ttl_pruned = stats.ttl_pruned,
                        dedup_removed = stats.dedup_removed,
                        cap_removed = stats.cap_removed,
                        "memory consolidated"
                    );
                }
                memory_state.last_consolidation = Some(MemoryConsolidationView {
                    ttl_pruned: stats.ttl_pruned,
                    dedup_removed: stats.dedup_removed,
                    cap_removed: stats.cap_removed,
                });
                emit_memory_state(&token_tx, &memory_state);
                if stats.ttl_pruned + stats.dedup_removed + stats.cap_removed > 0 {
                    emit_trace(
                        &token_tx,
                        ProgressStatus::Finished,
                        &format!(
                            "memory: consolidated -{} dup",
                            stats.dedup_removed + stats.cap_removed + stats.ttl_pruned
                        ),
                        true,
                    );
                }
                hooks.dispatch(HookEvent::MemoryConsolidated {
                    facts_pruned: stats.ttl_pruned,
                    facts_deduped: stats.dedup_removed,
                    facts_capped: stats.cap_removed,
                });
            }
            Err(e) => warn!(error = %e, "memory consolidation failed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::FactProvenance;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_project_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!("params-auto-inspect-{label}-{nonce}"))
    }

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

    #[test]
    fn detect_auto_inspect_intent_matches_repo_prompts() {
        assert_eq!(
            detect_auto_inspect_intent("What's in this repo?"),
            Some(AutoInspectIntent::RepoOverview)
        );
        assert_eq!(
            detect_auto_inspect_intent("Summarize this codebase"),
            Some(AutoInspectIntent::RepoOverview)
        );
    }

    #[test]
    fn detect_auto_inspect_intent_matches_directory_prompts() {
        assert_eq!(
            detect_auto_inspect_intent("What's in this directory?"),
            Some(AutoInspectIntent::DirectoryOverview)
        );
        assert_eq!(
            detect_auto_inspect_intent("What's here"),
            Some(AutoInspectIntent::DirectoryOverview)
        );
    }

    #[test]
    fn detect_auto_inspect_intent_skips_unrelated_prompts() {
        assert_eq!(
            detect_auto_inspect_intent("Where is the cache implemented?"),
            None
        );
        assert_eq!(detect_auto_inspect_intent("/read README.md"), None);
    }

    #[test]
    fn repo_auto_inspection_prefers_main_over_lib() {
        let root = temp_project_root("repo-main");
        fs::create_dir_all(root.join("src")).expect("create src");
        fs::write(root.join("README.md"), "# params").expect("write readme");
        fs::write(root.join("Cargo.toml"), "[package]").expect("write cargo");
        fs::write(root.join("src/main.rs"), "fn main() {}").expect("write main");
        fs::write(root.join("src/lib.rs"), "pub fn lib() {}").expect("write lib");

        let plan = plan_auto_inspection(AutoInspectIntent::RepoOverview, &root);
        let labels = plan
            .steps
            .iter()
            .map(|step| step.label.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            labels,
            vec![
                "List .",
                "Read README.md",
                "Read Cargo.toml",
                "Read src/main.rs"
            ]
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn directory_auto_inspection_is_bounded_and_chooses_single_manifest() {
        let root = temp_project_root("directory");
        fs::create_dir_all(&root).expect("create dir");
        fs::write(root.join("README.md"), "# params").expect("write readme");
        fs::write(root.join("Cargo.toml"), "[package]").expect("write cargo");
        fs::write(root.join("package.json"), "{}").expect("write package");

        let plan = plan_auto_inspection(AutoInspectIntent::DirectoryOverview, &root);
        let labels = plan
            .steps
            .iter()
            .map(|step| step.label.as_str())
            .collect::<Vec<_>>();

        assert_eq!(labels, vec!["List .", "Read README.md", "Read Cargo.toml"]);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn auto_inspection_hidden_context_has_expected_heading() {
        let plan = AutoInspectPlan {
            thinking: "Thinking: exploring the repo structure and key project docs.",
            status_label: "inspecting repo...",
            context_label: "this repo summary request",
            steps: vec![],
        };

        let hidden = ToolRegistry::format_results_with_limit(
            &[ToolResult {
                tool_name: "list_dir".to_string(),
                argument: ".".to_string(),
                output: "Directory: .\n\nsrc/\nREADME.md".to_string(),
            }],
            eco_tool_result_limit(false),
        )
        .map(|formatted| {
            format!(
                "Automatic inspection context for {}:\n\n{}",
                plan.context_label, formatted
            )
        })
        .expect("hidden context");

        assert!(hidden.starts_with("Automatic inspection context for this repo summary request:"));
        assert!(hidden.contains("--- list_dir(.) ---"));
    }
}
