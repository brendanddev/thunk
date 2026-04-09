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
    retrieval::{query_terms, score_text},
};
use crate::session::{
    display_name, list_label, short_id, SessionExcerptMatch, SessionExportFormat, SessionStore,
    SessionSummary,
};
use crate::tools::{
    BashTool, ListDir, ReadFile, SearchCode, Tool, ToolRegistry, ToolResult, ToolRunResult,
};

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
    WhereIsImplementation,
    FeatureTrace,
    ConfigLocate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AutoInspectStep {
    label: String,
    tool_name: &'static str,
    argument: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AutoInspectPlan {
    intent: AutoInspectIntent,
    thinking: &'static str,
    status_label: &'static str,
    context_label: &'static str,
    query: Option<String>,
    steps: Vec<AutoInspectStep>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AutoInspectBudget {
    total_chars: usize,
    top_level_entries: usize,
    code_entries: usize,
    readme_chars: usize,
    manifest_chars: usize,
    entrypoint_chars: usize,
    search_files: usize,
    read_files: usize,
    key_hits_per_file: usize,
    workflow_summary_chars: usize,
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

    if normalized.starts_with("where is ")
        && [" implemented", " defined", " handled"]
            .iter()
            .any(|suffix| normalized.ends_with(suffix))
    {
        return Some(AutoInspectIntent::WhereIsImplementation);
    }

    if normalized.starts_with("find ") || normalized.starts_with("which file has ") {
        return Some(AutoInspectIntent::WhereIsImplementation);
    }

    if normalized.starts_with("trace how ")
        || normalized.starts_with("how does ")
        || normalized.starts_with("what handles ")
        || normalized.starts_with("what writes to ")
    {
        return Some(AutoInspectIntent::FeatureTrace);
    }

    if normalized.starts_with("where is ")
        && [" configured", " set"]
            .iter()
            .any(|suffix| normalized.ends_with(suffix))
    {
        return Some(AutoInspectIntent::ConfigLocate);
    }

    if normalized.starts_with("which file configures ") {
        return Some(AutoInspectIntent::ConfigLocate);
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

fn push_read_step(steps: &mut Vec<AutoInspectStep>, project_root: &Path, rel: &str) {
    if project_root.join(rel).is_file() {
        steps.push(AutoInspectStep {
            label: format!("Read {rel}"),
            tool_name: "read_file",
            argument: rel.to_string(),
        });
    }
}

fn trim_query_noise(query: &str) -> String {
    let mut trimmed = query.trim().to_string();
    for prefix in ["the ", "a ", "an ", "this ", "that "] {
        if let Some(stripped) = trimmed.strip_prefix(prefix) {
            trimmed = stripped.trim().to_string();
        }
    }
    trimmed
}

fn trim_query_suffix<'a>(query: &'a str, suffixes: &[&str]) -> &'a str {
    for suffix in suffixes {
        if let Some(stripped) = query.strip_suffix(suffix) {
            return stripped.trim();
        }
    }
    query.trim()
}

fn singularize_token(token: &str) -> String {
    if token.len() > 4 && token.ends_with('s') {
        token[..token.len() - 1].to_string()
    } else {
        token.to_string()
    }
}

fn salient_search_token(phrase: &str, intent: AutoInspectIntent) -> Option<String> {
    let stopwords = match intent {
        AutoInspectIntent::WhereIsImplementation => &[
            "implemented",
            "define",
            "defined",
            "handle",
            "handled",
            "find",
            "which",
            "file",
            "has",
            "where",
            "is",
            "the",
        ][..],
        AutoInspectIntent::FeatureTrace => &[
            "trace", "how", "does", "work", "works", "flow", "what", "handles", "writes", "to",
            "are", "saved", "save", "restored", "restore", "the",
        ][..],
        AutoInspectIntent::ConfigLocate => &[
            "configured",
            "configures",
            "set",
            "mode",
            "which",
            "file",
            "where",
            "is",
            "the",
        ][..],
        AutoInspectIntent::RepoOverview | AutoInspectIntent::DirectoryOverview => &[][..],
    };

    phrase
        .split_whitespace()
        .map(trim_query_noise)
        .map(|token| singularize_token(&token))
        .find(|token| {
            !token.is_empty()
                && !stopwords.iter().any(|stop| stop == token)
                && token.chars().any(|ch| ch.is_ascii_alphanumeric())
        })
}

fn extract_auto_inspect_query(prompt: &str, intent: AutoInspectIntent) -> Option<String> {
    let normalized = normalize_intent_text(prompt);
    if normalized.contains("session") {
        if normalized.contains("save") || normalized.contains("saved") {
            // Target the actual persistence function (`save_messages` in
            // src/session/mod.rs) rather than the private thin wrapper
            // `save_session` in src/inference/session.rs.  The wrapper file is
            // ~150 KB — it exceeds the read_file limit and can never be
            // inspected.  `save_messages` is in the small, readable session
            // store and IS the true implementation.  Using it as the search
            // key also avoids finding the query-string literal
            // `"save_session(".to_string()` that appears in this file's own
            // query-builder code.
            return Some("save_messages".to_string());
        }
        if normalized.contains("restore")
            || normalized.contains("restored")
            || normalized.contains("resume")
        {
            return Some("load_most_recent".to_string());
        }
    }
    if intent == AutoInspectIntent::ConfigLocate && normalized.contains("eco") {
        return Some("eco.enabled".to_string());
    }

    let extracted = match intent {
        AutoInspectIntent::WhereIsImplementation => {
            if let Some(rest) = normalized.strip_prefix("where is ") {
                trim_query_suffix(rest, &[" implemented", " defined", " handled"]).to_string()
            } else if let Some(rest) = normalized.strip_prefix("find ") {
                rest.trim().to_string()
            } else if let Some(rest) = normalized.strip_prefix("which file has ") {
                rest.trim().to_string()
            } else {
                String::new()
            }
        }
        AutoInspectIntent::FeatureTrace => {
            if let Some(rest) = normalized.strip_prefix("trace how ") {
                trim_query_suffix(rest, &[" works", " work", " flows", " flow"]).to_string()
            } else if let Some(rest) = normalized.strip_prefix("how does ") {
                trim_query_suffix(rest, &[" work", " works", " flow", " flow through"]).to_string()
            } else if let Some(rest) = normalized.strip_prefix("what handles ") {
                rest.trim().to_string()
            } else if let Some(rest) = normalized.strip_prefix("what writes to ") {
                rest.trim().to_string()
            } else {
                String::new()
            }
        }
        AutoInspectIntent::ConfigLocate => {
            if let Some(rest) = normalized.strip_prefix("where is ") {
                trim_query_suffix(rest, &[" configured", " set"]).to_string()
            } else if let Some(rest) = normalized.strip_prefix("which file configures ") {
                rest.trim().to_string()
            } else {
                String::new()
            }
        }
        AutoInspectIntent::RepoOverview | AutoInspectIntent::DirectoryOverview => String::new(),
    };

    let cleaned =
        salient_search_token(&extracted, intent).unwrap_or_else(|| trim_query_noise(&extracted));
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

fn plan_auto_inspection(
    intent: AutoInspectIntent,
    prompt: &str,
    project_root: &Path,
) -> AutoInspectPlan {
    let mut steps = vec![AutoInspectStep {
        label: "List .".to_string(),
        tool_name: "list_dir",
        argument: ".".to_string(),
    }];

    match intent {
        AutoInspectIntent::RepoOverview => {
            if project_root.join("src").is_dir() {
                steps.push(AutoInspectStep {
                    label: "List src/".to_string(),
                    tool_name: "list_dir",
                    argument: "src".to_string(),
                });
            }
            push_read_step(&mut steps, project_root, "README.md");
            push_read_step(&mut steps, project_root, "Cargo.toml");
            if project_root.join("src/main.rs").is_file() {
                push_read_step(&mut steps, project_root, "src/main.rs");
            } else {
                push_read_step(&mut steps, project_root, "src/lib.rs");
            }

            AutoInspectPlan {
                intent,
                thinking: "Thinking: exploring the repo structure and key project docs.",
                status_label: "inspecting repo...",
                context_label: "this repo summary request",
                query: None,
                steps,
            }
        }
        AutoInspectIntent::DirectoryOverview => {
            push_read_step(&mut steps, project_root, "README.md");
            for manifest in ["Cargo.toml", "package.json", "pyproject.toml", "go.mod"] {
                if project_root.join(manifest).is_file() {
                    push_read_step(&mut steps, project_root, manifest);
                    break;
                }
            }

            AutoInspectPlan {
                intent,
                thinking: "Thinking: checking the current directory and its key files.",
                status_label: "inspecting directory...",
                context_label: "this directory summary request",
                query: None,
                steps,
            }
        }
        AutoInspectIntent::WhereIsImplementation => {
            let query = extract_auto_inspect_query(prompt, intent);
            let mut steps = Vec::new();
            if let Some(ref query) = query {
                steps.push(AutoInspectStep {
                    label: format!("Search {query}"),
                    tool_name: "search",
                    argument: query.clone(),
                });
            }
            AutoInspectPlan {
                intent,
                thinking: "Thinking: locating the most likely implementation files.",
                status_label: "locating implementation...",
                context_label: "this implementation lookup request",
                query,
                steps,
            }
        }
        AutoInspectIntent::FeatureTrace => {
            let query = extract_auto_inspect_query(prompt, intent);
            let mut steps = Vec::new();
            if let Some(ref query) = query {
                steps.push(AutoInspectStep {
                    label: format!("Search {query}"),
                    tool_name: "search",
                    argument: query.clone(),
                });
            }
            AutoInspectPlan {
                intent,
                thinking: "Thinking: tracing the main code path for this feature.",
                status_label: "tracing feature...",
                context_label: "this feature trace request",
                query,
                steps,
            }
        }
        AutoInspectIntent::ConfigLocate => {
            let query = extract_auto_inspect_query(prompt, intent);
            let mut steps = Vec::new();
            if let Some(ref query) = query {
                steps.push(AutoInspectStep {
                    label: format!("Search {query}"),
                    tool_name: "search",
                    argument: query.clone(),
                });
            }
            AutoInspectPlan {
                intent,
                thinking: "Thinking: checking the files that configure this behavior.",
                status_label: "locating configuration...",
                context_label: "this configuration lookup request",
                query,
                steps,
            }
        }
    }
}

fn auto_inspection_budget(
    intent: AutoInspectIntent,
    backend_name: &str,
    eco_enabled: bool,
) -> AutoInspectBudget {
    let constrained = backend_name.contains("llama.cpp");
    match (intent, constrained, eco_enabled) {
        (AutoInspectIntent::RepoOverview, true, true) => AutoInspectBudget {
            total_chars: 700,
            top_level_entries: 6,
            code_entries: 6,
            readme_chars: 120,
            manifest_chars: 160,
            entrypoint_chars: 160,
            search_files: 0,
            read_files: 0,
            key_hits_per_file: 0,
            workflow_summary_chars: 0,
        },
        (AutoInspectIntent::RepoOverview, true, false) => AutoInspectBudget {
            total_chars: 1000,
            top_level_entries: 8,
            code_entries: 8,
            readme_chars: 170,
            manifest_chars: 220,
            entrypoint_chars: 220,
            search_files: 0,
            read_files: 0,
            key_hits_per_file: 0,
            workflow_summary_chars: 0,
        },
        (AutoInspectIntent::RepoOverview, false, true) => AutoInspectBudget {
            total_chars: 1200,
            top_level_entries: 8,
            code_entries: 8,
            readme_chars: 180,
            manifest_chars: 240,
            entrypoint_chars: 240,
            search_files: 0,
            read_files: 0,
            key_hits_per_file: 0,
            workflow_summary_chars: 0,
        },
        (AutoInspectIntent::RepoOverview, false, false) => AutoInspectBudget {
            total_chars: 1700,
            top_level_entries: 10,
            code_entries: 10,
            readme_chars: 260,
            manifest_chars: 320,
            entrypoint_chars: 320,
            search_files: 0,
            read_files: 0,
            key_hits_per_file: 0,
            workflow_summary_chars: 0,
        },
        (AutoInspectIntent::DirectoryOverview, true, true) => AutoInspectBudget {
            total_chars: 550,
            top_level_entries: 6,
            code_entries: 0,
            readme_chars: 120,
            manifest_chars: 160,
            entrypoint_chars: 0,
            search_files: 0,
            read_files: 0,
            key_hits_per_file: 0,
            workflow_summary_chars: 0,
        },
        (AutoInspectIntent::DirectoryOverview, true, false) => AutoInspectBudget {
            total_chars: 800,
            top_level_entries: 8,
            code_entries: 0,
            readme_chars: 160,
            manifest_chars: 220,
            entrypoint_chars: 0,
            search_files: 0,
            read_files: 0,
            key_hits_per_file: 0,
            workflow_summary_chars: 0,
        },
        (AutoInspectIntent::DirectoryOverview, false, true) => AutoInspectBudget {
            total_chars: 950,
            top_level_entries: 8,
            code_entries: 0,
            readme_chars: 180,
            manifest_chars: 240,
            entrypoint_chars: 0,
            search_files: 0,
            read_files: 0,
            key_hits_per_file: 0,
            workflow_summary_chars: 0,
        },
        (AutoInspectIntent::DirectoryOverview, false, false) => AutoInspectBudget {
            total_chars: 1300,
            top_level_entries: 10,
            code_entries: 0,
            readme_chars: 240,
            manifest_chars: 320,
            entrypoint_chars: 0,
            search_files: 0,
            read_files: 0,
            key_hits_per_file: 0,
            workflow_summary_chars: 0,
        },
        (AutoInspectIntent::WhereIsImplementation, true, true)
        | (AutoInspectIntent::FeatureTrace, true, true)
        | (AutoInspectIntent::ConfigLocate, true, true) => AutoInspectBudget {
            total_chars: 650,
            top_level_entries: 0,
            code_entries: 0,
            readme_chars: 0,
            manifest_chars: 120,
            entrypoint_chars: 0,
            search_files: 3,
            read_files: 2,
            key_hits_per_file: 2,
            workflow_summary_chars: 140,
        },
        (AutoInspectIntent::WhereIsImplementation, true, false)
        | (AutoInspectIntent::FeatureTrace, true, false)
        | (AutoInspectIntent::ConfigLocate, true, false) => AutoInspectBudget {
            total_chars: 900,
            top_level_entries: 0,
            code_entries: 0,
            readme_chars: 0,
            manifest_chars: 160,
            entrypoint_chars: 0,
            search_files: 3,
            read_files: 2,
            key_hits_per_file: 2,
            workflow_summary_chars: 180,
        },
        (AutoInspectIntent::WhereIsImplementation, false, true)
        | (AutoInspectIntent::FeatureTrace, false, true)
        | (AutoInspectIntent::ConfigLocate, false, true) => AutoInspectBudget {
            total_chars: 1100,
            top_level_entries: 0,
            code_entries: 0,
            readme_chars: 0,
            manifest_chars: 180,
            entrypoint_chars: 0,
            search_files: 4,
            read_files: 2,
            key_hits_per_file: 2,
            workflow_summary_chars: 220,
        },
        (AutoInspectIntent::WhereIsImplementation, false, false)
        | (AutoInspectIntent::FeatureTrace, false, false)
        | (AutoInspectIntent::ConfigLocate, false, false) => AutoInspectBudget {
            total_chars: 1400,
            top_level_entries: 0,
            code_entries: 0,
            readme_chars: 0,
            manifest_chars: 220,
            entrypoint_chars: 0,
            search_files: 5,
            read_files: 3,
            key_hits_per_file: 2,
            workflow_summary_chars: 260,
        },
    }
}

fn clip_inline(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }

    let trimmed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if trimmed.chars().count() <= max_chars {
        return trimmed;
    }

    let clipped = trimmed
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    format!("{}…", clipped.trim_end())
}

fn parse_list_dir_output(output: &str) -> Vec<String> {
    let mut lines = output.lines();
    let _ = lines.next();
    let body = lines.collect::<Vec<_>>().join("\n");
    body.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect()
}

fn parse_read_file_output(output: &str) -> Option<(String, String)> {
    let path = output
        .lines()
        .next()
        .and_then(|line| line.strip_prefix("File: "))?
        .trim()
        .to_string();
    let start = output.find("```\n")?;
    let rest = &output[start + 4..];
    let end = rest.rfind("\n```")?;
    Some((path, rest[..end].to_string()))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SearchLineHit {
    line_number: usize,
    line_content: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SearchFileHit {
    path: String,
    hits: Vec<SearchLineHit>,
}

fn parse_search_output(output: &str) -> Vec<SearchFileHit> {
    let mut files = Vec::new();
    let mut current: Option<SearchFileHit> = None;

    for raw_line in output.lines() {
        let line = raw_line.trim_end();
        if line.is_empty() || line.starts_with("Search results for ") {
            continue;
        }

        if !raw_line.starts_with(' ') && line.ends_with(':') {
            if let Some(file) = current.take() {
                files.push(file);
            }
            current = Some(SearchFileHit {
                path: line.trim_end_matches(':').to_string(),
                hits: Vec::new(),
            });
            continue;
        }

        if let Some(file) = current.as_mut() {
            let trimmed = raw_line.trim_start();
            let Some((line_number, line_content)) = trimmed.split_once(':') else {
                continue;
            };
            let Ok(line_number) = line_number.trim().parse::<usize>() else {
                continue;
            };
            file.hits.push(SearchLineHit {
                line_number,
                line_content: line_content.trim().to_string(),
            });
        }
    }

    if let Some(file) = current {
        files.push(file);
    }

    files
}

fn is_config_path(path: &str) -> bool {
    matches!(
        path,
        ".params.toml"
            | ".local/config.toml"
            | "Cargo.toml"
            | "package.json"
            | "pyproject.toml"
            | "go.mod"
    ) || path.contains("config")
}

fn is_doc_path(path: &str) -> bool {
    path.ends_with(".md") || path.starts_with("docs/")
}

fn is_code_path(path: &str) -> bool {
    [
        ".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".kt", ".swift", ".rb", ".php",
        ".cs", ".toml", ".json", ".yaml", ".yml", ".sh",
    ]
    .iter()
    .any(|suffix| path.ends_with(suffix))
}

fn file_display_name(path: &str) -> String {
    format!("`{path}`")
}

fn declaration_lines_with_numbers(content: &str) -> Vec<(usize, String)> {
    content
        .lines()
        .enumerate()
        .map(|(idx, line)| (idx + 1, line.trim()))
        .filter(|(_, line)| {
            line.starts_with("pub struct ")
                || line.starts_with("struct ")
                || line.starts_with("pub enum ")
                || line.starts_with("enum ")
                || line.starts_with("pub fn ")
                || line.starts_with("fn ")
                || line.starts_with("impl ")
                || line.starts_with("mod ")
                || line.starts_with("pub mod ")
                || line.starts_with('[')
        })
        .take(4)
        .map(|(line_number, line)| (line_number, line.to_string()))
        .collect()
}

fn match_lines_with_numbers(
    content: &str,
    query_terms: &[crate::memory::retrieval::QueryTerm],
    limit: usize,
) -> Vec<(usize, String)> {
    content
        .lines()
        .enumerate()
        .map(|(idx, line)| (idx + 1, line.trim()))
        .filter(|(_, line)| !line.is_empty())
        .filter(|(_, line)| score_text(query_terms, line) > 0)
        .take(limit)
        .map(|(line_number, line)| (line_number, line.to_string()))
        .collect()
}

fn definition_match_lines_with_numbers(
    content: &str,
    query_terms: &[crate::memory::retrieval::QueryTerm],
    limit: usize,
) -> Vec<(usize, String)> {
    content
        .lines()
        .enumerate()
        .map(|(idx, line)| (idx + 1, line.trim()))
        .filter(|(_, line)| !line.is_empty())
        .filter(|(_, line)| score_text(query_terms, line) > 0)
        .filter(|(_, line)| is_implementation_definition_line(line))
        .take(limit)
        .map(|(line_number, line)| (line_number, line.to_string()))
        .collect()
}

fn primary_definition_location(
    path: &str,
    content: &str,
    query: &str,
    max_chars: usize,
) -> Option<String> {
    let terms = query_terms(query);
    let matches = definition_match_lines_with_numbers(content, &terms, 1);
    // Do NOT fall back to declaration_lines_with_numbers. If the file has no
    // definition line matching the query, it simply doesn't define the symbol —
    // reporting an unrelated declaration as the "Primary definition" produces
    // a false anchor the model will cite with a wrong line number.
    let (line_number, line) = matches.into_iter().next()?;
    Some(format!(
        "{}:{} `{}`",
        path,
        line_number,
        clip_inline(&line, max_chars)
    ))
}

fn format_numbered_hits(hits: &[(usize, String)], clip_chars: usize) -> String {
    hits.iter()
        .map(|(line_number, line)| format!("{line_number} `{}`", clip_inline(line, clip_chars)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn is_definition_like_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with("pub fn ")
        || trimmed.starts_with("fn ")
        || trimmed.starts_with("pub struct ")
        || trimmed.starts_with("struct ")
        || trimmed.starts_with("pub enum ")
        || trimmed.starts_with("enum ")
        || trimmed.starts_with("impl ")
        || trimmed.starts_with("pub mod ")
        || trimmed.starts_with("mod ")
        || trimmed.starts_with("pub use ")
}

fn is_implementation_definition_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with("pub fn ")
        || trimmed.starts_with("fn ")
        || trimmed.starts_with("pub struct ")
        || trimmed.starts_with("struct ")
        || trimmed.starts_with("pub enum ")
        || trimmed.starts_with("enum ")
        || trimmed.starts_with("impl ")
}

fn config_scope_label(path: &str) -> &'static str {
    match path {
        ".params.toml" => "project-local",
        ".local/config.toml" => "global",
        _ if path.contains("config") => "runtime/config code",
        _ => "repo config",
    }
}

fn score_search_file(intent: AutoInspectIntent, query: &str, file: &SearchFileHit) -> isize {
    let terms = query_terms(query);
    let exact = query.to_ascii_lowercase();
    let path_lower = file.path.to_ascii_lowercase();
    let path_score = score_text(&terms, &file.path) as isize * 6;
    let line_score = file
        .hits
        .iter()
        .take(4)
        .map(|hit| score_text(&terms, &hit.line_content) as isize)
        .sum::<isize>();
    let exact_path_bonus = if !exact.is_empty() && path_lower.contains(&exact) {
        30
    } else {
        0
    };
    let exact_line_bonus = if !exact.is_empty()
        && file
            .hits
            .iter()
            .any(|hit| hit.line_content.to_ascii_lowercase().contains(&exact))
    {
        16
    } else {
        0
    };
    let count_bonus = (file.hits.len().min(6) as isize) * 3;
    let workflow_bonus = match intent {
        AutoInspectIntent::WhereIsImplementation => {
            let mut bonus = 0;
            if is_code_path(&file.path) {
                bonus += 14;
            }
            if file.path.starts_with("src/") {
                bonus += 10;
            }
            if is_doc_path(&file.path) {
                bonus -= 10;
            }
            bonus
        }
        AutoInspectIntent::FeatureTrace => {
            let mut bonus = 0;
            if is_code_path(&file.path) {
                bonus += 12;
            }
            if file.path.ends_with("main.rs") || file.path.ends_with("mod.rs") {
                bonus += 8;
            }
            if is_doc_path(&file.path) {
                bonus -= 8;
            }
            bonus
        }
        AutoInspectIntent::ConfigLocate => {
            let mut bonus = 0;
            if is_config_path(&file.path) {
                bonus += 18;
            }
            if path_lower.contains("config") {
                bonus += 10;
            }
            bonus
        }
        AutoInspectIntent::RepoOverview | AutoInspectIntent::DirectoryOverview => 0,
    };

    path_score + line_score + exact_path_bonus + exact_line_bonus + count_bonus + workflow_bonus
}

fn rank_search_files(
    intent: AutoInspectIntent,
    query: &str,
    files: &[SearchFileHit],
) -> Vec<SearchFileHit> {
    let mut ranked = files.to_vec();
    ranked.sort_by(|a, b| {
        score_search_file(intent, query, b)
            .cmp(&score_search_file(intent, query, a))
            .then_with(|| a.path.cmp(&b.path))
    });
    ranked
}

fn preferred_config_paths(project_root: &Path) -> Vec<String> {
    [
        ".params.toml",
        ".local/config.toml",
        "src/config.rs",
        "config.rs",
        "Cargo.toml",
        "package.json",
        "pyproject.toml",
        "go.mod",
    ]
    .iter()
    .filter(|rel| project_root.join(rel).is_file())
    .map(|rel| (*rel).to_string())
    .collect()
}

fn preferred_workflow_paths(plan: &AutoInspectPlan, project_root: &Path) -> Vec<String> {
    let mut preferred = match plan.query.as_deref() {
        Some("load_most_recent") => {
            // src/inference/session.rs is ~150 KB — exceeds the read_file 100 KB
            // limit and never defines this function. Only list the file that
            // actually defines it.
            vec!["src/session/mod.rs".to_string()]
        }
        Some("save_messages") => {
            // The actual persistence logic lives in src/session/mod.rs.
            // src/inference/session.rs is ~150 KB and always fails to read.
            vec!["src/session/mod.rs".to_string()]
        }
        Some("eco.enabled") => vec![
            "src/config/profile.rs".to_string(),
            "src/inference/session.rs".to_string(),
            "src/tui/commands.rs".to_string(),
            "src/config.rs".to_string(),
        ],
        _ => Vec::new(),
    };

    if plan.intent == AutoInspectIntent::ConfigLocate {
        preferred.extend(preferred_config_paths(project_root));
    }

    preferred
        .into_iter()
        .filter(|rel| project_root.join(rel).is_file())
        .collect()
}

fn is_auto_inspection_read_candidate(project_root: &Path, rel: &str) -> bool {
    const MAX_READ_BYTES: u64 = 100_000;
    let path = project_root.join(rel);
    path.is_file()
        && path
            .metadata()
            .map(|meta| meta.len() <= MAX_READ_BYTES)
            .unwrap_or(false)
}

fn choose_followup_read_steps(
    plan: &AutoInspectPlan,
    project_root: &Path,
    search_hits: &[SearchFileHit],
    budget: AutoInspectBudget,
) -> Vec<AutoInspectStep> {
    let Some(query) = plan.query.as_deref() else {
        return Vec::new();
    };

    let ranked = rank_search_files(plan.intent, query, search_hits);
    let code_first_hits = match plan.intent {
        AutoInspectIntent::WhereIsImplementation | AutoInspectIntent::FeatureTrace => {
            let code_hits = ranked
                .iter()
                .filter(|file| file.path.starts_with("src/") || is_code_path(&file.path))
                .cloned()
                .collect::<Vec<_>>();
            if code_hits.is_empty() {
                ranked
            } else {
                code_hits
            }
        }
        AutoInspectIntent::ConfigLocate => {
            let config_hits = ranked
                .iter()
                .filter(|file| {
                    is_config_path(&file.path)
                        || file.path.starts_with("src/")
                        || file.path.starts_with("config/")
                })
                .cloned()
                .collect::<Vec<_>>();
            if config_hits.is_empty() {
                ranked
            } else {
                config_hits
            }
        }
        AutoInspectIntent::RepoOverview | AutoInspectIntent::DirectoryOverview => ranked,
    };

    let mut selected = code_first_hits
        .into_iter()
        .take(budget.search_files)
        .map(|file| file.path)
        .collect::<Vec<_>>();

    let mut preferred = preferred_workflow_paths(plan, project_root);
    preferred.reverse();
    for path in preferred {
        if let Some(existing_idx) = selected.iter().position(|existing| existing == &path) {
            let item = selected.remove(existing_idx);
            selected.insert(0, item);
        } else {
            selected.insert(0, path);
        }
    }

    if plan.intent == AutoInspectIntent::ConfigLocate {
        let mut preferred = preferred_config_paths(project_root);
        preferred.reverse();
        for path in preferred {
            if let Some(existing_idx) = selected.iter().position(|existing| existing == &path) {
                let item = selected.remove(existing_idx);
                selected.insert(0, item);
            } else {
                selected.insert(0, path);
            }
        }
    }

    selected.truncate(budget.read_files);
    selected
        .into_iter()
        .filter(|path| is_auto_inspection_read_candidate(project_root, path))
        .map(|path| AutoInspectStep {
            label: format!("Read {path}"),
            tool_name: "read_file",
            argument: path,
        })
        .collect()
}

fn summarize_workflow_read(
    path: &str,
    content: &str,
    query: &str,
    intent: AutoInspectIntent,
    max_chars: usize,
) -> Option<String> {
    let terms = query_terms(query);
    let declarations = declaration_lines_with_numbers(content);

    match intent {
        AutoInspectIntent::ConfigLocate => {
            let matches = match_lines_with_numbers(content, &terms, 2);
            let mut parts = vec![format!(
                "{} ({})",
                file_display_name(path),
                config_scope_label(path)
            )];
            if !matches.is_empty() {
                parts.push(format!(
                    "exact lines: {}",
                    format_numbered_hits(&matches, 48)
                ));
            } else if !declarations.is_empty() {
                parts.push(format!(
                    "declarations: {}",
                    format_numbered_hits(&declarations, 40)
                ));
            }
            Some(clip_inline(&parts.join("; "), max_chars))
        }
        AutoInspectIntent::WhereIsImplementation => {
            let matches = if intent == AutoInspectIntent::WhereIsImplementation {
                let preferred = definition_match_lines_with_numbers(content, &terms, 2);
                if preferred.is_empty() {
                    match_lines_with_numbers(content, &terms, 2)
                } else {
                    preferred
                }
            } else {
                match_lines_with_numbers(content, &terms, 2)
            };
            let mut parts = vec![file_display_name(path)];
            if !matches.is_empty() {
                parts.push(format!(
                    "exact lines: {}",
                    format_numbered_hits(&matches, 48)
                ));
            }
            // Only add the generic declarations list when no exact matches were
            // found. When we already have the target line, the first-N declarations
            // from the file are unrelated noise — small line numbers from structs
            // near the top of the file that the model will erroneously cite.
            if matches.is_empty() && !declarations.is_empty() {
                parts.push(format!(
                    "declarations: {}",
                    format_numbered_hits(&declarations, 36)
                ));
            }
            Some(clip_inline(&parts.join("; "), max_chars))
        }
        AutoInspectIntent::FeatureTrace => {
            let matches = match_lines_with_numbers(content, &terms, 3);
            if matches.is_empty() {
                return None;
            }

            let parts = vec![
                file_display_name(path),
                format!("flow lines: {}", format_numbered_hits(&matches, 48)),
            ];
            Some(clip_inline(&parts.join("; "), max_chars))
        }
        AutoInspectIntent::RepoOverview | AutoInspectIntent::DirectoryOverview => None,
    }
}

fn summarize_feature_trace_hits(
    query: &str,
    ranked: &[SearchFileHit],
    budget: AutoInspectBudget,
) -> Vec<String> {
    ranked
        .iter()
        .take(budget.search_files)
        .flat_map(|file| {
            file.hits
                .iter()
                .filter(move |hit| is_feature_trace_anchor_line(query, &hit.line_content))
                .take(budget.key_hits_per_file + 1)
                .map(move |hit| {
                    format!(
                        "{}:{} `{}`",
                        file.path,
                        hit.line_number,
                        clip_inline(&hit.line_content, 56)
                    )
                })
        })
        .take(budget.search_files * (budget.key_hits_per_file + 1))
        .collect()
}

fn is_feature_trace_anchor_line(query: &str, line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }

    let query_trimmed = query.trim();
    let bare_query = query_trimmed.trim_end_matches('(');
    let lower = trimmed.to_ascii_lowercase();
    let contains_symbol = lower.contains(query_trimmed)
        || (!bare_query.is_empty() && lower.contains(&format!("{bare_query}(")));

    if !contains_symbol {
        return false;
    }

    if trimmed.contains('"')
        || trimmed.contains('\'')
        || trimmed.contains("assert!")
        || trimmed.contains("return Some(")
        || trimmed.contains("Some(\"")
        || trimmed.contains("output:")
    {
        return false;
    }

    is_definition_like_line(trimmed) || trimmed.contains(&format!("{bare_query}("))
}

fn summarize_readme(content: &str, max_chars: usize) -> Option<String> {
    let mut focus = Vec::new();
    for line in content.lines().map(str::trim) {
        if line.is_empty() {
            continue;
        }
        if let Some(heading) = line.strip_prefix("# ") {
            focus.push(heading.to_string());
            continue;
        }
        if line.starts_with("- ") || line.starts_with("* ") {
            focus.push(line[2..].trim().to_string());
        } else {
            focus.push(line.to_string());
        }
        if focus.len() >= 4 {
            break;
        }
    }

    if focus.is_empty() {
        None
    } else {
        Some(format!(
            "README focus: {}",
            clip_inline(&focus.join("; "), max_chars)
        ))
    }
}

fn summarize_cargo_manifest(content: &str, max_chars: usize) -> Option<String> {
    let value = toml::from_str::<toml::Value>(content).ok()?;
    let mut parts = Vec::new();

    if let Some(name) = value
        .get("package")
        .and_then(|pkg| pkg.get("name"))
        .and_then(|name| name.as_str())
    {
        parts.push(format!("Rust package `{name}`"));
    } else if value.get("workspace").is_some() {
        parts.push("Rust workspace manifest".to_string());
    }

    if let Some(description) = value
        .get("package")
        .and_then(|pkg| pkg.get("description"))
        .and_then(|desc| desc.as_str())
    {
        parts.push(clip_inline(description, max_chars / 2));
    }

    let mut deps = value
        .get("dependencies")
        .and_then(|deps| deps.as_table())
        .map(|table| table.keys().take(6).cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    deps.sort();
    if !deps.is_empty() {
        parts.push(format!("key deps: {}", deps.join(", ")));
    }

    if parts.is_empty() {
        None
    } else {
        Some(format!(
            "Manifest: {}",
            clip_inline(&parts.join("; "), max_chars)
        ))
    }
}

fn summarize_entrypoint(path: &str, content: &str, max_chars: usize) -> Option<String> {
    if !path.ends_with(".rs") {
        return None;
    }

    let mut modules = Vec::new();
    let mut has_main = false;
    for line in content.lines().map(str::trim) {
        if let Some(name) = line
            .strip_prefix("mod ")
            .or_else(|| line.strip_prefix("pub mod "))
        {
            let name = name.trim_end_matches(';').trim();
            if !name.is_empty() {
                modules.push(name.to_string());
            }
        }
        if line.starts_with("fn main(") || line.starts_with("pub fn main(") {
            has_main = true;
        }
    }

    let mut parts = vec![format!(
        "{} `{}`",
        if has_main { "Entrypoint" } else { "Root file" },
        path
    )];
    if !modules.is_empty() {
        modules.truncate(8);
        parts.push(format!("modules: {}", modules.join(", ")));
    }

    if parts.len() == 1 && !has_main {
        None
    } else {
        Some(clip_inline(&parts.join("; "), max_chars))
    }
}

fn top_level_repo_type(entries: &[String]) -> Option<String> {
    if entries.iter().any(|entry| entry == "Cargo.toml") {
        Some("Repo type: Rust project".to_string())
    } else if entries.iter().any(|entry| entry == "package.json") {
        Some("Repo type: Node project".to_string())
    } else if entries.iter().any(|entry| entry == "pyproject.toml") {
        Some("Repo type: Python project".to_string())
    } else if entries.iter().any(|entry| entry == "go.mod") {
        Some("Repo type: Go project".to_string())
    } else {
        None
    }
}

fn format_entry_list(label: &str, entries: &[String], limit: usize) -> Option<String> {
    if entries.is_empty() || limit == 0 {
        return None;
    }

    let shown = entries
        .iter()
        .take(limit)
        .map(|entry| format!("`{entry}`"))
        .collect::<Vec<_>>();
    let extra = entries.len().saturating_sub(limit);
    let mut text = shown.join(", ");
    if extra > 0 {
        text.push_str(&format!(", +{extra} more"));
    }
    Some(format!("{label}: {text}"))
}

fn synthesize_auto_inspection_context(
    plan: &AutoInspectPlan,
    results: &[ToolResult],
    budget: AutoInspectBudget,
) -> Option<String> {
    if results.is_empty() {
        return None;
    }

    if matches!(
        plan.intent,
        AutoInspectIntent::WhereIsImplementation
            | AutoInspectIntent::FeatureTrace
            | AutoInspectIntent::ConfigLocate
    ) {
        let query = plan.query.as_deref()?;
        let mut search_hits = Vec::new();
        let mut read_summaries = Vec::new();
        let mut read_paths = Vec::new();
        let mut primary_locations = Vec::new();

        for result in results {
            match result.tool_name.as_str() {
                "search" => search_hits.extend(parse_search_output(&result.output)),
                "read_file" => {
                    if let Some((path, content)) = parse_read_file_output(&result.output) {
                        read_paths.push(path.clone());
                        if plan.intent == AutoInspectIntent::WhereIsImplementation {
                            if let Some(location) =
                                primary_definition_location(&path, &content, query, 72)
                            {
                                primary_locations.push(location);
                            }
                        }
                        if let Some(summary) = summarize_workflow_read(
                            &path,
                            &content,
                            query,
                            plan.intent,
                            budget.workflow_summary_chars,
                        ) {
                            read_summaries.push(summary);
                        }
                    }
                }
                _ => {}
            }
        }

        let ranked = rank_search_files(plan.intent, query, &search_hits);
        let mut likely_files = read_paths
            .iter()
            .map(|path| file_display_name(path))
            .collect::<Vec<_>>();
        for file in ranked.iter().take(budget.search_files) {
            let display = file_display_name(&file.path);
            if !likely_files.iter().any(|existing| existing == &display) {
                likely_files.push(display);
            }
        }
        likely_files.truncate(budget.search_files.max(budget.read_files));

        let supporting_hits = if read_paths.is_empty() {
            ranked.iter().take(budget.search_files).collect::<Vec<_>>()
        } else {
            ranked
                .iter()
                .filter(|file| read_paths.iter().any(|path| path == &file.path))
                .take(budget.search_files)
                .collect::<Vec<_>>()
        };
        let has_read_summaries = !read_summaries.is_empty();
        let flow_hits = if plan.intent == AutoInspectIntent::FeatureTrace {
            summarize_feature_trace_hits(query, &ranked, budget)
        } else {
            Vec::new()
        };

        let key_hits = supporting_hits
            .into_iter()
            .flat_map(|file| {
                file.hits
                    .iter()
                    .filter(move |hit| {
                        plan.intent != AutoInspectIntent::WhereIsImplementation
                            || !has_read_summaries
                            || is_definition_like_line(&hit.line_content)
                    })
                    .take(budget.key_hits_per_file)
                    .map(move |hit| {
                        format!(
                            "{}:{} `{}`",
                            file.path,
                            hit.line_number,
                            clip_inline(&hit.line_content, 56)
                        )
                    })
            })
            .take(budget.search_files * budget.key_hits_per_file)
            .collect::<Vec<_>>();

        let mut sections = vec![format!(
            "Automatic inspection context for {}:",
            plan.context_label
        )];
        let instruction = match plan.intent {
            AutoInspectIntent::WhereIsImplementation => {
                "Instruction: answer directly from this evidence. Prefer exact inspected-file evidence over supporting search hits. Report definition or implementation locations only, not use-sites, call-sites, tests, or later references. If multiple line numbers appear, cite the primary definition line and omit usage lines. Do not ask for more inspection unless the evidence is clearly insufficient. Do not emit tool calls or fenced code blocks. If exact code is not included below, answer in prose with file paths and line references only."
            }
            AutoInspectIntent::FeatureTrace => {
                "Instruction: answer directly from this evidence. Focus on the actual control flow using the flow anchors and inspected file hints below. Do not invent function bodies, placeholder snippets, or implementation details that are not present in the evidence — if the evidence shows only a function signature or a call site, describe what the name and signature tell you and cite the file:line location. Do not emit tool calls or fenced code blocks. Do not speculate from unrelated declarations."
            }
            _ => {
                "Instruction: answer directly from this evidence. Prefer exact inspected-file evidence over supporting search hits. Do not ask for more inspection unless the evidence is clearly insufficient. Do not emit tool calls or fenced code blocks. If exact code is not included below, answer in prose with file paths and line references only."
            }
        };
        sections.push(instruction.to_string());
        sections.push(format!("Query: {query}"));
        if !primary_locations.is_empty() {
            sections.push(format!(
                "Primary definition: {}",
                primary_locations.join(", ")
            ));
        }
        if !likely_files.is_empty() {
            sections.push(format!("Likely files: {}", likely_files.join(", ")));
        }
        if !flow_hits.is_empty() {
            sections.push(format!("Primary flow anchors: {}", flow_hits.join("; ")));
        }
        if !read_summaries.is_empty() {
            let label = match plan.intent {
                AutoInspectIntent::WhereIsImplementation => "Implementation hints",
                AutoInspectIntent::FeatureTrace => "Flow hints",
                AutoInspectIntent::ConfigLocate => "Config hints",
                AutoInspectIntent::RepoOverview | AutoInspectIntent::DirectoryOverview => {
                    unreachable!()
                }
            };
            sections.push(format!("{label}: {}", read_summaries.join(" | ")));
        }
        let include_supporting_hits = !key_hits.is_empty()
            && !(plan.intent == AutoInspectIntent::WhereIsImplementation && has_read_summaries);
        if include_supporting_hits {
            let label = if read_summaries.is_empty() {
                "Key hits"
            } else {
                "Supporting search hits"
            };
            sections.push(format!("{label}: {}", key_hits.join("; ")));
        }
        // When the workflow ran but could not inspect any file content (e.g.
        // every candidate file exceeded the read limit), tell the model
        // explicitly so it does not fill the gap with invented snippets.
        if !has_read_summaries
            && matches!(
                plan.intent,
                AutoInspectIntent::FeatureTrace
                    | AutoInspectIntent::WhereIsImplementation
                    | AutoInspectIntent::ConfigLocate
            )
            && (!flow_hits.is_empty() || !key_hits.is_empty())
        {
            sections.push(
                "Evidence: search anchors only — no file content was inspected. \
                 Cite only the locations above. Do not infer or invent \
                 function bodies or implementation details."
                    .to_string(),
            );
        }

        let mut output = String::new();
        for section in sections {
            let candidate = if output.is_empty() {
                section
            } else {
                format!("{output}\n- {section}")
            };
            if candidate.chars().count() > budget.total_chars {
                break;
            }
            output = candidate;
        }

        return if output.is_empty() {
            None
        } else {
            Some(output)
        };
    }

    let mut root_entries = Vec::new();
    let mut code_entries = Vec::new();
    let mut readme_summary = None;
    let mut manifest_summary = None;
    let mut entrypoint_summary = None;

    for result in results {
        match result.tool_name.as_str() {
            "list_dir" if result.argument == "." => {
                root_entries = parse_list_dir_output(&result.output);
            }
            "list_dir" if result.argument == "src" => {
                code_entries = parse_list_dir_output(&result.output);
            }
            "read_file" => {
                if let Some((path, content)) = parse_read_file_output(&result.output) {
                    match path.as_str() {
                        "README.md" => {
                            readme_summary = summarize_readme(&content, budget.readme_chars);
                        }
                        "Cargo.toml" => {
                            manifest_summary =
                                summarize_cargo_manifest(&content, budget.manifest_chars);
                        }
                        "src/main.rs" | "src/lib.rs" => {
                            entrypoint_summary =
                                summarize_entrypoint(&path, &content, budget.entrypoint_chars);
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    let mut sections = vec![format!(
        "Automatic inspection context for {}:",
        plan.context_label
    )];

    if let Some(repo_type) = top_level_repo_type(&root_entries) {
        sections.push(repo_type);
    }
    if let Some(top_level) = format_entry_list("Top level", &root_entries, budget.top_level_entries)
    {
        sections.push(top_level);
    }
    if let Some(code_areas) = format_entry_list("Code areas", &code_entries, budget.code_entries) {
        sections.push(code_areas);
    }
    if let Some(summary) = manifest_summary {
        sections.push(summary);
    }
    if let Some(summary) = entrypoint_summary {
        sections.push(summary);
    }
    if let Some(summary) = readme_summary {
        sections.push(summary);
    }

    let mut output = String::new();
    for section in sections {
        let candidate = if output.is_empty() {
            section
        } else {
            format!("{output}\n- {section}")
        };

        if candidate.chars().count() > budget.total_chars {
            break;
        }
        output = candidate;
    }

    if output.is_empty() {
        None
    } else if output.starts_with("Automatic inspection context") {
        Some(output)
    } else {
        Some(format!(
            "Automatic inspection context for {}:\n- {}",
            plan.context_label, output
        ))
    }
}

fn run_auto_inspection(
    plan: &AutoInspectPlan,
    token_tx: &Sender<InferenceEvent>,
    eco_enabled: bool,
    backend_name: &str,
    project_root: &Path,
) -> AutoInspectOutcome {
    let _ = token_tx.send(InferenceEvent::SystemMessage(plan.thinking.to_string()));

    let budget = auto_inspection_budget(plan.intent, backend_name, eco_enabled);
    let mut results = Vec::new();

    for step in &plan.steps {
        emit_trace(token_tx, ProgressStatus::Started, &step.label, false);
        let run_result = match step.tool_name {
            "list_dir" => ListDir.run(&step.argument),
            "read_file" => ReadFile.run(&step.argument),
            "search" => SearchCode.run(&step.argument),
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

    if matches!(
        plan.intent,
        AutoInspectIntent::WhereIsImplementation
            | AutoInspectIntent::FeatureTrace
            | AutoInspectIntent::ConfigLocate
    ) {
        let search_hits = results
            .iter()
            .filter(|result| result.tool_name == "search")
            .flat_map(|result| parse_search_output(&result.output))
            .collect::<Vec<_>>();
        let followup_reads = choose_followup_read_steps(plan, project_root, &search_hits, budget);
        for step in followup_reads {
            emit_trace(token_tx, ProgressStatus::Started, &step.label, false);
            match ReadFile.run(&step.argument) {
                Ok(ToolRunResult::Immediate(output)) => {
                    emit_trace(token_tx, ProgressStatus::Finished, &step.label, false);
                    results.push(ToolResult {
                        tool_name: step.tool_name.to_string(),
                        argument: step.argument,
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
    }

    let hidden_context = synthesize_auto_inspection_context(plan, &results, budget);

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

fn suppress_retrieval_for_auto_inspection(intent: AutoInspectIntent) -> bool {
    matches!(
        intent,
        AutoInspectIntent::WhereIsImplementation
            | AutoInspectIntent::FeatureTrace
            | AutoInspectIntent::ConfigLocate
    )
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
                    .map(|intent| plan_auto_inspection(intent, &prompt, &project_root));
                let auto_inspection_outcome = auto_inspection.as_ref().map(|plan| {
                    emit_generation_started(&token_tx, plan.status_label, false);
                    run_auto_inspection(
                        plan,
                        &token_tx,
                        eco_enabled,
                        &backend.name(),
                        &project_root,
                    )
                });
                let hidden_inspection_context = auto_inspection_outcome
                    .as_ref()
                    .and_then(|outcome| outcome.hidden_context.clone());

                let retrieval = if auto_inspection
                    .as_ref()
                    .map(|plan| suppress_retrieval_for_auto_inspection(plan.intent))
                    .unwrap_or(false)
                {
                    RetrievalBundle::default()
                } else {
                    collect_retrieval_bundle(
                        &prompt,
                        eco_enabled,
                        &project_name,
                        project_index.as_ref(),
                        fact_store.as_ref(),
                        session_store.as_ref(),
                        active_session.as_ref().map(|session| session.id.as_str()),
                        &memory_state.loaded_facts,
                    )
                };
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

                let mut generation_messages = session_messages.clone();
                if let Some(hidden_context) = hidden_inspection_context.as_ref() {
                    generation_messages.push(Message::user(hidden_context));
                }
                let generation_exact_cache = if auto_inspection.is_some() {
                    None
                } else {
                    exact_cache.as_ref()
                };

                emit_generation_started(&token_tx, "generating...", false);
                emit_trace(
                    &token_tx,
                    ProgressStatus::Started,
                    "drafting answer...",
                    false,
                );
                hooks.dispatch(HookEvent::BeforeGeneration {
                    backend: backend.name(),
                    message_count: generation_messages.len(),
                    eco: eco_enabled,
                    reflection: reflection_enabled,
                });
                let response = generate_with_cache(
                    &*backend,
                    &generation_messages,
                    &cfg,
                    &project_root,
                    token_tx.clone(),
                    !reflection_enabled,
                    generation_exact_cache,
                    &mut cache_stats,
                    CacheMode::PreferPromptLevel,
                );
                debug!(
                    reflection_enabled,
                    message_count = generation_messages.len(),
                    "generation started"
                );
                let prompt_tokens = estimate_message_tokens(&generation_messages);

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
    fn detect_auto_inspect_intent_matches_workflow_prompts() {
        assert_eq!(
            detect_auto_inspect_intent("Where is cache implemented?"),
            Some(AutoInspectIntent::WhereIsImplementation)
        );
        assert_eq!(
            detect_auto_inspect_intent("Trace how sessions are saved"),
            Some(AutoInspectIntent::FeatureTrace)
        );
        assert_eq!(
            detect_auto_inspect_intent("Where is eco mode configured?"),
            Some(AutoInspectIntent::ConfigLocate)
        );
    }

    #[test]
    fn detect_auto_inspect_intent_skips_unrelated_prompts() {
        assert_eq!(detect_auto_inspect_intent("Explain the cache"), None);
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

        let plan = plan_auto_inspection(
            AutoInspectIntent::RepoOverview,
            "What's in this repo?",
            &root,
        );
        let labels = plan
            .steps
            .iter()
            .map(|step| step.label.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            labels,
            vec![
                "List .",
                "List src/",
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

        let plan = plan_auto_inspection(AutoInspectIntent::DirectoryOverview, "What's here", &root);
        let labels = plan
            .steps
            .iter()
            .map(|step| step.label.as_str())
            .collect::<Vec<_>>();

        assert_eq!(labels, vec!["List .", "Read README.md", "Read Cargo.toml"]);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn workflow_auto_inspection_starts_with_search() {
        let root = temp_project_root("workflow-plan");
        fs::create_dir_all(root.join("src")).expect("create src");

        let where_plan = plan_auto_inspection(
            AutoInspectIntent::WhereIsImplementation,
            "Where is cache implemented?",
            &root,
        );
        let trace_plan = plan_auto_inspection(
            AutoInspectIntent::FeatureTrace,
            "Trace how sessions are saved",
            &root,
        );
        let config_plan = plan_auto_inspection(
            AutoInspectIntent::ConfigLocate,
            "Where is eco mode configured?",
            &root,
        );

        assert_eq!(where_plan.steps[0].tool_name, "search");
        assert_eq!(where_plan.steps[0].argument, "cache");
        assert_eq!(trace_plan.steps[0].tool_name, "search");
        // save_session is a thin wrapper in the ~150 KB unreadable file; the
        // real persistence function save_messages is in the readable session
        // store, so we search for that instead.
        assert_eq!(trace_plan.steps[0].argument, "save_messages");
        assert_eq!(config_plan.steps[0].tool_name, "search");
        assert_eq!(config_plan.steps[0].argument, "eco.enabled");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn workflow_query_extraction_prefers_salient_terms() {
        assert_eq!(
            extract_auto_inspect_query(
                "Where is session restore implemented?",
                AutoInspectIntent::WhereIsImplementation
            )
            .as_deref(),
            Some("load_most_recent")
        );
        assert_eq!(
            extract_auto_inspect_query(
                "Trace how sessions are saved",
                AutoInspectIntent::FeatureTrace
            )
            .as_deref(),
            Some("save_messages")
        );
        assert_eq!(
            extract_auto_inspect_query(
                "Where is eco mode configured?",
                AutoInspectIntent::ConfigLocate
            )
            .as_deref(),
            Some("eco.enabled")
        );
    }

    #[test]
    fn parse_search_output_groups_hits_by_file() {
        let hits = parse_search_output(
            "Search results for 'cache' (3 matches):\n\nsrc/main.rs:\n     4: mod cache;\n    18: cache::warm();\n\nsrc/cache/mod.rs:\n     2: pub fn warm() {}\n",
        );

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].path, "src/main.rs");
        assert_eq!(hits[0].hits.len(), 2);
        assert_eq!(hits[1].path, "src/cache/mod.rs");
    }

    #[test]
    fn config_candidate_ranking_prefers_config_paths() {
        let hits = vec![
            SearchFileHit {
                path: "docs/context/CLAUDE.md".to_string(),
                hits: vec![SearchLineHit {
                    line_number: 1,
                    line_content: "eco mode documentation".to_string(),
                }],
            },
            SearchFileHit {
                path: "src/config.rs".to_string(),
                hits: vec![SearchLineHit {
                    line_number: 22,
                    line_content: "pub struct EcoConfig".to_string(),
                }],
            },
        ];

        let ranked = rank_search_files(AutoInspectIntent::ConfigLocate, "eco mode", &hits);
        assert_eq!(ranked[0].path, "src/config.rs");
    }

    #[test]
    fn choose_followup_read_steps_prefers_code_paths_when_available() {
        let root = temp_project_root("workflow-select");
        fs::create_dir_all(root.join("src/session")).expect("create src/session");
        fs::write(
            root.join("src/session/mod.rs"),
            "pub fn load_session() {}\n",
        )
        .expect("write session mod");
        fs::write(root.join("docs.md"), "load_most_recent docs\n").expect("write docs");

        let plan = AutoInspectPlan {
            intent: AutoInspectIntent::WhereIsImplementation,
            thinking: "Thinking: locating the most likely implementation files.",
            status_label: "locating implementation...",
            context_label: "this implementation lookup request",
            query: Some("load_most_recent".to_string()),
            steps: vec![],
        };

        let hits = vec![
            SearchFileHit {
                path: "docs/context/CLAUDE.md".to_string(),
                hits: vec![SearchLineHit {
                    line_number: 10,
                    line_content: "load_most_recent docs".to_string(),
                }],
            },
            SearchFileHit {
                path: "src/session/mod.rs".to_string(),
                hits: vec![SearchLineHit {
                    line_number: 2,
                    line_content: "pub fn load_most_recent()".to_string(),
                }],
            },
        ];

        let steps = choose_followup_read_steps(
            &plan,
            &root,
            &hits,
            auto_inspection_budget(
                AutoInspectIntent::WhereIsImplementation,
                "llama.cpp · qwen",
                false,
            ),
        );

        assert_eq!(steps[0].argument, "src/session/mod.rs");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn auto_inspection_hidden_context_is_compact_and_structural() {
        let plan = AutoInspectPlan {
            intent: AutoInspectIntent::RepoOverview,
            thinking: "Thinking: exploring the repo structure and key project docs.",
            status_label: "inspecting repo...",
            context_label: "this repo summary request",
            query: None,
            steps: vec![],
        };

        let hidden = synthesize_auto_inspection_context(
            &plan,
            &[
                ToolResult {
                    tool_name: "list_dir".to_string(),
                    argument: ".".to_string(),
                    output: "Directory: .\n\nsrc/\ndocs/\nREADME.md\nCargo.toml".to_string(),
                },
                ToolResult {
                    tool_name: "list_dir".to_string(),
                    argument: "src".to_string(),
                    output: "Directory: src\n\ncache/\ninference/\ntui/\nmain.rs".to_string(),
                },
                ToolResult {
                    tool_name: "read_file".to_string(),
                    argument: "Cargo.toml".to_string(),
                    output: "File: Cargo.toml\nLines: 8\n\n```\n[package]\nname = \"params-cli\"\ndescription = \"Personal AI coding assistant CLI\"\n\n[dependencies]\ncrossterm = \"0.28\"\nllama-cpp-2 = \"0.1\"\nserde = \"1\"\n```\n".to_string(),
                },
                ToolResult {
                    tool_name: "read_file".to_string(),
                    argument: "src/main.rs".to_string(),
                    output: "File: src/main.rs\nLines: 5\n\n```\nmod cache;\nmod inference;\nmod tui;\n\nfn main() {}\n```\n".to_string(),
                },
            ],
            auto_inspection_budget(AutoInspectIntent::RepoOverview, "llama.cpp · test", false),
        )
        .expect("hidden context");

        assert!(hidden.starts_with("Automatic inspection context for this repo summary request:"));
        assert!(hidden.contains("Repo type: Rust project"));
        assert!(hidden.contains("Code areas:"));
        assert!(hidden.contains("Manifest:"));
        assert!(hidden.contains("Entrypoint `src/main.rs`; modules: cache, inference, tui"));
        assert!(!hidden.contains("--- list_dir(.) ---"));
        assert!(!hidden.contains("Tool results:"));
        assert!(hidden.chars().count() <= 1000);
    }

    #[test]
    fn auto_inspection_budget_is_tighter_for_llama_cpp() {
        let local =
            auto_inspection_budget(AutoInspectIntent::RepoOverview, "llama.cpp · qwen", false);
        let cloud = auto_inspection_budget(
            AutoInspectIntent::RepoOverview,
            "openai_compat · gpt",
            false,
        );

        assert!(local.total_chars < cloud.total_chars);
        assert!(local.readme_chars < cloud.readme_chars);
        assert!(local.entrypoint_chars < cloud.entrypoint_chars);
    }

    #[test]
    fn auto_inspection_prefers_code_structure_over_large_readme_excerpt() {
        let plan = AutoInspectPlan {
            intent: AutoInspectIntent::RepoOverview,
            thinking: "Thinking: exploring the repo structure and key project docs.",
            status_label: "inspecting repo...",
            context_label: "this repo summary request",
            query: None,
            steps: vec![],
        };
        let long_readme = "README intro ".repeat(120);

        let hidden = synthesize_auto_inspection_context(
            &plan,
            &[
                ToolResult {
                    tool_name: "list_dir".to_string(),
                    argument: ".".to_string(),
                    output: "Directory: .\n\nsrc/\nREADME.md\nCargo.toml".to_string(),
                },
                ToolResult {
                    tool_name: "list_dir".to_string(),
                    argument: "src".to_string(),
                    output: "Directory: src\n\ncache/\nconfig/\ninference/\nsession/\ntools/\ntui/\nmain.rs".to_string(),
                },
                ToolResult {
                    tool_name: "read_file".to_string(),
                    argument: "README.md".to_string(),
                    output: format!("File: README.md\nLines: 40\n\n```\n{}\n```\n", long_readme),
                },
                ToolResult {
                    tool_name: "list_dir".to_string(),
                    argument: ".".to_string(),
                    output: "Directory: .\n\nsrc/\nREADME.md\nCargo.toml".to_string(),
                },
            ],
            auto_inspection_budget(AutoInspectIntent::RepoOverview, "llama.cpp · qwen", false),
        )
        .expect("hidden context");

        assert!(hidden.contains("Code areas:"));
        assert!(hidden.contains("`cache/`"));
        assert!(hidden.contains("`inference/`"));
        assert!(hidden.chars().count() <= 1000);
    }

    #[test]
    fn workflow_hidden_context_is_compact_and_query_driven() {
        let plan = AutoInspectPlan {
            intent: AutoInspectIntent::WhereIsImplementation,
            thinking: "Thinking: locating the most likely implementation files.",
            status_label: "locating implementation...",
            context_label: "this implementation lookup request",
            query: Some("cache".to_string()),
            steps: vec![],
        };

        let hidden = synthesize_auto_inspection_context(
            &plan,
            &[
                ToolResult {
                    tool_name: "search".to_string(),
                    argument: "cache".to_string(),
                    output: "Search results for 'cache' (3 matches):\n\nsrc/main.rs:\n     4: mod cache;\n    18: cache::warm();\n\nsrc/cache/mod.rs:\n     2: pub fn warm() {}\n".to_string(),
                },
                ToolResult {
                    tool_name: "read_file".to_string(),
                    argument: "src/cache/mod.rs".to_string(),
                    output: "File: src/cache/mod.rs\nLines: 4\n\n```\npub fn warm() {}\npub fn clear() {}\n```\n".to_string(),
                },
            ],
            auto_inspection_budget(
                AutoInspectIntent::WhereIsImplementation,
                "llama.cpp · qwen",
                false,
            ),
        )
        .expect("workflow context");

        assert!(hidden
            .starts_with("Automatic inspection context for this implementation lookup request:"));
        assert!(hidden.contains("Instruction: answer directly from this evidence."));
        assert!(
            hidden.contains("Prefer exact inspected-file evidence over supporting search hits.")
        );
        assert!(hidden.contains("Do not emit tool calls or fenced code blocks."));
        assert!(hidden.contains("Query: cache"));
        assert!(hidden.contains("Likely files:"));
        assert!(hidden.contains("Implementation hints:"));
        assert!(hidden.contains("declarations: 1 `pub fn warm() {}`"));
        assert!(!hidden.contains("Supporting search hits:"));
        assert!(!hidden.contains("Tool results:"));
        assert!(hidden.chars().count() <= 900);
    }

    #[test]
    fn workflow_hidden_context_prefers_inspected_file_hits_over_doc_search_hits() {
        let plan = AutoInspectPlan {
            intent: AutoInspectIntent::WhereIsImplementation,
            thinking: "Thinking: locating the most likely implementation files.",
            status_label: "locating implementation...",
            context_label: "this implementation lookup request",
            query: Some("load_most_recent".to_string()),
            steps: vec![],
        };

        let hidden = synthesize_auto_inspection_context(
            &plan,
            &[
                ToolResult {
                    tool_name: "search".to_string(),
                    argument: "load_most_recent".to_string(),
                    output: "Search results for 'load_most_recent' (4 matches):\n\nsrc/session/mod.rs:\n   272: pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n   844: let loaded = store.load_most_recent().unwrap().unwrap();\n\ndocs/context/PLANS.md:\n  3189: load_most_recent overview\n".to_string(),
                },
                ToolResult {
                    tool_name: "read_file".to_string(),
                    argument: "src/session/mod.rs".to_string(),
                    output: "File: src/session/mod.rs\nLines: 8\n\n```\npub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    let Some(summary) = self.list_sessions()?.into_iter().next() else {\n        return Ok(None);\n    };\n    self.load_session_by_id(&summary.id)\n}\n\npub fn load_session(&self, selector: &str) -> Result<SavedSession> {\n```\n".to_string(),
                },
            ],
            auto_inspection_budget(
                AutoInspectIntent::WhereIsImplementation,
                "llama.cpp · qwen",
                false,
            ),
        )
        .expect("workflow context");

        assert!(hidden.contains("Likely files: `src/session/mod.rs`"));
        assert!(
            hidden.contains("Primary definition: src/session/mod.rs:1 `pub fn load_most_recent")
        );
        assert!(!hidden.contains("Supporting search hits:"));
        assert!(!hidden.contains("src/session/mod.rs:844"));
        assert!(!hidden.contains("docs/context/PLANS.md:3189"));
    }

    #[test]
    fn implementation_summary_prefers_definition_matches_over_use_sites() {
        let summary = summarize_workflow_read(
            "src/session/mod.rs",
            "fn unrelated() {}\n\npub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n\nfn later() {\n    let loaded = store.load_most_recent().unwrap().unwrap();\n}\n",
            "load_most_recent",
            AutoInspectIntent::WhereIsImplementation,
            260,
        )
        .expect("summary");

        assert!(summary.contains("exact lines: 3 `pub fn load_most_recent"));
        assert!(!summary.contains("7 `let loaded = store.load_most_recent().unwrap().unwrap();`"));
    }

    #[test]
    fn primary_definition_location_uses_definition_line() {
        let location = primary_definition_location(
            "src/session/mod.rs",
            "pub use self::session::load_most_recent;\n\npub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
            "load_most_recent",
            72,
        )
        .expect("location");

        assert_eq!(
            location,
            "src/session/mod.rs:3 `pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {`"
        );
    }

    #[test]
    fn implementation_workflows_suppress_general_retrieval_and_instruct_definition_only() {
        assert!(suppress_retrieval_for_auto_inspection(
            AutoInspectIntent::WhereIsImplementation
        ));
        assert!(suppress_retrieval_for_auto_inspection(
            AutoInspectIntent::FeatureTrace
        ));
        assert!(suppress_retrieval_for_auto_inspection(
            AutoInspectIntent::ConfigLocate
        ));
        assert!(!suppress_retrieval_for_auto_inspection(
            AutoInspectIntent::RepoOverview
        ));

        let plan = AutoInspectPlan {
            intent: AutoInspectIntent::WhereIsImplementation,
            thinking: "Thinking: locating the most likely implementation files.",
            status_label: "locating implementation...",
            context_label: "this implementation lookup request",
            query: Some("load_most_recent".to_string()),
            steps: vec![],
        };

        let hidden = synthesize_auto_inspection_context(
            &plan,
            &[ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/session/mod.rs".to_string(),
                output: "File: src/session/mod.rs\nLines: 5\n\n```\npub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n```\n".to_string(),
            }],
            auto_inspection_budget(
                AutoInspectIntent::WhereIsImplementation,
                "llama.cpp · qwen",
                false,
            ),
        )
        .expect("workflow context");

        assert!(hidden.contains("Report definition or implementation locations only"));
        assert!(hidden.contains("omit usage lines"));
    }

    // --- Tests that reproduce the actual runtime failure modes ---

    #[test]
    fn summarize_workflow_read_omits_unrelated_declarations_when_exact_match_found() {
        // Reproduces the "line 28 / line 34 / struct noise" failure: the old
        // code always appended the first-N declaration lines from the file even
        // when an exact definition match was already found. Those small line
        // numbers (top-of-file structs) were cited by the model instead of the
        // correct line.
        let content = "pub struct Other {}\n\npub struct AnotherThing {}\n\n\
                       pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n";
        let summary = summarize_workflow_read(
            "src/session/mod.rs",
            content,
            "load_most_recent",
            AutoInspectIntent::WhereIsImplementation,
            400,
        )
        .expect("summary");

        assert!(summary.contains("exact lines: 5 `pub fn load_most_recent"));
        // The unrelated structs at lines 1 and 3 must NOT appear.
        assert!(
            !summary.contains("declarations:"),
            "declarations section should be omitted when exact match exists"
        );
        assert!(!summary.contains("1 `pub struct Other"));
        assert!(!summary.contains("3 `pub struct AnotherThing"));
    }

    #[test]
    fn primary_definition_location_returns_none_for_file_that_only_calls_the_function() {
        // Reproduces the "line 12 / wrong anchor" failure: the old fallback in
        // primary_definition_location would pick the first fn/struct in the file
        // even when the file never *defines* the queried symbol, producing a
        // completely wrong "Primary definition" anchor.
        let content = "fn call_something() {\n    store.load_most_recent().unwrap();\n}\n\
                       fn other() {\n    let x = load_most_recent();\n}\n";
        let location = primary_definition_location(
            "src/inference/session.rs",
            content,
            "load_most_recent",
            72,
        );
        // Should return None — this file calls but does not define the function.
        assert!(
            location.is_none(),
            "expected None for a file that only calls the function, got: {location:?}"
        );
    }

    #[test]
    fn synthesize_context_deduplicates_search_and_read_paths_when_formats_match() {
        // Reproduces the absolute-vs-relative path mismatch: SearchCode previously
        // produced absolute paths while ReadFile produced relative paths, causing
        // every file to appear twice in Likely files and supporting_hits to be
        // always empty. After the fix, both use relative paths and this test
        // verifies they are deduplicated correctly.
        let plan = AutoInspectPlan {
            intent: AutoInspectIntent::WhereIsImplementation,
            thinking: "t",
            status_label: "s",
            context_label: "this implementation lookup request",
            query: Some("load_most_recent".to_string()),
            steps: vec![],
        };

        // Both the search output and the read output use the same relative path.
        let results = vec![
            ToolResult {
                tool_name: "search".to_string(),
                argument: "load_most_recent".to_string(),
                output: "Search results for 'load_most_recent' (1 match):\n\nsrc/session/mod.rs:\n   272: pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n".to_string(),
            },
            ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/session/mod.rs".to_string(),
                output: "File: src/session/mod.rs\nLines: 3\n\n```\npub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n```\n".to_string(),
            },
        ];

        let hidden = synthesize_auto_inspection_context(
            &plan,
            &results,
            auto_inspection_budget(AutoInspectIntent::WhereIsImplementation, "openai", false),
        )
        .expect("context");

        // The file should appear exactly once in Likely files — no duplicate
        // caused by one absolute and one relative representation.
        let likely_start = hidden.find("Likely files:").expect("Likely files section");
        let likely_end = hidden[likely_start..]
            .find('\n')
            .map(|i| likely_start + i)
            .unwrap_or(hidden.len());
        let likely_line = &hidden[likely_start..likely_end];
        let occurrences = likely_line.matches("src/session/mod.rs").count();
        assert_eq!(
            occurrences, 1,
            "src/session/mod.rs should appear exactly once in Likely files, got: {likely_line}"
        );
    }

    #[test]
    fn primary_definition_location_correct_for_deep_line_number() {
        // Verifies that the correct line number is reported when the target
        // function is deep in the file (e.g. line 272 in the real session store),
        // not confused by unrelated top-of-file structs.
        let mut content = String::new();
        // Pad with 271 blank lines so the function starts at line 272.
        for _ in 0..271 {
            content.push('\n');
        }
        content.push_str("pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n");
        content.push_str("    Ok(None)\n");
        content.push_str("}\n");

        let location =
            primary_definition_location("src/session/mod.rs", &content, "load_most_recent", 72)
                .expect("location");

        assert!(
            location.starts_with("src/session/mod.rs:272 "),
            "expected line 272, got: {location}"
        );
    }

    #[test]
    fn feature_trace_summary_skips_unrelated_declarations_without_matches() {
        let summary = summarize_workflow_read(
            "src/session/mod.rs",
            "pub struct SessionSummary {}\n\npub struct SavedSession {}\n",
            "save_session",
            AutoInspectIntent::FeatureTrace,
            240,
        );

        assert!(summary.is_none());
    }

    #[test]
    fn feature_trace_context_prefers_search_flow_anchors_when_large_file_cannot_be_read() {
        // When the target file is too large to read (e.g., src/inference/session.rs),
        // the workflow falls back to search-only evidence.  The synthesizer must
        // emit the "search anchors only" evidence-quality warning and the
        // anti-fabrication FeatureTrace instruction so the model does not invent
        // code bodies it never read.
        let plan = AutoInspectPlan {
            intent: AutoInspectIntent::FeatureTrace,
            thinking: "Thinking: tracing the main code path for this feature.",
            status_label: "tracing feature...",
            context_label: "this feature trace request",
            query: Some("save_messages".to_string()),
            steps: vec![],
        };

        // Use a non-constrained backend so the evidence warning is not truncated
        // by the tighter llama.cpp budget (900 chars).
        let hidden = synthesize_auto_inspection_context(
            &plan,
            &[ToolResult {
                tool_name: "search".to_string(),
                argument: "save_messages".to_string(),
                output: "Search results for 'save_messages' (4 matches):\n\nsrc/session/mod.rs:\n  224: pub fn save_messages(\n  241: save_messages(\n  260: save_messages(&conn, session_id, messages);\n  3635: assert_eq!(trace_plan.steps[0].argument, \"save_messages\");\n".to_string(),
            }],
            auto_inspection_budget(AutoInspectIntent::FeatureTrace, "openai_compat · gpt-4", false),
        )
        .expect("context");

        assert!(hidden.contains("Primary flow anchors:"));
        assert!(hidden.contains("src/session/mod.rs:224 `pub fn save_messages(`"));
        assert!(hidden.contains("src/session/mod.rs:241 `save_messages(`"));
        // assert_eq! call sites must be filtered out by is_feature_trace_anchor_line
        assert!(!hidden.contains("assert_eq!(trace_plan.steps[0].argument"));
        // Anti-fabrication instruction must be present
        assert!(hidden.contains("Do not invent function bodies, placeholder snippets"));
        // Evidence-quality warning must appear when no file content was read
        assert!(hidden.contains("Evidence: search anchors only"));
    }

    #[test]
    fn feature_trace_context_grounded_when_file_is_readable() {
        // Happy path: the readable src/session/mod.rs contains save_messages,
        // so the synthesizer should produce "Flow hints" from actual file
        // content and NOT emit the "search anchors only" evidence warning.
        let plan = AutoInspectPlan {
            intent: AutoInspectIntent::FeatureTrace,
            thinking: "Thinking: tracing session save.",
            status_label: "tracing feature...",
            context_label: "this feature trace request",
            query: Some("save_messages".to_string()),
            steps: vec![],
        };

        let file_content = "pub fn save_messages(\n    conn: &Connection,\n    session_id: i64,\n    messages: &[Message],\n) -> Result<()> {\n    for msg in messages {\n        conn.execute(INSERT, params![session_id, msg.role, msg.content])?;\n    }\n    Ok(())\n}";

        let hidden = synthesize_auto_inspection_context(
            &plan,
            &[
                ToolResult {
                    tool_name: "search".to_string(),
                    argument: "save_messages".to_string(),
                    output: "Search results for 'save_messages' (2 matches):\n\nsrc/session/mod.rs:\n  224: pub fn save_messages(\n  241: save_messages(&conn, session_id, messages);\n".to_string(),
                },
                ToolResult {
                    tool_name: "read_file".to_string(),
                    argument: "src/session/mod.rs".to_string(),
                    output: format!("File: src/session/mod.rs\nLines: 10\n\n```\n{file_content}\n```"),
                },
            ],
            auto_inspection_budget(AutoInspectIntent::FeatureTrace, "llama.cpp · qwen", false),
        )
        .expect("context");

        // File content was read, so flow hints must appear
        assert!(hidden.contains("Flow hints"));
        assert!(hidden.contains("src/session/mod.rs"));
        // The "search anchors only" warning must NOT appear — we have real evidence
        assert!(!hidden.contains("Evidence: search anchors only"));
        // Anti-fabrication instruction must still be present
        assert!(hidden.contains("Do not invent function bodies, placeholder snippets"));
    }

    #[test]
    fn choose_followup_read_steps_skips_oversized_auto_inspection_files() {
        let root = temp_project_root("workflow-large-skip");
        fs::create_dir_all(root.join("src")).expect("create src");
        fs::write(root.join("src/small.rs"), "fn helper() {}\n").expect("write small file");
        let oversized = "a".repeat(100_500);
        fs::write(root.join("src/large.rs"), oversized).expect("write large file");

        let plan = AutoInspectPlan {
            intent: AutoInspectIntent::FeatureTrace,
            thinking: "Thinking: tracing the main code path for this feature.",
            status_label: "tracing feature...",
            context_label: "this feature trace request",
            query: Some("save_session".to_string()),
            steps: vec![],
        };

        let hits = vec![
            SearchFileHit {
                path: "src/large.rs".to_string(),
                hits: vec![SearchLineHit {
                    line_number: 1,
                    line_content: "fn save_session(".to_string(),
                }],
            },
            SearchFileHit {
                path: "src/small.rs".to_string(),
                hits: vec![SearchLineHit {
                    line_number: 1,
                    line_content: "fn save_session(".to_string(),
                }],
            },
        ];

        let steps = choose_followup_read_steps(
            &plan,
            &root,
            &hits,
            auto_inspection_budget(AutoInspectIntent::FeatureTrace, "llama.cpp · qwen", false),
        );

        assert!(steps.iter().all(|step| step.argument != "src/large.rs"));
        assert!(steps.iter().any(|step| step.argument == "src/small.rs"));

        let _ = fs::remove_dir_all(root);
    }
}
