use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::cache::ExactCache;
use crate::config;
use crate::error::{ParamsError, Result};
use crate::events::{InferenceEvent, ProgressStatus};
use crate::tools::{ReadOnlyToolExecution, ToolRegistry, ToolResult};

use super::budget::{
    estimate_message_tokens, record_generation_budget, SessionBudget, SessionCacheStats,
};
use super::cache::{generate_with_cache, CacheMode};
use super::reflection::reflect_response;
use super::runtime::{eco_tool_result_limit, emit_generation_started, emit_trace};
use super::{system_prompt_with_tools, InferenceBackend, Message};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ToolLoopIntent {
    RepoOverview,
    DirectoryOverview,
    CodeNavigation,
    ConfigLocate,
}

#[derive(Clone, Copy)]
struct ToolLoopBudget {
    max_iterations: usize,
    max_duplicate_calls: usize,
}

pub(super) struct ToolLoopOutcome {
    pub final_response: String,
    pub tool_results: Vec<ToolResult>,
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

fn token_specificity_score(token: &str) -> usize {
    let len = token.len();
    let alpha_bonus = if token.chars().all(|ch| ch.is_ascii_alphabetic()) {
        1
    } else {
        0
    };
    let suffix_bonus = if token.ends_with("ing")
        || token.ends_with("tion")
        || token.ends_with("ment")
        || token.ends_with("al")
    {
        2
    } else {
        0
    };

    len + alpha_bonus + suffix_bonus
}

fn salient_search_token(phrase: &str, intent: ToolLoopIntent) -> Option<String> {
    let stopwords = match intent {
        ToolLoopIntent::CodeNavigation => &[
            "implemented",
            "implement",
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
            "project",
            "trace",
            "how",
            "does",
            "work",
            "works",
            "flow",
            "what",
            "writes",
            "to",
            "are",
            "saved",
            "save",
            "restored",
            "restore",
        ][..],
        ToolLoopIntent::ConfigLocate => &[
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
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview => &[][..],
    };

    phrase
        .split_whitespace()
        .map(trim_query_noise)
        .map(|token| singularize_token(&token))
        .filter(|token| {
            !token.is_empty()
                && !stopwords.iter().any(|stop| stop == token)
                && token.chars().any(|ch| ch.is_ascii_alphanumeric())
        })
        .max_by(|a, b| {
            token_specificity_score(a)
                .cmp(&token_specificity_score(b))
                .then_with(|| a.len().cmp(&b.len()))
        })
}

fn suggested_search_query(prompt: &str, intent: ToolLoopIntent) -> Option<String> {
    let normalized = normalize_intent_text(prompt);
    if normalized.contains("session") {
        if normalized.contains("save") || normalized.contains("saved") {
            return Some("save_messages".to_string());
        }
        if normalized.contains("restore")
            || normalized.contains("restored")
            || normalized.contains("resume")
        {
            return Some("load_most_recent".to_string());
        }
    }
    if intent == ToolLoopIntent::ConfigLocate && normalized.contains("eco") {
        return Some("eco.enabled".to_string());
    }

    let extracted = match intent {
        ToolLoopIntent::CodeNavigation => {
            if let Some(rest) = normalized.strip_prefix("where is ") {
                trim_query_suffix(
                    rest,
                    &[
                        " implemented",
                        " defined",
                        " handled",
                        " configured",
                        " configged",
                    ],
                )
                .to_string()
            } else if let Some(rest) = normalized.strip_prefix("find ") {
                rest.trim().to_string()
            } else if let Some(rest) = normalized.strip_prefix("which file has ") {
                rest.trim().to_string()
            } else if let Some(rest) = normalized.strip_prefix("what handles ") {
                rest.trim().to_string()
            } else if let Some(rest) = normalized.strip_prefix("what writes to ") {
                rest.trim().to_string()
            } else if let Some(rest) = normalized.strip_prefix("trace how ") {
                trim_query_suffix(rest, &[" works", " work", " flows", " flow"]).to_string()
            } else if let Some(rest) = normalized.strip_prefix("how does ") {
                trim_query_suffix(rest, &[" work", " works", " flow", " flow through"]).to_string()
            } else {
                String::new()
            }
        }
        ToolLoopIntent::ConfigLocate => {
            if let Some(rest) = normalized.strip_prefix("where is ") {
                trim_query_suffix(rest, &[" configured", " configged", " set"]).to_string()
            } else if let Some(rest) = normalized.strip_prefix("which file configures ") {
                rest.trim().to_string()
            } else {
                String::new()
            }
        }
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview => String::new(),
    };

    let cleaned =
        salient_search_token(&extracted, intent).unwrap_or_else(|| trim_query_noise(&extracted));
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

pub(super) fn detect_tool_loop_intent(prompt: &str) -> Option<ToolLoopIntent> {
    let normalized = normalize_intent_text(prompt);
    if normalized.starts_with('/') {
        return None;
    }

    let tokens = normalized
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let first_contains = |needle: &str| {
        tokens
            .first()
            .map(|token| token.contains(needle))
            .unwrap_or(false)
    };
    let second_is = |value: &str| tokens.get(1).map(|token| token == value).unwrap_or(false);
    let has_token = |value: &str| tokens.iter().any(|token| token == value);
    let has_prefix = |prefix: &str| tokens.iter().any(|token| token.starts_with(prefix));

    if ((first_contains("what") && second_is("is"))
        || (first_contains("whats") && (second_is("in") || second_is("here")))
        || (first_contains("summarize") && second_is("this")))
        && (has_token("repo")
            || has_token("project")
            || has_token("codebase")
            || has_token("directory")
            || has_token("folder")
            || has_token("here"))
    {
        if has_token("directory") || has_token("folder") || has_token("here") {
            return Some(ToolLoopIntent::DirectoryOverview);
        }
        return Some(ToolLoopIntent::RepoOverview);
    }

    if (first_contains("where") && second_is("is"))
        || normalized.starts_with("find ")
        || normalized.starts_with("which file has ")
        || normalized.starts_with("what handles ")
        || normalized.starts_with("what writes to ")
        || normalized.starts_with("trace how ")
        || normalized.starts_with("how does ")
    {
        if has_prefix("config") || has_token("set") {
            return Some(ToolLoopIntent::ConfigLocate);
        }
        if has_prefix("implement")
            || has_prefix("defin")
            || has_prefix("handl")
            || normalized.starts_with("trace how ")
            || normalized.starts_with("how does ")
            || normalized.starts_with("what handles ")
            || normalized.starts_with("what writes to ")
            || normalized.starts_with("find ")
            || normalized.starts_with("which file has ")
        {
            return Some(ToolLoopIntent::CodeNavigation);
        }
    }

    None
}

fn tool_loop_budget(eco_enabled: bool) -> ToolLoopBudget {
    if eco_enabled {
        ToolLoopBudget {
            max_iterations: 3,
            max_duplicate_calls: 1,
        }
    } else {
        ToolLoopBudget {
            max_iterations: 5,
            max_duplicate_calls: 2,
        }
    }
}

fn tool_loop_result_limit(backend_name: &str, eco_enabled: bool) -> Option<usize> {
    if backend_name.contains("llama.cpp") {
        if eco_enabled {
            Some(900)
        } else {
            Some(1800)
        }
    } else {
        eco_tool_result_limit(eco_enabled)
    }
}

fn with_progress_heartbeat<T, F>(
    token_tx: &Sender<InferenceEvent>,
    label: &str,
    run: F,
) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    with_progress_heartbeat_interval(token_tx, label, Duration::from_secs(3), run)
}

fn with_progress_heartbeat_interval<T, F>(
    token_tx: &Sender<InferenceEvent>,
    label: &str,
    interval: Duration,
    run: F,
) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();
    let tx = token_tx.clone();
    let label = label.to_string();

    let heartbeat = thread::spawn(move || {
        let mut elapsed = 0u64;
        loop {
            thread::sleep(interval);
            if stop_clone.load(Ordering::Relaxed) {
                break;
            }
            elapsed += interval.as_secs().max(1);
            emit_trace(
                &tx,
                ProgressStatus::Updated,
                &format!("{label} ({elapsed}s elapsed)"),
                false,
            );
        }
    });

    let result = run();
    stop.store(true, Ordering::Relaxed);
    let _ = heartbeat.join();
    result
}

fn thinking_label(intent: ToolLoopIntent) -> (&'static str, &'static str) {
    match intent {
        ToolLoopIntent::RepoOverview => (
            "Thinking: exploring the repo structure and key files.",
            "exploring repo...",
        ),
        ToolLoopIntent::DirectoryOverview => (
            "Thinking: checking the current directory and its key files.",
            "exploring directory...",
        ),
        ToolLoopIntent::CodeNavigation => (
            "Thinking: searching the repo and reading candidate files.",
            "investigating code...",
        ),
        ToolLoopIntent::ConfigLocate => (
            "Thinking: searching the repo and reading candidate config files.",
            "investigating config...",
        ),
    }
}

fn repo_context_paths(project_root: &Path) -> Vec<String> {
    [
        "README.md",
        "docs/context/CLAUDE.md",
        "AGENTS.md",
        "SKILLS.md",
    ]
    .iter()
    .filter(|path| project_root.join(path).is_file())
    .map(|path| (*path).to_string())
    .collect()
}

fn repo_context_summary(project_root: &Path) -> Option<String> {
    let paths = repo_context_paths(project_root);
    if paths.is_empty() {
        return None;
    }

    Some(format!(
        "Repo-local context files are available if useful: {}. Treat them as support context, not a substitute for live inspection.",
        paths.iter()
            .map(|path| format!("`{path}`"))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

fn build_tool_loop_system_prompt(
    tools: &ToolRegistry,
    project_root: &Path,
    eco_enabled: bool,
) -> String {
    let mut prompt = system_prompt_with_tools(&tools.read_only_tool_descriptions());
    prompt.push_str(
        "\n\nYou are in repo-navigation mode.\n\
         Work like a careful developer using live repo tools.\n\
         Search before guessing when a symbol or feature location is unclear.\n\
         Do not narrate intended tool use in prose. When you want to inspect something, emit the actual tool tag(s).\n\
         Read candidate files, then expand outward only if the evidence points elsewhere.\n\
         Prefer source files over docs when answering implementation questions.\n\
         Answer with concrete file paths and line references when available.\n\
         If the current evidence is insufficient, say so briefly instead of inventing details.",
    );
    if eco_enabled {
        prompt.push_str(
            "\nKeep the investigation compact: prefer the fewest tool calls that answer the question.",
        );
    }
    if let Some(summary) = repo_context_summary(project_root) {
        prompt.push_str("\n\n");
        prompt.push_str(&summary);
    }
    prompt
}

fn initial_tool_only_followup(intent: ToolLoopIntent, prompt: &str) -> String {
    let search_hint = suggested_search_query(prompt, intent);
    let starting_hint = match intent {
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview => {
            "Start by listing the current directory with `[list_dir: .]`."
        }
        ToolLoopIntent::CodeNavigation | ToolLoopIntent::ConfigLocate => match search_hint {
            Some(ref query) => {
                return format!(
                    "Repo-navigation mode requires live inspection before answering. \
                     Your next response must contain only one or more read-only tool tags and no prose. \
                     Start with `[search: {query}]`. Prefer the concrete symbol or setting name over a broad natural-language phrase."
                );
            }
            None => {
                "Start by searching for the most relevant symbol, term, or setting with `[search: ...]`."
            }
        },
    };

    format!(
        "Repo-navigation mode requires live inspection before answering. \
         Your next response must contain only one or more read-only tool tags and no prose. \
         {starting_hint}"
    )
}

fn initial_investigation_hint(intent: ToolLoopIntent, prompt: &str) -> Option<String> {
    let query = suggested_search_query(prompt, intent)?;
    let instruction = match intent {
        ToolLoopIntent::CodeNavigation => {
            "Prefer the concrete symbol over the full English question. After searching, read the strongest source-file candidate before answering. Ignore docs, tests, prompt strings, and use-sites unless the repo has no better source hits."
        }
        ToolLoopIntent::ConfigLocate => {
            "Prefer the concrete setting key over the full English question. After searching, read the strongest config or source file before answering."
        }
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview => return None,
    };

    Some(format!(
        "Investigation hint: the most promising search target for this request is `{query}`. \
         Start with `[search: {query}]`. {instruction}"
    ))
}

fn is_referential_follow_up(prompt: &str) -> bool {
    let normalized = normalize_intent_text(prompt);
    let referential_tokens = [
        "it", "its", "that", "this", "these", "those", "they", "them", "there", "same", "above",
        "previous", "former", "latter",
    ];

    normalized
        .split_whitespace()
        .any(|token| referential_tokens.contains(&token))
}

fn build_tool_loop_seed_messages(
    base_messages: &[Message],
    system_prompt: &str,
    prompt: &str,
) -> Vec<Message> {
    let mut messages = vec![Message::system(system_prompt)];

    let mut tail = if is_referential_follow_up(prompt) {
        base_messages
            .iter()
            .filter(|message| message.role != "system")
            .rev()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
    } else {
        base_messages
            .iter()
            .rev()
            .find(|message| message.role == "user")
            .cloned()
            .into_iter()
            .collect::<Vec<_>>()
    };
    tail.reverse();

    let already_has_prompt = tail
        .last()
        .map(|message| message.role == "user" && message.content == prompt)
        .unwrap_or(false);
    if !already_has_prompt {
        tail.push(Message::user(prompt));
    }

    messages.extend(tail);
    messages
}

fn repeated_tool_calls(
    counts: &mut HashMap<String, usize>,
    results: &[ToolResult],
    max_duplicate_calls: usize,
) -> bool {
    let mut repeated = false;
    for result in results {
        let key = format!("{}:{}", result.tool_name, result.argument);
        let count = counts.entry(key).or_insert(0);
        *count += 1;
        if *count > max_duplicate_calls {
            repeated = true;
        }
    }
    repeated
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

fn is_doc_path(path: &str) -> bool {
    path.ends_with(".md") || path.starts_with("docs/")
}

fn is_test_like_path(path: &str) -> bool {
    path.starts_with("tests/")
        || path.contains("/tests/")
        || path.contains("fixtures")
        || path.contains("snapshots")
        || path.ends_with("_test.rs")
        || path.ends_with("_tests.rs")
}

fn is_source_path(path: &str) -> bool {
    path.starts_with("src/")
        || path.ends_with(".rs")
        || path.ends_with(".py")
        || path.ends_with(".ts")
        || path.ends_with(".tsx")
        || path.ends_with(".js")
        || path.ends_with(".jsx")
        || path.ends_with(".go")
        || path.ends_with(".java")
        || path.ends_with(".kt")
        || path.ends_with(".swift")
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
        || trimmed.starts_with("def ")
        || trimmed.starts_with("class ")
        || trimmed.starts_with("interface ")
}

fn merge_search_hits(results: &[ToolResult]) -> Vec<SearchFileHit> {
    let mut by_path = HashMap::<String, Vec<SearchLineHit>>::new();
    for result in results {
        if result.tool_name != "search" {
            continue;
        }
        for file in parse_search_output(&result.output) {
            by_path.entry(file.path).or_default().extend(file.hits);
        }
    }

    let mut files = by_path
        .into_iter()
        .map(|(path, mut hits)| {
            hits.sort_by_key(|hit| hit.line_number);
            hits.dedup_by(|a, b| {
                a.line_number == b.line_number && a.line_content == b.line_content
            });
            SearchFileHit { path, hits }
        })
        .collect::<Vec<_>>();
    files.sort_by(|a, b| a.path.cmp(&b.path));
    files
}

fn score_search_candidate(intent: ToolLoopIntent, query: &str, file: &SearchFileHit) -> isize {
    let query = query.trim().to_ascii_lowercase();
    let path = file.path.to_ascii_lowercase();
    let mut score = 0isize;

    if is_source_path(&file.path) {
        score += 28;
    }
    if file.path.starts_with("src/") {
        score += 16;
    }
    if matches!(intent, ToolLoopIntent::ConfigLocate) && is_config_path(&file.path) {
        score += 24;
    }
    if is_doc_path(&file.path) {
        score -= 18;
    }
    if is_test_like_path(&file.path) {
        score -= 28;
    }
    if path.contains("prompt") || path.contains("fixture") {
        score -= 14;
    }
    if !query.is_empty() && path.contains(&query) {
        score += 10;
    }

    score += (file.hits.len().min(6) as isize) * 3;

    for hit in file.hits.iter().take(4) {
        let line = hit.line_content.trim();
        let line_lower = line.to_ascii_lowercase();
        if !query.is_empty() && line_lower.contains(&query) {
            score += 6;
        }
        if is_definition_like_line(line) {
            score += 24;
        }
        if line.contains("assert!")
            || line.contains("#[test]")
            || line.contains("mod tests")
            || line.contains("Search results for")
        {
            score -= 10;
        }
        if line.contains('"') && !is_definition_like_line(line) {
            score -= 4;
        }
    }

    score
}

fn preferred_candidate_path(intent: ToolLoopIntent, path: &str) -> bool {
    match intent {
        ToolLoopIntent::CodeNavigation => {
            is_source_path(path) && !is_doc_path(path) && !is_test_like_path(path)
        }
        ToolLoopIntent::ConfigLocate => {
            (is_config_path(path) || is_source_path(path))
                && !is_doc_path(path)
                && !is_test_like_path(path)
        }
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview => true,
    }
}

fn ranked_search_candidates(
    intent: ToolLoopIntent,
    prompt: &str,
    results: &[ToolResult],
) -> Vec<SearchFileHit> {
    let query =
        suggested_search_query(prompt, intent).unwrap_or_else(|| normalize_intent_text(prompt));
    let mut ranked = merge_search_hits(results);
    ranked.sort_by(|a, b| {
        preferred_candidate_path(intent, &b.path)
            .cmp(&preferred_candidate_path(intent, &a.path))
            .then_with(|| {
                score_search_candidate(intent, &query, b)
                    .cmp(&score_search_candidate(intent, &query, a))
            })
            .then_with(|| a.path.cmp(&b.path))
    });
    ranked
}

fn observed_read_paths(results: &[ToolResult]) -> HashSet<String> {
    results
        .iter()
        .filter(|result| result.tool_name == "read_file")
        .map(|result| result.argument.clone())
        .collect()
}

fn has_relevant_file_evidence(
    intent: ToolLoopIntent,
    prompt: &str,
    results: &[ToolResult],
) -> bool {
    match intent {
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview => {
            results.iter().any(|result| {
                matches!(
                    result.tool_name.as_str(),
                    "list_dir" | "read_file" | "lsp_definition" | "lsp_hover"
                )
            })
        }
        ToolLoopIntent::CodeNavigation | ToolLoopIntent::ConfigLocate => {
            if results
                .iter()
                .any(|result| matches!(result.tool_name.as_str(), "lsp_definition" | "lsp_hover"))
            {
                return true;
            }

            let read_paths = observed_read_paths(results);
            if read_paths.is_empty() {
                return false;
            }
            if read_paths
                .iter()
                .any(|path| preferred_candidate_path(intent, path))
            {
                return true;
            }

            let ranked = ranked_search_candidates(intent, prompt, results);
            let has_better_unread = ranked
                .iter()
                .any(|file| preferred_candidate_path(intent, &file.path));
            !has_better_unread
        }
    }
}

fn targeted_investigation_followup(
    intent: ToolLoopIntent,
    prompt: &str,
    results: &[ToolResult],
) -> Option<String> {
    let read_paths = observed_read_paths(results);
    let candidate = ranked_search_candidates(intent, prompt, results)
        .into_iter()
        .find(|file| !read_paths.contains(&file.path))?;
    let anchor = candidate
        .hits
        .iter()
        .find(|hit| is_definition_like_line(&hit.line_content))
        .or_else(|| candidate.hits.first());
    let anchor_text = anchor.map(|hit| format!("{}: {}", hit.line_number, hit.line_content));

    let body = match intent {
        ToolLoopIntent::CodeNavigation => {
            "Do not answer yet. Read this source candidate next and answer from the inspected implementation, not from docs, tests, prompt strings, or call-sites."
        }
        ToolLoopIntent::ConfigLocate => {
            "Do not answer yet. Read this config/source candidate next and answer from the inspected setting lines."
        }
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview => return None,
    };

    Some(match anchor_text {
        Some(anchor) => format!(
            "{body} Next read: `[read_file: {}]`. Strongest search anchor: `{}`.",
            candidate.path, anchor
        ),
        None => format!("{body} Next read: `[read_file: {}]`.", candidate.path),
    })
}

pub(super) fn run_read_only_tool_loop(
    intent: ToolLoopIntent,
    prompt: &str,
    base_messages: &[Message],
    backend: &dyn InferenceBackend,
    tools: &ToolRegistry,
    cfg: &config::Config,
    project_root: &Path,
    token_tx: &Sender<InferenceEvent>,
    exact_cache: Option<&ExactCache>,
    cache_stats: &mut SessionCacheStats,
    budget: &mut SessionBudget,
    eco_enabled: bool,
    reflection_enabled: bool,
) -> Result<ToolLoopOutcome> {
    let loop_budget = tool_loop_budget(eco_enabled);
    let result_limit = tool_loop_result_limit(&backend.name(), eco_enabled);
    let (thinking, status_label) = thinking_label(intent);
    let _ = token_tx.send(InferenceEvent::SystemMessage(thinking.to_string()));
    emit_generation_started(token_tx, status_label, false);

    let system_prompt = build_tool_loop_system_prompt(tools, project_root, eco_enabled);
    let mut loop_messages = build_tool_loop_seed_messages(base_messages, &system_prompt, prompt);
    if let Some(hint) = initial_investigation_hint(intent, prompt) {
        loop_messages.push(Message::user(&hint));
    }
    let mut all_tool_results = Vec::new();
    let mut tool_call_counts = HashMap::new();

    for iteration in 0..loop_budget.max_iterations {
        emit_trace(
            token_tx,
            ProgressStatus::Started,
            if iteration == 0 {
                "planning investigation..."
            } else {
                "continuing investigation..."
            },
            false,
        );
        let prompt_tokens = estimate_message_tokens(&loop_messages);
        let heartbeat_label = if iteration == 0 {
            "planning investigation..."
        } else {
            "continuing investigation..."
        };
        let draft = with_progress_heartbeat(token_tx, heartbeat_label, || {
            generate_with_cache(
                backend,
                &loop_messages,
                cfg,
                project_root,
                token_tx.clone(),
                false,
                exact_cache,
                cache_stats,
                CacheMode::PreferPromptLevel,
            )
        })?;

        if !draft.hit {
            record_generation_budget(cfg, budget, token_tx, prompt_tokens, &draft.text);
        }

        let ReadOnlyToolExecution {
            results,
            disallowed_calls,
        } = tools.execute_read_only_tool_calls(&draft.text);

        if results.is_empty() && disallowed_calls.is_empty() {
            if all_tool_results.is_empty() {
                emit_trace(
                    token_tx,
                    ProgressStatus::Updated,
                    "model replied without using a tool; requesting an actual tool call",
                    false,
                );
                loop_messages.push(Message::assistant(&draft.text));
                loop_messages.push(Message::user(&initial_tool_only_followup(intent, prompt)));
                continue;
            }
            if !has_relevant_file_evidence(intent, prompt, &all_tool_results) {
                emit_trace(
                    token_tx,
                    ProgressStatus::Updated,
                    "tool loop needs file-level evidence before answering",
                    false,
                );
                loop_messages.push(Message::assistant(&draft.text));
                let followup = targeted_investigation_followup(intent, prompt, &all_tool_results)
                    .unwrap_or_else(|| {
                        "You do not have enough file-level evidence yet. \
                         Do not answer from search hits alone. \
                         Read the most relevant candidate file or use an LSP read-only tool on the best location, then answer from that evidence."
                            .to_string()
                    });
                loop_messages.push(Message::user(&followup));
                continue;
            }
            let final_response = if reflection_enabled {
                emit_trace(
                    token_tx,
                    ProgressStatus::Updated,
                    "reflecting final answer...",
                    false,
                );
                reflect_response(
                    backend,
                    cfg,
                    project_root,
                    budget,
                    token_tx,
                    exact_cache,
                    cache_stats,
                    &loop_messages,
                    &draft.text,
                )?
            } else {
                draft.text
            };
            return Ok(ToolLoopOutcome {
                final_response,
                tool_results: all_tool_results,
            });
        }

        if !disallowed_calls.is_empty() {
            emit_trace(
                token_tx,
                ProgressStatus::Updated,
                "read-only tool loop rejected mutating tool request",
                false,
            );
            loop_messages.push(Message::assistant(&draft.text));
            loop_messages.push(Message::user(&format!(
                "Read-only repo navigation mode only allows read-only tools. \
                 Do not call {}. Continue by using read-only tools or answer from current evidence.",
                disallowed_calls
                    .iter()
                    .map(|name| format!("`{name}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
            continue;
        }

        for result in &results {
            emit_trace(
                token_tx,
                ProgressStatus::Finished,
                &format!("{} {}", result.tool_name, result.argument),
                false,
            );
        }

        let repeated = repeated_tool_calls(
            &mut tool_call_counts,
            &results,
            loop_budget.max_duplicate_calls,
        );
        all_tool_results.extend(results.clone());

        loop_messages.push(Message::assistant(&draft.text));
        let result_message = ToolRegistry::format_results_with_limit(&results, result_limit)
            .unwrap_or_else(|| "Tool results:\n".to_string());
        loop_messages.push(Message::user(&result_message));

        if repeated {
            loop_messages.push(Message::user(
                "You are repeating the same tool calls. Answer from the gathered evidence or explain briefly what is still missing.",
            ));
        }
        if let Some(followup) = targeted_investigation_followup(intent, prompt, &all_tool_results) {
            loop_messages.push(Message::user(&followup));
        } else if !repeated {
            loop_messages.push(Message::user(
                "Continue investigating only if you still need more evidence. Otherwise answer now using the observed file and line evidence.",
            ));
        }
    }

    emit_trace(
        token_tx,
        ProgressStatus::Updated,
        "tool loop hit its iteration limit; answering from gathered evidence...",
        false,
    );
    let prompt_tokens = estimate_message_tokens(&loop_messages);
    let final_draft =
        with_progress_heartbeat(token_tx, "answering from gathered evidence...", || {
            generate_with_cache(
                backend,
                &loop_messages,
                cfg,
                project_root,
                token_tx.clone(),
                false,
                exact_cache,
                cache_stats,
                CacheMode::PreferPromptLevel,
            )
        })?;
    if !final_draft.hit {
        record_generation_budget(cfg, budget, token_tx, prompt_tokens, &final_draft.text);
    }
    let final_response = if reflection_enabled {
        reflect_response(
            backend,
            cfg,
            project_root,
            budget,
            token_tx,
            exact_cache,
            cache_stats,
            &loop_messages,
            &final_draft.text,
        )?
    } else {
        final_draft.text
    };

    if final_response.trim().is_empty() {
        return Err(ParamsError::Inference(
            "Tool loop returned an empty final response".to_string(),
        ));
    }

    Ok(ToolLoopOutcome {
        final_response,
        tool_results: all_tool_results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::mpsc;
    use std::time::Duration;

    use crate::events::InferenceEvent;

    struct ScriptedBackend {
        responses: std::sync::Mutex<Vec<String>>,
    }

    impl ScriptedBackend {
        fn new(responses: Vec<&str>) -> Self {
            Self {
                responses: std::sync::Mutex::new(
                    responses.into_iter().rev().map(str::to_string).collect(),
                ),
            }
        }
    }

    impl InferenceBackend for ScriptedBackend {
        fn generate(&self, _messages: &[Message], tx: Sender<InferenceEvent>) -> Result<()> {
            let next = self
                .responses
                .lock()
                .expect("responses lock")
                .pop()
                .unwrap_or_default();
            let _ = tx.send(InferenceEvent::Token(next));
            Ok(())
        }

        fn name(&self) -> String {
            "scripted".to_string()
        }
    }

    struct InspectingBackend {
        steps: std::sync::Mutex<Vec<(Option<String>, String)>>,
    }

    impl InspectingBackend {
        fn new(steps: Vec<(Option<&str>, &str)>) -> Self {
            Self {
                steps: std::sync::Mutex::new(
                    steps
                        .into_iter()
                        .rev()
                        .map(|(expected, response)| {
                            (expected.map(str::to_string), response.to_string())
                        })
                        .collect(),
                ),
            }
        }
    }

    impl InferenceBackend for InspectingBackend {
        fn generate(&self, messages: &[Message], tx: Sender<InferenceEvent>) -> Result<()> {
            let (expected, response) = self
                .steps
                .lock()
                .expect("steps lock")
                .pop()
                .unwrap_or((None, String::new()));
            if let Some(expected) = expected {
                assert!(
                    messages
                        .iter()
                        .any(|message| message.content.contains(&expected)),
                    "expected tool-loop messages to contain `{expected}`, got:\n{}",
                    messages
                        .iter()
                        .map(|message| format!("{}: {}", message.role, message.content))
                        .collect::<Vec<_>>()
                        .join("\n\n")
                );
            }
            let _ = tx.send(InferenceEvent::Token(response));
            Ok(())
        }

        fn name(&self) -> String {
            "inspecting".to_string()
        }
    }

    #[test]
    fn detect_tool_loop_intent_handles_typoed_where_prompt() {
        assert_eq!(
            detect_tool_loop_intent("WWhere is memory retrieval implemented?"),
            Some(ToolLoopIntent::CodeNavigation)
        );
        assert_eq!(
            detect_tool_loop_intent("Where is eco mode configged"),
            Some(ToolLoopIntent::ConfigLocate)
        );
        assert_eq!(
            suggested_search_query(
                "Where is session restore implemented?",
                ToolLoopIntent::CodeNavigation
            )
            .as_deref(),
            Some("load_most_recent")
        );
    }

    #[test]
    fn tool_loop_system_prompt_uses_read_only_tools() {
        let dir = std::env::temp_dir().join("params-tool-loop-prompt");
        let _ = std::fs::create_dir_all(dir.join("docs/context"));
        let _ = std::fs::write(dir.join("README.md"), "# params-cli");
        let _ = std::fs::write(dir.join("docs/context/CLAUDE.md"), "# CLAUDE");

        let prompt = build_tool_loop_system_prompt(&ToolRegistry::default(), &dir, false);
        assert!(prompt.contains("read-only repo inspection tools"));
        assert!(prompt.contains("read_file"));
        assert!(prompt.contains("search"));
        assert!(!prompt.contains("write_file"));
        assert!(prompt.contains("README.md"));
        assert!(prompt.contains("docs/context/CLAUDE.md"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn llama_tool_loop_always_uses_bounded_tool_result_context() {
        assert_eq!(
            tool_loop_result_limit("llama.cpp · qwen", false),
            Some(1800)
        );
        assert_eq!(tool_loop_result_limit("llama.cpp · qwen", true), Some(900));
        assert_eq!(tool_loop_result_limit("openai_compat · gpt-5", false), None);
    }

    #[test]
    fn heartbeat_emits_progress_updates_during_long_internal_generation() {
        let (tx, rx) = mpsc::channel();
        let result = with_progress_heartbeat_interval(
            &tx,
            "planning investigation...",
            Duration::from_millis(25),
            || {
                std::thread::sleep(Duration::from_millis(70));
                Ok::<_, crate::error::ParamsError>("done")
            },
        )
        .expect("heartbeat wrapper should return inner result");

        assert_eq!(result, "done");
        let traces = rx
            .try_iter()
            .filter_map(|event| match event {
                InferenceEvent::Trace(trace) => Some(trace.label),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(
            traces
                .iter()
                .any(|label| label.contains("planning investigation...")),
            "expected heartbeat trace updates, got {traces:?}"
        );
    }

    #[test]
    fn tool_loop_seed_messages_drop_old_history_for_standalone_query() {
        let seed = build_tool_loop_seed_messages(
            &[
                Message::system("old system"),
                Message::user(&"previous context ".repeat(800)),
                Message::assistant(&"previous answer ".repeat(800)),
                Message::user("Where is session restore implemented?"),
            ],
            "tool loop system",
            "Where is session restore implemented?",
        );

        assert_eq!(seed.len(), 2);
        assert_eq!(seed[0].role, "system");
        assert_eq!(seed[0].content, "tool loop system");
        assert_eq!(seed[1].role, "user");
        assert_eq!(seed[1].content, "Where is session restore implemented?");
    }

    #[test]
    fn tool_loop_seed_messages_keep_short_context_for_referential_follow_up() {
        let seed = build_tool_loop_seed_messages(
            &[
                Message::system("old system"),
                Message::user("Where is session restore implemented?"),
                Message::assistant("It looks related to session persistence."),
                Message::user("What calls it?"),
            ],
            "tool loop system",
            "What calls it?",
        );

        assert_eq!(seed.len(), 4);
        assert_eq!(seed[1].content, "Where is session restore implemented?");
        assert_eq!(seed[2].content, "It looks related to session persistence.");
        assert_eq!(seed[3].content, "What calls it?");
    }

    #[test]
    fn read_only_tool_loop_bootstraps_with_shaped_search_target() {
        let dir = std::env::temp_dir().join("params-tool-loop-bootstrap-hint");
        let _ = fs::create_dir_all(dir.join("src/session"));
        let _ = fs::write(
            dir.join("src/session/mod.rs"),
            "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
        );

        let backend = InspectingBackend::new(vec![
            (
                Some("Start with `[search: load_most_recent]`"),
                "[search: load_most_recent]",
            ),
            (
                Some("Next read: `[read_file: src/session/mod.rs]`"),
                "[read_file: src/session/mod.rs]",
            ),
            (
                None,
                "The implementation is in `src/session/mod.rs` at line 1.",
            ),
        ]);
        let (tx, _rx) = mpsc::channel();
        let mut cache_stats = SessionCacheStats::default();
        let mut budget = SessionBudget::default();
        let outcome = run_read_only_tool_loop(
            ToolLoopIntent::CodeNavigation,
            "Where is session restore implemented?",
            &[
                Message::system("system"),
                Message::user("Where is session restore implemented?"),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
        .expect("tool loop");

        assert_eq!(
            outcome.final_response,
            "The implementation is in `src/session/mod.rs` at line 1."
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn read_only_tool_loop_runs_search_then_answers() {
        let dir = std::env::temp_dir().join("params-tool-loop-run");
        let _ = std::fs::create_dir_all(dir.join("src/session"));
        let _ = std::fs::write(
            dir.join("src/session/mod.rs"),
            "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
        );

        let backend = ScriptedBackend::new(vec![
            "[search: load_most_recent]",
            "[read_file: src/session/mod.rs]",
            "The implementation is in `src/session/mod.rs` at line 1.",
        ]);
        let (tx, _rx) = mpsc::channel();
        let mut cache_stats = SessionCacheStats::default();
        let mut budget = SessionBudget::default();
        let outcome = run_read_only_tool_loop(
            ToolLoopIntent::CodeNavigation,
            "Where is session restore implemented?",
            &[
                Message::system("system"),
                Message::user("Where is session restore implemented?"),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
        .expect("tool loop");

        assert_eq!(
            outcome.final_response,
            "The implementation is in `src/session/mod.rs` at line 1."
        );
        assert_eq!(outcome.tool_results.len(), 2);
        assert_eq!(outcome.tool_results[0].tool_name, "search");
        assert_eq!(outcome.tool_results[1].tool_name, "read_file");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn read_only_tool_loop_rejects_initial_prose_and_requires_tool_use() {
        let dir = std::env::temp_dir().join("params-tool-loop-requires-tool");
        let _ = std::fs::create_dir_all(dir.join("src/session"));
        let _ = std::fs::write(
            dir.join("src/session/mod.rs"),
            "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
        );

        let backend = ScriptedBackend::new(vec![
            "I will use the search tool to inspect the repo.",
            "[search: load_most_recent]",
            "[read_file: src/session/mod.rs]",
            "The implementation is in `src/session/mod.rs` at line 1.",
        ]);
        let (tx, _rx) = mpsc::channel();
        let mut cache_stats = SessionCacheStats::default();
        let mut budget = SessionBudget::default();
        let outcome = run_read_only_tool_loop(
            ToolLoopIntent::CodeNavigation,
            "Where is session restore implemented?",
            &[
                Message::system("system"),
                Message::user("Where is session restore implemented?"),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
        .expect("tool loop should recover after initial prose");

        assert_eq!(
            outcome.final_response,
            "The implementation is in `src/session/mod.rs` at line 1."
        );
        assert_eq!(outcome.tool_results.len(), 2);
        assert_eq!(outcome.tool_results[0].tool_name, "search");
        assert_eq!(outcome.tool_results[1].tool_name, "read_file");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn code_navigation_requires_file_evidence_before_answering() {
        let dir = std::env::temp_dir().join("params-tool-loop-requires-read");
        let _ = std::fs::create_dir_all(dir.join("src/session"));
        let _ = std::fs::write(
            dir.join("src/session/mod.rs"),
            "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
        );

        let backend = ScriptedBackend::new(vec![
            "[search: load_most_recent]",
            "The implementation is at `src/inference/session.rs:3765`.",
            "[read_file: src/session/mod.rs]",
            "The implementation is in `src/session/mod.rs` at line 1.",
        ]);
        let (tx, _rx) = mpsc::channel();
        let mut cache_stats = SessionCacheStats::default();
        let mut budget = SessionBudget::default();
        let outcome = run_read_only_tool_loop(
            ToolLoopIntent::CodeNavigation,
            "Where is session restore implemented?",
            &[
                Message::system("system"),
                Message::user("Where is session restore implemented?"),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
        .expect("tool loop should require a file read before final answer");

        assert_eq!(
            outcome.final_response,
            "The implementation is in `src/session/mod.rs` at line 1."
        );
        assert_eq!(outcome.tool_results.len(), 2);
        assert_eq!(outcome.tool_results[0].tool_name, "search");
        assert_eq!(outcome.tool_results[1].tool_name, "read_file");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn code_navigation_rejects_docs_only_read_and_forces_best_source_file() {
        let dir = std::env::temp_dir().join("params-tool-loop-noisy-search");
        let _ = fs::create_dir_all(dir.join("docs/context"));
        let _ = fs::create_dir_all(dir.join("src/inference"));
        let _ = fs::create_dir_all(dir.join("src/session"));
        let _ = fs::write(
            dir.join("docs/context/PLANS.md"),
            "load_most_recent overview\n",
        );
        let _ = fs::write(
            dir.join("src/inference/session.rs"),
            "fn prompt() {\n    let q = \"Where is session restore implemented?\";\n}\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn keeps_query() {\n        let x = \"load_most_recent\";\n    }\n}\n",
        );
        let _ = fs::write(
            dir.join("src/session/mod.rs"),
            "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
        );

        let backend = InspectingBackend::new(vec![
            (
                Some("Start with `[search: load_most_recent]`"),
                "[search: load_most_recent]",
            ),
            (
                Some("Next read: `[read_file: src/session/mod.rs]`"),
                "[read_file: docs/context/PLANS.md]",
            ),
            (None, "The implementation could not be clearly found."),
            (
                Some("Next read: `[read_file: src/session/mod.rs]`"),
                "[read_file: src/session/mod.rs]",
            ),
            (
                None,
                "The implementation is in `src/session/mod.rs` at line 1.",
            ),
        ]);
        let (tx, _rx) = mpsc::channel();
        let mut cache_stats = SessionCacheStats::default();
        let mut budget = SessionBudget::default();
        let outcome = run_read_only_tool_loop(
            ToolLoopIntent::CodeNavigation,
            "Where is session restore implemented?",
            &[
                Message::system("system"),
                Message::user("Where is session restore implemented?"),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
        .expect("tool loop should recover from docs-only evidence");

        assert_eq!(
            outcome.final_response,
            "The implementation is in `src/session/mod.rs` at line 1."
        );
        assert_eq!(
            outcome
                .tool_results
                .iter()
                .filter(|result| result.tool_name == "read_file")
                .count(),
            2
        );

        let _ = fs::remove_dir_all(dir);
    }
}
