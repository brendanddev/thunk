use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::error::Result;
use crate::events::{InferenceEvent, ProgressStatus};
use crate::skills;
use crate::tools::ToolRegistry;

use super::super::runtime::{eco_tool_result_limit, emit_trace};
use super::super::system_prompt_with_tools;
use super::super::Message;
use super::intent::{normalize_intent_text, suggested_search_query, ToolLoopIntent};
use super::ToolLoopBudget;

pub(super) fn tool_loop_budget(eco_enabled: bool) -> ToolLoopBudget {
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

pub(super) fn tool_loop_result_limit(backend_name: &str, eco_enabled: bool) -> Option<usize> {
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

pub(super) fn with_progress_heartbeat<T, F>(
    token_tx: &Sender<InferenceEvent>,
    label: &str,
    run: F,
) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    with_progress_heartbeat_interval(token_tx, label, Duration::from_secs(3), run)
}

pub(super) fn with_progress_heartbeat_interval<T, F>(
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
        let mut next_emit = Instant::now() + interval;
        loop {
            if stop_clone.load(Ordering::Relaxed) {
                break;
            }
            let now = Instant::now();
            if now >= next_emit {
                elapsed += interval.as_secs().max(1);
                emit_trace(
                    &tx,
                    ProgressStatus::Updated,
                    &format!("{label} ({elapsed}s elapsed)"),
                    false,
                );
                next_emit += interval;
                continue;
            }

            let sleep_for = std::cmp::min(
                Duration::from_millis(200),
                next_emit.saturating_duration_since(now),
            );
            thread::sleep(sleep_for);
        }
    });

    let result = run();
    stop.store(true, Ordering::Relaxed);
    let _ = heartbeat.join();
    result
}

pub(super) fn thinking_label(intent: ToolLoopIntent) -> (&'static str, &'static str) {
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
        ToolLoopIntent::CallSiteLookup => (
            "Thinking: searching for call-sites and reading caller files.",
            "tracing callers...",
        ),
        ToolLoopIntent::UsageLookup => (
            "Thinking: searching for usages and references.",
            "tracing usages...",
        ),
        ToolLoopIntent::FlowTrace => (
            "Thinking: tracing execution flow across files.",
            "tracing flow...",
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

pub(super) fn build_tool_loop_system_prompt(
    tools: &ToolRegistry,
    project_root: &Path,
    intent: ToolLoopIntent,
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
    match intent {
        ToolLoopIntent::CallSiteLookup => {
            prompt.push_str(
                "\n\nInvestigation mode: CALL-SITE LOOKUP.\n\
                 Your goal is to find WHERE this symbol is called from — not to describe what it does.\n\
                 - Search for the symbol name to find invocation lines (lines that call it, not define it).\n\
                 - After finding search hits, read one or more files containing call-sites.\n\
                 - List every file:line where the symbol is invoked.\n\
                 - Do NOT summarize the symbol's own implementation unless the caller's purpose is unclear.",
            );
        }
        ToolLoopIntent::UsageLookup => {
            prompt.push_str(
                "\n\nInvestigation mode: USAGE LOOKUP.\n\
                 Your goal is to find WHERE this symbol is used, imported, or referenced.\n\
                 - Search for the symbol name to find all non-definition references.\n\
                 - After finding search hits, read one or more files containing usages.\n\
                 - List every file:line where the symbol appears as a type, import, or reference.\n\
                 - Do NOT describe the symbol's implementation unless it explains a usage pattern.",
            );
        }
        ToolLoopIntent::FlowTrace => {
            prompt.push_str(
                "\n\nInvestigation mode: FLOW TRACE.\n\
                 Your goal is to trace the execution or data flow, which typically spans multiple files.\n\
                 - Start at the entry point; search for the main symbol or function to find its definition.\n\
                 - After reading the entry point, follow the function calls outward into related files.\n\
                 - Read at least two source files to build a cross-file picture of the flow.\n\
                 - Answer with a concrete, ordered sequence of steps showing how control or data moves.",
            );
        }
        ToolLoopIntent::CodeNavigation => {
            prompt.push_str(
                "\n\nInvestigation mode: IMPLEMENTATION LOOKUP.\n\
                 Your goal is to find where the symbol, feature, or behavior is defined or implemented.\n\
                 - Search for the symbol to find its definition file.\n\
                 - Read the definition file and identify the primary implementation.\n\
                 - Expand to related files only if the implementation delegates to them.",
            );
        }
        ToolLoopIntent::ConfigLocate => {
            prompt.push_str(
                "\n\nInvestigation mode: CONFIG LOOKUP.\n\
                 Your goal is to find where a setting, option, or configuration value is defined or applied.\n\
                 - Search for the config key name in source and config files.\n\
                 - Prefer .toml config files and source files that parse or apply the setting.",
            );
        }
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview => {}
    }
    if let Some(summary) = repo_context_summary(project_root) {
        prompt.push_str("\n\n");
        prompt.push_str(&summary);
    }
    skills::append_repo_navigation_skill_guidance(project_root, &mut prompt);
    prompt
}

pub(super) fn initial_tool_only_followup(intent: ToolLoopIntent, prompt: &str) -> String {
    let search_hint = suggested_search_query(prompt, intent);
    match intent {
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview => {
            "Repo-navigation mode requires live inspection before answering. \
             Your next response must contain only one or more read-only tool tags and no prose. \
             Start by listing the current directory with `[list_dir: .]`."
                .to_string()
        }
        ToolLoopIntent::CallSiteLookup => match search_hint {
            Some(ref query) => {
                format!(
                    "Repo-navigation mode requires live inspection before answering. \
                     Your goal is to find CALL-SITES — where the symbol is invoked, not defined. \
                     Your next response must contain only read-only tool tags and no prose. \
                     Start with `[search: {query}]`. Look for lines that call the symbol (contain `{query}(`), not lines that define it. \
                     After finding call-sites, read at least one file containing a caller."
                )
            }
            None => {
                "Start by searching for the symbol name with `[search: ...]` to find invocation lines."
                    .to_string()
            }
        },
        ToolLoopIntent::UsageLookup => match search_hint {
            Some(ref query) => {
                format!(
                    "Repo-navigation mode requires live inspection before answering. \
                     Your goal is to find USAGES — where the symbol is referenced, imported, or used as a type. \
                     Your next response must contain only read-only tool tags and no prose. \
                     Start with `[search: {query}]` to find all references."
                )
            }
            None => {
                "Start by searching for the symbol name with `[search: ...]` to find all references."
                    .to_string()
            }
        },
        ToolLoopIntent::FlowTrace => match search_hint {
            Some(ref query) => {
                format!(
                    "Repo-navigation mode requires live inspection before answering. \
                     Your goal is to trace the execution flow — this requires reading multiple source files. \
                     Your next response must contain only read-only tool tags and no prose. \
                     Start with `[search: {query}]` to locate the entry point. \
                     Then read the entry point file and follow the function calls to trace the complete flow."
                )
            }
            None => {
                "Start by searching for the entry-point symbol with `[search: ...]`, then follow the call chain across files."
                    .to_string()
            }
        },
        ToolLoopIntent::CodeNavigation | ToolLoopIntent::ConfigLocate => match search_hint {
            Some(ref query) => {
                format!(
                    "Repo-navigation mode requires live inspection before answering. \
                     Your next response must contain only one or more read-only tool tags and no prose. \
                     Start with `[search: {query}]`. Prefer the concrete symbol or setting name over a broad natural-language phrase."
                )
            }
            None => {
                "Repo-navigation mode requires live inspection before answering. \
                 Your next response must contain only one or more read-only tool tags and no prose. \
                 Start by searching for the most relevant symbol, term, or setting with `[search: ...]`."
                    .to_string()
            }
        },
    }
}

pub(super) fn initial_investigation_hint(intent: ToolLoopIntent, prompt: &str) -> Option<String> {
    let query = suggested_search_query(prompt, intent)?;
    match intent {
        ToolLoopIntent::CodeNavigation => Some(format!(
            "Investigation hint: the most promising search target for this request is `{query}`. \
             Start with `[search: {query}]`. \
             Prefer the concrete symbol over the full English question. After searching, read the strongest source-file candidate before answering. Ignore docs, tests, prompt strings, and use-sites unless the repo has no better source hits."
        )),
        ToolLoopIntent::ConfigLocate => Some(format!(
            "Investigation hint: the most promising search target for this request is `{query}`. \
             Start with `[search: {query}]`. \
             Prefer the concrete setting key over the full English question. After searching, read the strongest config or source file before answering."
        )),
        ToolLoopIntent::CallSiteLookup => Some(format!(
            "Investigation hint: search for `{query}` to find call-sites. \
             Start with `[search: {query}]`. \
             Look for lines that INVOKE the symbol (containing `{query}(`), not lines that define it. \
             After finding call-sites in search results, read at least one file containing a caller to confirm context."
        )),
        ToolLoopIntent::UsageLookup => Some(format!(
            "Investigation hint: search for `{query}` to find all usages. \
             Start with `[search: {query}]`. \
             Look for import lines, type annotations, and non-definition references."
        )),
        ToolLoopIntent::FlowTrace => Some(format!(
            "Investigation hint: start by locating the entry point with `[search: {query}]`. \
             After finding the entry point, read that file and follow the function calls into related files. \
             Trace across at least 2 source files to adequately describe the flow."
        )),
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview => None,
    }
}

fn is_referential_follow_up(prompt: &str) -> bool {
    let normalized = normalize_intent_text(prompt);
    let referential_tokens = [
        "it", "its", "that", "this", "these", "those", "they", "them", "there", "same", "above",
        "previous", "former", "latter", "now", "again", "more",
    ];

    normalized
        .split_whitespace()
        .any(|token| referential_tokens.contains(&token))
}

pub(super) fn build_tool_loop_seed_messages(
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
