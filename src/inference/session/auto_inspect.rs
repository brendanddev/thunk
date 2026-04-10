use std::path::Path;

use crate::memory::retrieval::{query_terms, score_text};
use crate::tools::ToolResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AutoInspectIntent {
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

fn detect_auto_inspect_intent(prompt: &str) -> Option<AutoInspectIntent> {
    let normalized = normalize_intent_text(prompt);
    let tokens = normalized
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    if normalized.starts_with('/') {
        return None;
    }

    let starts_with = |a: &str, b: &str| {
        tokens.first().map(|t| t == a).unwrap_or(false)
            && tokens.get(1).map(|t| t == b).unwrap_or(false)
    };
    let has_token = |value: &str| tokens.iter().any(|token| token == value);
    let has_prefix = |prefix: &str| tokens.iter().any(|token| token.starts_with(prefix));

    if (starts_with("what", "is") || starts_with("whats", "in") || starts_with("what", "does"))
        && (has_token("repo") || has_token("project") || has_token("codebase"))
    {
        return Some(AutoInspectIntent::RepoOverview);
    }

    if starts_with("summarize", "this")
        && (has_token("repo") || has_token("project") || has_token("codebase"))
    {
        return Some(AutoInspectIntent::RepoOverview);
    }

    if (starts_with("what", "is")
        || starts_with("whats", "in")
        || starts_with("whats", "here")
        || starts_with("what", "here"))
        && (has_token("directory") || has_token("folder") || has_token("here"))
    {
        return Some(AutoInspectIntent::DirectoryOverview);
    }

    if starts_with("where", "is")
        && (has_prefix("implement") || has_prefix("defin") || has_prefix("handl"))
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

    if starts_with("where", "is") && (has_prefix("config") || has_token("set")) {
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
            "project",
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
                trim_query_suffix(rest, &[" configured", " configged", " set"]).to_string()
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

fn test_module_start_line(content: &str) -> Option<usize> {
    content
        .lines()
        .enumerate()
        .find(|(_, line)| {
            let trimmed = line.trim();
            trimmed.starts_with("#[cfg(test)]") || trimmed == "mod tests {"
        })
        .map(|(idx, _)| idx + 1)
}

fn filter_non_test_hits(content: &str, hits: Vec<(usize, String)>) -> Vec<(usize, String)> {
    if let Some(test_start) = test_module_start_line(content) {
        hits.into_iter()
            .filter(|(line_number, _)| *line_number < test_start)
            .collect()
    } else {
        hits
    }
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

fn primary_config_locations(
    path: &str,
    content: &str,
    query: &str,
    max_chars: usize,
) -> Vec<String> {
    let terms = query_terms(query);
    filter_non_test_hits(content, match_lines_with_numbers(content, &terms, 2))
        .into_iter()
        .map(|(line_number, line)| {
            format!(
                "{}:{} `{}`",
                path,
                line_number,
                clip_inline(&line, max_chars)
            )
        })
        .collect()
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
            "src/config.rs".to_string(),
            "src/tui/commands.rs".to_string(),
            "src/inference/session.rs".to_string(),
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

    selected
        .into_iter()
        .filter(|path| is_auto_inspection_read_candidate(project_root, path))
        .take(budget.read_files)
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
            let matches =
                filter_non_test_hits(content, match_lines_with_numbers(content, &terms, 2));
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
                let preferred = filter_non_test_hits(
                    content,
                    definition_match_lines_with_numbers(content, &terms, 2),
                );
                if preferred.is_empty() {
                    filter_non_test_hits(content, match_lines_with_numbers(content, &terms, 2))
                } else {
                    preferred
                }
            } else {
                filter_non_test_hits(content, match_lines_with_numbers(content, &terms, 2))
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
            let matches =
                filter_non_test_hits(content, match_lines_with_numbers(content, &terms, 3));
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
    test_starts: &std::collections::HashMap<String, usize>,
) -> Vec<String> {
    ranked
        .iter()
        .take(budget.search_files)
        .filter_map(|file| {
            file.hits
                .iter()
                .filter(move |hit| is_feature_trace_anchor_line(query, &hit.line_content))
                .filter(move |hit| {
                    test_starts
                        .get(&file.path)
                        .map(|start| hit.line_number < *start)
                        .unwrap_or(true)
                })
                .next()
                .map(move |hit| {
                    format!(
                        "{}:{} `{}`",
                        file.path,
                        hit.line_number,
                        clip_inline(&hit.line_content, 56)
                    )
                })
        })
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
        let mut primary_config_lines = Vec::new();
        let mut read_test_starts = std::collections::HashMap::new();

        for result in results {
            match result.tool_name.as_str() {
                "search" => search_hits.extend(parse_search_output(&result.output)),
                "read_file" => {
                    if let Some((path, content)) = parse_read_file_output(&result.output) {
                        read_paths.push(path.clone());
                        if let Some(start) = test_module_start_line(&content) {
                            read_test_starts.insert(path.clone(), start);
                        }
                        if plan.intent == AutoInspectIntent::WhereIsImplementation {
                            if let Some(location) =
                                primary_definition_location(&path, &content, query, 72)
                            {
                                primary_locations.push(location);
                            }
                        } else if plan.intent == AutoInspectIntent::ConfigLocate {
                            primary_config_lines
                                .extend(primary_config_locations(&path, &content, query, 72));
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
            summarize_feature_trace_hits(query, &ranked, budget, &read_test_starts)
        } else {
            Vec::new()
        };
        let read_test_starts_ref = &read_test_starts;

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
                    .filter(move |hit| {
                        read_test_starts_ref
                            .get(&file.path)
                            .map(|start| hit.line_number < *start)
                            .unwrap_or(true)
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
            AutoInspectIntent::ConfigLocate => {
                "Instruction: answer directly from this evidence. Prefer exact config-setting or merge lines over broad section headings, struct declarations, or nearby docs. Cite the concrete file:line locations that set or merge the behavior. Do not emit tool calls or fenced code blocks."
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
        if !primary_config_lines.is_empty() {
            sections.push(format!(
                "Primary config lines: {}",
                primary_config_lines.join(", ")
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

/// Persistent model thread — loads the backend once, handles prompts in a loop.

#[cfg(test)]
mod tests;
