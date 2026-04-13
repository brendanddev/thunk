use std::path::Path;

use crate::memory::retrieval::{query_terms, score_text};

use super::parse::clip_inline;
use super::types::{
    AutoInspectBudget, AutoInspectIntent, AutoInspectPlan, AutoInspectStep, SearchFileHit,
};

pub(crate) fn is_config_path(path: &str) -> bool {
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

pub(crate) fn is_doc_path(path: &str) -> bool {
    path.ends_with(".md") || path.starts_with("docs/")
}

pub(crate) fn is_code_path(path: &str) -> bool {
    [
        ".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".kt", ".swift", ".rb", ".php",
        ".cs", ".toml", ".json", ".yaml", ".yml", ".sh",
    ]
    .iter()
    .any(|suffix| path.ends_with(suffix))
}

pub(crate) fn file_display_name(path: &str) -> String {
    format!("`{path}`")
}

pub(crate) fn declaration_lines_with_numbers(content: &str) -> Vec<(usize, String)> {
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

pub(crate) fn test_module_start_line(content: &str) -> Option<usize> {
    content
        .lines()
        .enumerate()
        .find(|(_, line)| {
            let trimmed = line.trim();
            trimmed.starts_with("#[cfg(test)]") || trimmed == "mod tests {"
        })
        .map(|(idx, _)| idx + 1)
}

pub(crate) fn filter_non_test_hits(
    content: &str,
    hits: Vec<(usize, String)>,
) -> Vec<(usize, String)> {
    if let Some(test_start) = test_module_start_line(content) {
        hits.into_iter()
            .filter(|(line_number, _)| *line_number < test_start)
            .collect()
    } else {
        hits
    }
}

pub(crate) fn match_lines_with_numbers(
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

pub(crate) fn definition_match_lines_with_numbers(
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

pub(crate) fn primary_definition_location(
    path: &str,
    content: &str,
    query: &str,
    max_chars: usize,
) -> Option<String> {
    let terms = query_terms(query);
    let matches = definition_match_lines_with_numbers(content, &terms, 1);
    let (line_number, line) = matches.into_iter().next()?;
    Some(format!(
        "{}:{} `{}`",
        path,
        line_number,
        clip_inline(&line, max_chars)
    ))
}

pub(crate) fn primary_config_locations(
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

pub(crate) fn format_numbered_hits(hits: &[(usize, String)], clip_chars: usize) -> String {
    hits.iter()
        .map(|(line_number, line)| format!("{line_number} `{}`", clip_inline(line, clip_chars)))
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn is_definition_like_line(line: &str) -> bool {
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

pub(crate) fn is_implementation_definition_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with("pub fn ")
        || trimmed.starts_with("fn ")
        || trimmed.starts_with("pub struct ")
        || trimmed.starts_with("struct ")
        || trimmed.starts_with("pub enum ")
        || trimmed.starts_with("enum ")
        || trimmed.starts_with("impl ")
}

pub(crate) fn config_scope_label(path: &str) -> &'static str {
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

pub(crate) fn rank_search_files(
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

pub(crate) fn preferred_config_paths(project_root: &Path) -> Vec<String> {
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

pub(crate) fn preferred_workflow_paths(plan: &AutoInspectPlan, project_root: &Path) -> Vec<String> {
    let mut preferred = match plan.query.as_deref() {
        Some("load_most_recent") => vec!["src/session/mod.rs".to_string()],
        Some("save_messages") => vec!["src/session/mod.rs".to_string()],
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

pub(crate) fn is_auto_inspection_read_candidate(project_root: &Path, rel: &str) -> bool {
    const MAX_READ_BYTES: u64 = 100_000;
    let path = project_root.join(rel);
    path.is_file()
        && path
            .metadata()
            .map(|meta| meta.len() <= MAX_READ_BYTES)
            .unwrap_or(false)
}

pub(crate) fn choose_followup_read_steps(
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

pub(crate) fn summarize_workflow_read(
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

pub(crate) fn summarize_feature_trace_hits(
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

pub(crate) fn is_feature_trace_anchor_line(query: &str, line: &str) -> bool {
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
