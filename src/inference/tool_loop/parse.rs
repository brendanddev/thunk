use crate::inference::session::investigation::InvestigationResolution;
use crate::memory::retrieval::{query_terms, score_text};
use crate::tools::ToolResult;

use super::super::intent::{
    is_referential_file_prompt, normalize_intent_text, suggested_search_query, ToolLoopIntent,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SearchLineHit {
    pub(super) line_number: usize,
    pub(super) line_content: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SearchFileHit {
    pub(super) path: String,
    pub(super) hits: Vec<SearchLineHit>,
}

pub(super) fn parse_search_output(output: &str) -> Vec<SearchFileHit> {
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

pub(super) fn is_doc_path(path: &str) -> bool {
    path.ends_with(".md") || path.starts_with("docs/")
}

pub(super) fn is_test_like_path(path: &str) -> bool {
    path.starts_with("tests/")
        || path.contains("/tests/")
        || path.ends_with("/tests.rs")
        || path.ends_with("/test.rs")
        || path.contains("fixtures")
        || path.contains("snapshots")
        || path.ends_with("_test.rs")
        || path.ends_with("_tests.rs")
}

pub(super) fn is_internal_tool_loop_path(path: &str) -> bool {
    path.starts_with("src/inference/tool_loop/")
}

pub(super) fn is_legacy_auto_inspect_path(path: &str) -> bool {
    path.starts_with("src/inference/session/auto_inspect")
}

pub(super) fn is_source_path(path: &str) -> bool {
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

pub(super) fn is_config_path(path: &str) -> bool {
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

pub(super) fn is_definition_like_line(line: &str) -> bool {
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

fn is_identifier_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn identifier_match_offsets(line: &str, symbol: &str) -> Vec<usize> {
    if symbol.is_empty() {
        return Vec::new();
    }

    line.match_indices(symbol)
        .filter_map(|(idx, _)| {
            let prev = line[..idx].chars().next_back();
            let next = line[idx + symbol.len()..].chars().next();
            let bounded = !prev.map(is_identifier_char).unwrap_or(false)
                && !next.map(is_identifier_char).unwrap_or(false);
            if bounded {
                Some(idx)
            } else {
                None
            }
        })
        .collect()
}

pub(super) fn prompt_mentions_tests(prompt: &str) -> bool {
    let normalized = prompt.to_ascii_lowercase();
    normalized.contains(" test")
        || normalized.contains("tests")
        || normalized.starts_with("test ")
        || normalized.contains(" unit test")
}

pub(super) fn line_contains_symbol_reference(line: &str, symbol: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with("//") {
        return false;
    }

    let lowered = trimmed.to_ascii_lowercase();
    let symbol = symbol.to_ascii_lowercase();
    identifier_match_offsets(&lowered, &symbol)
        .into_iter()
        .any(|idx| {
            let next = lowered[idx + symbol.len()..].chars().next();
            let prev = lowered[..idx].chars().next_back();
            !(matches!(prev, Some('"') | Some('\'')) && matches!(next, Some('"') | Some('\'')))
        })
}

pub(super) fn line_contains_symbol_invocation(line: &str, symbol: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with("//") {
        return false;
    }

    let lowered = trimmed.to_ascii_lowercase();
    let symbol = symbol.to_ascii_lowercase();
    identifier_match_offsets(&lowered, &symbol)
        .into_iter()
        .any(|idx| {
            let next = lowered[idx + symbol.len()..].trim_start();
            next.starts_with('(') || next.starts_with("::<")
        })
}

pub(super) fn clip_inline(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    let clipped = text
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    format!("{}…", clipped.trim_end())
}

pub(super) fn parse_read_file_output(output: &str) -> Option<(String, String)> {
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

pub(super) fn clip_tool_output(output: &str, max_chars: Option<usize>) -> String {
    let Some(limit) = max_chars else {
        return output.to_string();
    };

    let total = output.chars().count();
    if total <= limit {
        return output.to_string();
    }

    let keep = limit.saturating_sub(80);
    let truncated: String = output.chars().take(keep).collect();
    format!(
        "{truncated}\n\n[truncated {} chars for eco mode]",
        total.saturating_sub(keep)
    )
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

pub(super) fn filter_non_test_hits(
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

pub(super) fn definition_match_lines_with_numbers(
    content: &str,
    query: &str,
    limit: usize,
) -> Vec<(usize, String)> {
    let terms = query_terms(query);
    content
        .lines()
        .enumerate()
        .map(|(idx, line)| (idx + 1, line.trim()))
        .filter(|(_, line)| !line.is_empty())
        .filter(|(_, line)| score_text(&terms, line) > 0)
        .filter(|(_, line)| is_definition_like_line(line))
        .take(limit)
        .map(|(line_number, line)| (line_number, line.to_string()))
        .collect()
}

pub(super) fn query_match_lines_with_numbers(
    content: &str,
    query: &str,
    limit: usize,
) -> Vec<(usize, String)> {
    let terms = query_terms(query);
    content
        .lines()
        .enumerate()
        .map(|(idx, line)| (idx + 1, line.trim()))
        .filter(|(_, line)| !line.is_empty())
        .filter(|(_, line)| score_text(&terms, line) > 0)
        .take(limit)
        .map(|(line_number, line)| (line_number, line.to_string()))
        .collect()
}

pub(super) fn surrounding_body_lines(
    content: &str,
    anchor_line: usize,
    limit: usize,
) -> Vec<(usize, String)> {
    let test_start = test_module_start_line(content);
    content
        .lines()
        .enumerate()
        .skip(anchor_line)
        .map(|(idx, line)| (idx + 1, line.trim()))
        .filter(|(_, line)| !line.is_empty())
        .filter(|(line_number, _)| test_start.map(|start| *line_number < start).unwrap_or(true))
        .take(limit)
        .map(|(line_number, line)| (line_number, line.to_string()))
        .collect()
}

pub(super) fn first_non_empty_lines(content: &str, limit: usize) -> Vec<(usize, String)> {
    content
        .lines()
        .enumerate()
        .map(|(idx, line)| (idx + 1, line.trim()))
        .filter(|(_, line)| !line.is_empty())
        .take(limit)
        .map(|(line_number, line)| (line_number, line.to_string()))
        .collect()
}

pub(super) fn declaration_lines_with_numbers(content: &str, limit: usize) -> Vec<(usize, String)> {
    content
        .lines()
        .enumerate()
        .map(|(idx, line)| (idx + 1, line.trim()))
        .filter(|(_, line)| {
            !line.is_empty()
                && (line.starts_with("pub struct ")
                    || line.starts_with("struct ")
                    || line.starts_with("pub enum ")
                    || line.starts_with("enum ")
                    || line.starts_with("pub fn ")
                    || line.starts_with("fn ")
                    || line.starts_with("impl ")
                    || line.starts_with("mod ")
                    || line.starts_with("pub mod ")
                    || line.starts_with("use "))
        })
        .take(limit)
        .map(|(line_number, line)| (line_number, line.to_string()))
        .collect()
}

pub(super) fn compact_read_file_result(
    intent: ToolLoopIntent,
    prompt: &str,
    resolution: Option<&InvestigationResolution>,
    result: &ToolResult,
    max_chars_per_result: Option<usize>,
) -> Option<String> {
    let (path, content) = parse_read_file_output(&result.output)?;
    let query =
        suggested_search_query(prompt, intent).unwrap_or_else(|| normalize_intent_text(prompt));
    let mut sections = vec![format!("File: {path}")];

    match intent {
        ToolLoopIntent::CodeNavigation => {
            if is_referential_file_prompt(prompt)
                || resolution
                    .map(|resolution| {
                        resolution.prefer_answer_from_anchor
                            && resolution.anchored_file.as_deref() == Some(path.as_str())
                    })
                    .unwrap_or(false)
            {
                let declarations = declaration_lines_with_numbers(&content, 6);
                let excerpt = if declarations.is_empty() {
                    first_non_empty_lines(&content, 6)
                } else {
                    declarations
                };
                if !excerpt.is_empty() {
                    sections.push("Observed declarations:".to_string());
                    sections.extend(excerpt.into_iter().map(|(line_number, line)| {
                        format!("  {line_number}: `{}`", clip_inline(&line, 120))
                    }));
                }
            } else {
                let definition_hits = filter_non_test_hits(
                    &content,
                    definition_match_lines_with_numbers(&content, &query, 1),
                );
                if let Some((line_number, line)) = definition_hits.into_iter().next() {
                    sections.push(format!(
                        "Primary implementation: {}:{} `{}`",
                        path,
                        line_number,
                        clip_inline(&line, 120)
                    ));
                    let body_lines = surrounding_body_lines(&content, line_number, 4);
                    if !body_lines.is_empty() {
                        sections.push("Observed body lines:".to_string());
                        sections.extend(body_lines.into_iter().map(|(line_number, line)| {
                            format!("  {line_number}: `{}`", clip_inline(&line, 120))
                        }));
                    }
                } else {
                    let matches = filter_non_test_hits(
                        &content,
                        query_match_lines_with_numbers(&content, &query, 6),
                    );
                    let excerpt = if matches.is_empty() {
                        first_non_empty_lines(&content, 6)
                    } else {
                        matches
                    };
                    if !excerpt.is_empty() {
                        sections.push("Observed lines:".to_string());
                        sections.extend(excerpt.into_iter().map(|(line_number, line)| {
                            format!("  {line_number}: `{}`", clip_inline(&line, 120))
                        }));
                    }
                }
            }
        }
        ToolLoopIntent::FlowTrace => {
            let definition_hits = filter_non_test_hits(
                &content,
                definition_match_lines_with_numbers(&content, &query, 1),
            );
            if let Some((line_number, line)) = definition_hits.into_iter().next() {
                sections.push(format!(
                    "Primary definition: {}:{} `{}`",
                    path,
                    line_number,
                    clip_inline(&line, 120)
                ));
                let body_lines = surrounding_body_lines(&content, line_number, 6);
                if !body_lines.is_empty() {
                    sections.push("Observed body lines:".to_string());
                    sections.extend(body_lines.into_iter().map(|(line_number, line)| {
                        format!("  {line_number}: `{}`", clip_inline(&line, 120))
                    }));
                }
            } else {
                let matches = filter_non_test_hits(
                    &content,
                    query_match_lines_with_numbers(&content, &query, 8),
                );
                let excerpt = if matches.is_empty() {
                    first_non_empty_lines(&content, 6)
                } else {
                    matches
                };
                if !excerpt.is_empty() {
                    sections.push("Observed lines:".to_string());
                    sections.extend(excerpt.into_iter().map(|(line_number, line)| {
                        format!("  {line_number}: `{}`", clip_inline(&line, 120))
                    }));
                }
            }
        }
        ToolLoopIntent::CallSiteLookup | ToolLoopIntent::UsageLookup => {
            let all_matches = filter_non_test_hits(
                &content,
                query_match_lines_with_numbers(&content, &query, 12),
            );
            let is_call_site = matches!(intent, ToolLoopIntent::CallSiteLookup);
            let site_lines: Vec<(usize, String)> = all_matches
                .into_iter()
                .filter(|(_, line)| {
                    if is_call_site {
                        line.contains('(') && !is_definition_like_line(line)
                    } else {
                        !is_definition_like_line(line)
                    }
                })
                .take(6)
                .collect();
            if !site_lines.is_empty() {
                let label = if is_call_site {
                    "Call-sites:"
                } else {
                    "Usages:"
                };
                sections.push(label.to_string());
                sections.extend(site_lines.into_iter().map(|(line_number, line)| {
                    format!("  {line_number}: `{}`", clip_inline(&line, 120))
                }));
            } else {
                let excerpt = query_match_lines_with_numbers(&content, &query, 6);
                if !excerpt.is_empty() {
                    sections.push("Observed lines:".to_string());
                    sections.extend(excerpt.into_iter().map(|(line_number, line)| {
                        format!("  {line_number}: `{}`", clip_inline(&line, 120))
                    }));
                }
            }
        }
        ToolLoopIntent::ConfigLocate => {
            let matches = filter_non_test_hits(
                &content,
                query_match_lines_with_numbers(&content, &query, 6),
            );
            let excerpt = if matches.is_empty() {
                first_non_empty_lines(&content, 6)
            } else {
                matches
            };
            if !excerpt.is_empty() {
                sections.push("Observed lines:".to_string());
                sections.extend(excerpt.into_iter().map(|(line_number, line)| {
                    format!("  {line_number}: `{}`", clip_inline(&line, 120))
                }));
            }
        }
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview => {
            let excerpt = first_non_empty_lines(&content, 8);
            if !excerpt.is_empty() {
                sections.push("Observed lines:".to_string());
                sections.extend(excerpt.into_iter().map(|(line_number, line)| {
                    format!("  {line_number}: `{}`", clip_inline(&line, 120))
                }));
            }
        }
    }

    Some(clip_tool_output(&sections.join("\n"), max_chars_per_result))
}
