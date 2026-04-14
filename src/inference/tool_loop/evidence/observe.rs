use std::collections::{HashMap, HashSet};

use crate::tools::ToolResult;

use super::super::super::session::investigation::InvestigationResolution;
use super::super::super::Message;
use super::super::intent::{
    is_referential_file_prompt, normalize_intent_text, suggested_search_query, ToolLoopIntent,
};
use super::parse::{
    declaration_lines_with_numbers, definition_match_lines_with_numbers, filter_non_test_hits,
    first_non_empty_lines, is_definition_like_line, is_internal_tool_loop_path,
    is_legacy_auto_inspect_path, is_source_path, is_test_like_path,
    line_contains_symbol_invocation, line_contains_symbol_reference, parse_read_file_output,
    parse_search_output, query_match_lines_with_numbers, surrounding_body_lines, SearchFileHit,
    SearchLineHit,
};
use super::{
    FileSummaryEvidence, FlowTraceEvidence, ImplementationEvidence, ObservedLine, ObservedStep,
    ObservedStepKind, RepoOverviewEvidence,
};

pub(super) fn recent_loaded_file_context_path(base_messages: &[Message]) -> Option<String> {
    base_messages.iter().rev().find_map(|message| {
        if message.role != "user"
            || !message
                .content
                .starts_with("I've loaded this file for context:")
        {
            return None;
        }
        message.content.lines().find_map(|line| {
            line.strip_prefix("File: ")
                .map(|path| path.trim().to_string())
        })
    })
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

pub(super) fn search_candidates_in_output_order(results: &[ToolResult]) -> Vec<SearchFileHit> {
    let mut by_path = HashMap::<String, Vec<SearchLineHit>>::new();
    let mut order = Vec::<String>::new();

    for result in results {
        if result.tool_name != "search" {
            continue;
        }
        for file in parse_search_output(&result.output) {
            if !by_path.contains_key(&file.path) {
                order.push(file.path.clone());
            }
            by_path.entry(file.path).or_default().extend(file.hits);
        }
    }

    order
        .into_iter()
        .filter_map(|path| {
            let mut hits = by_path.remove(&path)?;
            hits.sort_by_key(|hit| hit.line_number);
            hits.dedup_by(|a, b| {
                a.line_number == b.line_number && a.line_content == b.line_content
            });
            Some(SearchFileHit { path, hits })
        })
        .collect()
}

fn lookup_hit_matches(intent: ToolLoopIntent, query: &str, line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() || is_definition_like_line(trimmed) {
        return false;
    }

    match intent {
        ToolLoopIntent::CallSiteLookup => line_contains_symbol_invocation(trimmed, query),
        ToolLoopIntent::UsageLookup => line_contains_symbol_reference(trimmed, query),
        _ => false,
    }
}

fn lookup_search_anchor_for_query(
    intent: ToolLoopIntent,
    query: &str,
    file: &SearchFileHit,
) -> Option<SearchLineHit> {
    if query.trim().is_empty() {
        return None;
    }

    if matches!(intent, ToolLoopIntent::UsageLookup) {
        if let Some(hit) = file.hits.iter().find(|hit| {
            let line = hit.line_content.trim();
            line.starts_with("use ") && lookup_hit_matches(intent, query, line)
        }) {
            return Some(hit.clone());
        }
    }

    file.hits
        .iter()
        .find(|hit| lookup_hit_matches(intent, query, hit.line_content.trim()))
        .cloned()
}

pub(super) fn lookup_search_anchor(
    intent: ToolLoopIntent,
    prompt: &str,
    file: &SearchFileHit,
) -> Option<SearchLineHit> {
    let query = suggested_search_query(prompt, intent).unwrap_or_default();
    lookup_search_anchor_for_query(intent, &query, file)
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
    if matches!(intent, ToolLoopIntent::ConfigLocate) && super::parse::is_config_path(&file.path) {
        score += 24;
    }
    if super::parse::is_doc_path(&file.path) {
        score -= 18;
    }
    if is_test_like_path(&file.path) {
        score -= 28;
    }
    if is_internal_tool_loop_path(&file.path) {
        score -= 32;
    }
    if is_legacy_auto_inspect_path(&file.path) {
        score -= 24;
    }
    if path.contains("prompt") || path.contains("fixture") {
        score -= 14;
    }
    if !query.is_empty() && path.contains(&query) {
        score += 10;
    }

    score += (file.hits.len().min(6) as isize) * 3;
    if matches!(
        intent,
        ToolLoopIntent::CallSiteLookup | ToolLoopIntent::UsageLookup
    ) {
        if let Some(anchor) = lookup_search_anchor_for_query(intent, &query, file) {
            score += 30;
            if matches!(intent, ToolLoopIntent::UsageLookup)
                && anchor.line_content.trim().starts_with("use ")
            {
                score += 18;
            }
        } else {
            score -= 18;
        }
    }

    for hit in file.hits.iter().take(6) {
        let line = hit.line_content.trim();
        let line_lower = line.to_ascii_lowercase();
        if !query.is_empty() && line_lower.contains(&query) {
            score += 6;
        }
        match intent {
            ToolLoopIntent::CallSiteLookup => {
                if lookup_hit_matches(intent, &query, line) {
                    score += 26;
                } else if line_contains_symbol_reference(line, &query)
                    && !is_definition_like_line(line)
                {
                    score += 8;
                }
                if is_definition_like_line(line) {
                    score -= 18;
                }
            }
            ToolLoopIntent::UsageLookup => {
                if lookup_hit_matches(intent, &query, line) {
                    score += if line.starts_with("use ") { 28 } else { 18 };
                }
                if is_definition_like_line(line) {
                    score -= 16;
                }
            }
            _ => {
                if is_definition_like_line(line) {
                    score += 24;
                }
            }
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

pub(super) fn preferred_candidate_path(intent: ToolLoopIntent, path: &str) -> bool {
    match intent {
        ToolLoopIntent::CodeNavigation => {
            is_source_path(path)
                && !super::parse::is_doc_path(path)
                && !is_test_like_path(path)
                && !is_internal_tool_loop_path(path)
                && !is_legacy_auto_inspect_path(path)
        }
        ToolLoopIntent::ConfigLocate => {
            (super::parse::is_config_path(path) || is_source_path(path))
                && !super::parse::is_doc_path(path)
                && !is_test_like_path(path)
                && !is_internal_tool_loop_path(path)
                && !is_legacy_auto_inspect_path(path)
        }
        ToolLoopIntent::FlowTrace
        | ToolLoopIntent::CallSiteLookup
        | ToolLoopIntent::UsageLookup => {
            is_source_path(path)
                && !super::parse::is_doc_path(path)
                && !is_test_like_path(path)
                && !is_internal_tool_loop_path(path)
                && !is_legacy_auto_inspect_path(path)
        }
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview => true,
    }
}

pub(super) fn ranked_search_candidates(
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

fn normalize_read_path(path: &str) -> String {
    let path = path.split(':').next().unwrap_or(path);
    let normalized = path.replace("\\", "/");
    if let Some(stripped) = normalized.strip_prefix("./") {
        stripped.to_string()
    } else {
        normalized
    }
}

pub(super) fn all_search_candidate_paths(
    intent: ToolLoopIntent,
    prompt: &str,
    results: &[ToolResult],
) -> Vec<String> {
    ranked_search_candidates(intent, prompt, results)
        .into_iter()
        .filter(|file| {
            is_source_path(&file.path)
                && !is_test_like_path(&file.path)
                && !is_internal_tool_loop_path(&file.path)
                && !is_legacy_auto_inspect_path(&file.path)
        })
        .map(|file| file.path)
        .collect()
}

pub(crate) fn all_candidates_fully_read(
    intent: ToolLoopIntent,
    prompt: &str,
    results: &[ToolResult],
) -> bool {
    let candidates = all_search_candidate_paths(intent, prompt, results);
    let read_paths = observed_read_paths(results);

    candidates.iter().all(|path| read_paths.contains(path)) && !candidates.is_empty()
}

pub(crate) fn observed_read_paths(results: &[ToolResult]) -> HashSet<String> {
    results
        .iter()
        .filter(|result| result.tool_name == "read_file")
        .map(|result| normalize_read_path(&result.argument))
        .collect()
}

pub(super) fn should_answer_from_anchor_file(
    prompt: &str,
    resolution: Option<&InvestigationResolution>,
) -> bool {
    is_referential_file_prompt(prompt)
        || resolution
            .map(|resolution| {
                resolution.prefer_answer_from_anchor && resolution.anchored_file.is_some()
            })
            .unwrap_or(false)
}

pub(super) fn observed_line(
    path: &str,
    line_number: usize,
    line_text: impl Into<String>,
) -> ObservedLine {
    ObservedLine {
        path: path.to_string(),
        line_number,
        line_text: line_text.into(),
    }
}

pub(super) fn read_result_lines_matching(
    result: &ToolResult,
    matcher: impl Fn(&str) -> bool,
    limit: usize,
) -> Vec<ObservedLine> {
    let Some((path, content)) = parse_read_file_output(&result.output) else {
        return Vec::new();
    };
    if !is_source_path(&path)
        || is_test_like_path(&path)
        || is_internal_tool_loop_path(&path)
        || is_legacy_auto_inspect_path(&path)
    {
        return Vec::new();
    }

    filter_non_test_hits(&content, first_matching_lines(&content, matcher, limit))
        .into_iter()
        .map(|(line_number, line)| observed_line(&path, line_number, line))
        .collect()
}

pub(super) fn first_matching_lines(
    content: &str,
    matcher: impl Fn(&str) -> bool,
    limit: usize,
) -> Vec<(usize, String)> {
    content
        .lines()
        .enumerate()
        .map(|(idx, line)| (idx + 1, line.trim()))
        .filter(|(_, line)| !line.is_empty() && matcher(line))
        .take(limit)
        .map(|(line_number, line)| (line_number, line.to_string()))
        .collect()
}

fn defined_symbol_name(line: &str) -> Option<String> {
    let trimmed = line.trim();
    for prefix in [
        "pub fn ",
        "fn ",
        "pub struct ",
        "struct ",
        "pub enum ",
        "enum ",
    ] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            return rest
                .split(|ch: char| ch == '(' || ch == '<' || ch.is_whitespace() || ch == '{')
                .find(|part| !part.trim().is_empty())
                .map(|name| name.trim().to_string());
        }
    }
    None
}

fn push_unique_excerpt_lines(
    excerpt: &mut Vec<(usize, String)>,
    lines: impl IntoIterator<Item = (usize, String)>,
    limit: usize,
) {
    for (line_number, line) in lines {
        if excerpt.len() >= limit {
            break;
        }
        if excerpt
            .iter()
            .any(|(existing_line_number, _)| *existing_line_number == line_number)
        {
            continue;
        }
        excerpt.push((line_number, line));
    }
}

fn anchored_file_summary_excerpt(content: &str) -> Vec<(usize, String)> {
    let mut excerpt = Vec::new();
    let limit = 24;
    push_unique_excerpt_lines(
        &mut excerpt,
        declaration_lines_with_numbers(content, 14),
        limit,
    );

    if let Some((line_number, line)) = first_matching_lines(
        content,
        |line| line.starts_with("struct Cli") || line.starts_with("pub struct Cli"),
        1,
    )
    .into_iter()
    .next()
    {
        push_unique_excerpt_lines(
            &mut excerpt,
            std::iter::once((line_number, line.clone())),
            limit,
        );
        push_unique_excerpt_lines(
            &mut excerpt,
            surrounding_body_lines(content, line_number, 6),
            limit,
        );
    }

    if let Some((line_number, line)) = first_matching_lines(
        content,
        |line| line.starts_with("enum Command") || line.starts_with("pub enum Command"),
        1,
    )
    .into_iter()
    .next()
    {
        push_unique_excerpt_lines(
            &mut excerpt,
            std::iter::once((line_number, line.clone())),
            limit,
        );
        push_unique_excerpt_lines(
            &mut excerpt,
            surrounding_body_lines(content, line_number, 8),
            limit,
        );
    }

    if let Some((line_number, line)) = first_matching_lines(
        content,
        |line| line.starts_with("fn main(") || line.starts_with("pub fn main("),
        1,
    )
    .into_iter()
    .next()
    {
        push_unique_excerpt_lines(
            &mut excerpt,
            std::iter::once((line_number, line.clone())),
            limit,
        );
        push_unique_excerpt_lines(
            &mut excerpt,
            surrounding_body_lines(content, line_number, 12),
            limit,
        );
    }

    excerpt.sort_by_key(|(line_number, _)| *line_number);
    excerpt
}

pub(super) fn observed_reference_lines_from_read_results(
    intent: ToolLoopIntent,
    prompt: &str,
    results: &[ToolResult],
) -> Vec<ObservedLine> {
    let query = suggested_search_query(prompt, intent).unwrap_or_default();
    let is_call_site = matches!(intent, ToolLoopIntent::CallSiteLookup);

    results
        .iter()
        .filter(|result| result.tool_name == "read_file")
        .map(|result| {
            read_result_lines_matching(
                result,
                |line| {
                    if is_call_site {
                        line_contains_symbol_invocation(line, &query)
                            && !is_definition_like_line(line)
                    } else {
                        line_contains_symbol_reference(line, &query)
                            && !is_definition_like_line(line)
                    }
                },
                4,
            )
        })
        .flatten()
        .collect()
}

pub(super) fn observed_definition_evidence(
    intent: ToolLoopIntent,
    prompt: &str,
    results: &[ToolResult],
) -> Option<ImplementationEvidence> {
    let query = suggested_search_query(prompt, intent).unwrap_or_default();
    let read_results = results
        .iter()
        .filter(|result| result.tool_name == "read_file")
        .filter_map(|result| {
            let (path, content) = parse_read_file_output(&result.output)?;
            if !preferred_candidate_path(intent, &path)
                || is_internal_tool_loop_path(&path)
                || is_legacy_auto_inspect_path(&path)
            {
                return None;
            }
            Some((path, content))
        })
        .collect::<Vec<_>>();

    let ranked_paths = ranked_search_candidates(intent, prompt, results)
        .into_iter()
        .map(|file| file.path)
        .collect::<Vec<_>>();

    let ordered_reads = ranked_paths
        .iter()
        .filter_map(|path| {
            read_results
                .iter()
                .find(|(candidate_path, _)| candidate_path == path)
                .cloned()
        })
        .chain(read_results.iter().cloned())
        .collect::<Vec<_>>();

    for (path, content) in ordered_reads {
        let primary = filter_non_test_hits(
            &content,
            definition_match_lines_with_numbers(&content, &query, 1),
        )
        .into_iter()
        .next()
        .or_else(|| {
            if matches!(intent, ToolLoopIntent::ConfigLocate) && !query.is_empty() {
                query_match_lines_with_numbers(&content, &query, 1)
                    .into_iter()
                    .next()
            } else {
                None
            }
        })
        .or_else(|| {
            first_matching_lines(&content, is_definition_like_line, 1)
                .into_iter()
                .next()
        });
        let Some((line_number, line_text)) = primary else {
            continue;
        };
        let supporting = surrounding_body_lines(&content, line_number, 4)
            .into_iter()
            .map(|(ln, line)| observed_line(&path, ln, line))
            .collect::<Vec<_>>();
        return Some(ImplementationEvidence {
            primary: observed_line(&path, line_number, line_text),
            supporting,
        });
    }
    None
}

pub(super) fn observed_file_summary_evidence(
    prompt: &str,
    resolution: Option<&InvestigationResolution>,
    results: &[ToolResult],
) -> Option<FileSummaryEvidence> {
    if !should_answer_from_anchor_file(prompt, resolution) {
        return None;
    }

    let anchored = resolution.and_then(|resolution| resolution.anchored_file.as_deref());
    let target = results
        .iter()
        .filter(|result| result.tool_name == "read_file")
        .find(|result| anchored.map(|path| path == result.argument).unwrap_or(true))?;
    let (path, content) = parse_read_file_output(&target.output)?;
    let excerpt = {
        let structured = anchored_file_summary_excerpt(&content);
        if structured.is_empty() {
            first_non_empty_lines(&content, 10)
        } else {
            structured
        }
    };
    if excerpt.is_empty() {
        return None;
    }

    Some(FileSummaryEvidence {
        path: path.clone(),
        declarations: excerpt
            .into_iter()
            .map(|(line_number, line)| observed_line(&path, line_number, line))
            .collect(),
    })
}

pub(super) fn observed_repo_overview_evidence(
    results: &[ToolResult],
) -> Option<RepoOverviewEvidence> {
    let mut package_line = None;
    let mut readme_line = None;
    let mut entrypoint_line = None;
    let mut subsystem_lines = Vec::new();

    for result in results
        .iter()
        .filter(|result| result.tool_name == "read_file")
    {
        let Some((path, content)) = parse_read_file_output(&result.output) else {
            continue;
        };
        match path.as_str() {
            "Cargo.toml" if package_line.is_none() => {
                package_line =
                    first_matching_lines(&content, |line| line.starts_with("name = "), 1)
                        .into_iter()
                        .next()
                        .map(|(line_number, line)| observed_line(&path, line_number, line));
            }
            "README.md" if readme_line.is_none() => {
                readme_line = first_non_empty_lines(&content, 1)
                    .into_iter()
                    .next()
                    .map(|(line_number, line)| observed_line(&path, line_number, line));
            }
            "src/main.rs" | "src/lib.rs" => {
                if entrypoint_line.is_none() {
                    entrypoint_line = first_matching_lines(
                        &content,
                        |line| line.starts_with("fn main(") || line.starts_with("pub fn main("),
                        1,
                    )
                    .into_iter()
                    .next()
                    .map(|(line_number, line)| observed_line(&path, line_number, line));
                }
                if subsystem_lines.is_empty() {
                    subsystem_lines = declaration_lines_with_numbers(&content, 8)
                        .into_iter()
                        .filter(|(_, line)| {
                            line.starts_with("mod ")
                                || line.starts_with("pub mod ")
                                || line.starts_with("use ")
                        })
                        .map(|(line_number, line)| observed_line(&path, line_number, line))
                        .take(6)
                        .collect();
                }
            }
            _ => {}
        }
    }

    if package_line.is_none()
        && readme_line.is_none()
        && entrypoint_line.is_none()
        && subsystem_lines.is_empty()
    {
        return None;
    }

    Some(RepoOverviewEvidence {
        package_line,
        readme_line,
        entrypoint_line,
        subsystem_lines,
    })
}

fn classify_flow_step(line_text: &str) -> ObservedStepKind {
    let trimmed = line_text.trim();
    if trimmed.contains("return ") {
        ObservedStepKind::Return
    } else if trimmed.contains("=>")
        || trimmed.starts_with("if ")
        || trimmed.starts_with("match ")
        || trimmed.starts_with("Some(")
        || trimmed.starts_with("None")
        || trimmed.starts_with("Ok(")
    {
        ObservedStepKind::Branch
    } else if trimmed.contains("load_")
        || trimmed.contains("save_")
        || trimmed.contains('.')
        || trimmed.contains("::")
    {
        ObservedStepKind::Delegation
    } else if trimmed.contains("else") {
        ObservedStepKind::Branch
    } else {
        ObservedStepKind::Definition
    }
}

pub(super) fn observed_flow_trace_evidence(
    prompt: &str,
    results: &[ToolResult],
) -> Option<FlowTraceEvidence> {
    let query = suggested_search_query(prompt, ToolLoopIntent::FlowTrace).unwrap_or_default();
    let implementation = observed_definition_evidence(ToolLoopIntent::FlowTrace, prompt, results)?;
    let defined_symbol = defined_symbol_name(&implementation.primary.line_text);
    let caller =
        observed_reference_lines_from_read_results(ToolLoopIntent::CallSiteLookup, prompt, results)
            .into_iter()
            .find(|line| {
                !is_internal_tool_loop_path(&line.path) && !is_legacy_auto_inspect_path(&line.path)
            })
            .or_else(|| {
                results
                    .iter()
                    .filter(|result| result.tool_name == "read_file")
                    .flat_map(|result| {
                        read_result_lines_matching(
                            result,
                            |line| {
                                line_contains_symbol_reference(line, &query)
                                    && !is_definition_like_line(line)
                            },
                            3,
                        )
                    })
                    .next()
            })
            .or_else(|| {
                let symbol = defined_symbol.as_deref()?;
                results
                    .iter()
                    .filter(|result| result.tool_name == "read_file")
                    .flat_map(|result| {
                        read_result_lines_matching(
                            result,
                            |line| {
                                line_contains_symbol_invocation(line, symbol)
                                    && !is_definition_like_line(line)
                            },
                            2,
                        )
                    })
                    .next()
            });
    let caller = caller?;
    if caller.path == implementation.primary.path {
        return None;
    }

    let mut steps = Vec::new();
    steps.push(ObservedStep {
        path: caller.path.clone(),
        line_number: caller.line_number,
        line_text: caller.line_text.clone(),
        step_kind: ObservedStepKind::EntryCall,
    });
    if let Some((_, caller_content)) = results
        .iter()
        .filter(|result| result.tool_name == "read_file")
        .filter_map(|result| parse_read_file_output(&result.output))
        .find(|(path, _)| path == &caller.path)
    {
        steps.extend(
            surrounding_body_lines(&caller_content, caller.line_number, 5)
                .into_iter()
                .filter(|(_, line)| !line.trim().is_empty())
                .map(|(line_number, line_text)| ObservedStep {
                    path: caller.path.clone(),
                    line_number,
                    line_text: line_text.clone(),
                    step_kind: classify_flow_step(&line_text),
                }),
        );
    }

    steps.push(ObservedStep {
        path: implementation.primary.path.clone(),
        line_number: implementation.primary.line_number,
        line_text: implementation.primary.line_text.clone(),
        step_kind: ObservedStepKind::Definition,
    });
    steps.extend(
        implementation
            .supporting
            .into_iter()
            .filter(|line| !line.line_text.trim().is_empty())
            .map(|line| ObservedStep {
                path: line.path,
                line_number: line.line_number,
                line_text: line.line_text.clone(),
                step_kind: classify_flow_step(&line.line_text),
            }),
    );
    let has_cross_file_handoff = steps.iter().any(|step| {
        step.path != caller.path
            && !is_test_like_path(&step.path)
            && !is_internal_tool_loop_path(&step.path)
            && !is_legacy_auto_inspect_path(&step.path)
    });

    if !has_cross_file_handoff {
        return None;
    }

    Some(FlowTraceEvidence {
        subject: if query.is_empty() {
            normalize_intent_text(prompt)
        } else {
            query
        },
        steps,
    })
}
