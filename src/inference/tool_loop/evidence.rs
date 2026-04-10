#[path = "parse.rs"]
mod parse;

use std::collections::HashMap;
use std::sync::mpsc::Sender;

use crate::events::{InferenceEvent, ProgressStatus};
use crate::tools::{ToolRegistry, ToolResult};

use super::super::runtime::emit_trace;
use super::intent::{normalize_intent_text, suggested_search_query, ToolLoopIntent};
use parse::{
    clip_inline, clip_tool_output, compact_read_file_result, definition_match_lines_with_numbers,
    filter_non_test_hits, is_config_path, is_definition_like_line, is_doc_path, is_source_path,
    is_test_like_path, parse_read_file_output, parse_search_output, surrounding_body_lines,
    SearchFileHit, SearchLineHit,
};

pub(super) fn format_tool_loop_results_with_limit(
    intent: ToolLoopIntent,
    prompt: &str,
    results: &[ToolResult],
    max_chars_per_result: Option<usize>,
) -> Option<String> {
    if results.is_empty() {
        return None;
    }

    let mut msg = String::from("Tool results:\n\n");
    for result in results {
        let output = if result.tool_name == "read_file" {
            compact_read_file_result(intent, prompt, result, max_chars_per_result)
                .unwrap_or_else(|| clip_tool_output(&result.output, max_chars_per_result))
        } else {
            clip_tool_output(&result.output, max_chars_per_result)
        };
        msg.push_str(&format!(
            "--- {}({}) ---\n{}\n\n",
            result.tool_name, result.argument, output
        ));
    }

    Some(msg)
}

pub(super) fn grounded_answer_guidance(
    intent: ToolLoopIntent,
    prompt: &str,
    results: &[ToolResult],
) -> Option<String> {
    let query = suggested_search_query(prompt, intent)?;
    match intent {
        ToolLoopIntent::CodeNavigation => {
            for result in results
                .iter()
                .filter(|result| result.tool_name == "read_file")
            {
                let Some((path, content)) = parse_read_file_output(&result.output) else {
                    continue;
                };
                let matches = filter_non_test_hits(
                    &content,
                    definition_match_lines_with_numbers(&content, &query, 1),
                );
                let Some((line_number, line)) = matches.into_iter().next() else {
                    continue;
                };
                let body_lines = surrounding_body_lines(&content, line_number, 4);
                let mut sections = vec![
                    "Grounded answer requirements: answer only from the inspected file evidence below. Do not include code fences. Do not quote full function bodies or paste large snippets. Do not invent placeholder code, omitted implementation comments, or extra body lines that are not present. Do not mention use-sites, tests, or search-only hits unless the user asked for them.".to_string(),
                    format!(
                        "Primary implementation: {}:{} `{}`",
                        path,
                        line_number,
                        clip_inline(&line, 120)
                    ),
                ];
                if !body_lines.is_empty() {
                    sections.push(
                        "Observed body lines (each line is listed separately \
                         with its exact line number):"
                            .to_string(),
                    );
                    for (ln, line_text) in &body_lines {
                        sections.push(format!(
                            "  {}:{} `{}`",
                            path,
                            ln,
                            clip_inline(line_text, 120)
                        ));
                    }
                }
                sections.push(
                    "Answer from the observed lines above only. Rules:\n\
                     1. Cite every fact with its exact file:line matching the lines listed above.\n\
                     2. Copy all method names, variable names, and expressions verbatim \
                        from the observed lines — do not rename or substitute them.\n\
                     3. Describe only what is visible in each specific observed line. \
                        Do not conflate separate lines or attribute behavior to the wrong line.\n\
                     4. Do not use hedging words (`presumably`, `likely`, `suggests`, \
                        `appears to`, `seems to`, `may`).\n\
                     5. One concrete sentence per key observed line."
                        .to_string(),
                );
                return Some(sections.join("\n"));
            }
            None
        }
        ToolLoopIntent::CallSiteLookup | ToolLoopIntent::UsageLookup => {
            let search_hits = merge_search_hits(results);
            let is_call_site = matches!(intent, ToolLoopIntent::CallSiteLookup);
            let mut sites: Vec<String> = Vec::new();
            for file_hit in &search_hits {
                if is_test_like_path(&file_hit.path) {
                    continue;
                }
                for hit in file_hit.hits.iter().take(4) {
                    let line = hit.line_content.trim();
                    let qualifies = if is_call_site {
                        line.contains('(') && !is_definition_like_line(line)
                    } else {
                        !is_definition_like_line(line)
                    };
                    if qualifies {
                        sites.push(format!(
                            "  {}:{} `{}`",
                            file_hit.path,
                            hit.line_number,
                            clip_inline(line, 120)
                        ));
                    }
                }
            }
            if sites.is_empty() {
                return None;
            }
            let mode = if is_call_site { "call-site" } else { "usage" };
            let rule = if is_call_site {
                "List each file:line where the symbol is invoked. Do NOT describe the symbol's own implementation."
            } else {
                "List each file:line where the symbol is used or referenced. Do NOT describe the symbol's own implementation."
            };
            let mut sections = vec![
                format!(
                    "Grounded answer requirements: list the {mode}s found in the observed evidence. \
                     Do not describe the symbol's implementation. Do not include code fences."
                ),
                format!("Observed {mode}s:"),
            ];
            sections.extend(sites);
            sections.push(format!(
                "Answer from the observed {mode}s above only. Rules:\n\
                 1. {rule}\n\
                 2. Cite every entry with its exact file:line from the list above.\n\
                 3. Do not invent additional {mode}s not observed.\n\
                 4. Do not use hedging words (`presumably`, `likely`, `suggests`, `appears to`, `seems to`, `may`)."
            ));
            Some(sections.join("\n"))
        }
        ToolLoopIntent::FlowTrace => {
            let mut file_sections: Vec<String> = Vec::new();
            for result in results.iter().filter(|r| r.tool_name == "read_file") {
                let Some((path, content)) = parse_read_file_output(&result.output) else {
                    continue;
                };
                let query_hits = filter_non_test_hits(
                    &content,
                    definition_match_lines_with_numbers(&content, &query, 1),
                );
                let fallback_def: Vec<(usize, String)> = content
                    .lines()
                    .enumerate()
                    .map(|(idx, line)| (idx + 1, line.trim().to_string()))
                    .filter(|(_, line)| is_definition_like_line(line) && !line.is_empty())
                    .take(1)
                    .collect();
                let hits = if !query_hits.is_empty() {
                    query_hits
                } else {
                    fallback_def
                };
                if let Some((line_number, line)) = hits.into_iter().next() {
                    file_sections.push(format!(
                        "{}:{} `{}`",
                        path,
                        line_number,
                        clip_inline(&line, 120)
                    ));
                    let body = surrounding_body_lines(&content, line_number, 4);
                    for (ln, line_text) in &body {
                        file_sections.push(format!(
                            "  {}:{} `{}`",
                            path,
                            ln,
                            clip_inline(line_text, 120)
                        ));
                    }
                }
            }
            if file_sections.is_empty() {
                return None;
            }
            let mut sections = vec![
                "Grounded answer requirements: describe the execution flow using only the observed file evidence. \
                 Do not include code fences. Describe the sequence as concrete ordered steps."
                    .to_string(),
                "Observed definitions across files:".to_string(),
            ];
            sections.extend(file_sections);
            sections.push(
                "Answer from the observed evidence above. Rules:\n\
                 1. Describe the flow as a concrete ordered sequence of steps.\n\
                 2. Cite file:line for each step.\n\
                 3. Do not invent steps not visible in the observed evidence.\n\
                 4. Do not use hedging words (`presumably`, `likely`, `suggests`, `appears to`, `seems to`, `may`)."
                    .to_string(),
            );
            Some(sections.join("\n"))
        }
        ToolLoopIntent::ConfigLocate
        | ToolLoopIntent::RepoOverview
        | ToolLoopIntent::DirectoryOverview => None,
    }
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
        ToolLoopIntent::FlowTrace
        | ToolLoopIntent::CallSiteLookup
        | ToolLoopIntent::UsageLookup => {
            is_source_path(path) && !is_doc_path(path) && !is_test_like_path(path)
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

fn observed_read_paths(results: &[ToolResult]) -> std::collections::HashSet<String> {
    results
        .iter()
        .filter(|result| result.tool_name == "read_file")
        .map(|result| result.argument.clone())
        .collect()
}

pub(super) fn has_relevant_file_evidence(
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
        ToolLoopIntent::CallSiteLookup | ToolLoopIntent::UsageLookup => {
            if results
                .iter()
                .any(|r| matches!(r.tool_name.as_str(), "lsp_definition" | "lsp_hover"))
            {
                return true;
            }
            let has_search = results.iter().any(|r| r.tool_name == "search");
            let read_paths = observed_read_paths(results);
            let has_source_read = read_paths.iter().any(|path| is_source_path(path));
            has_search && has_source_read
        }
        ToolLoopIntent::FlowTrace => {
            if results
                .iter()
                .any(|r| matches!(r.tool_name.as_str(), "lsp_definition" | "lsp_hover"))
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

pub(super) fn targeted_investigation_followup(
    intent: ToolLoopIntent,
    prompt: &str,
    results: &[ToolResult],
) -> Option<String> {
    match intent {
        ToolLoopIntent::CallSiteLookup | ToolLoopIntent::UsageLookup => {
            let read_paths = observed_read_paths(results);
            let is_call_site = matches!(intent, ToolLoopIntent::CallSiteLookup);
            let search_hits = merge_search_hits(results);
            let candidate = search_hits
                .into_iter()
                .filter(|file| {
                    !read_paths.contains(&file.path)
                        && is_source_path(&file.path)
                        && !is_test_like_path(&file.path)
                        && file.hits.iter().any(|hit| {
                            let line = hit.line_content.trim();
                            if is_call_site {
                                line.contains('(') && !is_definition_like_line(line)
                            } else {
                                !is_definition_like_line(line)
                            }
                        })
                })
                .next();
            if let Some(file) = candidate {
                let mode = if is_call_site { "call-site" } else { "usage" };
                let anchor = file
                    .hits
                    .iter()
                    .find(|hit| {
                        let l = hit.line_content.trim();
                        if is_call_site {
                            l.contains('(') && !is_definition_like_line(l)
                        } else {
                            !is_definition_like_line(l)
                        }
                    })
                    .or_else(|| file.hits.first());
                let anchor_text =
                    anchor.map(|hit| format!("{}: {}", hit.line_number, hit.line_content));
                return Some(match anchor_text {
                    Some(anchor) => format!(
                        "Do not answer yet. This file contains a {mode}: `[read_file: {}]`. \
                         Strongest {mode} anchor: `{}`.",
                        file.path, anchor
                    ),
                    None => format!(
                        "Do not answer yet. Read this file for {mode}s: `[read_file: {}]`.",
                        file.path
                    ),
                });
            }
            let read_paths = observed_read_paths(results);
            let fallback = ranked_search_candidates(intent, prompt, results)
                .into_iter()
                .find(|file| {
                    !read_paths.contains(&file.path) && preferred_candidate_path(intent, &file.path)
                })?;
            Some(format!(
                "Do not answer yet. Read this source candidate for {}: `[read_file: {}]`.",
                if is_call_site { "call-sites" } else { "usages" },
                fallback.path
            ))
        }
        ToolLoopIntent::FlowTrace => {
            let read_paths = observed_read_paths(results);
            let candidate = ranked_search_candidates(intent, prompt, results)
                .into_iter()
                .find(|file| {
                    !read_paths.contains(&file.path) && preferred_candidate_path(intent, &file.path)
                })?;
            let anchor = candidate
                .hits
                .iter()
                .find(|hit| is_definition_like_line(&hit.line_content))
                .or_else(|| candidate.hits.first());
            let anchor_text =
                anchor.map(|hit| format!("{}: {}", hit.line_number, hit.line_content));
            let body =
                "Do not answer yet — tracing the flow requires reading at least one more related file.";
            Some(match anchor_text {
                Some(anchor) => format!(
                    "{body} Read this related file next: `[read_file: {}]`. Evidence anchor: `{}`.",
                    candidate.path, anchor
                ),
                None => format!(
                    "{body} Read this related file next: `[read_file: {}]`.",
                    candidate.path
                ),
            })
        }
        ToolLoopIntent::CodeNavigation => {
            let read_paths = observed_read_paths(results);
            let candidate = ranked_search_candidates(intent, prompt, results)
                .into_iter()
                .find(|file| !read_paths.contains(&file.path))?;
            let anchor = candidate
                .hits
                .iter()
                .find(|hit| is_definition_like_line(&hit.line_content))
                .or_else(|| candidate.hits.first());
            let anchor_text =
                anchor.map(|hit| format!("{}: {}", hit.line_number, hit.line_content));
            let body = "Do not answer yet. Read this source candidate next and answer from the inspected implementation, not from docs, tests, prompt strings, or call-sites.";
            Some(match anchor_text {
                Some(anchor) => format!(
                    "{body} Next read: `[read_file: {}]`. Strongest search anchor: `{}`.",
                    candidate.path, anchor
                ),
                None => format!("{body} Next read: `[read_file: {}]`.", candidate.path),
            })
        }
        ToolLoopIntent::ConfigLocate => {
            let read_paths = observed_read_paths(results);
            let candidate = ranked_search_candidates(intent, prompt, results)
                .into_iter()
                .find(|file| !read_paths.contains(&file.path))?;
            let anchor = candidate
                .hits
                .iter()
                .find(|hit| is_definition_like_line(&hit.line_content))
                .or_else(|| candidate.hits.first());
            let anchor_text =
                anchor.map(|hit| format!("{}: {}", hit.line_number, hit.line_content));
            let body = "Do not answer yet. Read this config/source candidate next and answer from the inspected setting lines.";
            Some(match anchor_text {
                Some(anchor) => format!(
                    "{body} Next read: `[read_file: {}]`. Strongest search anchor: `{}`.",
                    candidate.path, anchor
                ),
                None => format!("{body} Next read: `[read_file: {}]`.", candidate.path),
            })
        }
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview => None,
    }
}

fn auto_read_best_candidate(
    intent: ToolLoopIntent,
    prompt: &str,
    tools: &ToolRegistry,
    existing_results: &[ToolResult],
    token_tx: &Sender<InferenceEvent>,
) -> Option<ToolResult> {
    let read_paths = observed_read_paths(existing_results);
    let candidate = ranked_search_candidates(intent, prompt, existing_results)
        .into_iter()
        .find(|file| !read_paths.contains(&file.path))?;

    emit_trace(
        token_tx,
        ProgressStatus::Started,
        &format!("reading top candidate {}...", candidate.path),
        false,
    );
    let execution = tools.execute_read_only_tool_calls(&format!("[read_file: {}]", candidate.path));
    let result = execution.results.into_iter().next()?;
    emit_trace(
        token_tx,
        ProgressStatus::Finished,
        &format!("read_file {}", result.argument),
        false,
    );
    Some(result)
}

pub(super) fn bootstrap_tool_results(
    intent: ToolLoopIntent,
    prompt: &str,
    backend_name: &str,
    tools: &ToolRegistry,
    token_tx: &Sender<InferenceEvent>,
) -> Vec<ToolResult> {
    if !backend_name.contains("llama.cpp") {
        return Vec::new();
    }
    if !matches!(
        intent,
        ToolLoopIntent::CodeNavigation
            | ToolLoopIntent::ConfigLocate
            | ToolLoopIntent::CallSiteLookup
            | ToolLoopIntent::UsageLookup
            | ToolLoopIntent::FlowTrace
    ) {
        return Vec::new();
    }

    let Some(query) = suggested_search_query(prompt, intent) else {
        return Vec::new();
    };

    emit_trace(
        token_tx,
        ProgressStatus::Started,
        &format!("bootstrapping search {}...", query),
        false,
    );
    let execution = tools.execute_read_only_tool_calls(&format!("[search: {query}]"));
    let mut results = execution.results;
    if let Some(search_result) = results.first() {
        emit_trace(
            token_tx,
            ProgressStatus::Finished,
            &format!("search {}", search_result.argument),
            false,
        );
    }

    if matches!(
        intent,
        ToolLoopIntent::CodeNavigation | ToolLoopIntent::ConfigLocate | ToolLoopIntent::FlowTrace
    ) && !has_relevant_file_evidence(intent, prompt, &results)
    {
        if let Some(read_result) =
            auto_read_best_candidate(intent, prompt, tools, &results, token_tx)
        {
            results.push(read_result);
        }
    }

    results
}
