#[path = "parse.rs"]
mod parse;

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::mpsc::Sender;

use crate::events::{InferenceEvent, ProgressStatus};
use crate::tools::{ToolRegistry, ToolResult};

use super::super::runtime::emit_trace;
use super::super::session::investigation::InvestigationResolution;
use super::super::Message;
use super::intent::{
    is_referential_file_prompt, normalize_intent_text, suggested_search_query, ToolLoopIntent,
};
use parse::{
    clip_inline, clip_tool_output, compact_read_file_result, declaration_lines_with_numbers,
    definition_match_lines_with_numbers, filter_non_test_hits, first_non_empty_lines,
    is_config_path, is_definition_like_line, is_doc_path, is_internal_tool_loop_path,
    is_legacy_auto_inspect_path, is_source_path, is_test_like_path,
    line_contains_symbol_invocation, line_contains_symbol_reference, parse_read_file_output,
    parse_search_output, prompt_mentions_tests, query_match_lines_with_numbers,
    surrounding_body_lines, SearchFileHit, SearchLineHit,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ObservedLine {
    pub(super) path: String,
    pub(super) line_number: usize,
    pub(super) line_text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ObservedStepKind {
    EntryCall,
    Definition,
    Branch,
    Return,
    Delegation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ObservedStep {
    pub(super) path: String,
    pub(super) line_number: usize,
    pub(super) line_text: String,
    pub(super) step_kind: ObservedStepKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RepoOverviewEvidence {
    pub(super) package_line: Option<ObservedLine>,
    pub(super) readme_line: Option<ObservedLine>,
    pub(super) entrypoint_line: Option<ObservedLine>,
    pub(super) subsystem_lines: Vec<ObservedLine>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct FileSummaryEvidence {
    pub(super) path: String,
    pub(super) declarations: Vec<ObservedLine>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ImplementationEvidence {
    pub(super) primary: ObservedLine,
    pub(super) supporting: Vec<ObservedLine>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ConfigEvidence {
    pub(super) lines: Vec<ObservedLine>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CallSiteEvidence {
    pub(super) symbol: String,
    pub(super) sites: Vec<ObservedLine>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct UsageEvidence {
    pub(super) symbol: String,
    pub(super) usages: Vec<ObservedLine>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct FlowTraceEvidence {
    pub(super) subject: String,
    pub(super) steps: Vec<ObservedStep>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum StructuredEvidence {
    RepoOverview(RepoOverviewEvidence),
    FileSummary(FileSummaryEvidence),
    Implementation(ImplementationEvidence),
    Config(ConfigEvidence),
    CallSites(CallSiteEvidence),
    Usages(UsageEvidence),
    FlowTrace(FlowTraceEvidence),
}

pub(super) enum InvestigationOutcome {
    NeedsMore {
        required_next_step: String,
    },
    Ready {
        evidence: StructuredEvidence,
        stop_reason: &'static str,
    },
    Insufficient {
        reason: String,
    },
}

pub(super) type InvestigationReadiness = InvestigationOutcome;

pub(super) fn format_tool_loop_results_with_limit(
    intent: ToolLoopIntent,
    prompt: &str,
    resolution: Option<&InvestigationResolution>,
    results: &[ToolResult],
    max_chars_per_result: Option<usize>,
) -> Option<String> {
    if results.is_empty() {
        return None;
    }

    let mut msg = String::from("Tool results:\n\n");
    for result in results {
        let output = if result.tool_name == "read_file" {
            compact_read_file_result(intent, prompt, resolution, result, max_chars_per_result)
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
    resolution: Option<&InvestigationResolution>,
    results: &[ToolResult],
) -> Option<String> {
    match intent {
        ToolLoopIntent::CodeNavigation => {
            if should_answer_from_anchor_file(prompt, resolution) {
                for result in results
                    .iter()
                    .filter(|result| result.tool_name == "read_file")
                {
                    let Some((path, content)) = parse_read_file_output(&result.output) else {
                        continue;
                    };
                    // Use a higher limit so fn main and other key declarations
                    // appear even in files with many leading mod lines.
                    let mut declarations = declaration_lines_with_numbers(&content, 14);
                    let has_main = declarations
                        .iter()
                        .any(|(_, l)| l.starts_with("fn main") || l.starts_with("pub fn main"));
                    if !has_main {
                        if let Some(main_line) = first_matching_lines(
                            &content,
                            |l| l.starts_with("fn main(") || l.starts_with("pub fn main("),
                            1,
                        )
                        .into_iter()
                        .next()
                        {
                            declarations.push(main_line);
                        }
                    }
                    let excerpt = if declarations.is_empty() {
                        first_non_empty_lines(&content, 10)
                    } else {
                        declarations
                    };
                    if excerpt.is_empty() {
                        continue;
                    }
                    let mut sections = vec![
                        "Grounded answer requirements: answer what this loaded file does using only the observed lines below. Do not include code fences. Do not mention search results, wrapper prompt text, or unrelated prompt strings. Do not suggest inspecting another file unless the observed lines are genuinely insufficient.".to_string(),
                        format!("Loaded file: `{path}`"),
                        "Observed declarations:".to_string(),
                    ];
                    sections.extend(excerpt.into_iter().map(|(line_number, line)| {
                        format!("  {}:{} `{}`", path, line_number, clip_inline(&line, 120))
                    }));
                    sections.push(
                        "Answer from the observed lines above only. Rules:\n\
                         1. Keep the answer to 2-4 short sentences or a flat 3-bullet list.\n\
                         2. If `fn main` is visible, mention that this is the binary entrypoint and describe its role.\n\
                         3. Summarize the file's role from the declarations/imports/modules actually shown.\n\
                         4. Cite every concrete fact with exact file:line references.\n\
                         5. Copy identifiers verbatim — do not rename methods, modules, or types.\n\
                         6. Do not use hedging words (`presumably`, `likely`, `suggests`, `appears to`, `seems to`, `may`).\n\
                         7. Do not pivot to other files or offer next-step advice unless the observed lines clearly delegate elsewhere."
                            .to_string(),
                    );
                    return Some(sections.join("\n"));
                }
                return None;
            }

            let query = suggested_search_query(prompt, intent)?;
            for result in results
                .iter()
                .filter(|result| result.tool_name == "read_file")
            {
                let Some((path, content)) = parse_read_file_output(&result.output) else {
                    continue;
                };
                if is_test_like_path(&path) && !prompt_mentions_tests(prompt) {
                    continue;
                }
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
            let is_call_site = matches!(intent, ToolLoopIntent::CallSiteLookup);
            let sites = observed_reference_lines_from_read_results(intent, prompt, results)
                .into_iter()
                .map(|line| {
                    format!(
                        "  {}:{} `{}`",
                        line.path,
                        line.line_number,
                        clip_inline(&line.line_text, 120)
                    )
                })
                .collect::<Vec<_>>();
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
                 4. Keep the answer concise: one short sentence or one flat bullet per observed file.\n\
                 5. Do not use hedging words (`presumably`, `likely`, `suggests`, `appears to`, `seems to`, `may`)."
            ));
            Some(sections.join("\n"))
        }
        ToolLoopIntent::FlowTrace => {
            let query = suggested_search_query(prompt, intent)?;
            let mut file_sections: Vec<String> = Vec::new();
            for result in results.iter().filter(|r| r.tool_name == "read_file") {
                let Some((path, content)) = parse_read_file_output(&result.output) else {
                    continue;
                };
                if is_test_like_path(&path) && !prompt_mentions_tests(prompt) {
                    continue;
                }
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
                "Grounded answer requirements: explain the execution flow in plain language using \
                 only the observed file evidence below. Do not list raw code lines. \
                 Write a short natural-language explanation of what happens, in order, \
                 with file:line citations. Do not include code fences."
                    .to_string(),
                "Observed cross-file evidence:".to_string(),
            ];
            sections.extend(file_sections);
            sections.push(
                "Answer from the observed evidence above. Rules:\n\
                 1. Write a SHORT explanation in plain language (2–4 sentences), not a code dump.\n\
                 2. Describe what each key step does and where it lives (file:line).\n\
                 3. Mention branch behavior (early return, None path) if visible.\n\
                 4. Do not copy raw source lines verbatim into the answer — paraphrase them.\n\
                 5. Do not use hedging words (`presumably`, `likely`, `suggests`, `appears to`, `seems to`, `may`)."
                    .to_string(),
            );
            Some(sections.join("\n"))
        }
        ToolLoopIntent::ConfigLocate => {
            let query = suggested_search_query(prompt, intent)?;
            // Try each read file; prefer the one with query-matching lines.
            for result in results.iter().filter(|r| r.tool_name == "read_file") {
                let Some((path, content)) = parse_read_file_output(&result.output) else {
                    continue;
                };
                if is_test_like_path(&path) && !prompt_mentions_tests(prompt) {
                    continue;
                }
                // Find the most relevant line — a definition match or any query match.
                let best = filter_non_test_hits(
                    &content,
                    definition_match_lines_with_numbers(&content, &query, 1),
                )
                .into_iter()
                .next()
                .or_else(|| {
                    query_match_lines_with_numbers(&content, &query, 1)
                        .into_iter()
                        .next()
                });
                let Some((line_number, line)) = best else {
                    continue;
                };
                let body_lines = surrounding_body_lines(&content, line_number, 3);
                let mut sections = vec![
                    "Grounded answer requirements: identify where the config setting is defined, \
                     parsed, or applied from the observed evidence. \
                     Do not include code fences."
                        .to_string(),
                    format!(
                        "Primary evidence: {}:{} `{}`",
                        path,
                        line_number,
                        clip_inline(&line, 120)
                    ),
                ];
                if !body_lines.is_empty() {
                    sections.push("Surrounding lines:".to_string());
                    for (ln, lt) in &body_lines {
                        sections.push(format!("  {}:{} `{}`", path, ln, clip_inline(lt, 120)));
                    }
                }
                sections.push(
                    "Answer from the observed evidence only. Rules:\n\
                     1. State exactly where the setting is defined or used (file:line).\n\
                     2. Name the field, struct, or config key as it appears in the code.\n\
                     3. Keep the answer to 1–2 sentences.\n\
                     4. Do not use hedging words."
                        .to_string(),
                );
                return Some(sections.join("\n"));
            }
            None
        }
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview => {
            let directories = results
                .iter()
                .filter(|result| result.tool_name == "list_dir")
                .map(|result| clip_inline(&result.output.replace('\n', " | "), 180))
                .take(2)
                .collect::<Vec<_>>();
            let observed_files = results
                .iter()
                .filter(|result| result.tool_name == "read_file")
                .filter_map(|result| {
                    let (path, content) = parse_read_file_output(&result.output)?;
                    let declarations = declaration_lines_with_numbers(&content, 5);
                    let excerpt = if declarations.is_empty() {
                        first_non_empty_lines(&content, 5)
                    } else {
                        declarations
                    };
                    Some((path, excerpt))
                })
                .collect::<Vec<_>>();
            if directories.is_empty() && observed_files.is_empty() {
                return None;
            }
            let mut sections = vec![
                "Grounded answer requirements: summarize the project or directory using only the inspected structure and key file evidence below. Do not include code fences. Do not ask the user to provide files if the repo is already accessible.".to_string(),
            ];
            if !directories.is_empty() {
                sections.push("Observed structure:".to_string());
                for listing in directories {
                    sections.push(format!("  `{listing}`"));
                }
            }
            if !observed_files.is_empty() {
                sections.push("Observed key files:".to_string());
                for (path, excerpt) in observed_files {
                    sections.push(format!("  File: `{path}`"));
                    for (line_number, line) in excerpt {
                        sections.push(format!(
                            "    {}:{} `{}`",
                            path,
                            line_number,
                            clip_inline(&line, 120)
                        ));
                    }
                }
            }
            sections.push(
                "Answer from the observed structure and file lines above. Rules:\n\
                 1. Keep the answer to 2-4 short sentences or a flat 4-bullet list.\n\
                 2. Summarize what the project is, where startup/entrypoints are, and what the main subsystems appear to be from inspected evidence only.\n\
                 3. Cite concrete facts with file:line when available.\n\
                 4. Mention dependency versions only if the user asked about dependencies or they are essential to the runtime shape.\n\
                 5. Do not ask the user to provide files or guess missing paths.\n\
                 6. Do not use hedging words (`presumably`, `likely`, `suggests`, `appears to`, `seems to`, `may`)."
                    .to_string(),
            );
            Some(sections.join("\n"))
        }
    }
}

fn recent_loaded_file_context_path(base_messages: &[Message]) -> Option<String> {
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
            is_source_path(path)
                && !is_doc_path(path)
                && !is_test_like_path(path)
                && !is_internal_tool_loop_path(path)
                && !is_legacy_auto_inspect_path(path)
        }
        ToolLoopIntent::ConfigLocate => {
            (is_config_path(path) || is_source_path(path))
                && !is_doc_path(path)
                && !is_test_like_path(path)
                && !is_internal_tool_loop_path(path)
                && !is_legacy_auto_inspect_path(path)
        }
        ToolLoopIntent::FlowTrace
        | ToolLoopIntent::CallSiteLookup
        | ToolLoopIntent::UsageLookup => {
            is_source_path(path)
                && !is_doc_path(path)
                && !is_test_like_path(path)
                && !is_internal_tool_loop_path(path)
                && !is_legacy_auto_inspect_path(path)
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

fn should_answer_from_anchor_file(
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

fn observed_line(path: &str, line_number: usize, line_text: impl Into<String>) -> ObservedLine {
    ObservedLine {
        path: path.to_string(),
        line_number,
        line_text: line_text.into(),
    }
}

fn read_result_lines_matching(
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

fn first_matching_lines(
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

fn observed_reference_lines_from_read_results(
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

fn observed_definition_evidence(
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
        // For config lookup, also accept non-definition lines that contain the
        // query term (e.g. field declarations like `pub enabled: bool`).
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

fn observed_file_summary_evidence(
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
    // Use a higher limit so files with many `mod` declarations before `fn main` still
    // capture the entrypoint function.
    let mut declarations = declaration_lines_with_numbers(&content, 14);
    // Ensure `fn main` is always present for entrypoint files even if it falls
    // outside the declaration window.
    let has_main = declarations
        .iter()
        .any(|(_, l)| l.starts_with("fn main") || l.starts_with("pub fn main"));
    if !has_main {
        if let Some(main_line) = first_matching_lines(
            &content,
            |l| l.starts_with("fn main(") || l.starts_with("pub fn main("),
            1,
        )
        .into_iter()
        .next()
        {
            declarations.push(main_line);
        }
    }
    let excerpt = if declarations.is_empty() {
        first_non_empty_lines(&content, 10)
    } else {
        declarations
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

fn observed_repo_overview_evidence(results: &[ToolResult]) -> Option<RepoOverviewEvidence> {
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
    } else if trimmed.contains("load_")
        || trimmed.contains("save_")
        || trimmed.contains(".")
        || trimmed.contains("::")
    {
        ObservedStepKind::Delegation
    } else if trimmed.contains("else")
        || trimmed.starts_with("if ")
        || trimmed.starts_with("match ")
    {
        ObservedStepKind::Branch
    } else {
        ObservedStepKind::Definition
    }
}

fn observed_flow_trace_evidence(prompt: &str, results: &[ToolResult]) -> Option<FlowTraceEvidence> {
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
    if caller.is_none() && implementation.supporting.is_empty() {
        return None;
    }

    let mut steps = Vec::new();
    if let Some(caller) = caller {
        steps.push(ObservedStep {
            path: caller.path,
            line_number: caller.line_number,
            line_text: caller.line_text,
            step_kind: ObservedStepKind::EntryCall,
        });
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
    if let Some((path, content)) = results
        .iter()
        .filter(|result| result.tool_name == "read_file")
        .filter_map(|result| parse_read_file_output(&result.output))
        .find(|(path, _)| {
            path != &implementation.primary.path
                && preferred_candidate_path(ToolLoopIntent::FlowTrace, path)
                && !is_internal_tool_loop_path(path)
                && !is_legacy_auto_inspect_path(path)
        })
    {
        if let Some((line_number, line_text)) =
            first_matching_lines(&content, is_definition_like_line, 1)
                .into_iter()
                .next()
        {
            steps.push(ObservedStep {
                path: path.clone(),
                line_number,
                line_text,
                step_kind: ObservedStepKind::Delegation,
            });
        }
    }

    let has_cross_file_handoff = steps
        .first()
        .map(|first| {
            steps.iter().any(|step| {
                step.path != first.path
                    && !is_test_like_path(&step.path)
                    && !is_internal_tool_loop_path(&step.path)
                    && !is_legacy_auto_inspect_path(&step.path)
            })
        })
        .unwrap_or(false);

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

pub(super) fn investigation_outcome(
    intent: ToolLoopIntent,
    prompt: &str,
    resolution: Option<&InvestigationResolution>,
    results: &[ToolResult],
) -> InvestigationOutcome {
    let default_followup = || {
        "You do not have enough grounded file evidence yet. Read the most relevant source file and answer only after you have concrete lines to cite."
            .to_string()
    };
    match intent {
        ToolLoopIntent::CodeNavigation => {
            if let Some(evidence) = observed_file_summary_evidence(prompt, resolution, results) {
                return InvestigationOutcome::Ready {
                    evidence: StructuredEvidence::FileSummary(evidence),
                    stop_reason: "anchored file summary ready",
                };
            }
            if let Some(evidence) = observed_definition_evidence(intent, prompt, results) {
                InvestigationOutcome::Ready {
                    evidence: StructuredEvidence::Implementation(evidence),
                    stop_reason: "implementation evidence ready",
                }
            } else {
                InvestigationOutcome::NeedsMore {
                    required_next_step: targeted_investigation_followup(intent, prompt, results)
                        .unwrap_or_else(default_followup),
                }
            }
        }
        ToolLoopIntent::ConfigLocate => {
            if let Some(evidence) = observed_definition_evidence(intent, prompt, results) {
                InvestigationOutcome::Ready {
                    evidence: StructuredEvidence::Config(ConfigEvidence {
                        lines: std::iter::once(evidence.primary)
                            .chain(evidence.supporting)
                            .collect(),
                    }),
                    stop_reason: "config evidence ready",
                }
            } else {
                InvestigationOutcome::NeedsMore {
                    required_next_step: targeted_investigation_followup(intent, prompt, results)
                        .unwrap_or_else(default_followup),
                }
            }
        }
        ToolLoopIntent::CallSiteLookup | ToolLoopIntent::UsageLookup => {
            let observed = observed_reference_lines_from_read_results(intent, prompt, results);
            if !observed.is_empty() {
                let symbol = suggested_search_query(prompt, intent).unwrap_or_default();
                return InvestigationOutcome::Ready {
                    evidence: if matches!(intent, ToolLoopIntent::CallSiteLookup) {
                        StructuredEvidence::CallSites(CallSiteEvidence {
                            symbol,
                            sites: observed.into_iter().take(2).collect(),
                        })
                    } else {
                        StructuredEvidence::Usages(UsageEvidence {
                            symbol,
                            usages: observed.into_iter().take(2).collect(),
                        })
                    },
                    stop_reason: if matches!(intent, ToolLoopIntent::CallSiteLookup) {
                        "caller evidence ready"
                    } else {
                        "usage evidence ready"
                    },
                };
            }

            if !ranked_search_candidates(intent, prompt, results)
                .into_iter()
                .any(|file| {
                    !is_test_like_path(&file.path)
                        && !is_internal_tool_loop_path(&file.path)
                        && file.hits.iter().any(|hit| {
                            let line = hit.line_content.trim();
                            if matches!(intent, ToolLoopIntent::CallSiteLookup) {
                                line_contains_symbol_invocation(
                                    line,
                                    &suggested_search_query(prompt, intent).unwrap_or_default(),
                                ) && !is_definition_like_line(line)
                            } else {
                                line_contains_symbol_reference(
                                    line,
                                    &suggested_search_query(prompt, intent).unwrap_or_default(),
                                ) && !is_definition_like_line(line)
                            }
                        })
                })
                && !results.is_empty()
            {
                InvestigationOutcome::Insufficient {
                    reason: format!(
                        "I couldn't confirm a non-test source {} for `{}` within the current read budget.",
                        if matches!(intent, ToolLoopIntent::CallSiteLookup) {
                            "call-site"
                        } else {
                            "usage"
                        },
                        suggested_search_query(prompt, intent).unwrap_or_else(|| "the symbol".to_string())
                    ),
                }
            } else {
                InvestigationOutcome::NeedsMore {
                    required_next_step: targeted_investigation_followup(intent, prompt, results)
                        .unwrap_or_else(default_followup),
                }
            }
        }
        ToolLoopIntent::FlowTrace => {
            if let Some(evidence) = observed_flow_trace_evidence(prompt, results) {
                InvestigationOutcome::Ready {
                    evidence: StructuredEvidence::FlowTrace(evidence),
                    stop_reason: "flow chain ready",
                }
            } else if observed_read_paths(results)
                .iter()
                .filter(|path| {
                    is_source_path(path)
                        && !is_test_like_path(path)
                        && !is_internal_tool_loop_path(path)
                })
                .count()
                >= 3
            {
                InvestigationOutcome::Insufficient {
                    reason: format!(
                        "I couldn't confirm a source-only cross-file flow for `{}` within the current investigation budget.",
                        suggested_search_query(prompt, intent).unwrap_or_else(|| "that flow".to_string())
                    ),
                }
            } else {
                InvestigationOutcome::NeedsMore {
                    required_next_step: targeted_investigation_followup(intent, prompt, results)
                        .unwrap_or_else(default_followup),
                }
            }
        }
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview => {
            if let Some(evidence) = observed_repo_overview_evidence(results) {
                InvestigationOutcome::Ready {
                    evidence: StructuredEvidence::RepoOverview(evidence),
                    stop_reason: "repo overview ready",
                }
            } else {
                InvestigationOutcome::NeedsMore {
                    required_next_step: targeted_investigation_followup(intent, prompt, results)
                        .unwrap_or_else(default_followup),
                }
            }
        }
    }
}

pub(super) fn investigation_readiness(
    intent: ToolLoopIntent,
    prompt: &str,
    resolution: Option<&InvestigationResolution>,
    results: &[ToolResult],
) -> InvestigationReadiness {
    investigation_outcome(intent, prompt, resolution, results)
}

pub(super) fn has_relevant_file_evidence(
    intent: ToolLoopIntent,
    prompt: &str,
    results: &[ToolResult],
) -> bool {
    matches!(
        investigation_outcome(intent, prompt, None, results),
        InvestigationOutcome::Ready { .. }
    )
}

fn render_line_ref(line: &ObservedLine) -> String {
    format!("`{}:{}`", line.path, line.line_number)
}

fn render_code_ref(line: &ObservedLine) -> String {
    format!(
        "{} `{}`",
        render_line_ref(line),
        clip_inline(&line.line_text, 120)
    )
}

fn extract_declared_names(lines: &[ObservedLine], prefixes: &[&str], limit: usize) -> Vec<String> {
    lines
        .iter()
        .filter_map(|line| {
            let trimmed = line.line_text.trim();
            prefixes.iter().find_map(|prefix| {
                trimmed.strip_prefix(prefix).map(|rest| {
                    rest.trim_end_matches(';')
                        .split_whitespace()
                        .next()
                        .unwrap_or(rest.trim())
                        .trim_matches('{')
                        .trim_matches('(')
                        .trim()
                        .to_string()
                })
            })
        })
        .filter(|name| !name.is_empty())
        .take(limit)
        .collect()
}

fn join_names(names: &[String]) -> String {
    match names.len() {
        0 => String::new(),
        1 => names[0].clone(),
        2 => format!("{}, {}", names[0], names[1]),
        _ => {
            let mut values = names[..names.len() - 1].join(", ");
            values.push_str(", and ");
            if let Some(last) = names.last() {
                values.push_str(last);
            }
            values
        }
    }
}

pub(super) fn render_structured_answer(_prompt: &str, evidence: &StructuredEvidence) -> String {
    match evidence {
        StructuredEvidence::RepoOverview(evidence) => {
            let mut sentences = Vec::new();
            if let Some(package_line) = &evidence.package_line {
                sentences.push(format!(
                    "This repo is a Rust CLI project defined in {}.",
                    render_code_ref(package_line)
                ));
            }
            if let Some(entrypoint_line) = &evidence.entrypoint_line {
                sentences.push(format!(
                    "Startup runs through {}.",
                    render_code_ref(entrypoint_line)
                ));
            }
            if !evidence.subsystem_lines.is_empty() {
                let modules = extract_declared_names(
                    &evidence.subsystem_lines,
                    &["mod ", "pub mod ", "use "],
                    4,
                );
                if !modules.is_empty() {
                    sentences.push(format!(
                        "Top-level modules visible so far are {} from {}.",
                        join_names(&modules),
                        render_line_ref(&evidence.subsystem_lines[0])
                    ));
                }
            }
            if let Some(readme_line) = &evidence.readme_line {
                sentences.push(format!(
                    "The project description is summarized in {}.",
                    render_code_ref(readme_line)
                ));
            }
            sentences.into_iter().take(4).collect::<Vec<_>>().join(" ")
        }
        StructuredEvidence::FileSummary(evidence) => {
            let module_names =
                extract_declared_names(&evidence.declarations, &["mod ", "pub mod "], 6);
            let fn_names = extract_declared_names(&evidence.declarations, &["pub fn ", "fn "], 3);
            let import_names = extract_declared_names(&evidence.declarations, &["use "], 3)
                .into_iter()
                .map(|name| name.split("::").last().unwrap_or(&name).to_string())
                .collect::<Vec<_>>();
            let fn_anchor = evidence
                .declarations
                .iter()
                .find(|line| {
                    line.line_text.starts_with("pub fn ") || line.line_text.starts_with("fn ")
                })
                .unwrap_or(&evidence.declarations[0]);
            if !module_names.is_empty() && fn_names.iter().any(|name| name == "main") {
                return format!(
                    "{} declares modules {}, and {} defines `main`, so this file is the CLI entrypoint.",
                    render_line_ref(&evidence.declarations[0]),
                    join_names(&module_names),
                    render_line_ref(fn_anchor)
                );
            }
            let mut sentences = Vec::new();
            if !module_names.is_empty() {
                sentences.push(format!(
                    "{} declares modules {}.",
                    render_line_ref(&evidence.declarations[0]),
                    join_names(&module_names)
                ));
            }
            if !fn_names.is_empty() {
                sentences.push(format!(
                    "It defines {} in {}.",
                    join_names(&fn_names),
                    render_line_ref(fn_anchor)
                ));
            }
            if !import_names.is_empty() {
                sentences.push(format!(
                    "It imports {} in {}.",
                    join_names(&import_names),
                    render_line_ref(
                        evidence
                            .declarations
                            .iter()
                            .find(|line| line.line_text.starts_with("use "))
                            .unwrap_or(&evidence.declarations[0])
                    )
                ));
            }
            if sentences.is_empty() {
                let refs = evidence
                    .declarations
                    .iter()
                    .take(3)
                    .map(render_code_ref)
                    .collect::<Vec<_>>()
                    .join(", ");
                sentences.push(format!(
                    "{} anchors the file summary with {}.",
                    render_line_ref(&evidence.declarations[0]),
                    refs
                ));
            }
            sentences.join(" ")
        }
        StructuredEvidence::Implementation(evidence) => {
            let mut parts = vec![format!(
                "The implementation is in `{}` at line {}.",
                evidence.primary.path, evidence.primary.line_number
            )];
            let informative = evidence
                .supporting
                .iter()
                .filter(|line| {
                    let trimmed = line.line_text.trim();
                    !trimmed.is_empty()
                        && trimmed != "}"
                        && trimmed != "{"
                        && (trimmed.contains("return ")
                            || trimmed.contains("else")
                            || trimmed.contains(".")
                            || trimmed.contains("::"))
                })
                .take(3)
                .map(render_code_ref)
                .collect::<Vec<_>>();
            if informative.len() >= 2 {
                parts.push(format!(
                    "Relevant observed lines: {}.",
                    informative.join(" ")
                ));
            }
            parts.join(" ")
        }
        StructuredEvidence::Config(evidence) => evidence
            .lines
            .iter()
            .take(3)
            .map(|line| {
                format!(
                    "{} is part of the relevant config evidence.",
                    render_code_ref(line)
                )
            })
            .collect::<Vec<_>>()
            .join(" "),
        StructuredEvidence::CallSites(evidence) => evidence
            .sites
            .iter()
            .map(|line| format!("{} calls `{}`.", render_line_ref(line), evidence.symbol))
            .collect::<Vec<_>>()
            .join("\n"),
        StructuredEvidence::Usages(evidence) => evidence
            .usages
            .iter()
            .map(|line| format!("{} uses `{}`.", render_line_ref(line), evidence.symbol))
            .collect::<Vec<_>>()
            .join("\n"),
        StructuredEvidence::FlowTrace(evidence) => {
            // Build a short prose description of the flow as a fallback.
            let steps = evidence.steps.iter().take(5).collect::<Vec<_>>();
            if steps.is_empty() {
                return String::new();
            }
            let mut sentences = Vec::new();
            for step in &steps {
                let s = match step.step_kind {
                    ObservedStepKind::EntryCall => format!(
                        "The call originates at `{}:{}` (`{}`).",
                        step.path,
                        step.line_number,
                        clip_inline(&step.line_text, 80)
                    ),
                    ObservedStepKind::Definition => format!(
                        "`{}` is defined at `{}:{}` (`{}`).",
                        evidence.subject,
                        step.path,
                        step.line_number,
                        clip_inline(&step.line_text, 80)
                    ),
                    ObservedStepKind::Branch => format!(
                        "At `{}:{}`, the flow branches: `{}`.",
                        step.path,
                        step.line_number,
                        clip_inline(&step.line_text, 80)
                    ),
                    ObservedStepKind::Return => format!(
                        "It returns at `{}:{}` (`{}`).",
                        step.path,
                        step.line_number,
                        clip_inline(&step.line_text, 80)
                    ),
                    ObservedStepKind::Delegation => format!(
                        "The implementation delegates at `{}:{}` (`{}`).",
                        step.path,
                        step.line_number,
                        clip_inline(&step.line_text, 80)
                    ),
                };
                sentences.push(s);
            }
            sentences.join(" ")
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
            let query = suggested_search_query(prompt, intent).unwrap_or_default();
            let candidate = ranked_search_candidates(intent, prompt, results)
                .into_iter()
                .filter(|file| {
                    !read_paths.contains(&file.path)
                        && is_source_path(&file.path)
                        && !is_test_like_path(&file.path)
                        && !is_internal_tool_loop_path(&file.path)
                        && !is_legacy_auto_inspect_path(&file.path)
                        && file.hits.iter().any(|hit| {
                            let line = hit.line_content.trim();
                            if is_call_site {
                                line_contains_symbol_invocation(line, &query)
                                    && !is_definition_like_line(line)
                            } else {
                                line_contains_symbol_reference(line, &query)
                                    && !is_definition_like_line(line)
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
                            line_contains_symbol_invocation(l, &query)
                                && !is_definition_like_line(l)
                        } else {
                            line_contains_symbol_reference(l, &query) && !is_definition_like_line(l)
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
            None
        }
        ToolLoopIntent::FlowTrace => {
            let read_paths = observed_read_paths(results);
            let candidate = ranked_search_candidates(intent, prompt, results)
                .into_iter()
                .find(|file| {
                    !read_paths.contains(&file.path)
                        && preferred_candidate_path(intent, &file.path)
                        && !is_internal_tool_loop_path(&file.path)
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
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview => {
            let read_paths = observed_read_paths(results);
            repo_bootstrap_read_targets(Path::new("."), intent)
                .into_iter()
                .find(|path| !read_paths.contains(path))
                .map(|path| {
                    format!(
                        "Do not answer yet. Read this repo file next to ground the overview: `[read_file: {}]`.",
                        path
                    )
                })
        }
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

/// Like `auto_read_best_candidate` but specifically for CallSiteLookup and
/// UsageLookup: only reads files that have confirmed non-definition hits for the
/// symbol so we never waste a read on a file that only contains its definition.
fn auto_read_best_caller_candidate(
    intent: ToolLoopIntent,
    prompt: &str,
    tools: &ToolRegistry,
    existing_results: &[ToolResult],
    token_tx: &Sender<InferenceEvent>,
) -> Option<ToolResult> {
    let query = suggested_search_query(prompt, intent)?;
    let is_call_site = matches!(intent, ToolLoopIntent::CallSiteLookup);
    let read_paths = observed_read_paths(existing_results);

    // Pick the highest-ranked candidate that has at least one non-definition hit.
    let candidate = ranked_search_candidates(intent, prompt, existing_results)
        .into_iter()
        .find(|file| {
            !read_paths.contains(&file.path)
                && preferred_candidate_path(intent, &file.path)
                && !is_internal_tool_loop_path(&file.path)
                && !is_legacy_auto_inspect_path(&file.path)
                && file.hits.iter().any(|hit| {
                    let line = hit.line_content.trim();
                    if is_call_site {
                        line_contains_symbol_invocation(line, &query)
                            && !is_definition_like_line(line)
                    } else {
                        line_contains_symbol_reference(line, &query)
                            && !is_definition_like_line(line)
                    }
                })
        })?;

    emit_trace(
        token_tx,
        ProgressStatus::Started,
        &format!(
            "reading top {} candidate {}...",
            if is_call_site { "caller" } else { "usage" },
            candidate.path
        ),
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

fn read_file_result_from_project_root(project_root: &Path, path: &str) -> Option<ToolResult> {
    let resolved = if Path::new(path).is_absolute() {
        Path::new(path).to_path_buf()
    } else {
        project_root.join(path)
    };
    let content = fs::read_to_string(&resolved).ok()?;
    Some(ToolResult {
        tool_name: "read_file".to_string(),
        argument: path.to_string(),
        output: format!(
            "File: {}\nLines: {}\n\n```\n{}\n```",
            path,
            content.lines().count(),
            content
        ),
    })
}

pub(super) fn bootstrap_tool_results(
    intent: ToolLoopIntent,
    prompt: &str,
    resolution: Option<&InvestigationResolution>,
    base_messages: &[Message],
    project_root: &Path,
    backend_name: &str,
    tools: &ToolRegistry,
    token_tx: &Sender<InferenceEvent>,
) -> Vec<ToolResult> {
    if let Some(path) = resolution
        .and_then(|resolution| resolution.anchored_file.as_deref())
        .map(str::to_string)
        .or_else(|| {
            if should_answer_from_anchor_file(prompt, resolution) {
                recent_loaded_file_context_path(base_messages)
            } else {
                None
            }
        })
    {
        emit_trace(
            token_tx,
            ProgressStatus::Started,
            &format!("bootstrapping read {}...", path),
            false,
        );
        if let Some(read_result) = read_file_result_from_project_root(project_root, &path) {
            emit_trace(
                token_tx,
                ProgressStatus::Finished,
                &format!("read_file {}", read_result.argument),
                false,
            );
            return vec![read_result];
        }
        let execution = tools.execute_read_only_tool_calls(&format!("[read_file: {path}]"));
        let results = execution.results;
        if let Some(read_result) = results.first() {
            emit_trace(
                token_tx,
                ProgressStatus::Finished,
                &format!("read_file {}", read_result.argument),
                false,
            );
        }
        return results;
    }

    if matches!(
        intent,
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview
    ) {
        return bootstrap_repo_overview_results(intent, project_root, tools, token_tx);
    }

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

    // For caller / usage lookup, auto-read the best file that has confirmed
    // non-definition hits so the loop can answer immediately instead of burning
    // iterations while the model tries to figure out which file to read.
    if matches!(
        intent,
        ToolLoopIntent::CallSiteLookup | ToolLoopIntent::UsageLookup
    ) && !has_relevant_file_evidence(intent, prompt, &results)
    {
        if let Some(read_result) =
            auto_read_best_caller_candidate(intent, prompt, tools, &results, token_tx)
        {
            results.push(read_result);
        }
    }

    results
}

fn bootstrap_repo_overview_results(
    intent: ToolLoopIntent,
    project_root: &Path,
    tools: &ToolRegistry,
    token_tx: &Sender<InferenceEvent>,
) -> Vec<ToolResult> {
    let mut results = tools.execute_read_only_tool_calls("[list_dir: .]").results;
    if !results.is_empty() {
        emit_trace(token_tx, ProgressStatus::Finished, "list_dir .", false);
    }

    for path in repo_bootstrap_read_targets(project_root, intent) {
        let execution = tools.execute_read_only_tool_calls(&format!("[read_file: {path}]"));
        if let Some(result) = execution.results.into_iter().next() {
            emit_trace(
                token_tx,
                ProgressStatus::Finished,
                &format!("read_file {}", result.argument),
                false,
            );
            results.push(result);
        }
    }

    results
}

fn repo_bootstrap_read_targets(project_root: &Path, intent: ToolLoopIntent) -> Vec<String> {
    let candidates = match intent {
        ToolLoopIntent::RepoOverview => {
            vec!["Cargo.toml", "README.md", "src/main.rs", "src/lib.rs"]
        }
        ToolLoopIntent::DirectoryOverview => vec!["Cargo.toml", "src/main.rs", "src/lib.rs"],
        _ => Vec::new(),
    };

    candidates
        .into_iter()
        .filter(|path| project_root.join(path).is_file())
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_overview_bootstrap_reads_manifest_and_entrypoint() {
        let dir =
            std::env::temp_dir().join(format!("params-repo-bootstrap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("Cargo.toml"), "[package]\nname = \"demo\"\n").unwrap();
        std::fs::write(dir.join("README.md"), "# demo\n").unwrap();
        std::fs::write(dir.join("src/main.rs"), "fn main() {}\n").unwrap();

        let (tx, _rx) = std::sync::mpsc::channel();
        let results = bootstrap_tool_results(
            ToolLoopIntent::RepoOverview,
            "Can you see my project?",
            None,
            &[Message::user("Can you see my project?")],
            &dir,
            "llama.cpp",
            &ToolRegistry::default(),
            &tx,
        );

        assert!(results.iter().any(|r| r.tool_name == "list_dir"));
        assert!(results.iter().any(|r| r.argument == "Cargo.toml"));
        assert!(results.iter().any(|r| r.argument == "src/main.rs"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn callsite_lookup_requires_non_definition_read() {
        let definition_result = ToolResult {
            tool_name: "read_file".to_string(),
            argument: "src/session/mod.rs".to_string(),
            output:
                "File: src/session/mod.rs\nLines: 3\n\n```\npub fn load_most_recent() {\n}\n```"
                    .to_string(),
        };
        let search_result = ToolResult {
            tool_name: "search".to_string(),
            argument: "load_most_recent".to_string(),
            output: "src/session/mod.rs:\n  1: pub fn load_most_recent() {\n\nsrc/main.rs:\n  12: store.load_most_recent();\n"
                .to_string(),
        };

        assert!(
            !has_relevant_file_evidence(
                ToolLoopIntent::CallSiteLookup,
                "what calls load_most_recent",
                &[search_result.clone(), definition_result]
            ),
            "definition-only reads should not satisfy call-site lookup"
        );

        let caller_result = ToolResult {
            tool_name: "read_file".to_string(),
            argument: "src/main.rs".to_string(),
            output: "File: src/main.rs\nLines: 5\n\n```\nfn start() {\n    store.load_most_recent();\n}\n```"
                .to_string(),
        };
        assert!(has_relevant_file_evidence(
            ToolLoopIntent::CallSiteLookup,
            "what calls load_most_recent",
            &[search_result, caller_result]
        ));
    }

    #[test]
    fn flow_trace_requires_cross_file_evidence() {
        let one_read = ToolResult {
            tool_name: "read_file".to_string(),
            argument: "src/main.rs".to_string(),
            output: "File: src/main.rs\nLines: 4\n\n```\nfn main() {\n    init_logging();\n}\n```"
                .to_string(),
        };
        let search = ToolResult {
            tool_name: "search".to_string(),
            argument: "logging".to_string(),
            output: "src/main.rs:\n  2: init_logging();\n\nsrc/logging.rs:\n  1: pub fn init_logging() {}\n"
                .to_string(),
        };

        assert!(
            !has_relevant_file_evidence(
                ToolLoopIntent::FlowTrace,
                "Trace how logging works.",
                &[search.clone(), one_read.clone()]
            ),
            "single-file evidence should not satisfy flow tracing"
        );

        let second_read = ToolResult {
            tool_name: "read_file".to_string(),
            argument: "src/logging.rs".to_string(),
            output: "File: src/logging.rs\nLines: 4\n\n```\npub fn init_logging() {\n    configure_sink();\n}\n```"
                .to_string(),
        };
        assert!(has_relevant_file_evidence(
            ToolLoopIntent::FlowTrace,
            "Trace how logging works.",
            &[search, one_read, second_read]
        ));
    }
}
