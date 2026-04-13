use crate::tools::ToolResult;

use super::super::super::session::investigation::InvestigationResolution;
use super::super::intent::{suggested_search_query, ToolLoopIntent};
use super::observe::{
    observed_file_summary_evidence, observed_flow_trace_evidence,
    observed_reference_lines_from_read_results, should_answer_from_anchor_file,
};
use super::parse::{
    clip_inline, clip_tool_output, compact_read_file_result, declaration_lines_with_numbers,
    definition_match_lines_with_numbers, filter_non_test_hits, first_non_empty_lines,
    is_test_like_path, parse_read_file_output, prompt_mentions_tests,
    query_match_lines_with_numbers, surrounding_body_lines,
};

pub(crate) fn format_tool_loop_results_with_limit(
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

pub(crate) fn grounded_answer_guidance(
    intent: ToolLoopIntent,
    prompt: &str,
    resolution: Option<&InvestigationResolution>,
    results: &[ToolResult],
) -> Option<String> {
    match intent {
        ToolLoopIntent::CodeNavigation => {
            if should_answer_from_anchor_file(prompt, resolution) {
                if let Some(evidence) = observed_file_summary_evidence(prompt, resolution, results)
                {
                    let mut sections = vec![
                        "Grounded answer requirements: answer what this loaded file does using only the observed lines below. Do not include code fences. Do not mention search results, wrapper prompt text, or unrelated prompt strings. Do not suggest inspecting another file unless the observed lines are genuinely insufficient.".to_string(),
                        format!("Loaded file: `{}`", evidence.path),
                        "Observed file structure:".to_string(),
                    ];
                    sections.extend(evidence.declarations.iter().map(|line| {
                        format!(
                            "  {}:{} `{}`",
                            line.path,
                            line.line_number,
                            clip_inline(&line.line_text, 120)
                        )
                    }));
                    sections.push(
                        "Answer from the observed lines above only. Rules:\n\
                         1. Keep the answer to 2-4 short sentences or a flat 3-bullet list.\n\
                         2. If `fn main` is visible, mention that this is the binary entrypoint and describe its role.\n\
                         3. If `struct Cli` or `enum Command` are visible, explain the CLI shape and subcommand surface from those lines.\n\
                         4. If startup lines like `Cli::parse()` or `match cli.command` are visible, describe the startup/orchestration role.\n\
                         5. Cite every concrete fact with exact file:line references.\n\
                         6. Copy identifiers verbatim — do not rename methods, modules, or types.\n\
                         7. Do not use hedging words (`presumably`, `likely`, `suggests`, `appears to`, `seems to`, `may`).\n\
                         8. Do not pivot to other files or offer next-step advice unless the observed lines clearly delegate elsewhere."
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
                "List each file:line where the symbol is used or referenced. Import lines count as real usages when they appear in source files. Do NOT describe the symbol's own implementation."
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
            let evidence = observed_flow_trace_evidence(prompt, results)?;
            let mut sections = vec![
                "Grounded answer requirements: explain the execution flow in plain language using \
                 only the observed file evidence below. Do not list raw code lines. \
                 Write a short natural-language explanation of what happens, in order, \
                 with file:line citations. Do not include code fences."
                    .to_string(),
                "Observed cross-file evidence:".to_string(),
            ];
            sections.extend(evidence.steps.iter().map(|step| {
                format!(
                    "  {}:{} `{}`",
                    step.path,
                    step.line_number,
                    clip_inline(&step.line_text, 120)
                )
            }));
            sections.push(
                "Answer from the observed evidence above. Rules:\n\
                 1. Write a SHORT explanation in plain language (2–4 sentences), not a code dump.\n\
                 2. Connect the runtime caller to the definition and keep the steps in execution order.\n\
                 3. Mention branch behavior (early return, None path) and the successful handoff if visible.\n\
                 4. Do not copy raw source lines verbatim into the answer — paraphrase them.\n\
                 5. Do not use hedging words (`presumably`, `likely`, `suggests`, `appears to`, `seems to`, `may`)."
                    .to_string(),
            );
            Some(sections.join("\n"))
        }
        ToolLoopIntent::ConfigLocate => {
            let query = suggested_search_query(prompt, intent)?;
            for result in results.iter().filter(|r| r.tool_name == "read_file") {
                let Some((path, content)) = parse_read_file_output(&result.output) else {
                    continue;
                };
                if is_test_like_path(&path) && !prompt_mentions_tests(prompt) {
                    continue;
                }
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
