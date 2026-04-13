use crate::tools::ToolResult;

use super::followup::{
    file_display_name, primary_config_locations, primary_definition_location, rank_search_files,
    summarize_feature_trace_hits, summarize_workflow_read, test_module_start_line,
};
use super::parse::{
    clip_inline, parse_list_dir_output, parse_read_file_output, parse_search_output,
};
use super::types::{AutoInspectBudget, AutoInspectIntent, AutoInspectPlan};

pub(crate) fn summarize_readme(content: &str, max_chars: usize) -> Option<String> {
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

pub(crate) fn summarize_cargo_manifest(content: &str, max_chars: usize) -> Option<String> {
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

pub(crate) fn summarize_entrypoint(path: &str, content: &str, max_chars: usize) -> Option<String> {
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

pub(crate) fn top_level_repo_type(entries: &[String]) -> Option<String> {
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

pub(crate) fn format_entry_list(label: &str, entries: &[String], limit: usize) -> Option<String> {
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

pub(crate) fn synthesize_auto_inspection_context(
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
                            || super::followup::is_definition_like_line(&hit.line_content)
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
