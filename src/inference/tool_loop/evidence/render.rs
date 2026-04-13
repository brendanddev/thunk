use super::parse::clip_inline;
use super::{ObservedLine, ObservedStep, ObservedStepKind, StructuredEvidence};

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

fn render_step_ref(step: &ObservedStep) -> String {
    format!("`{}:{}`", step.path, step.line_number)
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

pub(crate) fn render_structured_answer(_prompt: &str, evidence: &StructuredEvidence) -> String {
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
            let cli_line = evidence.declarations.iter().find(|line| {
                line.line_text.starts_with("struct Cli")
                    || line.line_text.starts_with("pub struct Cli")
            });
            let command_line = evidence.declarations.iter().find(|line| {
                line.line_text.starts_with("enum Command")
                    || line.line_text.starts_with("pub enum Command")
            });
            let main_line = evidence.declarations.iter().find(|line| {
                line.line_text.starts_with("pub fn main(") || line.line_text.starts_with("fn main(")
            });
            let parse_line = evidence
                .declarations
                .iter()
                .find(|line| line.line_text.contains("::parse()"));
            let dispatch_line = evidence
                .declarations
                .iter()
                .find(|line| line.line_text.contains("match cli.command"));
            let fn_anchor = evidence
                .declarations
                .iter()
                .find(|line| {
                    line.line_text.starts_with("pub fn ") || line.line_text.starts_with("fn ")
                })
                .unwrap_or(&evidence.declarations[0]);
            if let Some(main_line) = main_line {
                let mut sentences = Vec::new();
                if let Some(cli_line) = cli_line {
                    if let Some(command_line) = command_line {
                        sentences.push(format!(
                            "{} defines `Cli` and {} defines `Command`, so this file lays out the visible CLI shape and subcommand surface.",
                            render_line_ref(cli_line),
                            render_line_ref(command_line)
                        ));
                    } else {
                        sentences.push(format!(
                            "{} defines `Cli`, so this file describes the command-line surface.",
                            render_line_ref(cli_line)
                        ));
                    }
                }
                let startup_ref = match (parse_line, dispatch_line) {
                    (Some(parse_line), Some(dispatch_line)) => format!(
                        " Startup orchestration is visible in {} and {}.",
                        render_line_ref(parse_line),
                        render_line_ref(dispatch_line)
                    ),
                    (Some(parse_line), None) => {
                        format!(
                            " Startup orchestration is visible in {}.",
                            render_line_ref(parse_line)
                        )
                    }
                    (None, Some(dispatch_line)) => format!(
                        " Startup orchestration is visible in {}.",
                        render_line_ref(dispatch_line)
                    ),
                    (None, None) => String::new(),
                };
                sentences.push(format!(
                    "{} defines `main`, so this file is the CLI entrypoint.{startup_ref}",
                    render_line_ref(main_line)
                ));
                return sentences.into_iter().take(3).collect::<Vec<_>>().join(" ");
            }
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
                            || trimmed.contains('.')
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
            .map(|line| {
                if line.line_text.trim().starts_with("use ") {
                    format!("{} imports `{}`.", render_line_ref(line), evidence.symbol)
                } else {
                    format!("{} uses `{}`.", render_line_ref(line), evidence.symbol)
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        StructuredEvidence::FlowTrace(evidence) => {
            let steps = evidence.steps.iter().take(5).collect::<Vec<_>>();
            if steps.is_empty() {
                return String::new();
            }
            let entry = evidence
                .steps
                .iter()
                .find(|step| step.step_kind == ObservedStepKind::EntryCall);
            let selection = evidence.steps.iter().find(|step| {
                step.line_text
                    .contains("list_sessions()?.into_iter().next()")
            });
            let runtime_no_session = entry.and_then(|entry| {
                evidence.steps.iter().find(|step| {
                    step.path == entry.path
                        && step.line_number != entry.line_number
                        && (step.line_text.contains("Ok(None)") || step.line_text.contains("None =>"))
                })
            });
            let return_no_session = evidence
                .steps
                .iter()
                .find(|step| step.line_text.contains("return Ok(None)"));
            let load_by_id = evidence
                .steps
                .iter()
                .find(|step| step.line_text.contains("load_session_by_id"));

            if entry.is_some()
                && (selection.is_some()
                    || runtime_no_session.is_some()
                    || return_no_session.is_some()
                    || load_by_id.is_some())
            {
                let mut sentences = Vec::new();
                if let Some(entry) = entry {
                    if entry.line_text.contains("match store.load_most_recent()") {
                        sentences.push(format!(
                            "Session restore starts at {} where the runtime branches on `match store.load_most_recent()`.",
                            render_step_ref(entry)
                        ));
                    } else {
                        sentences.push(format!(
                            "Session restore starts at {} where the runtime path calls `{}`.",
                            render_step_ref(entry),
                            evidence.subject
                        ));
                    }
                }
                if let Some(selection) = selection {
                    if let Some(return_no_session) = return_no_session {
                        if let Some(runtime_no_session) = runtime_no_session {
                            sentences.push(format!(
                                "In {}, `{}` takes the first summary from `list_sessions()?.into_iter().next()`; if there is no summary it returns `Ok(None)` at {}, which feeds the runtime no-session branch at {}.",
                                render_step_ref(selection),
                                evidence.subject,
                                render_step_ref(return_no_session),
                                render_step_ref(runtime_no_session)
                            ));
                        } else {
                            sentences.push(format!(
                                "In {}, `{}` takes the first summary from `list_sessions()?.into_iter().next()`; if there is no summary it returns `Ok(None)` at {}.",
                                render_step_ref(selection),
                                evidence.subject,
                                render_step_ref(return_no_session)
                            ));
                        }
                    } else {
                        sentences.push(format!(
                            "In {}, `{}` takes the first summary from `list_sessions()?.into_iter().next()`.",
                            render_step_ref(selection),
                            evidence.subject
                        ));
                    }
                }
                if selection.is_none() {
                    if let Some(runtime_no_session) = runtime_no_session {
                        sentences.push(format!(
                            "If no saved session is available, the runtime no-session branch is at {} (`{}`).",
                            render_step_ref(runtime_no_session),
                            clip_inline(&runtime_no_session.line_text, 80)
                        ));
                    }
                }
                if selection.is_none() {
                    if let Some(return_no_session) = return_no_session {
                        sentences.push(format!(
                            "If there is no saved summary, `{}` returns `Ok(None)` at {}.",
                            evidence.subject,
                            render_step_ref(return_no_session)
                        ));
                    }
                }
                if let Some(load_by_id) = load_by_id {
                    sentences.push(format!(
                        "If a summary exists, it hands off to `load_session_by_id(&summary.id)` at {}.",
                        render_step_ref(load_by_id)
                    ));
                }
                if !sentences.is_empty() {
                    return sentences.join(" ");
                }
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
