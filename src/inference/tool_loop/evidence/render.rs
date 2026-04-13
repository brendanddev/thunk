use super::parse::clip_inline;
use super::{ObservedLine, ObservedStepKind, StructuredEvidence};

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
            .map(|line| format!("{} uses `{}`.", render_line_ref(line), evidence.symbol))
            .collect::<Vec<_>>()
            .join("\n"),
        StructuredEvidence::FlowTrace(evidence) => {
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
