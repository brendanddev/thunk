use crate::tools::ToolResult;

use super::super::super::session::investigation::InvestigationResolution;
use super::super::intent::{suggested_search_query, ToolLoopIntent};
use super::bootstrap::repo_bootstrap_read_targets;
use super::observe::{
    observed_definition_evidence, observed_file_summary_evidence, observed_flow_trace_evidence,
    observed_read_paths, observed_reference_lines_from_read_results,
    observed_repo_overview_evidence, preferred_candidate_path, ranked_search_candidates,
};
use super::parse::{
    is_definition_like_line, is_internal_tool_loop_path, is_source_path, is_test_like_path,
    line_contains_symbol_invocation, line_contains_symbol_reference,
};
use super::{
    CallSiteEvidence, ConfigEvidence, InvestigationOutcome, InvestigationReadiness,
    StructuredEvidence, UsageEvidence,
};
use std::path::Path;

pub(crate) fn investigation_outcome(
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
                        suggested_search_query(prompt, intent)
                            .unwrap_or_else(|| "the symbol".to_string())
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
                        suggested_search_query(prompt, intent)
                            .unwrap_or_else(|| "that flow".to_string())
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

pub(crate) fn investigation_readiness(
    intent: ToolLoopIntent,
    prompt: &str,
    resolution: Option<&InvestigationResolution>,
    results: &[ToolResult],
) -> InvestigationReadiness {
    investigation_outcome(intent, prompt, resolution, results)
}

pub(crate) fn has_relevant_file_evidence(
    intent: ToolLoopIntent,
    prompt: &str,
    results: &[ToolResult],
) -> bool {
    matches!(
        investigation_outcome(intent, prompt, None, results),
        InvestigationOutcome::Ready { .. }
    )
}

pub(crate) fn targeted_investigation_followup(
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
                        && !super::parse::is_legacy_auto_inspect_path(&file.path)
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
