use crate::tools::ToolResult;

use super::super::super::session::investigation::InvestigationResolution;
use super::super::intent::{suggested_search_query, ToolLoopIntent};
use super::bootstrap::repo_bootstrap_read_targets;
use super::observe::{
    all_candidates_fully_read, lookup_search_anchor, observed_definition_evidence,
    observed_file_summary_evidence, observed_flow_trace_evidence, observed_read_paths,
    observed_reference_lines_from_read_results, observed_repo_overview_evidence,
    preferred_candidate_path, ranked_search_candidates, search_candidates_in_output_order,
};
use super::parse::{
    is_definition_like_line, is_internal_tool_loop_path, is_legacy_auto_inspect_path,
    is_source_path, is_test_like_path,
};
use super::types::{
    CallSiteEvidence, ConfigEvidence, FlowTraceEvidence, InvestigationOutcome,
    InvestigationReadiness, StructuredEvidence, UsageEvidence,
};
use std::path::Path;

fn normalize_path(path: &str) -> String {
    let normalized = path.replace("\\", "/");
    if let Some(stripped) = normalized.strip_prefix("./") {
        stripped.to_string()
    } else {
        normalized
    }
}

fn validate_evidence_readiness(
    intent: ToolLoopIntent,
    evidence: &StructuredEvidence,
    results: &[ToolResult],
) -> bool {
    match evidence {
        StructuredEvidence::FileSummary(fe) => {
            let read_paths = observed_read_paths(results);
            let fe_path = normalize_path(&fe.path);
            let has_valid_read = read_paths
                .iter()
                .any(|path| path == &fe_path && is_source_path(path));
            if !has_valid_read {
                return false;
            }
            let valid_declarations = fe.declarations.iter().filter(|line| {
                is_source_path(&line.path)
                    && !is_test_like_path(&line.path)
                    && !is_internal_tool_loop_path(&line.path)
                    && !is_legacy_auto_inspect_path(&line.path)
            });
            valid_declarations.count() >= 1
        }
        StructuredEvidence::Implementation(ie) => {
            is_source_path(&ie.primary.path)
                && !is_test_like_path(&ie.primary.path)
                && !is_internal_tool_loop_path(&ie.primary.path)
                && !is_legacy_auto_inspect_path(&ie.primary.path)
                && ie.primary.line_number > 0
        }
        StructuredEvidence::Config(ce) => {
            ce.lines
                .iter()
                .all(|line| is_source_path(&line.path) || line.path.contains("config"))
                && !ce.lines.is_empty()
        }
        StructuredEvidence::CallSites(cse) => {
            cse.sites.iter().all(|line| {
                is_source_path(&line.path)
                    && !is_internal_tool_loop_path(&line.path)
                    && !is_legacy_auto_inspect_path(&line.path)
            }) && !cse.sites.is_empty()
        }
        StructuredEvidence::Usages(ue) => {
            ue.usages.iter().all(|line| {
                is_source_path(&line.path)
                    && !is_internal_tool_loop_path(&line.path)
                    && !is_legacy_auto_inspect_path(&line.path)
            }) && !ue.usages.is_empty()
        }
        StructuredEvidence::FlowTrace(fte) => validate_flow_trace_completeness(fte),
        StructuredEvidence::RepoOverview(_) => true,
    }
}

fn validate_flow_trace_completeness(evidence: &FlowTraceEvidence) -> bool {
    if evidence.steps.is_empty() {
        return false;
    }
    let has_cross_file = evidence.steps.windows(2).any(|window| {
        let a = &window[0];
        let b = &window[1];
        a.path != b.path
            && !is_test_like_path(&a.path)
            && !is_test_like_path(&b.path)
            && !is_internal_tool_loop_path(&a.path)
            && !is_internal_tool_loop_path(&b.path)
            && !is_legacy_auto_inspect_path(&a.path)
            && !is_legacy_auto_inspect_path(&b.path)
    });
    let source_only = evidence.steps.iter().all(|step| {
        is_source_path(&step.path)
            && !is_test_like_path(&step.path)
            && !is_internal_tool_loop_path(&step.path)
            && !is_legacy_auto_inspect_path(&step.path)
    });
    has_cross_file && source_only
}

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
                let structured = StructuredEvidence::FileSummary(evidence);
                if validate_evidence_readiness(intent, &structured, results) {
                    return InvestigationOutcome::Ready {
                        evidence: structured,
                        stop_reason: "anchored file summary ready",
                    };
                }
            }
            if let Some(evidence) = observed_definition_evidence(intent, prompt, results) {
                let structured = StructuredEvidence::Implementation(evidence);
                if validate_evidence_readiness(intent, &structured, results) {
                    return InvestigationOutcome::Ready {
                        evidence: structured,
                        stop_reason: "implementation evidence ready",
                    };
                }
            }
            InvestigationOutcome::NeedsMore {
                required_next_step: targeted_investigation_followup(intent, prompt, results)
                    .unwrap_or_else(default_followup),
            }
        }
        ToolLoopIntent::ConfigLocate => {
            if let Some(evidence) = observed_definition_evidence(intent, prompt, results) {
                let structured = StructuredEvidence::Config(ConfigEvidence {
                    lines: std::iter::once(evidence.primary)
                        .chain(evidence.supporting)
                        .collect(),
                });
                if validate_evidence_readiness(intent, &structured, results) {
                    return InvestigationOutcome::Ready {
                        evidence: structured,
                        stop_reason: "config evidence ready",
                    };
                }
            }
            InvestigationOutcome::NeedsMore {
                required_next_step: targeted_investigation_followup(intent, prompt, results)
                    .unwrap_or_else(default_followup),
            }
        }
        ToolLoopIntent::CallSiteLookup | ToolLoopIntent::UsageLookup => {
            let observed = observed_reference_lines_from_read_results(intent, prompt, results);
            if !observed.is_empty() {
                let symbol = suggested_search_query(prompt, intent).unwrap_or_default();
                let evidence = if matches!(intent, ToolLoopIntent::CallSiteLookup) {
                    StructuredEvidence::CallSites(CallSiteEvidence {
                        symbol,
                        sites: observed.into_iter().collect(),
                    })
                } else {
                    StructuredEvidence::Usages(UsageEvidence {
                        symbol,
                        usages: observed.into_iter().collect(),
                    })
                };
                if validate_evidence_readiness(intent, &evidence, results) {
                    return InvestigationOutcome::Ready {
                        evidence,
                        stop_reason: if matches!(intent, ToolLoopIntent::CallSiteLookup) {
                            "caller evidence ready"
                        } else {
                            "usage evidence ready"
                        },
                    };
                }
            }

            let read_paths = observed_read_paths(results);
            let source_reads = read_paths
                .iter()
                .filter(|path| {
                    is_source_path(path)
                        && !is_test_like_path(path)
                        && !is_internal_tool_loop_path(path)
                })
                .count();
            let candidates = ranked_search_candidates(intent, prompt, results)
                .into_iter()
                .filter(|file| {
                    preferred_candidate_path(intent, &file.path)
                        && !is_test_like_path(&file.path)
                        && !is_internal_tool_loop_path(&file.path)
                        && lookup_search_anchor(intent, prompt, file).is_some()
                })
                .collect::<Vec<_>>();

            let has_all_candidates_read = all_candidates_fully_read(intent, prompt, results);

            if candidates
                .iter()
                .any(|file| !read_paths.contains(&file.path))
            {
                return InvestigationOutcome::NeedsMore {
                    required_next_step: targeted_investigation_followup(intent, prompt, results)
                        .unwrap_or_else(default_followup),
                };
            }

            if !candidates.is_empty() && has_all_candidates_read && !results.is_empty() {
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
            } else if source_reads > 0 && !results.is_empty() && candidates.is_empty() {
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
                let structured = StructuredEvidence::FlowTrace(evidence);
                if validate_evidence_readiness(intent, &structured, results) {
                    return InvestigationOutcome::Ready {
                        evidence: structured,
                        stop_reason: "flow chain ready",
                    };
                }
            }
            let source_reads = observed_read_paths(results)
                .iter()
                .filter(|path| {
                    is_source_path(path)
                        && !is_test_like_path(path)
                        && !is_internal_tool_loop_path(path)
                })
                .count();
            if source_reads >= 3 {
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
                let structured = StructuredEvidence::RepoOverview(evidence);
                if validate_evidence_readiness(intent, &structured, results) {
                    return InvestigationOutcome::Ready {
                        evidence: structured,
                        stop_reason: "repo overview ready",
                    };
                }
            }
            InvestigationOutcome::NeedsMore {
                required_next_step: targeted_investigation_followup(intent, prompt, results)
                    .unwrap_or_else(default_followup),
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
            let candidate = ranked_search_candidates(intent, prompt, results)
                .into_iter()
                .find(|file| {
                    !read_paths.contains(&file.path)
                        && preferred_candidate_path(intent, &file.path)
                        && lookup_search_anchor(intent, prompt, file).is_some()
                })
                .or_else(|| {
                    ranked_search_candidates(intent, prompt, results)
                        .into_iter()
                        .find(|file| {
                            !read_paths.contains(&file.path)
                                && is_source_path(&file.path)
                                && !is_test_like_path(&file.path)
                                && !is_internal_tool_loop_path(&file.path)
                                && !super::parse::is_legacy_auto_inspect_path(&file.path)
                        })
                });
            if let Some(file) = candidate {
                let mode = if is_call_site { "call-site" } else { "usage" };
                let anchor = lookup_search_anchor(intent, prompt, &file)
                    .or_else(|| file.hits.first().cloned());
                let anchor_text = anchor
                    .as_ref()
                    .map(|hit| format!("{}: {}", hit.line_number, hit.line_content));
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
            let candidate = search_candidates_in_output_order(results)
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
