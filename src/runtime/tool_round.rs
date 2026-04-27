use std::collections::HashSet;

use crate::tools::{ExecutionKind, PendingAction, ToolInput, ToolRegistry, ToolRunResult};

use super::anchors::AnchorState;
use super::investigation::{InvestigationMode, InvestigationState, RecoveryKind};
use super::paths::{normalize_evidence_path, path_is_within_scope, path_matches_requested};
use super::response_text::*;
use super::search_query::{simplify_search_input, weak_search_query_reason};
use super::tool_codec;
use super::tool_surface::{is_git_read_only_tool_input, tool_allowed_for_surface, ToolSurface};
use super::trace::trace_runtime_decision;
use super::types::{RuntimeEvent, RuntimeTerminalReason};

/// Maximum number of successful read_file calls allowed in a single turn.
/// Each read injects up to MAX_LINES lines into the prompt; this cap bounds worst-case
/// context growth when the model reads speculatively or drifts into repeated reads.
/// 3 is conservative: a correct investigation needs 1 (search → read → answer);
/// 2-3 accommodates a reasonable follow-up read without runaway context expansion.
const MAX_READS_PER_TURN: usize = 3;

/// Maximum number of distinct search-candidate files that may be read in a single
/// investigation turn.  After two candidate reads, if evidence is still not ready,
/// the runtime terminates cleanly rather than allowing another correction cycle.
pub(super) const MAX_CANDIDATE_READS_PER_INVESTIGATION: usize = 2;

/// Tracks search_code usage within a single turn.
/// Rules: 1 search always permitted; a second search is permitted only when the first
/// returned zero matches; any further searches are blocked.
pub(super) struct SearchBudget {
    pub(super) calls: usize,
    last_was_empty: bool,
}

impl SearchBudget {
    pub(super) fn new() -> Self {
        Self {
            calls: 0,
            last_was_empty: false,
        }
    }

    fn is_allowed(&self) -> bool {
        self.calls == 0 || (self.calls == 1 && self.last_was_empty)
    }

    fn record(&mut self, was_empty: bool) {
        self.calls += 1;
        self.last_was_empty = was_empty;
    }

    pub(super) fn is_closed(&self) -> bool {
        self.calls >= 2 || (self.calls == 1 && !self.last_was_empty)
    }

    pub(super) fn empty_retry_exhausted(&self) -> bool {
        self.calls >= 2 && self.last_was_empty
    }

    pub(super) fn closed_message(&self) -> &'static str {
        if self.calls >= 2 && self.last_was_empty {
            SEARCH_CLOSED_AFTER_EMPTY_RETRY
        } else {
            SEARCH_CLOSED_AFTER_RESULTS
        }
    }
}

/// Returns a stable fingerprint for a tool call, used for consecutive-cycle detection.
/// Null bytes separate fields; they cannot appear in paths, queries, or file content
/// on any supported platform, so false matches are impossible.
fn call_fingerprint(input: &ToolInput) -> String {
    match input {
        ToolInput::ReadFile { path } => format!("read_file\x00{path}"),
        ToolInput::ListDir { path } => format!("list_dir\x00{path}"),
        ToolInput::SearchCode { query, path } => {
            format!(
                "search_code\x00{query}\x00{}",
                path.as_deref().unwrap_or("")
            )
        }
        ToolInput::GitStatus => "git_status".to_string(),
        ToolInput::GitDiff => "git_diff".to_string(),
        ToolInput::GitLog => "git_log".to_string(),
        ToolInput::EditFile {
            path,
            search,
            replace,
        } => {
            format!("edit_file\x00{path}\x00{search}\x00{replace}")
        }
        ToolInput::WriteFile { path, content } => {
            format!("write_file\x00{path}\x00{content}")
        }
    }
}

fn is_mutating_tool(input: &ToolInput) -> bool {
    matches!(
        input,
        ToolInput::EditFile { .. } | ToolInput::WriteFile { .. }
    )
}

/// Outcome of dispatching one round of tool calls.
pub(super) enum ToolRoundOutcome {
    /// All tools in this round completed immediately; results are ready to push.
    Completed {
        results: String,
        git_acquisition_answer: Option<String>,
    },
    /// The runtime has enough information to end the turn without asking the model
    /// for another synthesis pass.
    TerminalAnswer {
        results: String,
        answer: String,
        reason: RuntimeTerminalReason,
    },
    /// A tool requested approval. Results accumulated before it are preserved.
    /// The turn is now suspended; the caller must store pending and fire the event.
    ApprovalRequired {
        accumulated: String,
        pending: PendingAction,
    },

    /// Runtime has selected the next tool call itself.
    /// The caller must re-enter the normal tool execution loop with this call;
    /// it must not dispatch the tool inline.
    RuntimeDispatch {
        accumulated: String,
        call: ToolInput,
    },
}

/// Dispatches one round of tool calls, accumulating results.
/// Stops at the first tool that requires approval and returns any results
/// accumulated before it alongside the PendingAction.
/// ToolCallStarted is fired for each tool, but ToolCallFinished is NOT fired
/// for the approval-requiring tool — handle_approve/reject fires it after resolution.
///
/// `last_call_key` carries the fingerprint of the most recently executed call across
/// rounds. If the current call matches it, a cycle error is injected instead of
/// dispatching. The key is updated after every non-cycle, non-approval dispatch.
pub(super) fn run_tool_round(
    registry: &ToolRegistry,
    calls: Vec<ToolInput>,
    last_call_key: &mut Option<String>,
    search_budget: &mut SearchBudget,
    investigation: &mut InvestigationState,
    reads_this_turn: &mut HashSet<String>,
    anchors: &mut AnchorState,
    tool_surface: ToolSurface,
    disallowed_tool_attempts: &mut usize,
    weak_search_query_attempts: &mut usize,
    mutation_allowed: bool,
    investigation_required: bool,
    investigation_mode: InvestigationMode,
    requested_read_path: Option<&str>,
    requested_read_completed: &mut bool,
    investigation_path_scope: Option<&str>,
    on_event: &mut dyn FnMut(RuntimeEvent),
) -> ToolRoundOutcome {
    let mut accumulated = String::new();
    let mut git_answer_sections = Vec::new();

    for mut input in calls {
        simplify_search_input(&mut input);
        // Enforce the prompt-derived path scope as an upper bound on search dispatch.
        // None → inject scope (9.1.2 behavior).
        // Some(p) within scope → keep; model narrowed correctly.
        // Some(p) broader than or orthogonal to scope → clamp silently to scope.
        if let (Some(scope), ToolInput::SearchCode { path, .. }) =
            (investigation_path_scope, &mut input)
        {
            match path {
                None => {
                    trace_runtime_decision(
                        on_event,
                        "search_scope_applied",
                        &[
                            ("action", "inject".into()),
                            ("original_path", "none".into()),
                            ("scope", scope.to_string()),
                            ("final_path", scope.to_string()),
                        ],
                    );
                    *path = Some(scope.to_string());
                }
                Some(ref p) if !path_is_within_scope(p, scope) => {
                    trace_runtime_decision(
                        on_event,
                        "search_scope_applied",
                        &[
                            ("action", "clamp".into()),
                            ("original_path", p.to_string()),
                            ("scope", scope.to_string()),
                            ("final_path", scope.to_string()),
                        ],
                    );
                    *path = Some(scope.to_string());
                }
                _ => {}
            }
        }
        let effective_search_input = match &input {
            ToolInput::SearchCode { query, path } => Some((query.clone(), path.clone())),
            _ => None,
        };
        let read_path = match &input {
            ToolInput::ReadFile { path } => Some(path.clone()),
            _ => None,
        };
        let name = input.tool_name().to_string();
        let key = call_fingerprint(&input);
        let is_git_read_only_tool = is_git_read_only_tool_input(&input);
        on_event(RuntimeEvent::ToolCallStarted { name: name.clone() });

        if !tool_allowed_for_surface(&input, tool_surface) {
            *disallowed_tool_attempts += 1;
            trace_runtime_decision(
                on_event,
                "tool_disallowed",
                &[
                    ("tool", name.clone()),
                    ("surface", tool_surface.as_str().into()),
                    ("attempts", disallowed_tool_attempts.to_string()),
                ],
            );
            on_event(RuntimeEvent::ToolCallFinished {
                name: name.clone(),
                summary: None,
            });
            if *disallowed_tool_attempts == 1 {
                accumulated.push_str(&tool_codec::format_tool_error(
                    &name,
                    surface_policy_correction(tool_surface),
                ));
                continue;
            }
            accumulated.push_str(&tool_codec::format_tool_error(
                &name,
                repeated_disallowed_tool_error(tool_surface),
            ));
            return ToolRoundOutcome::TerminalAnswer {
                results: accumulated,
                answer: repeated_disallowed_tool_final_answer().to_string(),
                reason: RuntimeTerminalReason::RepeatedDisallowedTool,
            };
        }

        if tool_surface == ToolSurface::RetrievalFirst && investigation_required {
            if let ToolInput::SearchCode { query, .. } = &input {
                if let Some(reason) = weak_search_query_reason(query) {
                    *weak_search_query_attempts += 1;
                    trace_runtime_decision(
                        on_event,
                        "weak_search_query_rejected",
                        &[
                            ("query", query.clone()),
                            ("reason", reason.into()),
                            ("attempts", weak_search_query_attempts.to_string()),
                        ],
                    );
                    on_event(RuntimeEvent::ToolCallFinished {
                        name: name.clone(),
                        summary: None,
                    });
                    if *weak_search_query_attempts == 1 {
                        let correction = weak_search_query_correction(reason);
                        accumulated.push_str(&tool_codec::format_tool_error(&name, &correction));
                        continue;
                    }
                    accumulated.push_str(&tool_codec::format_tool_error(
                        &name,
                        "repeated weak search query for this investigation turn.",
                    ));
                    return ToolRoundOutcome::TerminalAnswer {
                        results: accumulated,
                        answer: repeated_weak_search_query_final_answer().to_string(),
                        reason: RuntimeTerminalReason::RepeatedWeakSearchQuery,
                    };
                }
            }
        }

        if is_mutating_tool(&input) && !mutation_allowed {
            on_event(RuntimeEvent::ToolCallFinished {
                name: name.clone(),
                summary: None,
            });
            accumulated.push_str(&tool_codec::format_tool_error(
                &name,
                READ_ONLY_TOOL_POLICY_ERROR,
            ));
            continue;
        }

        if matches!(input, ToolInput::ListDir { .. })
            && investigation_required
            && !investigation.search_attempted()
        {
            on_event(RuntimeEvent::ToolCallFinished {
                name: name.clone(),
                summary: None,
            });
            accumulated.push_str(&tool_codec::format_tool_error(
                &name,
                LIST_DIR_BEFORE_SEARCH_BLOCKED,
            ));
            continue;
        }

        if let (Some(requested), ToolInput::ReadFile { path }) = (requested_read_path, &input) {
            if !path_matches_requested(path, requested) {
                let error = format!(
                    "read_file path `{path}` does not match the requested path `{requested}`"
                );
                on_event(RuntimeEvent::ToolCallFinished {
                    name: name.clone(),
                    summary: None,
                });
                accumulated.push_str(&tool_codec::format_tool_error(&name, &error));
                return ToolRoundOutcome::TerminalAnswer {
                    results: accumulated,
                    answer: read_path_mismatch_final_answer(requested, path),
                    reason: RuntimeTerminalReason::ReadFileFailed,
                };
            }
        }

        // Per-turn search budget: 1 search always allowed; a second only when the first
        // returned no results; further searches are always blocked.
        if matches!(input, ToolInput::SearchCode { .. }) && !search_budget.is_allowed() {
            if search_budget.empty_retry_exhausted()
                && !investigation.search_produced_results()
                && investigation.files_read_count() == 0
            {
                trace_runtime_decision(
                    on_event,
                    "terminal_insufficient_evidence",
                    &[
                        ("reason", "empty_search_retry_exhausted".into()),
                        ("search_calls", search_budget.calls.to_string()),
                        ("files_read", investigation.files_read_count().to_string()),
                    ],
                );
                on_event(RuntimeEvent::ToolCallFinished {
                    name: name.clone(),
                    summary: None,
                });
                return ToolRoundOutcome::TerminalAnswer {
                    results: accumulated,
                    answer: insufficient_evidence_final_answer().to_string(),
                    reason: RuntimeTerminalReason::InsufficientEvidence,
                };
            }
            on_event(RuntimeEvent::ToolCallFinished {
                name: name.clone(),
                summary: None,
            });
            accumulated.push_str(&tool_codec::format_tool_error(
                &name,
                SEARCH_BUDGET_EXCEEDED,
            ));
            continue;
        }

        // Dedup: block re-reads of the same file within the same turn.
        // The file's contents are already in context; re-reading only inflates the prompt.
        if let Some(rp) = read_path.as_deref() {
            let normalized = normalize_evidence_path(rp);
            if reads_this_turn.contains(&normalized) {
                on_event(RuntimeEvent::ToolCallFinished {
                    name: name.clone(),
                    summary: None,
                });
                accumulated.push_str(&tool_codec::format_tool_error(
                    &name,
                    DUPLICATE_READ_REJECTED,
                ));
                continue;
            }
        }

        // Candidate-read cap: once two matched candidates have been read without
        // useful evidence, do not allow the model to keep reading current candidates.
        if investigation_required
            && !investigation.evidence_ready()
            && investigation.candidate_reads_count() >= MAX_CANDIDATE_READS_PER_INVESTIGATION
        {
            if let Some(rp) = read_path.as_deref() {
                if investigation.is_search_candidate_path(rp) {
                    trace_runtime_decision(
                        on_event,
                        "read_evidence",
                        &[
                            ("path", normalize_evidence_path(rp)),
                            ("accepted", "false".into()),
                            ("reason", "candidate_read_limit_exhausted".into()),
                            (
                                "candidate_reads",
                                investigation.candidate_reads_count().to_string(),
                            ),
                        ],
                    );
                    trace_runtime_decision(
                        on_event,
                        "terminal_insufficient_evidence",
                        &[
                            ("reason", "candidate_read_limit_exhausted".into()),
                            (
                                "candidate_reads",
                                investigation.candidate_reads_count().to_string(),
                            ),
                        ],
                    );
                    on_event(RuntimeEvent::ToolCallFinished {
                        name: name.clone(),
                        summary: None,
                    });
                    accumulated.push_str(&tool_codec::format_tool_error(
                        &name,
                        CANDIDATE_READ_CAP_EXCEEDED,
                    ));
                    return ToolRoundOutcome::TerminalAnswer {
                        results: accumulated,
                        answer: ungrounded_investigation_final_answer().to_string(),
                        reason: RuntimeTerminalReason::InsufficientEvidence,
                    };
                }
            }
        }

        // Per-turn read cap: block new reads once MAX_READS_PER_TURN unique files have been read.
        // reads_this_turn.len() counts only successful reads, so the cap is exact.
        if read_path.is_some() && reads_this_turn.len() >= MAX_READS_PER_TURN {
            on_event(RuntimeEvent::ToolCallFinished {
                name: name.clone(),
                summary: None,
            });
            accumulated.push_str(&tool_codec::format_tool_error(&name, READ_CAP_EXCEEDED));
            continue;
        }

        if last_call_key.as_deref() == Some(key.as_str()) {
            if matches!(input, ToolInput::SearchCode { .. })
                && search_budget.calls > 0
                && search_budget.last_was_empty
                && !investigation.search_produced_results()
                && investigation.files_read_count() == 0
            {
                trace_runtime_decision(
                    on_event,
                    "terminal_insufficient_evidence",
                    &[
                        ("reason", "empty_search_duplicate_retry".into()),
                        ("search_calls", search_budget.calls.to_string()),
                        ("files_read", investigation.files_read_count().to_string()),
                    ],
                );
                on_event(RuntimeEvent::ToolCallFinished {
                    name: name.clone(),
                    summary: None,
                });
                return ToolRoundOutcome::TerminalAnswer {
                    results: accumulated,
                    answer: insufficient_evidence_final_answer().to_string(),
                    reason: RuntimeTerminalReason::InsufficientEvidence,
                };
            }
            let msg = format!("{name} called with identical arguments twice in a row");
            on_event(RuntimeEvent::ToolCallFinished {
                name: name.clone(),
                summary: None,
            });
            accumulated.push_str(&tool_codec::format_tool_error(&name, &msg));
            // Do not update last_call_key: keep the same fingerprint so a third
            // consecutive identical call is also blocked.
            continue;
        }

        match registry.dispatch(input) {
            Ok(ToolRunResult::Immediate(output)) => {
                // Guard: spec must agree that this tool is Immediate.
                // A mismatch means the spec() and run() implementations are out of sync.
                debug_assert!(
                    registry
                        .spec_for(&name)
                        .map(|s| s.execution_kind == ExecutionKind::Immediate)
                        .unwrap_or(true),
                    "tool '{name}' returned Immediate but spec declares RequiresApproval"
                );
                // Record search results against the per-turn budget and investigation state.
                let search_closed_message = if name == "search_code" {
                    if let Some((query, scope)) = effective_search_input.clone() {
                        if let Some((query, scope)) =
                            anchors.record_successful_search(&output, query, scope)
                        {
                            trace_runtime_decision(
                                on_event,
                                "anchor_updated",
                                &[
                                    ("kind", "last_search".into()),
                                    ("query", query),
                                    ("scope", scope.unwrap_or_else(|| "none".into())),
                                ],
                            );
                        }
                    }
                    let was_empty = investigation.record_search_results(
                        &output,
                        effective_search_input.as_ref().map(|(q, _)| q.as_str()),
                        on_event,
                    );
                    search_budget.record(was_empty);
                    search_budget
                        .is_closed()
                        .then(|| search_budget.closed_message())
                } else {
                    None
                };
                // Track successful file reads for evidence grounding and dedup.
                let read_recovery = if name == "read_file" {
                    if let Some(path) = anchors.record_successful_read(&output) {
                        trace_runtime_decision(
                            on_event,
                            "anchor_updated",
                            &[("kind", "last_read_file".into()), ("path", path)],
                        );
                    }
                    let recovery =
                        investigation.record_read_result(&output, investigation_mode, on_event);
                    if let Some(requested) = requested_read_path {
                        if let Some(rp) = read_path.as_deref() {
                            if normalize_evidence_path(rp) == normalize_evidence_path(requested) {
                                *requested_read_completed = true;
                            }
                        }
                    }
                    // Record path so a repeat read in the same turn is blocked.
                    if let Some(rp) = read_path.as_deref() {
                        reads_this_turn.insert(normalize_evidence_path(rp));
                    }
                    recovery
                } else {
                    None
                };
                let summary = tool_codec::render_compact_summary(&output);
                on_event(RuntimeEvent::ToolCallFinished {
                    name: name.clone(),
                    summary: Some(summary),
                });
                if is_git_read_only_tool {
                    git_answer_sections.push(git_acquisition_answer_section(
                        &name,
                        &tool_codec::render_output(&output),
                    ));
                }
                let result_formatted =
                    if name == "search_code"
                        && matches!(investigation_mode, InvestigationMode::DefinitionLookup)
                    {
                        tool_codec::format_tool_result_definition_ordered(&name, &output)
                    } else {
                        tool_codec::format_tool_result(&name, &output)
                    };
                accumulated.push_str(&result_formatted);
                if name == "search_code" {
                    if let Some(hint) =
                        investigation.candidate_preference_hint(investigation_mode)
                    {
                        accumulated.push_str(&hint);
                        accumulated.push_str("\n\n");
                    }
                }
                if let Some((path, kind)) = read_recovery {
                    trace_runtime_decision(
                        on_event,
                        "recovery_issued",
                        &[("kind", kind.as_str().into()), ("path", path.clone())],
                    );
                    let correction = match kind {
                        RecoveryKind::DefinitionOnly | RecoveryKind::NonDefinitionSite => {
                            return ToolRoundOutcome::RuntimeDispatch {
                                accumulated,
                                call: ToolInput::ReadFile { path },
                            };
                        }
                        RecoveryKind::ImportOnly => import_read_recovery_correction(&path),
                        RecoveryKind::ConfigFile => config_read_recovery_correction(&path),
                        RecoveryKind::Initialization => {
                            initialization_read_recovery_correction(&path)
                        }
                        RecoveryKind::Create => create_read_recovery_correction(&path),
                        RecoveryKind::Register => register_read_recovery_correction(&path),
                        RecoveryKind::Load => load_read_recovery_correction(&path),
                        RecoveryKind::Save => save_read_recovery_correction(&path),
                        RecoveryKind::Lockfile => lockfile_read_recovery_correction(&path),
                    };
                    accumulated.push_str(&correction);
                    accumulated.push_str("\n\n");
                }
                if let Some(message) = search_closed_message {
                    accumulated.push_str(message);
                    accumulated.push_str("\n\n");
                }
                *last_call_key = Some(key);
            }
            Ok(ToolRunResult::Approval(pending)) => {
                // Guard: spec must agree that this tool requires approval.
                debug_assert!(
                    registry
                        .spec_for(&name)
                        .map(|s| s.execution_kind == ExecutionKind::RequiresApproval)
                        .unwrap_or(true),
                    "tool '{name}' returned Approval but spec declares Immediate"
                );
                return ToolRoundOutcome::ApprovalRequired {
                    accumulated,
                    pending,
                };
            }
            Err(e) => {
                let error = e.to_string();
                on_event(RuntimeEvent::ToolCallFinished {
                    name: name.clone(),
                    summary: None,
                });
                if is_git_read_only_tool {
                    git_answer_sections.push(git_acquisition_answer_section(&name, &error));
                }
                accumulated.push_str(&tool_codec::format_tool_error(&name, &error));
                if let Some(path) = read_path {
                    return ToolRoundOutcome::TerminalAnswer {
                        results: accumulated,
                        answer: read_failure_final_answer(&path, &error),
                        reason: RuntimeTerminalReason::ReadFileFailed,
                    };
                }
                // Do NOT update last_call_key on error: a failed call should not block
                // an identical retry. Cycle detection applies only to successful executions.
            }
        }
    }

    ToolRoundOutcome::Completed {
        results: accumulated,
        git_acquisition_answer: render_git_acquisition_answer(git_answer_sections),
    }
}
