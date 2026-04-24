use std::collections::HashSet;
use std::path::Path;

use crate::app::config::Config;
use crate::llm::backend::ModelBackend;
use crate::tools::{ExecutionKind, PendingAction, ToolInput, ToolRegistry, ToolRunResult};

use super::anchors::{
    has_same_scope_reference, is_last_read_file_anchor_prompt, is_last_search_anchor_prompt,
    AnchorState,
};
use super::conversation::Conversation;
use super::generation::{emit_visible_assistant_message, run_generate_turn};
use super::investigation::{detect_investigation_mode, InvestigationMode, InvestigationState};
use super::prompt;
use super::tool_codec;
use super::tool_round::{
    run_tool_round, SearchBudget, ToolRoundOutcome, MAX_CANDIDATE_READS_PER_INVESTIGATION,
};
use super::types::{Activity, AnswerSource, RuntimeEvent, RuntimeRequest, RuntimeTerminalReason};

/// Maximum tool rounds per turn. Prevents runaway loops when the model keeps
/// producing tool calls without reaching a final answer.
const MAX_TOOL_ROUNDS: usize = 10;

/// Maximum automatic corrections per turn. One correction is enough — if the
/// model fabricates twice in a row the prompt fix is insufficient and we surface
/// the failure rather than looping silently.
const MAX_CORRECTIONS: usize = 1;

use super::response_text::*;
use super::trace::trace_runtime_decision;

fn trace_insufficient_evidence_terminal(
    reason: &str,
    tool_rounds: usize,
    search_budget: &SearchBudget,
    investigation: &InvestigationState,
    on_event: &mut dyn FnMut(RuntimeEvent),
) {
    trace_runtime_decision(
        on_event,
        "terminal_insufficient_evidence",
        &[
            ("reason", reason.to_string()),
            ("rounds", tool_rounds.to_string()),
            ("search_calls", search_budget.calls.to_string()),
            (
                "search_produced_results",
                investigation.search_produced_results().to_string(),
            ),
            ("files_read", investigation.files_read_count().to_string()),
            (
                "candidate_reads",
                investigation.candidate_reads_count().to_string(),
            ),
            ("evidence_ready", investigation.evidence_ready().to_string()),
        ],
    );
}

use super::tool_surface::{select_tool_surface, ToolSurface};

/// Returns true if the prompt contains a token that looks like a code identifier.
/// Only two structural patterns are checked — no NLP, no heuristics.
use super::prompt_analysis::{
    extract_investigation_path_scope, prompt_requires_investigation, requested_read_path,
    user_requested_mutation,
};

pub struct Runtime {
    conversation: Conversation,
    backend: Box<dyn ModelBackend>,
    registry: ToolRegistry,
    system_prompt: String,
    anchors: AnchorState,
    /// Holds a mutating tool action that is waiting for user approval.
    /// Set when a tool round suspends; cleared by Approve or Reject.
    /// At most one pending action exists at any time.
    pending_action: Option<PendingAction>,
}

impl Runtime {
    pub fn new(
        config: &Config,
        project_root: &Path,
        backend: Box<dyn ModelBackend>,
        registry: ToolRegistry,
    ) -> Self {
        let specs = registry.specs();
        let system_prompt = prompt::build_system_prompt(&config.app.name, project_root, &specs);
        Self {
            conversation: Conversation::new(system_prompt.clone()),
            backend,
            registry,
            system_prompt,
            anchors: AnchorState::default(),
            pending_action: None,
        }
    }

    /// Returns a snapshot of all current conversation messages for persistence.
    pub fn messages_snapshot(&self) -> Vec<crate::llm::backend::Message> {
        self.conversation.snapshot()
    }

    /// Appends historical messages into the conversation after the system prompt.
    /// Called once at startup when restoring a prior session. Not for use mid-turn.
    pub fn load_history(&mut self, messages: Vec<crate::llm::backend::Message>) {
        self.conversation.extend_history(messages);
    }

    /// Handles a RuntimeRequest by updating the conversation, invoking the backend,
    /// and firing RuntimeEvents to drive the UI. Each request type has its own
    /// handler method for clarity.
    pub fn handle(&mut self, request: RuntimeRequest, on_event: &mut dyn FnMut(RuntimeEvent)) {
        match request {
            RuntimeRequest::Submit { text } => self.handle_submit(text, on_event),
            RuntimeRequest::Reset => self.handle_reset(on_event),
            RuntimeRequest::Approve => self.handle_approve(on_event),
            RuntimeRequest::Reject => self.handle_reject(on_event),
        }
    }

    fn handle_reset(&mut self, on_event: &mut dyn FnMut(RuntimeEvent)) {
        self.pending_action = None;
        self.anchors.clear();
        trace_runtime_decision(
            on_event,
            "anchor_cleared",
            &[("kind", "last_read_file".into())],
        );
        trace_runtime_decision(
            on_event,
            "anchor_cleared",
            &[("kind", "last_search".into())],
        );
        self.conversation.reset(self.system_prompt.clone());
        on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
    }

    fn handle_submit(&mut self, text: String, on_event: &mut dyn FnMut(RuntimeEvent)) {
        if self.pending_action.is_some() {
            on_event(RuntimeEvent::Failed {
                message:
                    "Cannot submit while a tool approval is pending. Use /approve or /reject first."
                        .to_string(),
            });
            return;
        }

        let trimmed = text.trim();
        if trimmed.is_empty() {
            on_event(RuntimeEvent::Failed {
                message: "Cannot submit an empty prompt.".to_string(),
            });
            return;
        }

        let is_last_read_file_anchor = is_last_read_file_anchor_prompt(trimmed);
        let is_last_search_anchor = is_last_search_anchor_prompt(trimmed);
        self.conversation.push_user(text);
        on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
        if is_last_read_file_anchor {
            trace_runtime_decision(
                on_event,
                "anchor_prompt_matched",
                &[("kind", "last_read_file".into())],
            );
            if let Some(path) = self.anchors.last_read_file().map(str::to_string) {
                trace_runtime_decision(
                    on_event,
                    "anchor_resolved",
                    &[("kind", "last_read_file".into()), ("path", path.clone())],
                );
                self.run_last_read_file_anchor(path, on_event);
            } else {
                trace_runtime_decision(
                    on_event,
                    "anchor_missing",
                    &[("kind", "last_read_file".into())],
                );
                self.finish_with_runtime_answer(
                    NO_LAST_READ_FILE_AVAILABLE,
                    AnswerSource::RuntimeTerminal {
                        reason: RuntimeTerminalReason::ReadFileFailed,
                        rounds: 0,
                    },
                    on_event,
                );
            }
            return;
        }
        if is_last_search_anchor {
            trace_runtime_decision(
                on_event,
                "anchor_prompt_matched",
                &[("kind", "last_search".into())],
            );
            if let Some((query, scope)) = self.anchors.last_search() {
                trace_runtime_decision(
                    on_event,
                    "anchor_resolved",
                    &[
                        ("kind", "last_search".into()),
                        ("query", query.clone()),
                        ("scope", scope.clone().unwrap_or_else(|| "none".into())),
                    ],
                );
                self.run_last_search_anchor(query, scope, on_event);
            } else {
                trace_runtime_decision(
                    on_event,
                    "anchor_missing",
                    &[("kind", "last_search".into())],
                );
                self.finish_with_runtime_answer(
                    NO_LAST_SEARCH_AVAILABLE,
                    AnswerSource::RuntimeTerminal {
                        reason: RuntimeTerminalReason::InsufficientEvidence,
                        rounds: 0,
                    },
                    on_event,
                );
            }
            return;
        }
        self.run_turns(0, on_event);
    }

    fn run_last_read_file_anchor(&mut self, path: String, on_event: &mut dyn FnMut(RuntimeEvent)) {
        let mut last_call_key: Option<String> = None;
        let mut search_budget = SearchBudget::new();
        let mut investigation = InvestigationState::new();
        let mut reads_this_turn: HashSet<String> = HashSet::new();
        let mut requested_read_completed = false;
        let mut disallowed_tool_attempts = 0usize;
        let mut weak_search_query_attempts = 0usize;

        on_event(RuntimeEvent::ActivityChanged(Activity::ExecutingTools));
        match run_tool_round(
            &self.registry,
            vec![ToolInput::ReadFile { path }],
            &mut last_call_key,
            &mut search_budget,
            &mut investigation,
            &mut reads_this_turn,
            &mut self.anchors,
            ToolSurface::RetrievalFirst,
            &mut disallowed_tool_attempts,
            &mut weak_search_query_attempts,
            false,
            false,
            InvestigationMode::General,
            None,
            &mut requested_read_completed,
            None,
            on_event,
        ) {
            ToolRoundOutcome::Completed { results, .. } => {
                self.conversation.push_user(results);
                self.conversation.trim_tool_exchanges_if_needed();
                on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
                self.run_turns_with_initial_reads(1, reads_this_turn, on_event);
            }
            ToolRoundOutcome::TerminalAnswer {
                results,
                answer,
                reason,
            } => {
                self.conversation.push_user(results);
                self.conversation.trim_tool_exchanges_if_needed();
                self.finish_with_runtime_answer(
                    &answer,
                    AnswerSource::RuntimeTerminal { reason, rounds: 1 },
                    on_event,
                );
            }
            ToolRoundOutcome::ApprovalRequired {
                accumulated,
                pending,
            } => {
                if !accumulated.is_empty() {
                    self.conversation.push_user(accumulated);
                    self.conversation.trim_tool_exchanges_if_needed();
                }
                self.pending_action = Some(pending.clone());
                on_event(RuntimeEvent::ApprovalRequired(pending));
                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
            }
        }
    }

    fn run_last_search_anchor(
        &mut self,
        query: String,
        scope: Option<String>,
        on_event: &mut dyn FnMut(RuntimeEvent),
    ) {
        let input = ToolInput::SearchCode {
            query: query.clone(),
            path: scope.clone(),
        };
        let name = input.tool_name().to_string();

        on_event(RuntimeEvent::ActivityChanged(Activity::ExecutingTools));
        on_event(RuntimeEvent::ToolCallStarted { name: name.clone() });

        match self.registry.dispatch(input) {
            Ok(ToolRunResult::Immediate(output)) => {
                debug_assert!(
                    self.registry
                        .spec_for(&name)
                        .map(|s| s.execution_kind == ExecutionKind::Immediate)
                        .unwrap_or(true),
                    "tool '{name}' returned Immediate but spec declares RequiresApproval"
                );
                if let Some((query, scope)) =
                    self.anchors
                        .record_successful_search(&output, query.clone(), scope.clone())
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
                let summary = tool_codec::render_compact_summary(&output);
                on_event(RuntimeEvent::ToolCallFinished {
                    name: name.clone(),
                    summary: Some(summary),
                });
                self.conversation
                    .push_user(tool_codec::format_tool_result(&name, &output));
                self.conversation.trim_tool_exchanges_if_needed();
                self.finish_with_runtime_answer(
                    LAST_SEARCH_REPLAYED,
                    AnswerSource::ToolAssisted { rounds: 1 },
                    on_event,
                );
            }
            Ok(ToolRunResult::Approval(pending)) => {
                debug_assert!(
                    self.registry
                        .spec_for(&name)
                        .map(|s| s.execution_kind == ExecutionKind::RequiresApproval)
                        .unwrap_or(false),
                    "tool '{name}' requested approval but spec declares Immediate"
                );
                self.pending_action = Some(pending.clone());
                on_event(RuntimeEvent::ApprovalRequired(pending));
                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
            }
            Err(e) => {
                on_event(RuntimeEvent::ToolCallFinished {
                    name: name.clone(),
                    summary: None,
                });
                self.conversation
                    .push_user(tool_codec::format_tool_error(&name, &e.to_string()));
                self.conversation.trim_tool_exchanges_if_needed();
                self.finish_with_runtime_answer(
                    LAST_SEARCH_REPLAY_FAILED,
                    AnswerSource::RuntimeTerminal {
                        reason: RuntimeTerminalReason::InsufficientEvidence,
                        rounds: 1,
                    },
                    on_event,
                );
            }
        }
    }

    fn handle_approve(&mut self, on_event: &mut dyn FnMut(RuntimeEvent)) {
        let pending = match self.pending_action.take() {
            Some(p) => p,
            None => {
                on_event(RuntimeEvent::Failed {
                    message: "No pending action to approve.".to_string(),
                });
                return;
            }
        };

        on_event(RuntimeEvent::ActivityChanged(Activity::ExecutingTools));
        let tool_name = pending.tool_name.clone();

        match self.registry.execute_approved(&pending) {
            Ok(output) => {
                let summary = tool_codec::render_compact_summary(&output);
                let final_answer = mutation_complete_final_answer(&tool_name, &summary);
                on_event(RuntimeEvent::ToolCallFinished {
                    name: tool_name.clone(),
                    summary: Some(summary),
                });
                let result_text = tool_codec::format_tool_result(&tool_name, &output);
                self.conversation.push_user(result_text);
                self.conversation.trim_tool_exchanges_if_needed();
                self.finish_with_runtime_answer(
                    &final_answer,
                    AnswerSource::ToolAssisted { rounds: 1 },
                    on_event,
                );
            }
            Err(e) => {
                on_event(RuntimeEvent::ToolCallFinished {
                    name: tool_name.clone(),
                    summary: None,
                });
                let error_text = tool_codec::format_tool_error(&tool_name, &e.to_string());
                self.conversation.push_user(error_text);
                // On failure, let the model respond — it may want to retry.
                on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
                self.run_turns(0, on_event);
            }
        }
    }

    fn handle_reject(&mut self, on_event: &mut dyn FnMut(RuntimeEvent)) {
        let pending = match self.pending_action.take() {
            Some(p) => p,
            None => {
                on_event(RuntimeEvent::Failed {
                    message: "No pending action to reject.".to_string(),
                });
                return;
            }
        };

        let tool_name = pending.tool_name.clone();
        on_event(RuntimeEvent::ToolCallFinished {
            name: tool_name.clone(),
            summary: None,
        });
        let rejection = tool_codec::format_tool_error(
            &tool_name,
            "user rejected this action — do not retry or re-propose it. \
             Acknowledge the cancellation in plain text and wait for the user's next instruction.",
        );
        self.conversation.push_user(rejection);
        self.finish_with_runtime_answer(
            rejection_final_answer(&tool_name),
            AnswerSource::RuntimeTerminal {
                reason: RuntimeTerminalReason::RejectedMutation,
                rounds: 1,
            },
            on_event,
        );
    }

    /// Runs the generate -> tool-round loop until the model produces a final answer,
    /// the tool round limit is reached, or a tool action requires approval.
    /// `tool_rounds` is the count already consumed before this call (0 for a fresh turn).
    fn run_turns(&mut self, tool_rounds: usize, on_event: &mut dyn FnMut(RuntimeEvent)) {
        self.run_turns_with_initial_reads(tool_rounds, HashSet::new(), on_event);
    }

    fn run_turns_with_initial_reads(
        &mut self,
        mut tool_rounds: usize,
        mut reads_this_turn: HashSet<String>,
        on_event: &mut dyn FnMut(RuntimeEvent),
    ) {
        #[derive(Clone, Copy)]
        enum AnswerPhaseKind {
            PostRead,
            InvestigationEvidenceReady,
        }

        let mut corrections = 0usize;
        let mut last_call_key: Option<String> = None;
        let mut search_budget = SearchBudget::new();
        let mut investigation = InvestigationState::new();
        let mut requested_read_completed = false;
        let mut read_request_correction_issued = false;
        let mut disallowed_tool_attempts = 0usize;
        let mut weak_search_query_attempts = 0usize;
        let mut answer_phase: Option<AnswerPhaseKind> = None;
        let mut post_answer_phase_tool_attempts = 0usize;
        // Computed once from the original user message. Excludes tool result/error injections
        // and correction messages so the approve-failure path (run_turns(0,...)) is safe.
        let original_user_prompt = self.conversation.last_user_content().filter(|c| {
            !c.starts_with("=== tool_result:")
                && !c.starts_with("=== tool_error:")
                && !c.starts_with("[runtime:correction]")
        });
        let requested_read_path = original_user_prompt.and_then(requested_read_path);
        let investigation_required = original_user_prompt
            .map(|prompt| {
                requested_read_path.is_none()
                    && !user_requested_mutation(prompt)
                    && prompt_requires_investigation(prompt)
            })
            .unwrap_or(false);
        let mutation_allowed = original_user_prompt
            .map(user_requested_mutation)
            .unwrap_or(false);
        let tool_surface = original_user_prompt
            .map(select_tool_surface)
            .unwrap_or(ToolSurface::RetrievalFirst);
        let investigation_mode = original_user_prompt
            .map(detect_investigation_mode)
            .unwrap_or(InvestigationMode::General);
        let explicit_investigation_path_scope: Option<String> = if investigation_required {
            original_user_prompt.and_then(extract_investigation_path_scope)
        } else {
            None
        };
        let same_scope_reference = investigation_required
            && explicit_investigation_path_scope.is_none()
            && original_user_prompt.is_some_and(has_same_scope_reference);
        let investigation_path_scope: Option<String> =
            if let Some(scope) = explicit_investigation_path_scope {
                Some(scope)
            } else if same_scope_reference {
                trace_runtime_decision(
                    on_event,
                    "anchor_prompt_matched",
                    &[("kind", "same_scope".into())],
                );
                match self.anchors.last_scoped_search_scope().map(str::to_string) {
                    Some(scope) => {
                        trace_runtime_decision(
                            on_event,
                            "anchor_resolved",
                            &[("kind", "same_scope".into()), ("scope", scope.clone())],
                        );
                        Some(scope)
                    }
                    None => {
                        trace_runtime_decision(
                            on_event,
                            "anchor_missing",
                            &[("kind", "same_scope".into())],
                        );
                        self.finish_with_runtime_answer(
                            NO_LAST_SCOPED_SEARCH_AVAILABLE,
                            AnswerSource::RuntimeTerminal {
                                reason: RuntimeTerminalReason::InsufficientEvidence,
                                rounds: tool_rounds,
                            },
                            on_event,
                        );
                        return;
                    }
                }
            } else {
                None
            };
        trace_runtime_decision(
            on_event,
            "investigation_mode_detected",
            &[
                ("mode", investigation_mode.as_str().into()),
                ("required", investigation_required.to_string()),
            ],
        );
        trace_runtime_decision(
            on_event,
            "investigation_path_scope",
            &[(
                "scope",
                investigation_path_scope
                    .as_deref()
                    .unwrap_or("none")
                    .to_string(),
            )],
        );
        trace_runtime_decision(
            on_event,
            "tool_surface_selected",
            &[("surface", tool_surface.as_str().into())],
        );
        loop {
            let response = match run_generate_turn(
                self.backend.as_mut(),
                &mut self.conversation,
                tool_surface,
                on_event,
            ) {
                Ok(Some(r)) => r,
                Ok(None) => {
                    on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                    on_event(RuntimeEvent::Failed {
                        message: format!("{} returned no output.", self.backend.name()),
                    });
                    return;
                }
                Err(e) => {
                    on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                    on_event(RuntimeEvent::Failed {
                        message: e.to_string(),
                    });
                    return;
                }
            };

            let calls = tool_codec::parse_all_tool_inputs(&response);

            if let Some(phase) = answer_phase {
                if !calls.is_empty() {
                    post_answer_phase_tool_attempts += 1;
                    if matches!(phase, AnswerPhaseKind::InvestigationEvidenceReady) {
                        trace_runtime_decision(
                            on_event,
                            "post_evidence_tool_call_rejected",
                            &[
                                ("attempts", post_answer_phase_tool_attempts.to_string()),
                                ("tool_count", calls.len().to_string()),
                            ],
                        );
                    }
                    self.conversation.discard_last_if_assistant();
                    if post_answer_phase_tool_attempts == 1 {
                        self.conversation.push_user(
                            match phase {
                                AnswerPhaseKind::PostRead => TURN_COMPLETE_ANSWER_ONLY,
                                AnswerPhaseKind::InvestigationEvidenceReady => {
                                    EVIDENCE_READY_ANSWER_ONLY
                                }
                            }
                            .to_string(),
                        );
                        continue;
                    }
                    let (answer, reason) = match phase {
                        AnswerPhaseKind::PostRead => (
                            repeated_tool_after_answer_phase_final_answer(),
                            RuntimeTerminalReason::RepeatedToolAfterAnswerPhase,
                        ),
                        AnswerPhaseKind::InvestigationEvidenceReady => (
                            repeated_tool_after_evidence_ready_final_answer(),
                            RuntimeTerminalReason::RepeatedToolAfterEvidenceReady,
                        ),
                    };
                    self.finish_with_runtime_answer(
                        answer,
                        AnswerSource::RuntimeTerminal {
                            reason,
                            rounds: tool_rounds,
                        },
                        on_event,
                    );
                    return;
                }
            }

            if search_budget.is_closed()
                && calls
                    .iter()
                    .any(|c| matches!(c, ToolInput::SearchCode { .. }))
            {
                if search_budget.empty_retry_exhausted()
                    && !investigation.search_produced_results()
                    && investigation.files_read_count() == 0
                {
                    trace_insufficient_evidence_terminal(
                        "empty_search_retry_exhausted",
                        tool_rounds,
                        &search_budget,
                        &investigation,
                        on_event,
                    );
                    self.conversation.discard_last_if_assistant();
                    self.finish_with_runtime_answer(
                        insufficient_evidence_final_answer(),
                        AnswerSource::RuntimeTerminal {
                            reason: RuntimeTerminalReason::InsufficientEvidence,
                            rounds: tool_rounds,
                        },
                        on_event,
                    );
                    return;
                }
                if corrections < MAX_CORRECTIONS {
                    corrections += 1;
                    self.conversation.discard_last_if_assistant();
                    self.conversation
                        .push_user(search_budget.closed_message().to_string());
                    continue;
                }
                on_event(RuntimeEvent::Failed {
                    message: "Model kept searching after the search budget was closed.".to_string(),
                });
                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                return;
            }

            if calls.is_empty() {
                // If the previous tool round ended in an edit_file error and the model's repair
                // attempt contains edit_file tag syntax but produced no parseable tool calls,
                // inject a targeted correction rather than silently accepting as Direct.
                if tool_codec::contains_edit_attempt(&response)
                    && last_injected_was_edit_error(&self.conversation)
                    && corrections < MAX_CORRECTIONS
                {
                    corrections += 1;
                    self.conversation.discard_last_if_assistant();
                    self.conversation
                        .push_user(EDIT_REPAIR_CORRECTION.to_string());
                    continue;
                }

                // Fabricated [tool_result:] / [tool_error:] blocks mean the model bypassed the
                // protocol. Attempt one automatic correction before surfacing the error.
                if tool_codec::contains_fabricated_exchange(&response) {
                    if corrections < MAX_CORRECTIONS {
                        corrections += 1;
                        self.conversation.discard_last_if_assistant();
                        self.conversation
                            .push_user(FABRICATION_CORRECTION.to_string());
                        continue;
                    }
                    on_event(RuntimeEvent::Failed {
                        message: "Model repeatedly produced fabricated tool results. Try rephrasing your request.".to_string(),
                    });
                    on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                    return;
                }
                // Malformed block: a known closing tag ([/write_file], [/edit_file], etc.)
                // is present without the matching opening tag. The model used a wrong tag name.
                // Attempt one correction before giving up.
                if tool_codec::contains_malformed_block(&response) {
                    if corrections < MAX_CORRECTIONS {
                        corrections += 1;
                        self.conversation.discard_last_if_assistant();
                        self.conversation
                            .push_user(MALFORMED_BLOCK_CORRECTION.to_string());
                        continue;
                    }
                    on_event(RuntimeEvent::Failed {
                        message:
                            "Model used incorrect tool tag names. Try rephrasing your request."
                                .to_string(),
                    });
                    on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                    return;
                }

                if let Some(path) = requested_read_path.as_deref() {
                    if !requested_read_completed {
                        if !read_request_correction_issued && corrections < MAX_CORRECTIONS {
                            corrections += 1;
                            read_request_correction_issued = true;
                            self.conversation.push_user(format!(
                                "{READ_REQUEST_TOOL_REQUIRED} Requested path: `{path}`"
                            ));
                            continue;
                        }

                        self.finish_with_runtime_answer(
                            &unread_requested_file_final_answer(path),
                            AnswerSource::RuntimeTerminal {
                                reason: RuntimeTerminalReason::ReadFileFailed,
                                rounds: tool_rounds,
                            },
                            on_event,
                        );
                        return;
                    }
                }

                // R4: insufficient-evidence terminal.
                // Search was attempted this turn, all results were empty, and no file
                // was read. The model cannot have any grounded evidence to synthesize from.
                // Discard whatever the model produced and emit the runtime-owned answer.
                if search_budget.calls > 0
                    && !investigation.search_produced_results()
                    && investigation.files_read_count() == 0
                {
                    trace_insufficient_evidence_terminal(
                        "empty_search_no_read",
                        tool_rounds,
                        &search_budget,
                        &investigation,
                        on_event,
                    );
                    self.finish_with_runtime_answer(
                        insufficient_evidence_final_answer(),
                        AnswerSource::RuntimeTerminal {
                            reason: RuntimeTerminalReason::InsufficientEvidence,
                            rounds: tool_rounds,
                        },
                        on_event,
                    );
                    return;
                }

                if investigation_required && !investigation.evidence_ready() {
                    if search_budget.calls == 0 {
                        if investigation.issue_direct_answer_correction() {
                            self.conversation
                                .push_user(SEARCH_BEFORE_ANSWERING.to_string());
                            continue;
                        }

                        trace_insufficient_evidence_terminal(
                            "no_search_after_direct_answer_correction",
                            tool_rounds,
                            &search_budget,
                            &investigation,
                            on_event,
                        );
                        self.finish_with_runtime_answer(
                            ungrounded_investigation_final_answer(),
                            AnswerSource::RuntimeTerminal {
                                reason: RuntimeTerminalReason::InsufficientEvidence,
                                rounds: tool_rounds,
                            },
                            on_event,
                        );
                        return;
                    }

                    if investigation.search_produced_results() {
                        // Both candidate-read slots exhausted and evidence is still not ready.
                        // Do not attempt another correction cycle — terminate cleanly.
                        if investigation.candidate_reads_count()
                            >= MAX_CANDIDATE_READS_PER_INVESTIGATION
                        {
                            trace_insufficient_evidence_terminal(
                                "candidate_read_limit_exhausted",
                                tool_rounds,
                                &search_budget,
                                &investigation,
                                on_event,
                            );
                            self.finish_with_runtime_answer(
                                ungrounded_investigation_final_answer(),
                                AnswerSource::RuntimeTerminal {
                                    reason: RuntimeTerminalReason::InsufficientEvidence,
                                    rounds: tool_rounds,
                                },
                                on_event,
                            );
                            return;
                        }

                        if corrections < MAX_CORRECTIONS
                            && investigation.issue_premature_synthesis_correction()
                        {
                            corrections += 1;
                            self.conversation.discard_last_if_assistant();
                            self.conversation
                                .push_user(READ_BEFORE_ANSWERING.to_string());
                            continue;
                        }

                        trace_insufficient_evidence_terminal(
                            "read_required_correction_unavailable",
                            tool_rounds,
                            &search_budget,
                            &investigation,
                            on_event,
                        );
                        self.finish_with_runtime_answer(
                            ungrounded_investigation_final_answer(),
                            AnswerSource::RuntimeTerminal {
                                reason: RuntimeTerminalReason::InsufficientEvidence,
                                rounds: tool_rounds,
                            },
                            on_event,
                        );
                        return;
                    }
                }

                let source = if tool_rounds == 0 {
                    AnswerSource::Direct
                } else {
                    AnswerSource::ToolAssisted {
                        rounds: tool_rounds,
                    }
                };
                emit_visible_assistant_message(&response, on_event);
                on_event(RuntimeEvent::AnswerReady(source));
                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                return;
            }

            tool_rounds += 1;

            if tool_rounds >= MAX_TOOL_ROUNDS {
                on_event(RuntimeEvent::AnswerReady(AnswerSource::ToolLimitReached));
                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                return;
            }

            on_event(RuntimeEvent::ActivityChanged(Activity::ExecutingTools));

            match run_tool_round(
                &self.registry,
                calls,
                &mut last_call_key,
                &mut search_budget,
                &mut investigation,
                &mut reads_this_turn,
                &mut self.anchors,
                tool_surface,
                &mut disallowed_tool_attempts,
                &mut weak_search_query_attempts,
                mutation_allowed,
                investigation_required,
                investigation_mode,
                requested_read_path.as_deref(),
                &mut requested_read_completed,
                investigation_path_scope.as_deref(),
                on_event,
            ) {
                ToolRoundOutcome::Completed {
                    results,
                    git_acquisition_answer,
                } => {
                    self.conversation.push_user(results);
                    self.conversation.trim_tool_exchanges_if_needed();
                    if tool_surface == ToolSurface::GitReadOnly {
                        if let Some(answer) = git_acquisition_answer {
                            trace_runtime_decision(
                                on_event,
                                "git_acquisition_completed",
                                &[("rounds", tool_rounds.to_string())],
                            );
                            self.finish_with_runtime_answer(
                                &answer,
                                AnswerSource::ToolAssisted {
                                    rounds: tool_rounds,
                                },
                                on_event,
                            );
                            return;
                        }
                    }
                    if answer_phase.is_none() {
                        if investigation_required && investigation.evidence_ready() {
                            answer_phase = Some(AnswerPhaseKind::InvestigationEvidenceReady);
                        } else if !investigation_required && !reads_this_turn.is_empty() {
                            answer_phase = Some(AnswerPhaseKind::PostRead);
                        }
                    }
                    // Signal re-entry before the next generate so the status bar
                    // transitions cleanly from "executing tools" → "processing" → …
                    on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
                    // Do not return — loop continues so the model is re-invoked
                    // with the tool results in context to produce a synthesis response.
                }
                ToolRoundOutcome::TerminalAnswer {
                    results,
                    answer,
                    reason,
                } => {
                    self.conversation.push_user(results);
                    self.conversation.trim_tool_exchanges_if_needed();
                    self.finish_with_runtime_answer(
                        &answer,
                        AnswerSource::RuntimeTerminal {
                            reason,
                            rounds: tool_rounds,
                        },
                        on_event,
                    );
                    return;
                }
                ToolRoundOutcome::ApprovalRequired {
                    accumulated,
                    pending,
                } => {
                    if !accumulated.is_empty() {
                        self.conversation.push_user(accumulated);
                        self.conversation.trim_tool_exchanges_if_needed();
                    }
                    self.pending_action = Some(pending.clone());
                    on_event(RuntimeEvent::ApprovalRequired(pending));
                    on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                    return;
                }
            }
        }
    }

    fn finish_with_runtime_answer(
        &mut self,
        answer: &str,
        source: AnswerSource,
        on_event: &mut dyn FnMut(RuntimeEvent),
    ) {
        on_event(RuntimeEvent::ActivityChanged(Activity::Responding));
        self.conversation.begin_assistant_reply();
        on_event(RuntimeEvent::AssistantMessageStarted);
        self.conversation.push_assistant_chunk(answer);
        on_event(RuntimeEvent::AssistantMessageChunk(answer.to_string()));
        on_event(RuntimeEvent::AssistantMessageFinished);
        on_event(RuntimeEvent::AnswerReady(source));
        on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
    }

    #[cfg(test)]
    pub(crate) fn set_pending_for_test(&mut self, action: PendingAction) {
        self.pending_action = Some(action);
    }
}

/// Returns true when the most recent user message in the conversation is an edit_file
/// tool error injected by the runtime. Used to detect the edit-repair failure pattern:
/// model emits garbled edit syntax after a failed edit, producing zero parsed tool calls.
fn last_injected_was_edit_error(conversation: &Conversation) -> bool {
    conversation
        .last_user_content()
        .map(|c| c.starts_with("=== tool_error: edit_file ==="))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::Config;
    use crate::llm::backend::{BackendEvent, GenerateRequest};
    use crate::tools::default_registry;

    struct TestBackend {
        responses: Vec<String>,
        call_count: usize,
    }

    impl TestBackend {
        fn new(responses: Vec<impl Into<String>>) -> Self {
            Self {
                responses: responses.into_iter().map(Into::into).collect(),
                call_count: 0,
            }
        }
    }

    impl ModelBackend for TestBackend {
        fn name(&self) -> &str {
            "test"
        }

        fn generate(
            &mut self,
            _request: GenerateRequest,
            on_event: &mut dyn FnMut(BackendEvent),
        ) -> crate::app::Result<()> {
            let reply = self
                .responses
                .get(self.call_count)
                .cloned()
                .unwrap_or_default();
            self.call_count += 1;
            if !reply.is_empty() {
                on_event(BackendEvent::TextDelta(reply));
            }
            on_event(BackendEvent::Finished);
            Ok(())
        }
    }

    fn make_runtime_in(responses: Vec<impl Into<String>>, root: &std::path::Path) -> Runtime {
        Runtime::new(
            &Config::default(),
            root,
            Box::new(TestBackend::new(responses)),
            default_registry(root.to_path_buf()),
        )
    }

    fn collect_events(runtime: &mut Runtime, request: RuntimeRequest) -> Vec<RuntimeEvent> {
        let mut events = Vec::new();
        runtime.handle(request, &mut |e| events.push(e));
        events
    }

    fn has_failed(events: &[RuntimeEvent]) -> bool {
        events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::Failed { .. }))
    }

    #[test]
    fn search_anchor_stores_effective_clamped_scope() {
        use std::collections::HashSet;
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("sandbox/in_scope.py"), "needle = True\n").unwrap();
        fs::write(tmp.path().join("src/outside.py"), "needle = False\n").unwrap();

        let registry = default_registry(tmp.path().to_path_buf());
        let mut last_call_key = None;
        let mut search_budget = SearchBudget::new();
        let mut investigation = InvestigationState::new();
        let mut reads_this_turn = HashSet::new();
        let mut anchors = AnchorState::default();
        let mut requested_read_completed = false;
        let mut disallowed_tool_attempts = 0usize;
        let mut weak_search_query_attempts = 0usize;
        let mut events = Vec::new();

        let outcome = run_tool_round(
            &registry,
            vec![ToolInput::SearchCode {
                query: "needle".into(),
                path: Some("src/".into()),
            }],
            &mut last_call_key,
            &mut search_budget,
            &mut investigation,
            &mut reads_this_turn,
            &mut anchors,
            ToolSurface::RetrievalFirst,
            &mut disallowed_tool_attempts,
            &mut weak_search_query_attempts,
            false,
            true,
            InvestigationMode::UsageLookup,
            None,
            &mut requested_read_completed,
            Some("sandbox/"),
            &mut |e| events.push(e),
        );

        assert!(
            matches!(outcome, ToolRoundOutcome::Completed { .. }),
            "search round must complete"
        );
        assert_eq!(anchors.last_search_query(), Some("needle"));
        assert_eq!(anchors.last_search_scope(), Some("sandbox/"));
    }

    #[test]
    fn failed_search_code_does_not_update_last_search_anchor() {
        use std::collections::HashSet;
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("a.rs"), "fn needle() {}\n").unwrap();
        let registry = default_registry(tmp.path().to_path_buf());
        let mut last_call_key = None;
        let mut search_budget = SearchBudget::new();
        let mut investigation = InvestigationState::new();
        let mut reads_this_turn = HashSet::new();
        let mut anchors = AnchorState::default();
        let mut requested_read_completed = false;
        let mut disallowed_tool_attempts = 0usize;
        let mut weak_search_query_attempts = 0usize;
        let mut events = Vec::new();

        let seed_outcome = run_tool_round(
            &registry,
            vec![ToolInput::SearchCode {
                query: "needle".into(),
                path: Some("sandbox/".into()),
            }],
            &mut last_call_key,
            &mut search_budget,
            &mut investigation,
            &mut reads_this_turn,
            &mut anchors,
            ToolSurface::RetrievalFirst,
            &mut disallowed_tool_attempts,
            &mut weak_search_query_attempts,
            false,
            false,
            InvestigationMode::General,
            None,
            &mut requested_read_completed,
            None,
            &mut |e| events.push(e),
        );
        assert!(
            matches!(seed_outcome, ToolRoundOutcome::Completed { .. }),
            "seed search round must complete"
        );
        assert_eq!(anchors.last_search_query(), Some("needle"));
        assert_eq!(anchors.last_search_scope(), Some("sandbox/"));

        let outcome = run_tool_round(
            &registry,
            vec![ToolInput::SearchCode {
                query: "".into(),
                path: None,
            }],
            &mut last_call_key,
            &mut search_budget,
            &mut investigation,
            &mut reads_this_turn,
            &mut anchors,
            ToolSurface::RetrievalFirst,
            &mut disallowed_tool_attempts,
            &mut weak_search_query_attempts,
            false,
            false,
            InvestigationMode::General,
            None,
            &mut requested_read_completed,
            None,
            &mut |e| events.push(e),
        );

        assert!(
            matches!(outcome, ToolRoundOutcome::Completed { .. }),
            "failed non-read tool should return completed with tool error"
        );
        assert_eq!(anchors.last_search_query(), Some("needle"));
        assert_eq!(anchors.last_search_scope(), Some("sandbox/"));
    }
    #[test]
    fn unsupported_search_anchor_phrases_do_not_resolve() {
        assert!(!is_last_search_anchor_prompt("search it again"));
        assert!(!is_last_search_anchor_prompt("search for that thing again"));
        assert!(!is_last_search_anchor_prompt("search again"));
        assert!(is_last_search_anchor_prompt("search that again"));
        assert!(is_last_search_anchor_prompt("repeat the last search"));
    }

    #[test]
    fn same_scope_followup_after_empty_scope_search_fails_deterministically() {
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let mut rt = make_runtime_in(Vec::<String>::new(), tmp.path());
        let output =
            crate::tools::ToolOutput::SearchResults(crate::tools::types::SearchResultsOutput {
                query: "needle".into(),
                matches: Vec::new(),
                total_matches: 0,
                truncated: false,
            });

        rt.anchors
            .record_successful_search(&output, "needle".into(), Some("   ".into()));
        assert_eq!(rt.anchors.last_search_query(), Some("needle"));
        assert_eq!(rt.anchors.last_search_scope(), None);
        assert_eq!(rt.anchors.last_scoped_search_scope(), None);

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where database is configured in the same folder".into(),
            },
        );

        assert!(
            events.iter().any(|e| matches!(
                e,
                RuntimeEvent::AssistantMessageChunk(chunk)
                    if chunk == NO_LAST_SCOPED_SEARCH_AVAILABLE
            )),
            "empty stored scope must not provide same-scope continuity: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { .. })),
            "empty stored scope must not dispatch tools: {events:?}"
        );
    }

    #[test]
    fn unsupported_same_scope_phrases_do_not_match() {
        assert!(!has_same_scope_reference("Find database in the same place"));
        assert!(!has_same_scope_reference("Find it there"));
        assert!(!has_same_scope_reference("Search the same place"));
        assert!(!has_same_scope_reference("Find database in this folder"));
        assert!(!has_same_scope_reference(
            "Find database in the same folderish"
        ));
        assert!(!has_same_scope_reference(
            "Find database within the same scopekeeper"
        ));
        assert!(has_same_scope_reference("Find database in the same folder"));
        assert!(has_same_scope_reference(
            "Find database within the same directory"
        ));
        assert!(has_same_scope_reference(
            "Find database within the same scope"
        ));
    }

    #[test]
    fn same_scope_forced_broader_path_clamps_to_prior_scoped_search() {
        use std::collections::HashSet;
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(
            tmp.path().join("sandbox/services/logging.py"),
            "def initialize_logging():\n    pass\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("sandbox/services/database.yaml"),
            "database: sqlite:///service.db\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("src/database.yaml"),
            "database: sqlite:///wrong.db\n",
        )
        .unwrap();

        let registry = default_registry(tmp.path().to_path_buf());
        let mut anchors = AnchorState::default();
        let mut events = Vec::new();

        let mut seed_last_call_key = None;
        let mut seed_search_budget = SearchBudget::new();
        let mut seed_investigation = InvestigationState::new();
        let mut seed_reads_this_turn = HashSet::new();
        let mut seed_requested_read_completed = false;
        let mut seed_disallowed_tool_attempts = 0usize;
        let mut seed_weak_search_query_attempts = 0usize;
        let seed_outcome = run_tool_round(
            &registry,
            vec![ToolInput::SearchCode {
                query: "logging".into(),
                path: Some("sandbox/services/".into()),
            }],
            &mut seed_last_call_key,
            &mut seed_search_budget,
            &mut seed_investigation,
            &mut seed_reads_this_turn,
            &mut anchors,
            ToolSurface::RetrievalFirst,
            &mut seed_disallowed_tool_attempts,
            &mut seed_weak_search_query_attempts,
            false,
            true,
            InvestigationMode::InitializationLookup,
            None,
            &mut seed_requested_read_completed,
            None,
            &mut |e| events.push(e),
        );
        assert!(
            matches!(seed_outcome, ToolRoundOutcome::Completed { .. }),
            "seed scoped search must complete"
        );
        assert_eq!(
            anchors.last_scoped_search_scope(),
            Some("sandbox/services/")
        );

        let same_scope = anchors
            .last_scoped_search_scope()
            .map(str::to_string)
            .expect("seeded scoped search");
        let mut last_call_key = None;
        let mut search_budget = SearchBudget::new();
        let mut investigation = InvestigationState::new();
        let mut reads_this_turn = HashSet::new();
        let mut requested_read_completed = false;
        let mut disallowed_tool_attempts = 0usize;
        let mut weak_search_query_attempts = 0usize;
        let outcome = run_tool_round(
            &registry,
            vec![ToolInput::SearchCode {
                query: "database".into(),
                path: Some("src/".into()),
            }],
            &mut last_call_key,
            &mut search_budget,
            &mut investigation,
            &mut reads_this_turn,
            &mut anchors,
            ToolSurface::RetrievalFirst,
            &mut disallowed_tool_attempts,
            &mut weak_search_query_attempts,
            false,
            true,
            InvestigationMode::ConfigLookup,
            None,
            &mut requested_read_completed,
            Some(&same_scope),
            &mut |e| events.push(e),
        );

        let results = match outcome {
            ToolRoundOutcome::Completed { results, .. } => results,
            _ => panic!("forced same-scope clamp should complete"),
        };
        assert!(
            results.contains("sandbox/services/database.yaml"),
            "clamped same-scope search must include prior scoped path: {results}"
        );
        assert!(
            !results.contains("src/database.yaml"),
            "broader model path must be clamped away from src/: {results}"
        );
        assert_eq!(
            anchors.last_scoped_search_scope(),
            Some("sandbox/services/")
        );
    }

    // Phase 9.1.1 — bounded multi-step investigation

    #[test]
    fn two_candidate_reads_both_insufficient_terminates_cleanly() {
        // Usage lookup: three search candidates (two definition-only + one usage).
        // First read is definition-only → recovery correction fires pointing to usage file.
        // Model ignores correction and reads a second definition-only file.
        // After two candidate reads with evidence still not ready the runtime must
        // terminate cleanly with InsufficientEvidence — no further correction cycles.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("models")).unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("models").join("enums.py"),
            "class TaskStatus(str, Enum):\n    TODO = \"todo\"\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("models").join("alt_enums.py"),
            "class TaskStatus:\n    DONE = \"done\"\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("task_service.py"),
            "from models.enums import TaskStatus\nif task.status == TaskStatus.TODO:\n    pass\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: TaskStatus]",
                // Round 2: reads first definition file.
                // Runtime injects recovery correction pointing to task_service.py.
                "[read_file: models/enums.py]",
                // Round 3: model ignores correction and reads second definition file.
                // candidate_reads_count reaches 2, evidence still not ready.
                "[read_file: models/alt_enums.py]",
                // Round 4: model tries to synthesize without reading usage evidence.
                // Runtime must terminate with InsufficientEvidence — not fire another correction.
                "TaskStatus is defined in models/enums.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is TaskStatus used?".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "turn must terminate cleanly: {events:?}"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(
                answer_source,
                Some(AnswerSource::RuntimeTerminal {
                    reason: RuntimeTerminalReason::InsufficientEvidence,
                    ..
                })
            ),
            "two insufficient candidate reads must produce InsufficientEvidence: {answer_source:?}"
        );

        // The model's premature synthesis must not appear as the last assistant message.
        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some(ungrounded_investigation_final_answer()),
            "last assistant must be the runtime terminal, not model synthesis"
        );
    }

    // Phase 9.1.2 — Path-Scoped Investigation

    // Phase 9.1.4 — Prompt Scope as Search Upper Bound

    // Phase 9.1.3 — Candidate Selection Quality (import-only weak candidate rejection)

    #[test]
    fn config_lookup_second_non_config_candidate_after_recovery_is_not_accepted() {
        // Config lookup: config candidate exists, but the model ignores the config recovery
        // and reads a second non-config candidate. The second read must remain insufficient;
        // after two candidate reads the bounded investigation terminates cleanly.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::create_dir_all(tmp.path().join("config")).unwrap();
        fs::write(
            tmp.path().join("services").join("database.py"),
            "database = os.getenv(\"DATABASE_URL\")\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("database_alt.py"),
            "database = load_from_environment()\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("config").join("database.yaml"),
            "database:\n  url: postgres://localhost/mydb\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: database]",
                "[read_file: services/database.py]",
                "[read_file: services/database_alt.py]",
                "The database is configured in services/database_alt.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is the database configured?".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "turn must terminate cleanly: {events:?}"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(
                answer_source,
                Some(AnswerSource::RuntimeTerminal {
                    reason: RuntimeTerminalReason::InsufficientEvidence,
                    ..
                })
            ),
            "second non-config candidate must not satisfy config evidence: {answer_source:?}"
        );

        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some(ungrounded_investigation_final_answer()),
            "last assistant must be the runtime terminal, not model synthesis"
        );
    }

    // Phase 9.2.2 — Narrow Action-Specific Lookup Satisfaction: Initialization Lookup

    #[test]
    fn initialization_lookup_second_non_initialization_after_recovery_is_not_accepted() {
        // Initialization lookup: initialization candidate exists, but the model ignores
        // recovery and reads a second non-initialization candidate. That second read must
        // remain insufficient; after two candidate reads the runtime terminates cleanly.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("services").join("logging_factory.py"),
            "logger = logging.getLogger(__name__)\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("logging_reader.py"),
            "logging.getLogger(\"reader\")\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("logging_setup.py"),
            "def initialize_logging():\n    logging.basicConfig(level=logging.INFO)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: logging]",
                "[read_file: services/logging_factory.py]",
                "[read_file: services/logging_reader.py]",
                "Logging is initialized in services/logging_reader.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where logging is initialized".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "turn must terminate cleanly: {events:?}"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(
                answer_source,
                Some(AnswerSource::RuntimeTerminal {
                    reason: RuntimeTerminalReason::InsufficientEvidence,
                    ..
                })
            ),
            "second non-initialization candidate must not satisfy evidence: {answer_source:?}"
        );

        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some(ungrounded_investigation_final_answer()),
            "last assistant must be the runtime terminal, not model synthesis"
        );
    }

    #[test]
    fn initialization_lookup_path_scope_keeps_candidates_inside_scope() {
        // Prompt scope must remain the upper bound. The out-of-scope initialization
        // file is stronger-looking but must not appear in search candidates.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/other")).unwrap();
        fs::write(
            tmp.path()
                .join("sandbox/services")
                .join("logging_factory.py"),
            "logger = logging.getLogger(__name__)\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("sandbox/services").join("logging_setup.py"),
            "def initialize_logging():\n    logging.basicConfig(level=logging.INFO)\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("sandbox/other").join("logging_setup.py"),
            "def initialize_logging():\n    logging.basicConfig(level=logging.DEBUG)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: logging]",
                "[read_file: sandbox/services/logging_factory.py]",
                "[read_file: sandbox/services/logging_setup.py]",
                "Logging is initialized in sandbox/services/logging_setup.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where logging is initialized in sandbox/services/".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        let search_result = snapshot
            .iter()
            .find(|m| m.content.contains("=== tool_result: search_code ==="))
            .map(|m| m.content.as_str())
            .unwrap_or("");
        assert!(
            search_result.contains("sandbox/services/logging_factory.py"),
            "scoped search must include in-scope non-initialization candidate: {search_result}"
        );
        assert!(
            search_result.contains("sandbox/services/logging_setup.py"),
            "scoped search must include in-scope initialization candidate: {search_result}"
        );
        assert!(
            !search_result.contains("sandbox/other/logging_setup.py"),
            "scoped search must exclude out-of-scope initialization candidate: {search_result}"
        );

        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Logging is initialized in sandbox/services/logging_setup.py.")
        );
    }

    // Phase 9.2.3 — CreateLookup

    // Phase 9.2.4 — RegisterLookup

    #[test]
    fn register_lookup_path_scope_keeps_candidates_inside_scope() {
        // Prompt scope must remain the upper bound. The out-of-scope registration
        // file is stronger-looking but must not appear in search candidates.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/cli")).unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
        fs::write(
            tmp.path().join("sandbox/cli").join("commands.py"),
            "def command_handler(command):\n    return command.run()\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("sandbox/cli").join("registry.py"),
            "def wire_command(command):\n    registry.register(command)\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("sandbox/services").join("registry.py"),
            "def wire_command(command):\n    registry.register(command)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: command]",
                "[read_file: sandbox/cli/commands.py]",
                "[read_file: sandbox/cli/registry.py]",
                "Commands are registered in sandbox/cli/registry.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where commands are registered in sandbox/cli/".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        let search_result = snapshot
            .iter()
            .find(|m| m.content.contains("=== tool_result: search_code ==="))
            .map(|m| m.content.as_str())
            .unwrap_or("");
        assert!(
            search_result.contains("sandbox/cli/commands.py"),
            "scoped search must include in-scope non-register candidate: {search_result}"
        );
        assert!(
            search_result.contains("sandbox/cli/registry.py"),
            "scoped search must include in-scope register candidate: {search_result}"
        );
        assert!(
            !search_result.contains("sandbox/services/registry.py"),
            "scoped search must exclude out-of-scope register candidate: {search_result}"
        );

        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Commands are registered in sandbox/cli/registry.py.")
        );
    }

    // Phase 9.2.5 — LoadLookup

    #[test]
    fn load_lookup_path_scope_keeps_candidates_inside_scope() {
        // Prompt scope must remain the upper bound. The out-of-scope load
        // file is stronger-looking but must not appear in search candidates.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/controllers")).unwrap();
        fs::write(
            tmp.path()
                .join("sandbox/services")
                .join("session_handler.py"),
            "def handle_session(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path()
                .join("sandbox/services")
                .join("session_loader.py"),
            "def get_session(session_id):\n    return load_session(session_id)\n",
        )
        .unwrap();
        fs::write(
            tmp.path()
                .join("sandbox/controllers")
                .join("session_loader.py"),
            "def get_session(session_id):\n    return load_session(session_id)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: session]",
                "[read_file: sandbox/services/session_handler.py]",
                "[read_file: sandbox/services/session_loader.py]",
                "Sessions are loaded in sandbox/services/session_loader.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where sessions are loaded in sandbox/services/".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        let search_result = snapshot
            .iter()
            .find(|m| m.content.contains("=== tool_result: search_code ==="))
            .map(|m| m.content.as_str())
            .unwrap_or("");
        assert!(
            search_result.contains("sandbox/services/session_handler.py"),
            "scoped search must include in-scope non-load candidate: {search_result}"
        );
        assert!(
            search_result.contains("sandbox/services/session_loader.py"),
            "scoped search must include in-scope load candidate: {search_result}"
        );
        assert!(
            !search_result.contains("sandbox/controllers/session_loader.py"),
            "scoped search must exclude out-of-scope load candidate: {search_result}"
        );

        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Sessions are loaded in sandbox/services/session_loader.py.")
        );
    }

    #[test]
    fn load_lookup_read_cap_still_applies() {
        // MaxReadsPerTurn must still apply under LoadLookup.
        // After 3 reads the runtime blocks further reads regardless of mode.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        for dir in &["a", "b", "c", "d"] {
            fs::create_dir_all(tmp.path().join(dir)).unwrap();
        }
        fs::write(
            tmp.path().join("a").join("session.py"),
            "def session_a(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("b").join("session.py"),
            "def session_b(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("c").join("session.py"),
            "def session_c(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("d").join("session.py"),
            "session = load_session(session_id)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: session]",
                // Reads 3 non-load files — hits cap before reaching load file.
                "[read_file: a/session.py]",
                "[read_file: b/session.py]",
                "[read_file: c/session.py]",
                "[read_file: d/session.py]",
                "Sessions are loaded in d/session.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are sessions loaded?".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "must not fail (cap is a correction): {events:?}"
        );
        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_error: read_file ===")
                    && m.content.contains("read limit")),
            "read cap must block the 4th read"
        );
    }

    // Phase 9.2.6 — SaveLookup

    #[test]
    fn save_lookup_path_scope_keeps_candidates_inside_scope() {
        // Prompt scope must remain the upper bound. The out-of-scope save
        // file is stronger-looking but must not appear in search candidates.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/controllers")).unwrap();
        fs::write(
            tmp.path()
                .join("sandbox/services")
                .join("session_handler.py"),
            "def handle_session(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("sandbox/services").join("session_store.py"),
            "def store_session(session):\n    save_session(session)\n",
        )
        .unwrap();
        fs::write(
            tmp.path()
                .join("sandbox/controllers")
                .join("session_store.py"),
            "def store_session(session):\n    save_session(session)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: session]",
                "[read_file: sandbox/services/session_handler.py]",
                "[read_file: sandbox/services/session_store.py]",
                "Sessions are saved in sandbox/services/session_store.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where sessions are saved in sandbox/services/".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        let search_result = snapshot
            .iter()
            .find(|m| m.content.contains("=== tool_result: search_code ==="))
            .map(|m| m.content.as_str())
            .unwrap_or("");
        assert!(
            search_result.contains("sandbox/services/session_handler.py"),
            "scoped search must include in-scope non-save candidate: {search_result}"
        );
        assert!(
            search_result.contains("sandbox/services/session_store.py"),
            "scoped search must include in-scope save candidate: {search_result}"
        );
        assert!(
            !search_result.contains("sandbox/controllers/session_store.py"),
            "scoped search must exclude out-of-scope save candidate: {search_result}"
        );

        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Sessions are saved in sandbox/services/session_store.py.")
        );
    }

    #[test]
    fn save_lookup_read_cap_still_applies() {
        // MaxReadsPerTurn must still apply under SaveLookup.
        // After 3 reads the runtime blocks further reads regardless of mode.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        for dir in &["a", "b", "c", "d"] {
            fs::create_dir_all(tmp.path().join(dir)).unwrap();
        }
        fs::write(
            tmp.path().join("a").join("session.py"),
            "def session_a(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("b").join("session.py"),
            "def session_b(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("c").join("session.py"),
            "def session_c(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("d").join("session.py"),
            "save_session(session)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: session]",
                // Reads 3 non-save files — hits cap before reaching save file.
                "[read_file: a/session.py]",
                "[read_file: b/session.py]",
                "[read_file: c/session.py]",
                "[read_file: d/session.py]",
                "Sessions are saved in d/session.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are sessions saved?".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "must not fail (cap is a correction): {events:?}"
        );
        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_error: read_file ===")
                    && m.content.contains("read limit")),
            "read cap must block the 4th read"
        );
    }

    // Phase 9.2.3 — regression tests for earlier modes/invariants

    #[test]
    fn create_lookup_read_cap_still_applies() {
        // MaxReadsPerTurn must still apply under CreateLookup.
        // After 3 reads the runtime blocks further reads regardless of mode.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        for dir in &["a", "b", "c", "d"] {
            fs::create_dir_all(tmp.path().join(dir)).unwrap();
        }
        fs::write(
            tmp.path().join("a").join("task.py"),
            "def task_a():\n    pass\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("b").join("task.py"),
            "def task_b():\n    pass\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("c").join("task.py"),
            "def task_c():\n    pass\n",
        )
        .unwrap();
        fs::write(tmp.path().join("d").join("task.py"), "db.create(task)\n").unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: task]",
                // Reads 3 non-create files — hits cap before reaching create file.
                "[read_file: a/task.py]",
                "[read_file: b/task.py]",
                "[read_file: c/task.py]",
                "[read_file: d/task.py]",
                "Tasks are created in d/task.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are tasks created?".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "must not fail (cap is a correction): {events:?}"
        );
        let snapshot = rt.messages_snapshot();
        // The 4th read must be blocked by the cap.
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_error: read_file ===")
                    && m.content.contains("read limit")),
            "read cap must block the 4th read"
        );
    }
}
