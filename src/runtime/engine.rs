use std::collections::HashSet;
use std::path::Path;

use crate::app::config::Config;
use crate::llm::backend::{BackendCapabilities, ModelBackend, Role};
use crate::tools::{
    ExecutionKind, PendingAction, ToolError, ToolInput, ToolRegistry, ToolRunResult,
};

use super::anchors::{
    has_same_scope_reference, is_last_read_file_anchor_prompt, is_last_search_anchor_prompt,
    AnchorState,
};
use super::conversation::Conversation;
use super::generation::{emit_visible_assistant_message, run_generate_turn};
use super::investigation::{detect_investigation_mode, InvestigationMode, InvestigationState};
use super::project_root::ProjectRoot;
use super::prompt;
use super::resolve;
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

/// Bounds for /history output. Limits messages shown and chars per message to
/// prevent unbounded InfoMessage output from long or tool-heavy sessions.
const MAX_HISTORY_MESSAGES: usize = 10;
const MAX_MESSAGE_CHARS: usize = 200;

/// Policy values derived once from backend capabilities at construction time.
/// Both layers of capability-aware context management read from this struct.
struct ContextPolicy {
    /// Message count threshold at which conversation trimming fires (Layer 2).
    trim_threshold: usize,
    /// Maximum content lines per tool result block before it is capped (Layer 1).
    tool_result_max_lines: usize,
}

impl ContextPolicy {
    fn from_capabilities(caps: BackendCapabilities) -> Self {
        match caps.context_window_tokens {
            Some(t) if t >= 16_384 => Self {
                trim_threshold: 40,
                tool_result_max_lines: 200,
            },
            Some(t) if t >= 8_192 => Self {
                trim_threshold: 30,
                tool_result_max_lines: 150,
            },
            Some(t) if t >= 4_096 => Self {
                trim_threshold: 20,
                tool_result_max_lines: 80,
            },
            Some(_) => Self {
                trim_threshold: 12,
                tool_result_max_lines: 40,
            },
            None => Self {
                trim_threshold: 40,
                tool_result_max_lines: 200,
            },
        }
    }
}

/// Explicit allowlist of tools that slash commands may invoke via the runtime.
/// All command-to-registry dispatch passes through this type — no command handler
/// calls registry.dispatch() directly or constructs ToolInput outside this enum.
/// Mutating tools are excluded by omission; adding one requires an explicit variant.
enum CommandTool {
    ReadFile { path: String },
    SearchCode { query: String },
}

impl CommandTool {
    fn into_input(self) -> ToolInput {
        match self {
            Self::ReadFile { path } => ToolInput::ReadFile { path },
            Self::SearchCode { query } => ToolInput::SearchCode { query, path: None },
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::ReadFile { .. } => "read_file",
            Self::SearchCode { .. } => "search_code",
        }
    }
}

use super::response_text::*;
use super::trace::{trace_runtime_decision, RUNTIME_TRACE_ENV};

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

fn usage_lookup_is_broad(
    mode: InvestigationMode,
    requested_read_path: Option<&str>,
    investigation_path_scope: Option<&str>,
) -> bool {
    if !matches!(mode, InvestigationMode::UsageLookup) || requested_read_path.is_some() {
        return false;
    }

    match investigation_path_scope {
        None => true,
        Some(scope) => !path_scope_looks_like_file(scope),
    }
}

fn path_scope_looks_like_file(scope: &str) -> bool {
    Path::new(scope)
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.contains('.'))
}

#[derive(Clone, Copy)]
enum GenerationRoundLabel {
    Initial,
    PostTool,
    PostEvidenceRetry,
    CorrectionRetry,
}

impl GenerationRoundLabel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Initial => "initial",
            Self::PostTool => "post-tool",
            Self::PostEvidenceRetry => "post-evidence-retry",
            Self::CorrectionRetry => "correction-retry",
        }
    }
}

#[derive(Clone, Copy)]
enum GenerationRoundCause {
    Initial,
    ToolResults,
    Recovery,
    SearchRetry,
    PostEvidenceToolCallRejected,
    AnswerPhaseToolCallRejected,
    SearchBudgetClosedCorrection,
    EditRepairCorrection,
    FabricationCorrection,
    MalformedBlockCorrection,
    ReadRequestToolRequired,
    SearchBeforeAnsweringCorrection,
    ReadBeforeAnsweringCorrection,
}

impl GenerationRoundCause {
    fn as_str(self) -> &'static str {
        match self {
            Self::Initial => "initial",
            Self::ToolResults => "tool-results",
            Self::Recovery => "recovery",
            Self::SearchRetry => "search-retry",
            Self::PostEvidenceToolCallRejected => "post_evidence_tool_call_rejected",
            Self::AnswerPhaseToolCallRejected => "answer_phase_tool_call_rejected",
            Self::SearchBudgetClosedCorrection => "search_budget_closed_correction",
            Self::EditRepairCorrection => "edit_repair_correction",
            Self::FabricationCorrection => "fabrication_correction",
            Self::MalformedBlockCorrection => "malformed_block_correction",
            Self::ReadRequestToolRequired => "read_request_tool_required",
            Self::SearchBeforeAnsweringCorrection => "search_before_answering",
            Self::ReadBeforeAnsweringCorrection => "read_before_answering",
        }
    }
}

struct TurnPerformance {
    enabled: bool,
    turn_start: Option<std::time::Instant>,
    rounds: usize,
    round_labels: Vec<GenerationRoundLabel>,
    round_causes: Vec<GenerationRoundCause>,
    prompt_sizes: Vec<usize>,
    ctx_ms: u64,
    tokenize_ms: u64,
    prefill_ms: u64,
    generation_ms: u64,
    model_load_ms: u64,
    tool_ms: u64,
}

impl TurnPerformance {
    fn new() -> Self {
        let enabled = std::env::var_os(RUNTIME_TRACE_ENV).is_some();
        Self {
            enabled,
            turn_start: enabled.then(std::time::Instant::now),
            rounds: 0,
            round_labels: Vec::new(),
            round_causes: Vec::new(),
            prompt_sizes: Vec::new(),
            ctx_ms: 0,
            tokenize_ms: 0,
            prefill_ms: 0,
            generation_ms: 0,
            model_load_ms: 0,
            tool_ms: 0,
        }
    }

    fn start_round(
        &mut self,
        label: GenerationRoundLabel,
        cause: GenerationRoundCause,
        prompt_chars: usize,
        on_event: &mut dyn FnMut(RuntimeEvent),
    ) {
        if !self.enabled {
            return;
        }

        self.rounds += 1;
        self.round_labels.push(label);
        self.round_causes.push(cause);
        self.prompt_sizes.push(prompt_chars);
        on_event(RuntimeEvent::RuntimeTrace(format!(
            "[runtime:perf] round={} label={} cause={} prompt_chars={}",
            self.rounds,
            label.as_str(),
            cause.as_str(),
            prompt_chars
        )));
    }

    fn record_backend_timing(&mut self, stage: &str, elapsed_ms: u64) {
        if !self.enabled {
            return;
        }

        match stage {
            "ctx_create" => self.ctx_ms += elapsed_ms,
            "tokenize" => self.tokenize_ms += elapsed_ms,
            "prefill_done" => self.prefill_ms += elapsed_ms,
            "generation_done" => self.generation_ms += elapsed_ms,
            "model_load" => self.model_load_ms += elapsed_ms,
            _ => {}
        }
    }

    fn record_tool_elapsed(&mut self, elapsed_ms: u64) {
        if !self.enabled {
            return;
        }
        self.tool_ms += elapsed_ms;
    }

    fn emit_summary(&self, on_event: &mut dyn FnMut(RuntimeEvent)) {
        if !self.enabled {
            return;
        }

        let round_labels = if self.round_labels.is_empty() {
            "none".to_string()
        } else {
            self.round_labels
                .iter()
                .map(|label| label.as_str())
                .collect::<Vec<_>>()
                .join(",")
        };
        let causes = if self.round_causes.is_empty() {
            "none".to_string()
        } else {
            self.round_causes
                .iter()
                .map(|cause| cause.as_str())
                .collect::<Vec<_>>()
                .join(",")
        };
        let prompt_sizes = if self.prompt_sizes.is_empty() {
            "none".to_string()
        } else {
            self.prompt_sizes
                .iter()
                .map(|size| size.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };

        let model_ms = self.ctx_ms + self.tokenize_ms + self.prefill_ms + self.generation_ms;
        let total_turn_ms = self
            .turn_start
            .map(|t| t.elapsed().as_millis() as u64)
            .unwrap_or(0);

        on_event(RuntimeEvent::RuntimeTrace(format!(
            "[runtime:perf] rounds={} round_labels={} causes={} prompt_sizes={} prefill_ms={} generation_ms={} ctx_ms={} tokenize_ms={} model_load_ms={} tool_ms={} model_ms={} total_turn_ms={}",
            self.rounds,
            round_labels,
            causes,
            prompt_sizes,
            self.prefill_ms,
            self.generation_ms,
            self.ctx_ms,
            self.tokenize_ms,
            self.model_load_ms,
            self.tool_ms,
            model_ms,
            total_turn_ms
        )));
    }
}

fn estimate_generation_prompt_chars(
    conversation: &Conversation,
    tool_surface: ToolSurface,
) -> usize {
    let hint = prompt::render_tool_surface_hint(
        tool_surface.as_str(),
        tool_surface
            .allowed_tool_names()
            .chain(tool_surface.mutation_tool_names().iter().copied()),
    );
    conversation
        .snapshot()
        .into_iter()
        .map(|message| message.content.len())
        .sum::<usize>()
        + hint.len()
}

fn infer_post_tool_round_cause(results: &str) -> GenerationRoundCause {
    if results.contains("=== tool_result: search_code ===") && results.contains("No matches found.")
    {
        GenerationRoundCause::SearchRetry
    } else if results.contains("This is a usage lookup")
        || results.contains("This is a config lookup")
        || results.contains("This is an initialization lookup")
        || results.contains("This is a creation lookup")
        || results.contains("This is a registration lookup")
        || results.contains("This is a load lookup")
        || results.contains("This is a save lookup")
        || results.contains("The file just read contained only import matches")
        || results.contains("The file just read is a lockfile")
    {
        GenerationRoundCause::Recovery
    } else {
        GenerationRoundCause::ToolResults
    }
}

use super::tool_surface::{select_tool_surface, ToolSurface};

/// Returns true if the prompt contains a token that looks like a code identifier.
/// Only two structural patterns are checked — no NLP, no heuristics.
use super::prompt_analysis::{
    classify_retrieval_intent, extract_investigation_path_scope, prompt_requires_investigation,
    user_requested_mutation, RetrievalIntent,
};

pub struct Runtime {
    #[allow(dead_code)]
    project_root: ProjectRoot,
    conversation: Conversation,
    backend: Box<dyn ModelBackend>,
    registry: ToolRegistry,
    system_prompt: String,
    anchors: AnchorState,
    context_policy: ContextPolicy,
    /// Holds a mutating tool action that is waiting for user approval.
    /// Set when a tool round suspends; cleared by Approve or Reject.
    /// At most one pending action exists at any time.
    pending_action: Option<PendingAction>,
}

impl Runtime {
    pub fn new(
        config: &Config,
        project_root: ProjectRoot,
        backend: Box<dyn ModelBackend>,
        registry: ToolRegistry,
    ) -> Self {
        let specs = registry.specs();
        let system_prompt =
            prompt::build_system_prompt(&config.app.name, project_root.path(), &specs);
        let context_policy = ContextPolicy::from_capabilities(backend.capabilities());
        Self {
            project_root,
            conversation: Conversation::new(system_prompt.clone()),
            backend,
            registry,
            system_prompt,
            anchors: AnchorState::default(),
            context_policy,
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
            RuntimeRequest::QueryLast => self.handle_query_last(on_event),
            RuntimeRequest::QueryAnchors => self.handle_query_anchors(on_event),
            RuntimeRequest::QueryHistory => self.handle_query_history(on_event),
            RuntimeRequest::ReadFile { path } => self.handle_read_file(path, on_event),
            RuntimeRequest::SearchCode { query } => self.handle_search_code(query, on_event),
        }
    }

    fn handle_query_last(&mut self, on_event: &mut dyn FnMut(RuntimeEvent)) {
        let text = match self.conversation.last_assistant_content() {
            Some(content) => content.to_string(),
            None => "No previous response.".to_string(),
        };
        on_event(RuntimeEvent::InfoMessage(text));
    }

    fn handle_query_anchors(&mut self, on_event: &mut dyn FnMut(RuntimeEvent)) {
        let mut parts = Vec::new();
        if let Some(path) = self.anchors.last_read_file() {
            parts.push(format!("last read:   {path}"));
        }
        if let Some((query, scope)) = self.anchors.last_search() {
            match scope {
                Some(s) => parts.push(format!("last search: {query} (in {s})")),
                None => parts.push(format!("last search: {query}")),
            }
        }
        let text = if parts.is_empty() {
            "no anchors set".to_string()
        } else {
            parts.join("\n")
        };
        on_event(RuntimeEvent::InfoMessage(text));
    }

    fn handle_query_history(&mut self, on_event: &mut dyn FnMut(RuntimeEvent)) {
        let messages = self.conversation.human_visible_snapshot();

        if messages.is_empty() {
            on_event(RuntimeEvent::InfoMessage(
                "no conversation history".to_string(),
            ));
            return;
        }

        let tail = if messages.len() > MAX_HISTORY_MESSAGES {
            messages[messages.len() - MAX_HISTORY_MESSAGES..].to_vec()
        } else {
            messages
        };

        let mut lines = vec!["history:".to_string()];
        let mut first = true;
        for msg in &tail {
            let label = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => continue,
            };
            if msg.role == Role::User && !first {
                lines.push(String::new());
            }
            let content = if msg.content.chars().count() > MAX_MESSAGE_CHARS {
                let truncated: String = msg.content.chars().take(MAX_MESSAGE_CHARS).collect();
                format!("{truncated}...")
            } else {
                msg.content.clone()
            };
            lines.push(format!("[{label}] {content}"));
            first = false;
        }

        on_event(RuntimeEvent::InfoMessage(lines.join("\n")));
    }

    /// Applies the Layer 1 context cap then commits the results to the conversation.
    /// Must be used for all tool-origin push_user calls so the cap is applied consistently.
    fn commit_tool_results(&mut self, results: String) {
        let capped = cap_tool_result_blocks(&results, self.context_policy.tool_result_max_lines);
        self.conversation.push_user(capped);
    }

    fn dispatch_command_tool(&mut self, tool: CommandTool, on_event: &mut dyn FnMut(RuntimeEvent)) {
        if self.pending_action.is_some() {
            on_event(RuntimeEvent::Failed {
                message: "cannot run command while a tool approval is pending".to_string(),
            });
            return;
        }
        let search_query = match &tool {
            CommandTool::SearchCode { query } => Some(query.clone()),
            CommandTool::ReadFile { .. } => None,
        };
        let name = tool.name();
        let input = tool.into_input();
        let resolved = match resolve(&self.project_root, &input) {
            Ok(resolved) => resolved,
            Err(error) => {
                let tool_error: ToolError = error.into();
                on_event(RuntimeEvent::InfoMessage(format!("error: {}", tool_error)));
                return;
            }
        };
        match self.registry.dispatch(resolved) {
            Ok(ToolRunResult::Immediate(output)) => {
                self.anchors.record_successful_read(&output);
                if let Some(query) = search_query {
                    self.anchors.record_successful_search(&output, query, None);
                }
                on_event(RuntimeEvent::InfoMessage(tool_codec::format_tool_result(
                    name, &output,
                )));
            }
            Ok(ToolRunResult::Approval(pending)) => {
                self.pending_action = Some(pending.clone());
                on_event(RuntimeEvent::ApprovalRequired(pending));
            }
            Err(e) => {
                on_event(RuntimeEvent::InfoMessage(format!("error: {e}")));
            }
        }
    }

    fn handle_read_file(&mut self, path: String, on_event: &mut dyn FnMut(RuntimeEvent)) {
        let p = std::path::Path::new(&path);
        if p.is_absolute() {
            on_event(RuntimeEvent::InfoMessage(
                "error: path must be relative".to_string(),
            ));
            return;
        }
        if p.components().any(|c| c == std::path::Component::ParentDir) {
            on_event(RuntimeEvent::InfoMessage(
                "error: path must not contain '..' components".to_string(),
            ));
            return;
        }
        self.dispatch_command_tool(CommandTool::ReadFile { path }, on_event);
    }

    fn handle_search_code(&mut self, query: String, on_event: &mut dyn FnMut(RuntimeEvent)) {
        if query.trim().len() < 2 {
            on_event(RuntimeEvent::InfoMessage(
                "error: search query must be at least 2 characters".to_string(),
            ));
            return;
        }
        self.dispatch_command_tool(CommandTool::SearchCode { query }, on_event);
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
            &self.project_root,
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
                self.commit_tool_results(results);
                self.conversation
                    .trim_tool_exchanges_if_needed(self.context_policy.trim_threshold);
                on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
                self.run_turns_with_initial_reads(1, reads_this_turn, on_event);
            }
            ToolRoundOutcome::TerminalAnswer {
                results,
                answer,
                reason,
            } => {
                self.commit_tool_results(results);
                self.conversation
                    .trim_tool_exchanges_if_needed(self.context_policy.trim_threshold);
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
                    self.commit_tool_results(accumulated);
                    self.conversation
                        .trim_tool_exchanges_if_needed(self.context_policy.trim_threshold);
                }
                self.pending_action = Some(pending.clone());
                on_event(RuntimeEvent::ApprovalRequired(pending));
                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
            }
            ToolRoundOutcome::RuntimeDispatch { .. } => {
                debug_assert!(
                    false,
                    "RuntimeDispatch is not expected during last-read anchor replay"
                );
                on_event(RuntimeEvent::Failed {
                    message: "Unexpected runtime dispatch during last-read replay.".to_string(),
                });
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

        let resolved = match resolve(&self.project_root, &input) {
            Ok(resolved) => resolved,
            Err(error) => {
                let tool_error: ToolError = error.into();
                on_event(RuntimeEvent::ToolCallFinished {
                    name: name.clone(),
                    summary: None,
                });
                self.conversation.push_user(tool_codec::format_tool_error(
                    &name,
                    &tool_error.to_string(),
                ));
                self.conversation
                    .trim_tool_exchanges_if_needed(self.context_policy.trim_threshold);
                self.finish_with_runtime_answer(
                    LAST_SEARCH_REPLAY_FAILED,
                    AnswerSource::RuntimeTerminal {
                        reason: RuntimeTerminalReason::InsufficientEvidence,
                        rounds: 1,
                    },
                    on_event,
                );
                return;
            }
        };

        match self.registry.dispatch(resolved) {
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
                self.commit_tool_results(tool_codec::format_tool_result(&name, &output));
                self.conversation
                    .trim_tool_exchanges_if_needed(self.context_policy.trim_threshold);
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
                self.conversation
                    .trim_tool_exchanges_if_needed(self.context_policy.trim_threshold);
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
                self.commit_tool_results(tool_codec::format_tool_result(&tool_name, &output));
                self.conversation
                    .trim_tool_exchanges_if_needed(self.context_policy.trim_threshold);
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
        struct PendingRuntimeCall {
            input: ToolInput,
            seeded_pre_generation: bool,
        }

        #[derive(Clone, Copy)]
        enum AnswerPhaseKind {
            PostRead,
            InvestigationEvidenceReady,
        }

        let mut corrections = 0usize;
        let mut last_call_key: Option<String> = None;
        let mut pending_runtime_call: Option<PendingRuntimeCall> = None;
        let mut search_budget = SearchBudget::new();
        let mut investigation = InvestigationState::new();
        let mut turn_perf = TurnPerformance::new();
        let mut next_round_label = GenerationRoundLabel::Initial;
        let mut next_round_cause = GenerationRoundCause::Initial;
        let mut requested_read_completed = false;
        let mut read_request_correction_issued = false;
        let mut disallowed_tool_attempts = 0usize;
        let mut weak_search_query_attempts = 0usize;
        let mut answer_phase: Option<AnswerPhaseKind> = None;
        let mut post_answer_phase_tool_attempts = 0usize;
        let mut post_answer_phase_correction_echo_retries = 0usize;
        let mut seeded_tool_executed = false;

        macro_rules! finish_turn {
            () => {{
                turn_perf.emit_summary(on_event);
                return;
            }};
        }
        // Computed once from the original user message. Excludes tool result/error injections
        // and correction messages so the approve-failure path (run_turns(0,...)) is safe.
        let original_user_prompt = self.conversation.last_user_content().filter(|c| {
            !c.starts_with("=== tool_result:")
                && !c.starts_with("=== tool_error:")
                && !c.starts_with("[runtime:correction]")
        });
        let retrieval_intent = original_user_prompt
            .map(classify_retrieval_intent)
            .unwrap_or(RetrievalIntent::None);
        let requested_read_path: Option<String> = match &retrieval_intent {
            RetrievalIntent::DirectRead { path } => Some(path.clone()),
            _ => None,
        };
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
            .map(|p| {
                select_tool_surface(
                    p,
                    investigation_required,
                    mutation_allowed,
                    requested_read_path.is_some() || !reads_this_turn.is_empty(),
                )
            })
            .unwrap_or(if reads_this_turn.is_empty() {
                ToolSurface::AnswerOnly
            } else {
                ToolSurface::RetrievalFirst
            });
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
                        finish_turn!();
                    }
                }
            } else {
                None
            };
        investigation.configure_usage_evidence_policy(usage_lookup_is_broad(
            investigation_mode,
            requested_read_path.as_deref(),
            investigation_path_scope.as_deref(),
        ));
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
        if !investigation_required {
            match &retrieval_intent {
                RetrievalIntent::DirectRead { path } => {
                    pending_runtime_call = Some(PendingRuntimeCall {
                        input: ToolInput::ReadFile { path: path.clone() },
                        seeded_pre_generation: true,
                    });
                }
                RetrievalIntent::DirectoryListing { path } => {
                    pending_runtime_call = Some(PendingRuntimeCall {
                        input: ToolInput::ListDir { path: path.clone() },
                        seeded_pre_generation: true,
                    });
                }
                RetrievalIntent::None => {}
            }
        }
        loop {
            // Bind answer-phase synthesis to a no-tool surface so the model is never offered
            // tool access after evidence is accepted. This eliminates the extra generation
            // round that would otherwise occur when the model attempts a tool call and the
            // runtime has to issue a post_evidence_tool_call_rejected correction.
            let effective_surface = if answer_phase.is_some() {
                ToolSurface::AnswerOnly
            } else {
                tool_surface
            };
            if matches!(effective_surface, ToolSurface::AnswerOnly) {
                trace_runtime_decision(
                    on_event,
                    "answer_phase_synthesis_bounded",
                    &[("surface", "AnswerOnly".into())],
                );
            }
            let prompt_chars = if turn_perf.enabled {
                estimate_generation_prompt_chars(&self.conversation, effective_surface)
            } else {
                0
            };

            turn_perf.start_round(next_round_label, next_round_cause, prompt_chars, on_event);

            let (calls, response, seeded_pre_generation) =
                if let Some(pending) = pending_runtime_call.take() {
                    (vec![pending.input], None, pending.seeded_pre_generation)
                } else {
                    let response = {
                        let turn_perf = &mut turn_perf;
                        let mut perf_on_event = |event| {
                            if let RuntimeEvent::BackendTiming { stage, elapsed_ms } = &event {
                                turn_perf.record_backend_timing(stage, *elapsed_ms);
                            }
                            on_event(event);
                        };

                        match run_generate_turn(
                            self.backend.as_mut(),
                            &mut self.conversation,
                            effective_surface,
                            &mut perf_on_event,
                        ) {
                            Ok(Some(r)) => r,
                            Ok(None) => {
                                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                                on_event(RuntimeEvent::Failed {
                                    message: format!("{} returned no output.", self.backend.name()),
                                });
                                finish_turn!();
                            }
                            Err(e) => {
                                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                                on_event(RuntimeEvent::Failed {
                                    message: e.to_string(),
                                });
                                finish_turn!();
                            }
                        }
                    };

                    let calls = tool_codec::parse_all_tool_inputs(&response);
                    (calls, Some(response), false)
                };

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
                        let (label, cause) = match phase {
                            AnswerPhaseKind::PostRead => (
                                GenerationRoundLabel::CorrectionRetry,
                                GenerationRoundCause::AnswerPhaseToolCallRejected,
                            ),
                            AnswerPhaseKind::InvestigationEvidenceReady => (
                                GenerationRoundLabel::PostEvidenceRetry,
                                GenerationRoundCause::PostEvidenceToolCallRejected,
                            ),
                        };
                        next_round_label = label;
                        next_round_cause = cause;
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
                    finish_turn!();
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
                    finish_turn!();
                }
                if corrections < MAX_CORRECTIONS {
                    corrections += 1;
                    self.conversation.discard_last_if_assistant();
                    self.conversation
                        .push_user(search_budget.closed_message().to_string());
                    next_round_label = GenerationRoundLabel::CorrectionRetry;
                    next_round_cause = GenerationRoundCause::SearchBudgetClosedCorrection;
                    continue;
                }
                on_event(RuntimeEvent::Failed {
                    message: "Model kept searching after the search budget was closed.".to_string(),
                });
                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                finish_turn!();
            }

            if calls.is_empty() {
                let response = response.expect("response exists when calls are empty");

                if let Some(phase) = answer_phase {
                    // Detect correction echoes by sentinel prefix OR by known correction
                    // substrings. The latter catches cases where the model parrots the
                    // correction text back without the [runtime:correction] prefix.
                    let is_correction_echo =
                        response.trim_start().starts_with("[runtime:correction]")
                            || response.contains("The file was already read this turn")
                            || response.contains("Evidence is already ready from the file");
                    if is_correction_echo {
                        self.conversation.discard_last_if_assistant();
                        if post_answer_phase_correction_echo_retries == 0 {
                            post_answer_phase_correction_echo_retries += 1;
                            let (label, cause) = match phase {
                                AnswerPhaseKind::PostRead => (
                                    GenerationRoundLabel::CorrectionRetry,
                                    GenerationRoundCause::AnswerPhaseToolCallRejected,
                                ),
                                AnswerPhaseKind::InvestigationEvidenceReady => (
                                    GenerationRoundLabel::PostEvidenceRetry,
                                    GenerationRoundCause::PostEvidenceToolCallRejected,
                                ),
                            };
                            next_round_label = label;
                            next_round_cause = cause;
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
                        finish_turn!();
                    }
                }

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
                    next_round_label = GenerationRoundLabel::CorrectionRetry;
                    next_round_cause = GenerationRoundCause::EditRepairCorrection;
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
                        next_round_label = GenerationRoundLabel::CorrectionRetry;
                        next_round_cause = GenerationRoundCause::FabricationCorrection;
                        continue;
                    }
                    on_event(RuntimeEvent::Failed {
                        message: "Model repeatedly produced fabricated tool results. Try rephrasing your request.".to_string(),
                    });
                    on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                    finish_turn!();
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
                        next_round_label = GenerationRoundLabel::CorrectionRetry;
                        next_round_cause = GenerationRoundCause::MalformedBlockCorrection;
                        continue;
                    }
                    on_event(RuntimeEvent::Failed {
                        message:
                            "Model used incorrect tool tag names. Try rephrasing your request."
                                .to_string(),
                    });
                    on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                    finish_turn!();
                }

                if let Some(path) = requested_read_path.as_deref() {
                    if !requested_read_completed {
                        if !read_request_correction_issued && corrections < MAX_CORRECTIONS {
                            corrections += 1;
                            read_request_correction_issued = true;
                            self.conversation.push_user(format!(
                                "{READ_REQUEST_TOOL_REQUIRED} Requested path: `{path}`"
                            ));
                            next_round_label = GenerationRoundLabel::CorrectionRetry;
                            next_round_cause = GenerationRoundCause::ReadRequestToolRequired;
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
                        finish_turn!();
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
                    finish_turn!();
                }

                if investigation_required && !investigation.evidence_ready() {
                    if search_budget.calls == 0 {
                        if investigation.issue_direct_answer_correction() {
                            self.conversation
                                .push_user(SEARCH_BEFORE_ANSWERING.to_string());
                            next_round_label = GenerationRoundLabel::CorrectionRetry;
                            next_round_cause =
                                GenerationRoundCause::SearchBeforeAnsweringCorrection;
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
                        finish_turn!();
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
                            finish_turn!();
                        }

                        if corrections < MAX_CORRECTIONS
                            && investigation.issue_premature_synthesis_correction()
                        {
                            corrections += 1;
                            self.conversation.discard_last_if_assistant();
                            self.conversation
                                .push_user(READ_BEFORE_ANSWERING.to_string());
                            next_round_label = GenerationRoundLabel::CorrectionRetry;
                            next_round_cause = GenerationRoundCause::ReadBeforeAnsweringCorrection;
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
                        finish_turn!();
                    }
                }

                let source = if tool_rounds == 0 {
                    if seeded_tool_executed {
                        AnswerSource::ToolAssisted { rounds: 1 }
                    } else {
                        AnswerSource::Direct
                    }
                } else {
                    AnswerSource::ToolAssisted {
                        rounds: tool_rounds,
                    }
                };
                emit_visible_assistant_message(&response, on_event);
                on_event(RuntimeEvent::AnswerReady(source));
                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                finish_turn!();
            }

            if !seeded_pre_generation {
                tool_rounds += 1;

                if tool_rounds >= MAX_TOOL_ROUNDS {
                    on_event(RuntimeEvent::AnswerReady(AnswerSource::ToolLimitReached));
                    on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                    finish_turn!();
                }
            }

            on_event(RuntimeEvent::ActivityChanged(Activity::ExecutingTools));
            let t_tool_start = if turn_perf.enabled {
                Some(std::time::Instant::now())
            } else {
                None
            };

            match run_tool_round(
                &self.project_root,
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
                    if seeded_pre_generation {
                        seeded_tool_executed = true;
                        last_call_key = None;
                        if matches!(retrieval_intent, RetrievalIntent::DirectoryListing { .. }) {
                            answer_phase = Some(AnswerPhaseKind::PostRead);
                        }
                    }
                    if let Some(t) = t_tool_start {
                        turn_perf.record_tool_elapsed(t.elapsed().as_millis() as u64);
                    }
                    let post_tool_cause = infer_post_tool_round_cause(&results);
                    self.commit_tool_results(results);
                    self.conversation
                        .trim_tool_exchanges_if_needed(self.context_policy.trim_threshold);
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
                            finish_turn!();
                        }
                    }
                    if answer_phase.is_none() {
                        if investigation_required && investigation.evidence_ready() {
                            answer_phase = Some(AnswerPhaseKind::InvestigationEvidenceReady);
                        } else if !investigation_required
                            && !mutation_allowed
                            && !reads_this_turn.is_empty()
                        {
                            answer_phase = Some(AnswerPhaseKind::PostRead);
                        }
                    }
                    next_round_label = GenerationRoundLabel::PostTool;
                    next_round_cause = post_tool_cause;
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
                    if let Some(t) = t_tool_start {
                        turn_perf.record_tool_elapsed(t.elapsed().as_millis() as u64);
                    }
                    self.commit_tool_results(results);
                    self.conversation
                        .trim_tool_exchanges_if_needed(self.context_policy.trim_threshold);
                    self.finish_with_runtime_answer(
                        &answer,
                        AnswerSource::RuntimeTerminal {
                            reason,
                            rounds: tool_rounds,
                        },
                        on_event,
                    );
                    finish_turn!();
                }
                ToolRoundOutcome::ApprovalRequired {
                    accumulated,
                    pending,
                } => {
                    if let Some(t) = t_tool_start {
                        turn_perf.record_tool_elapsed(t.elapsed().as_millis() as u64);
                    }
                    if !accumulated.is_empty() {
                        self.commit_tool_results(accumulated);
                        self.conversation
                            .trim_tool_exchanges_if_needed(self.context_policy.trim_threshold);
                    }
                    self.pending_action = Some(pending.clone());
                    on_event(RuntimeEvent::ApprovalRequired(pending));
                    on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                    finish_turn!();
                }
                ToolRoundOutcome::RuntimeDispatch { accumulated, call } => {
                    if let Some(t) = t_tool_start {
                        turn_perf.record_tool_elapsed(t.elapsed().as_millis() as u64);
                    }
                    if !accumulated.is_empty() {
                        self.commit_tool_results(accumulated);
                        self.conversation
                            .trim_tool_exchanges_if_needed(self.context_policy.trim_threshold);
                    }
                    pending_runtime_call = Some(PendingRuntimeCall {
                        input: call,
                        seeded_pre_generation: false,
                    });
                    on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
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

/// Caps tool result blocks in an accumulated results string to `max_lines` content lines each.
///
/// Only `=== tool_result: ... ===` blocks are affected. Error blocks, corrections, and other
/// injected messages pass through unchanged. Top-aligned truncation: the first `max_lines`
/// content lines are kept; a metadata note is appended when capping occurs.
fn cap_tool_result_blocks(text: &str, max_lines: usize) -> String {
    const HDR: &str = "=== tool_result:";
    const FTR: &str = "=== /tool_result ===";

    let mut out = String::with_capacity(text.len());
    let mut pos = 0;

    while pos < text.len() {
        match text[pos..].find(HDR) {
            None => {
                out.push_str(&text[pos..]);
                break;
            }
            Some(rel) => {
                let hdr_start = pos + rel;
                out.push_str(&text[pos..hdr_start]);

                let body_start = text[hdr_start..]
                    .find('\n')
                    .map(|i| hdr_start + i + 1)
                    .unwrap_or(text.len());
                out.push_str(&text[hdr_start..body_start]);

                match text[body_start..].find(FTR) {
                    None => {
                        out.push_str(&text[body_start..]);
                        pos = text.len();
                    }
                    Some(rel_ftr) => {
                        let ftr_start = body_start + rel_ftr;
                        let body = &text[body_start..ftr_start];
                        let body_line_count = body.lines().count();

                        if body_line_count > max_lines {
                            for line in body.lines().take(max_lines) {
                                out.push_str(line);
                                out.push('\n');
                            }
                            out.push_str(&format!(
                                "[capped at {max_lines} lines — original: {body_line_count} lines]\n"
                            ));
                        } else {
                            out.push_str(body);
                        }

                        let ftr_end = ftr_start + FTR.len();
                        let trailing = text[ftr_end..]
                            .find(|c: char| c != '\n')
                            .map(|i| ftr_end + i)
                            .unwrap_or(text.len());
                        out.push_str(&text[ftr_start..trailing]);
                        pos = trailing;
                    }
                }
            }
        }
    }

    out
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
    use crate::llm::backend::{BackendCapabilities, BackendEvent, GenerateRequest};
    use crate::runtime::ProjectRoot;
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

        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                context_window_tokens: None,
                max_output_tokens: None,
            }
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
        let project_root = ProjectRoot::new(root.to_path_buf()).unwrap();
        Runtime::new(
            &Config::default(),
            project_root.clone(),
            Box::new(TestBackend::new(responses)),
            default_registry().with_project_root(project_root.as_path_buf()),
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

    // ── ContextPolicy tests ──────────────────────────────────────────────────

    #[test]
    fn context_policy_none_uses_defaults() {
        let policy = ContextPolicy::from_capabilities(BackendCapabilities {
            context_window_tokens: None,
            max_output_tokens: None,
        });
        assert_eq!(policy.trim_threshold, 40);
        assert_eq!(policy.tool_result_max_lines, 200);
    }

    #[test]
    fn context_policy_small_context_uses_tight_limits() {
        let policy = ContextPolicy::from_capabilities(BackendCapabilities {
            context_window_tokens: Some(2048),
            max_output_tokens: None,
        });
        assert_eq!(policy.trim_threshold, 12);
        assert_eq!(policy.tool_result_max_lines, 40);
    }

    #[test]
    fn context_policy_mid_context_uses_intermediate_limits() {
        let policy = ContextPolicy::from_capabilities(BackendCapabilities {
            context_window_tokens: Some(4096),
            max_output_tokens: None,
        });
        assert_eq!(policy.trim_threshold, 20);
        assert_eq!(policy.tool_result_max_lines, 80);
    }

    #[test]
    fn context_policy_large_context_uses_defaults() {
        let policy = ContextPolicy::from_capabilities(BackendCapabilities {
            context_window_tokens: Some(32768),
            max_output_tokens: None,
        });
        assert_eq!(policy.trim_threshold, 40);
        assert_eq!(policy.tool_result_max_lines, 200);
    }

    // ── cap_tool_result_blocks tests ─────────────────────────────────────────

    #[test]
    fn cap_under_limit_is_noop() {
        let text = "=== tool_result: read_file ===\nline1\nline2\n=== /tool_result ===\n\n";
        assert_eq!(cap_tool_result_blocks(text, 5), text);
    }

    #[test]
    fn cap_over_limit_truncates_and_adds_note() {
        let body_lines: Vec<String> = (1..=5).map(|i| format!("line{i}")).collect();
        let body = body_lines.join("\n") + "\n";
        let text = format!("=== tool_result: read_file ===\n{body}=== /tool_result ===\n\n");
        let result = cap_tool_result_blocks(&text, 3);
        assert!(
            result.contains("line1\nline2\nline3\n"),
            "first 3 lines must be kept"
        );
        assert!(!result.contains("line4"), "line4 must be removed");
        assert!(result.contains("[capped at 3 lines — original: 5 lines]"));
        assert!(result.contains("=== tool_result: read_file ==="));
        assert!(result.contains("=== /tool_result ==="));
    }

    #[test]
    fn cap_leaves_non_tool_result_content_unchanged() {
        let text = "[runtime:correction] must not fabricate tool calls\n";
        assert_eq!(cap_tool_result_blocks(text, 5), text);
    }

    #[test]
    fn cap_processes_multi_block_independently() {
        let block = |n: usize| {
            let body: String = (1..=n).map(|i| format!("line{i}\n")).collect();
            format!("=== tool_result: read_file ===\n{body}=== /tool_result ===\n\n")
        };
        // Two blocks, both over the limit of 2
        let text = format!("{}{}", block(4), block(3));
        let result = cap_tool_result_blocks(&text, 2);
        assert_eq!(result.matches("[capped at 2 lines").count(), 2);
    }

    #[test]
    fn cap_error_blocks_pass_through_unchanged() {
        let text = "=== tool_error: read_file ===\nfile not found\n=== /tool_error ===\n\n";
        assert_eq!(cap_tool_result_blocks(text, 1), text);
    }

    #[test]
    fn perf_summary_includes_cold_start_and_tool_fields() {
        // Phase 11.3.4 + 11.3.5: verify model_load_ms, tool_ms, model_ms, total_turn_ms
        // appear in the [runtime:perf] summary when tracing is enabled.
        //
        // Uses env-var isolation: set before constructing TurnPerformance (which captures
        // enabled at construction), removed immediately after so parallel tests are unaffected.
        std::env::set_var(RUNTIME_TRACE_ENV, "1");
        let mut perf = TurnPerformance::new();
        std::env::remove_var(RUNTIME_TRACE_ENV);

        perf.record_backend_timing("model_load", 4200);
        perf.record_backend_timing("ctx_create", 50);
        perf.record_backend_timing("tokenize", 20);
        perf.record_backend_timing("prefill_done", 1000);
        perf.record_backend_timing("generation_done", 800);
        perf.record_tool_elapsed(300);
        perf.record_tool_elapsed(150);

        let mut lines = Vec::new();
        perf.emit_summary(&mut |e| {
            if let RuntimeEvent::RuntimeTrace(line) = e {
                lines.push(line);
            }
        });

        assert_eq!(lines.len(), 1, "expect exactly one summary line");
        let summary = &lines[0];
        assert!(
            summary.contains("model_load_ms=4200"),
            "cold-start field missing: {summary}"
        );
        assert!(
            summary.contains("tool_ms=450"),
            "tool aggregation field missing: {summary}"
        );
        // model_ms = ctx_ms(50) + tokenize_ms(20) + prefill_ms(1000) + generation_ms(800) = 1870
        assert!(
            summary.contains("model_ms=1870"),
            "model-side aggregate missing: {summary}"
        );
        assert!(
            summary.contains("total_turn_ms="),
            "wall-clock turn time missing: {summary}"
        );
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

        let project_root = ProjectRoot::new(tmp.path().to_path_buf()).unwrap();
        let registry = default_registry().with_project_root(project_root.as_path_buf());
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
            &project_root,
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

        match outcome {
            ToolRoundOutcome::RuntimeDispatch {
                call: ToolInput::ReadFile { path },
                ..
            } => assert!(
                path.ends_with("sandbox/in_scope.py"),
                "usage lookup should auto-read the in-scope preferred candidate: {path}"
            ),
            _ => panic!("usage lookup search should now runtime-dispatch a preferred read"),
        }
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
        fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
        let project_root = ProjectRoot::new(tmp.path().to_path_buf()).unwrap();
        let registry = default_registry().with_project_root(project_root.as_path_buf());
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
            &project_root,
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
            &project_root,
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

        let project_root = ProjectRoot::new(tmp.path().to_path_buf()).unwrap();
        let registry = default_registry().with_project_root(project_root.as_path_buf());
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
            &project_root,
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
            &project_root,
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
            "from models.enums import TaskStatus\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: TaskStatus]",
                // Round 2: reads first definition file.
                // Runtime auto-dispatches task_service.py (import-only, no usage evidence).
                "[read_file: models/enums.py]",
                // Round 3: model tries second definition file.
                // candidate_reads_count reaches 2 after the auto-dispatch; read is blocked.
                "[read_file: models/alt_enums.py]",
                // Round 4 would be model synthesis — not reached; runtime terminates first.
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

    #[test]
    fn read_file_command_rejects_absolute_path() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let mut rt = make_runtime_in(Vec::<String>::new(), tmp.path());
        let events = collect_events(
            &mut rt,
            RuntimeRequest::ReadFile {
                path: "/etc/passwd".to_string(),
            },
        );
        let info: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let RuntimeEvent::InfoMessage(m) = e {
                    Some(m.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert!(
            info.iter().any(|m| m.contains("path must be relative")),
            "expected absolute path error, got: {info:?}"
        );
        assert!(
            rt.anchors.last_read_file().is_none(),
            "anchor must not be updated on rejected path"
        );
    }

    #[test]
    fn read_file_command_rejects_parent_traversal() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let mut rt = make_runtime_in(Vec::<String>::new(), tmp.path());
        let events = collect_events(
            &mut rt,
            RuntimeRequest::ReadFile {
                path: "src/../../etc/passwd".to_string(),
            },
        );
        let info: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let RuntimeEvent::InfoMessage(m) = e {
                    Some(m.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert!(
            info.iter().any(|m| m.contains("'..' components")),
            "expected parent traversal error, got: {info:?}"
        );
        assert!(
            rt.anchors.last_read_file().is_none(),
            "anchor must not be updated on rejected path"
        );
    }

    #[test]
    fn search_code_command_rejects_short_query() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let mut rt = make_runtime_in(Vec::<String>::new(), tmp.path());
        let events = collect_events(
            &mut rt,
            RuntimeRequest::SearchCode {
                query: "a".to_string(),
            },
        );
        let info: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let RuntimeEvent::InfoMessage(m) = e {
                    Some(m.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert!(
            info.iter().any(|m| m.contains("at least 2 characters")),
            "expected short query error, got: {info:?}"
        );
        assert!(
            rt.anchors.last_search_query().is_none(),
            "anchor must not be updated on rejected query"
        );
    }
}
