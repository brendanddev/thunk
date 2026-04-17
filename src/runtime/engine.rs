use std::path::Path;

use crate::app::config::Config;
use crate::app::Result;
use crate::llm::backend::{BackendEvent, BackendStatus, GenerateRequest, ModelBackend};
use crate::tools::{ExecutionKind, PendingAction, ToolInput, ToolRegistry, ToolRunResult};

use super::conversation::Conversation;
use super::prompt;
use super::tool_codec;
use super::types::{Activity, AnswerSource, RuntimeEvent, RuntimeRequest};

/// Maximum tool rounds per turn. Prevents runaway loops when the model keeps
/// producing tool calls without reaching a final answer.
const MAX_TOOL_ROUNDS: usize = 10;

/// Maximum automatic corrections per turn. One correction is enough — if the
/// model fabricates twice in a row the prompt fix is insufficient and we surface
/// the failure rather than looping silently.
const MAX_CORRECTIONS: usize = 1;

/// Injected into the conversation when a fabricated tool-result block is detected.
/// Shown to the model only; not displayed in the TUI.
/// The [runtime:correction] sentinel prefix lets session restore detect and strip these messages
/// so they do not pollute future conversation context.
const FABRICATION_CORRECTION: &str =
    "[runtime:correction] Your response contained a result block which is forbidden. \
     You must emit ONLY a tool call tag (e.g. [read_file: path]) or answer directly in plain text. \
     Output the tool call tag now, with no other text.";

/// Injected when the model uses a wrong opening tag for a block tool (e.g. [test_file] instead
/// of [write_file]). Tag names are fixed — the model must use the exact names from the protocol.
const MALFORMED_BLOCK_CORRECTION: &str =
    "[runtime:correction] Your response contained a block with an unrecognized opening tag. \
     Tag names are exact — you must use [write_file], [edit_file], etc. exactly as shown. \
     Do not rename or abbreviate them. Emit the correct tool call now with no other text.";

/// Returns a stable fingerprint for a tool call, used for consecutive-cycle detection.
/// Null bytes separate fields; they cannot appear in paths, queries, or file content
/// on any supported platform, so false matches are impossible.
fn call_fingerprint(input: &ToolInput) -> String {
    match input {
        ToolInput::ReadFile { path } => format!("read_file\x00{path}"),
        ToolInput::ListDir { path } => format!("list_dir\x00{path}"),
        ToolInput::SearchCode { query, path } => {
            format!("search_code\x00{query}\x00{}", path.as_deref().unwrap_or(""))
        }
        ToolInput::EditFile { path, search, replace } => {
            format!("edit_file\x00{path}\x00{search}\x00{replace}")
        }
        ToolInput::WriteFile { path, content } => {
            format!("write_file\x00{path}\x00{content}")
        }
    }
}

pub struct Runtime {
    conversation: Conversation,
    backend: Box<dyn ModelBackend>,
    registry: ToolRegistry,
    system_prompt: String,
    /// Holds a mutating tool action that is waiting for user approval.
    /// Set when a tool round suspends; cleared by Approve or Reject.
    /// At most one pending action exists at any time.
    pending_action: Option<PendingAction>,
}

/// Outcome of dispatching one round of tool calls.
enum ToolRoundOutcome {
    /// All tools in this round completed immediately; results are ready to push.
    Completed { results: String },
    /// A tool requested approval. Results accumulated before it are preserved.
    /// The turn is now suspended; the caller must store pending and fire the event.
    ApprovalRequired { accumulated: String, pending: PendingAction },
}

impl Runtime {
    pub fn new(
        config: &Config,
        project_root: &Path,
        backend: Box<dyn ModelBackend>,
        registry: ToolRegistry,
    ) -> Self {
        let specs = registry.specs();
        let system_prompt =
            prompt::build_system_prompt(&config.app.name, project_root, &specs);
        Self {
            conversation: Conversation::new(system_prompt.clone()),
            backend,
            registry,
            system_prompt,
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
        self.conversation.reset(self.system_prompt.clone());
        on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
    }

    fn handle_submit(&mut self, text: String, on_event: &mut dyn FnMut(RuntimeEvent)) {
        if self.pending_action.is_some() {
            on_event(RuntimeEvent::Failed {
                message: "Cannot submit while a tool approval is pending. Use /approve or /reject first.".to_string(),
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

        self.conversation.push_user(text);
        on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
        self.run_turns(0, on_event);
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
                on_event(RuntimeEvent::ToolCallFinished { name: tool_name.clone(), summary: Some(summary) });
                let result_text = tool_codec::format_tool_result(&tool_name, &output);
                self.conversation.push_user(result_text);
                self.conversation.trim_tool_exchanges_if_needed();
                // Re-enter the generation loop so the model synthesizes a response
                // confirming what was done — mirrors the Immediate tool path in run_turns.
                on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
                self.run_turns(1, on_event);
            }
            Err(e) => {
                on_event(RuntimeEvent::ToolCallFinished { name: tool_name.clone(), summary: None });
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
        on_event(RuntimeEvent::ToolCallFinished { name: tool_name.clone(), summary: None });
        let rejection = tool_codec::format_tool_error(
            &tool_name,
            "user rejected this action — do not retry or re-propose it. \
             Acknowledge the cancellation in plain text and wait for the user's next instruction.",
        );
        self.conversation.push_user(rejection);

        on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
        self.run_turns(0, on_event);
    }

    /// Runs the generate -> tool-round loop until the model produces a final answer,
    /// the tool round limit is reached, or a tool action requires approval.
    /// `tool_rounds` is the count already consumed before this call (0 for a fresh turn).
    fn run_turns(&mut self, mut tool_rounds: usize, on_event: &mut dyn FnMut(RuntimeEvent)) {
        let mut corrections = 0usize;
        // Tracks the fingerprint of the last executed tool call within this run.
        // Resets to None at the start of every run_turns invocation so cycle detection
        // is scoped to one turn and never blocks calls across separate user submissions.
        let mut last_call_key: Option<String> = None;
        loop {
            let response = match run_generate_turn(
                self.backend.as_mut(),
                &mut self.conversation,
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
                    on_event(RuntimeEvent::Failed { message: e.to_string() });
                    return;
                }
            };

            let calls = tool_codec::parse_all_tool_inputs(&response);

            if calls.is_empty() {
                // Fabricated [tool_result:] / [tool_error:] blocks mean the model bypassed the
                // protocol. Attempt one automatic correction before surfacing the error.
                if tool_codec::contains_fabricated_exchange(&response) {
                    if corrections < MAX_CORRECTIONS {
                        corrections += 1;
                        self.conversation.discard_last_if_assistant();
                        self.conversation.push_user(FABRICATION_CORRECTION.to_string());
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
                        self.conversation.push_user(MALFORMED_BLOCK_CORRECTION.to_string());
                        continue;
                    }
                    on_event(RuntimeEvent::Failed {
                        message: "Model used incorrect tool tag names. Try rephrasing your request.".to_string(),
                    });
                    on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                    return;
                }
                let source = if tool_rounds == 0 {
                    AnswerSource::Direct
                } else {
                    AnswerSource::ToolAssisted { rounds: tool_rounds }
                };
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

            match run_tool_round(&self.registry, calls, &mut last_call_key, on_event) {
                ToolRoundOutcome::Completed { results } => {
                    self.conversation.push_user(results);
                    self.conversation.trim_tool_exchanges_if_needed();
                    // Signal re-entry before the next generate so the status bar
                    // transitions cleanly from "executing tools" → "processing" → …
                    on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
                    // Do not return — loop continues so the model is re-invoked
                    // with the tool results in context to produce a synthesis response.
                }
                ToolRoundOutcome::ApprovalRequired { accumulated, pending } => {
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

    #[cfg(test)]
    pub(crate) fn set_pending_for_test(&mut self, action: PendingAction) {
        self.pending_action = Some(action);
    }
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
fn run_tool_round(
    registry: &ToolRegistry,
    calls: Vec<ToolInput>,
    last_call_key: &mut Option<String>,
    on_event: &mut dyn FnMut(RuntimeEvent),
) -> ToolRoundOutcome {
    let mut accumulated = String::new();

    for input in calls {
        let name = input.tool_name().to_string();
        let key = call_fingerprint(&input);
        on_event(RuntimeEvent::ToolCallStarted { name: name.clone() });

        if last_call_key.as_deref() == Some(key.as_str()) {
            let msg = format!("{name} called with identical arguments twice in a row");
            on_event(RuntimeEvent::ToolCallFinished { name: name.clone(), summary: None });
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
                    registry.spec_for(&name)
                        .map(|s| s.execution_kind == ExecutionKind::Immediate)
                        .unwrap_or(true),
                    "tool '{name}' returned Immediate but spec declares RequiresApproval"
                );
                let summary = tool_codec::render_compact_summary(&output);
                on_event(RuntimeEvent::ToolCallFinished { name: name.clone(), summary: Some(summary) });
                accumulated.push_str(&tool_codec::format_tool_result(&name, &output));
                *last_call_key = Some(key);
            }
            Ok(ToolRunResult::Approval(pending)) => {
                // Guard: spec must agree that this tool requires approval.
                debug_assert!(
                    registry.spec_for(&name)
                        .map(|s| s.execution_kind == ExecutionKind::RequiresApproval)
                        .unwrap_or(true),
                    "tool '{name}' returned Approval but spec declares Immediate"
                );
                return ToolRoundOutcome::ApprovalRequired { accumulated, pending };
            }
            Err(e) => {
                on_event(RuntimeEvent::ToolCallFinished { name: name.clone(), summary: None });
                accumulated.push_str(&tool_codec::format_tool_error(&name, &e.to_string()));
                // Do NOT update last_call_key on error: a failed call should not block
                // an identical retry. Cycle detection applies only to successful executions.
            }
        }
    }

    ToolRoundOutcome::Completed { results: accumulated }
}

/// Runs a single generation turn: sends the current conversation to the backend,
/// streams tokens into the conversation and fires events, then returns the
/// complete assistant response text, or None if the backend produced no output.
fn run_generate_turn(
    backend: &mut dyn ModelBackend,
    conversation: &mut Conversation,
    on_event: &mut dyn FnMut(RuntimeEvent),
) -> Result<Option<String>> {
    let request = GenerateRequest::new(conversation.snapshot());
    let mut started = false;

    let result = backend.generate(request, &mut |event| match event {
        BackendEvent::StatusChanged(status) => {
            on_event(RuntimeEvent::ActivityChanged(map_backend_status(status)));
        }
        BackendEvent::TextDelta(chunk) => {
            if !started {
                started = true;
                conversation.begin_assistant_reply();
                on_event(RuntimeEvent::ActivityChanged(Activity::Responding));
                on_event(RuntimeEvent::AssistantMessageStarted);
            }
            conversation.push_assistant_chunk(&chunk);
            on_event(RuntimeEvent::AssistantMessageChunk(chunk));
        }
        BackendEvent::Timing { stage, elapsed_ms } => {
            on_event(RuntimeEvent::BackendTiming { stage, elapsed_ms });
        }
        BackendEvent::Finished => {}
    });

    result?;

    if started {
        on_event(RuntimeEvent::AssistantMessageFinished);
        Ok(conversation.last_assistant_content().map(|s| s.to_string()))
    } else {
        Ok(None)
    }
}

fn map_backend_status(status: BackendStatus) -> Activity {
    match status {
        BackendStatus::LoadingModel => Activity::LoadingModel,
        BackendStatus::Generating => Activity::Generating,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::app::config::Config;
    use crate::tools::{default_registry, RiskLevel};

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
            let reply = self.responses.get(self.call_count).cloned().unwrap_or_default();
            self.call_count += 1;
            if !reply.is_empty() {
                on_event(BackendEvent::TextDelta(reply));
            }
            on_event(BackendEvent::Finished);
            Ok(())
        }
    }

    fn make_runtime(responses: Vec<impl Into<String>>) -> Runtime {
        Runtime::new(
            &Config::default(),
            &PathBuf::from("."),
            Box::new(TestBackend::new(responses)),
            default_registry(PathBuf::from(".")),
        )
    }

    fn collect_events(runtime: &mut Runtime, request: RuntimeRequest) -> Vec<RuntimeEvent> {
        let mut events = Vec::new();
        runtime.handle(request, &mut |e| events.push(e));
        events
    }

    fn has_failed(events: &[RuntimeEvent]) -> bool {
        events.iter().any(|e| matches!(e, RuntimeEvent::Failed { .. }))
    }

    fn failed_message(events: &[RuntimeEvent]) -> Option<String> {
        events.iter().find_map(|e| {
            if let RuntimeEvent::Failed { message } = e {
                Some(message.clone())
            } else {
                None
            }
        })
    }

    #[test]
    fn approve_with_no_pending_fires_failed() {
        let mut rt = make_runtime(vec!["hello"]);
        let events = collect_events(&mut rt, RuntimeRequest::Approve);
        assert!(has_failed(&events), "expected Failed, got: {events:?}");
        assert_eq!(
            failed_message(&events).as_deref(),
            Some("No pending action to approve.")
        );
    }

    #[test]
    fn reject_with_no_pending_fires_failed() {
        let mut rt = make_runtime(vec!["hello"]);
        let events = collect_events(&mut rt, RuntimeRequest::Reject);
        assert!(has_failed(&events), "expected Failed, got: {events:?}");
        assert_eq!(
            failed_message(&events).as_deref(),
            Some("No pending action to reject.")
        );
    }

    #[test]
    fn submit_while_pending_fires_failed() {
        let mut rt = make_runtime(vec!["hello"]);
        rt.set_pending_for_test(PendingAction {
            tool_name: "edit_file".into(),
            summary: "edit src/lib.rs".into(),
            risk: RiskLevel::Medium,
            payload: "{}".into(),
        });
        let events = collect_events(&mut rt, RuntimeRequest::Submit { text: "continue".into() });
        assert!(has_failed(&events), "expected Failed, got: {events:?}");
        assert!(
            failed_message(&events)
                .as_deref()
                .unwrap_or("")
                .contains("pending"),
        );
    }

    #[test]
    fn reset_clears_pending_state() {
        let mut rt = make_runtime(vec!["hello"]);
        rt.set_pending_for_test(PendingAction {
            tool_name: "write_file".into(),
            summary: "write src/new.rs".into(),
            risk: RiskLevel::High,
            payload: "{}".into(),
        });
        collect_events(&mut rt, RuntimeRequest::Reset);
        // After reset, approve should fail with "no pending" — not "submit blocked"
        let events = collect_events(&mut rt, RuntimeRequest::Approve);
        assert!(has_failed(&events), "expected Failed after reset, got: {events:?}");
        assert_eq!(
            failed_message(&events).as_deref(),
            Some("No pending action to approve.")
        );
    }

    #[test]
    fn cycle_detection_blocks_second_identical_call() {
        // Model emits the same list_dir call twice in one response.
        // First call executes; second is blocked with a cycle error.
        // A synthesis response is provided so the loop can complete normally.
        let mut rt = make_runtime(vec!["[list_dir: .][list_dir: .]", "Done."]);
        collect_events(&mut rt, RuntimeRequest::Submit { text: "list it twice".into() });

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| m.content.contains("[tool_result: list_dir]")),
            "first call must produce a tool result"
        );
        assert!(
            snapshot.iter().any(|m|
                m.content.contains("[tool_error: list_dir]") &&
                m.content.contains("identical arguments twice in a row")
            ),
            "second identical call must produce a cycle error"
        );
    }

    #[test]
    fn cycle_detection_allows_different_args() {
        // Two list_dir calls with different paths — neither should be blocked.
        // A synthesis response is provided so the loop can complete normally.
        let mut rt = make_runtime(vec!["[list_dir: .][list_dir: src/]", "Listed both."]);
        collect_events(&mut rt, RuntimeRequest::Submit { text: "list both".into() });

        let snapshot = rt.messages_snapshot();
        assert!(
            !snapshot.iter().any(|m| m.content.contains("identical arguments twice in a row")),
            "different args must not trigger cycle detection"
        );
    }

    #[test]
    fn tool_round_followed_by_synthesized_answer() {
        // After a tool call completes, the model is re-invoked in the same turn.
        // The synthesis response (no tool calls) produces the final AnswerReady event.
        let mut rt = make_runtime(vec!["[list_dir: .]", "The root contains several files."]);
        let events = collect_events(&mut rt, RuntimeRequest::Submit { text: "what is in root?".into() });

        assert!(!has_failed(&events), "unexpected failure: {events:?}");

        // AnswerReady must be ToolAssisted from the synthesis response
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e { Some(src.clone()) } else { None }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { rounds: 1 })),
            "expected ToolAssisted(1), got: {answer_source:?}"
        );

        // Conversation must contain: user prompt, assistant tool call, user tool result,
        // assistant synthesis — i.e. two assistant messages.
        let snapshot = rt.messages_snapshot();
        let assistant_msgs: Vec<_> = snapshot
            .iter()
            .filter(|m| m.role == crate::llm::backend::Role::Assistant)
            .collect();
        assert_eq!(assistant_msgs.len(), 2, "expected tool-call + synthesis, got: {assistant_msgs:?}");
        assert!(
            assistant_msgs[1].content.contains("several files"),
            "synthesis must contain model's response text"
        );
    }

    #[test]
    fn multi_tool_round_synthesizes_after_all_rounds() {
        // Model calls list_dir twice across two separate rounds, then synthesizes.
        // tool_rounds must reflect both rounds in the AnswerReady source.
        let mut rt = make_runtime(vec![
            "[list_dir: .]",
            "[list_dir: src/]",
            "Found everything I need.",
        ]);
        let events = collect_events(&mut rt, RuntimeRequest::Submit { text: "explore".into() });

        assert!(!has_failed(&events), "unexpected failure: {events:?}");

        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e { Some(src.clone()) } else { None }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { rounds: 2 })),
            "expected ToolAssisted(2), got: {answer_source:?}"
        );

        let snapshot = rt.messages_snapshot();
        let assistant_msgs: Vec<_> = snapshot
            .iter()
            .filter(|m| m.role == crate::llm::backend::Role::Assistant)
            .collect();
        assert_eq!(assistant_msgs.len(), 3, "expected two tool calls + synthesis");
    }

    #[test]
    fn malformed_block_triggers_correction_and_retries() {
        // Model emits [test_file]...[/write_file] — wrong opening tag, correct closing tag.
        // The engine should detect the malformed block, discard the response, inject
        // a correction, and re-invoke the model. The synthesis response closes the loop.
        let malformed = "[test_file]\npath: f.txt\n---content---\nhello\n[/write_file]";
        let mut rt = make_runtime(vec![malformed, "Done."]);
        let events = collect_events(&mut rt, RuntimeRequest::Submit { text: "create f.txt".into() });

        assert!(!has_failed(&events), "must not fail permanently: {events:?}");

        // The final answer must be the synthesis response, not the malformed block.
        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot.iter().rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(last_assistant, Some("Done."), "last assistant message must be synthesis");

        // The correction message must have been injected into the conversation.
        assert!(
            snapshot.iter().any(|m| m.content.starts_with("[runtime:correction]") &&
                m.content.contains("unrecognized opening tag")),
            "malformed block correction must be in conversation"
        );
    }

    #[test]
    fn cycle_detection_allows_retry_after_tool_error() {
        // A tool fails (bad path), then the model retries with the same args.
        // The retry must NOT be blocked as a cycle — only successful calls set the key.
        // list_dir returns an IO error for a non-existent path, then succeeds on ".".
        // The synthesis response closes the loop.
        let mut rt = make_runtime(vec!["[list_dir: /nonexistent/path][list_dir: .]", "Done."]);
        collect_events(&mut rt, RuntimeRequest::Submit { text: "list both".into() });

        let snapshot = rt.messages_snapshot();
        // The error from the first call must appear
        assert!(
            snapshot.iter().any(|m| m.content.contains("[tool_error: list_dir]")),
            "first call error must be in conversation"
        );
        // The successful retry must produce a result — no cycle error
        assert!(
            snapshot.iter().any(|m| m.content.contains("[tool_result: list_dir]")),
            "successful retry must not be blocked by cycle detection"
        );
        assert!(
            !snapshot.iter().any(|m|
                m.content.contains("identical arguments") &&
                m.content.contains("[tool_error: list_dir]") &&
                m.content.contains(".")
            ),
            "retry with same args after error must not trigger cycle detection"
        );
    }

    #[test]
    fn approve_synthesizes_after_successful_mutation() {
        // After approving a write_file call, the model must be re-invoked for synthesis.
        // The synthesis response closes the loop and becomes the final assistant message.
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Write an existing file so edit_file has something to edit.
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "hello").unwrap();
        let path = f.path().to_string_lossy().into_owned();

        // edit_file payload: path\x00search\x00replace (null-byte separated)
        let payload = format!("{}\x00hello\x00world", path);

        // The synthesis response — model confirms what was done.
        let mut rt = make_runtime(vec!["Done, the edit has been applied."]);
        let before_count = rt.messages_snapshot().len();

        rt.set_pending_for_test(PendingAction {
            tool_name: "edit_file".into(),
            summary: format!("edit {path}"),
            risk: RiskLevel::Medium,
            payload,
        });

        let events = collect_events(&mut rt, RuntimeRequest::Approve);
        assert!(!has_failed(&events), "approve must not fail: {events:?}");

        // Synthesis must have fired — AssistantMessageChunk is emitted during synthesis.
        assert!(
            events.iter().any(|e| matches!(e, RuntimeEvent::AssistantMessageChunk(_))),
            "approve must trigger synthesis: no AssistantMessageChunk in events"
        );

        let snapshot = rt.messages_snapshot();
        // Snapshot must have grown: at minimum tool result + synthesis assistant message.
        assert!(
            snapshot.len() > before_count,
            "snapshot must grow after approve + synthesis"
        );
        // Tool result must be present.
        assert!(
            snapshot.iter().any(|m| m.content.contains("[tool_result: edit_file]")),
            "tool result must be in conversation after approve"
        );
        // The final assistant message must be the synthesis response, not a tool call.
        let last_assistant = snapshot.iter().rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant);
        assert!(
            last_assistant.map(|m| m.content.contains("Done")).unwrap_or(false),
            "last assistant message must be the synthesis response"
        );
    }
}
