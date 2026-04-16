use std::path::Path;

use crate::app::config::Config;
use crate::app::Result;
use crate::llm::backend::{BackendEvent, BackendStatus, GenerateRequest, ModelBackend};
use crate::tools::{PendingAction, ToolInput, ToolRegistry, ToolRunResult};

use super::conversation::Conversation;
use super::prompt;
use super::tool_codec;
use super::types::{Activity, AnswerSource, RuntimeEvent, RuntimeRequest};

/// Maximum tool rounds per turn. Prevents runaway loops when the model keeps
/// producing tool calls without reaching a final answer.
const MAX_TOOL_ROUNDS: usize = 10;

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
                on_event(RuntimeEvent::ToolCallFinished { name: tool_name.clone(), success: true });
                let result_text = tool_codec::format_tool_result(&tool_name, &output);
                self.conversation.push_user(result_text);
            }
            Err(e) => {
                on_event(RuntimeEvent::ToolCallFinished { name: tool_name.clone(), success: false });
                let error_text = tool_codec::format_tool_error(&tool_name, &e.to_string());
                self.conversation.push_user(error_text);
            }
        }

        on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
        self.run_turns(0, on_event);
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
        on_event(RuntimeEvent::ToolCallFinished { name: tool_name.clone(), success: false });
        let rejection = tool_codec::format_tool_error(&tool_name, "user rejected the proposed change");
        self.conversation.push_user(rejection);

        on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
        self.run_turns(0, on_event);
    }

    /// Runs the generate → tool-round loop until the model produces a final answer,
    /// the tool round limit is reached, or a tool action requires approval.
    /// `tool_rounds` is the count already consumed before this call (0 for a fresh turn).
    fn run_turns(&mut self, mut tool_rounds: usize, on_event: &mut dyn FnMut(RuntimeEvent)) {
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

            let calls = tool_codec::parse_tool_calls(&response);

            if calls.is_empty() {
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

            if tool_rounds > MAX_TOOL_ROUNDS {
                on_event(RuntimeEvent::AnswerReady(AnswerSource::ToolLimitReached));
                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                return;
            }

            on_event(RuntimeEvent::ActivityChanged(Activity::ExecutingTools));

            match run_tool_round(&self.registry, calls, on_event) {
                ToolRoundOutcome::Completed { results } => {
                    self.conversation.push_user(results);
                }
                ToolRoundOutcome::ApprovalRequired { accumulated, pending } => {
                    if !accumulated.is_empty() {
                        self.conversation.push_user(accumulated);
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
    fn set_pending_for_test(&mut self, action: PendingAction) {
        self.pending_action = Some(action);
    }
}

/// Dispatches one round of tool calls, accumulating results.
/// Stops at the first tool that requires approval and returns any results
/// accumulated before it alongside the PendingAction.
/// ToolCallStarted is fired for each tool, but ToolCallFinished is NOT fired
/// for the approval-requiring tool — handle_approve/reject fires it after resolution.
fn run_tool_round(
    registry: &ToolRegistry,
    calls: Vec<ToolInput>,
    on_event: &mut dyn FnMut(RuntimeEvent),
) -> ToolRoundOutcome {
    let mut accumulated = String::new();

    for input in calls {
        let name = input.tool_name().to_string();
        on_event(RuntimeEvent::ToolCallStarted { name: name.clone() });
        match registry.dispatch(input) {
            Ok(ToolRunResult::Immediate(output)) => {
                on_event(RuntimeEvent::ToolCallFinished { name: name.clone(), success: true });
                accumulated.push_str(&tool_codec::format_tool_result(&name, &output));
            }
            Ok(ToolRunResult::Approval(pending)) => {
                return ToolRoundOutcome::ApprovalRequired { accumulated, pending };
            }
            Err(e) => {
                on_event(RuntimeEvent::ToolCallFinished { name: name.clone(), success: false });
                accumulated.push_str(&tool_codec::format_tool_error(&name, &e.to_string()));
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
}
