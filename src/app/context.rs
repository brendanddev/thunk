use std::path::Path;
use std::time::Instant;

use crate::logging::SessionLog;
use crate::runtime::{Runtime, RuntimeEvent, RuntimeRequest};
use crate::tools::ToolRegistry;

use super::config::Config;
use super::session::ActiveSession;
use super::Result;

/// Application-level orchestrator. Owns both the runtime and the active session.
///
/// The TUI works with AppContext rather than Runtime directly. This keeps session
/// persistence invisible to the TUI — from its perspective it calls handle() and
/// receives RuntimeEvents. The save happens after each Submit turn; failures are
/// returned as errors rather than silently written to stderr.
pub struct AppContext {
    runtime: Runtime,
    session: ActiveSession,
    log: Option<SessionLog>,
}

impl AppContext {
    /// Delegates to the runtime, then auto-saves after a Submit turn completes.
    /// Returns an error if the session save fails so callers can surface it.
    pub fn handle(
        &mut self,
        request: RuntimeRequest,
        on_event: &mut dyn FnMut(RuntimeEvent),
    ) -> Result<()> {
        // Save after any request that can mutate the conversation. Reset is excluded:
        // begin_new() handles its own session lifecycle separately.
        let should_save = matches!(
            request,
            RuntimeRequest::Submit { .. } | RuntimeRequest::Approve | RuntimeRequest::Reject
        );

        // Take log out of self so we can borrow self.runtime simultaneously.
        let mut log = self.log.take();
        if let Some(ref mut l) = log {
            l.log(&format!("request: {}", request_label(&request)));
        }

        // Stage timers: track generation and per-tool durations.
        let mut gen_start: Option<Instant> = None;
        let mut tool_start: Option<(String, Instant)> = None;

        self.runtime.handle(request, &mut |event| {
            if let Some(ref mut l) = log {
                match &event {
                    RuntimeEvent::AssistantMessageStarted => {
                        gen_start = Some(Instant::now());
                        l.log("generation: started");
                    }
                    RuntimeEvent::AssistantMessageFinished => {
                        if let Some(t) = gen_start.take() {
                            l.log_timed("generation: finished", t.elapsed());
                        }
                    }
                    RuntimeEvent::ToolCallStarted { name } => {
                        tool_start = Some((name.clone(), Instant::now()));
                        l.log(&format!("tool started: {name}"));
                    }
                    RuntimeEvent::ToolCallFinished { name, summary } => {
                        if let Some((_, t)) = tool_start.take() {
                            let label = match summary {
                                Some(s) => format!("tool done: {name} — {s}"),
                                None => format!("tool failed: {name}"),
                            };
                            l.log_timed(&label, t.elapsed());
                        }
                    }
                    RuntimeEvent::BackendTiming { stage, elapsed_ms } => {
                        l.log(&format!("backend: {stage} ({elapsed_ms}ms)"));
                        // Do not forward — TUI has no use for internal backend timings.
                        return;
                    }
                    other => {
                        if let Some(label) = event_label(other) {
                            l.log(&label);
                        }
                    }
                }
            }
            on_event(event);
        });

        self.log = log;

        if should_save {
            self.session.save(&self.runtime.messages_snapshot())?;
        }
        Ok(())
    }

    /// Resets the runtime conversation and starts a new session.
    /// The TUI handles its own message-list clearing separately.
    pub fn reset(&mut self) -> Result<()> {
        self.runtime.handle(RuntimeRequest::Reset, &mut |_| {});
        self.session.begin_new()?;
        Ok(())
    }

    /// Initializes the AppContext by building a Runtime and loading the session history.
    pub fn build(
        config: &Config,
        project_root: &Path,
        backend: Box<dyn crate::llm::backend::ModelBackend>,
        registry: ToolRegistry,
        session: ActiveSession,
        history: Vec<crate::llm::backend::Message>,
        log: Option<SessionLog>,
    ) -> Result<Self> {
        let mut runtime = Runtime::new(config, project_root, backend, registry);
        if !history.is_empty() {
            runtime.load_history(history);
        }
        Ok(Self { runtime, session, log })
    }
}

/// Defines labels for requests and events for logging purposes.
fn request_label(request: &RuntimeRequest) -> &'static str {
    match request {
        RuntimeRequest::Submit { .. } => "submit",
        RuntimeRequest::Reset => "reset",
        RuntimeRequest::Approve => "approve",
        RuntimeRequest::Reject => "reject",
    }
}

/// Labels for events that are not already handled with timing in handle().
fn event_label(event: &RuntimeEvent) -> Option<String> {
    match event {
        RuntimeEvent::ActivityChanged(a) => Some(format!("activity: {}", a.label())),
        RuntimeEvent::AnswerReady(source) => Some(format!("answer ready: {source:?}")),
        RuntimeEvent::Failed { message } => Some(format!("failed: {message}")),
        RuntimeEvent::ApprovalRequired(p) => Some(format!("approval required: {}", p.summary)),
        // Handled with timing in handle():
        RuntimeEvent::AssistantMessageStarted
        | RuntimeEvent::AssistantMessageFinished
        | RuntimeEvent::ToolCallStarted { .. }
        | RuntimeEvent::ToolCallFinished { .. }
        | RuntimeEvent::AssistantMessageChunk(_)
        | RuntimeEvent::BackendTiming { .. } => None,
    }
}
