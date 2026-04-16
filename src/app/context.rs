use std::path::Path;

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
}

impl AppContext {
    /// Delegates to the runtime, then auto-saves after a Submit turn completes.
    /// Returns an error if the session save fails so callers can surface it.
    pub fn handle(
        &mut self,
        request: RuntimeRequest,
        on_event: &mut dyn FnMut(RuntimeEvent),
    ) -> Result<()> {
        let is_submit = matches!(request, RuntimeRequest::Submit { .. });
        self.runtime.handle(request, on_event);
        if is_submit {
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
    ) -> Result<Self> {
        let mut runtime = Runtime::new(config, project_root, backend, registry);
        if !history.is_empty() {
            runtime.load_history(history);
        }
        Ok(Self { runtime, session })
    }
}
