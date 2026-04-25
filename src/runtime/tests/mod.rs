use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::app::config::Config;
use crate::llm::backend::{BackendEvent, GenerateRequest, ModelBackend};
use crate::tools::default_registry;

pub use super::{AnswerSource, PendingAction, RiskLevel, Runtime, RuntimeEvent, RuntimeRequest};

mod anchors;
mod approval;
mod candidate_selection;
mod finalization;
mod git_acquisition;
mod integration_misc;
mod investigation;
mod investigation_modes;
mod path_scope;
mod read_bounds;
mod search_budget;
mod search_guardrails;
mod tool_round;
mod tool_surface;

pub struct TestBackend {
    responses: Vec<String>,
    call_count: usize,
}

impl TestBackend {
    pub fn new(responses: Vec<impl Into<String>>) -> Self {
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

pub struct RecordingBackend {
    responses: Vec<String>,
    call_count: usize,
    requests: Arc<Mutex<Vec<GenerateRequest>>>,
}

impl RecordingBackend {
    pub fn new(
        responses: Vec<impl Into<String>>,
        requests: Arc<Mutex<Vec<GenerateRequest>>>,
    ) -> Self {
        Self {
            responses: responses.into_iter().map(Into::into).collect(),
            call_count: 0,
            requests,
        }
    }
}

impl ModelBackend for RecordingBackend {
    fn name(&self) -> &str {
        "recording-test"
    }

    fn generate(
        &mut self,
        request: GenerateRequest,
        on_event: &mut dyn FnMut(BackendEvent),
    ) -> crate::app::Result<()> {
        self.requests.lock().unwrap().push(request);
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

pub fn make_runtime(responses: Vec<impl Into<String>>) -> Runtime {
    Runtime::new(
        &Config::default(),
        &PathBuf::from("."),
        Box::new(TestBackend::new(responses)),
        default_registry(PathBuf::from(".")),
    )
}

pub fn make_runtime_in(responses: Vec<impl Into<String>>, root: &std::path::Path) -> Runtime {
    Runtime::new(
        &Config::default(),
        root,
        Box::new(TestBackend::new(responses)),
        default_registry(root.to_path_buf()),
    )
}

pub fn make_runtime_with_recorded_requests(
    responses: Vec<impl Into<String>>,
) -> (Runtime, Arc<Mutex<Vec<GenerateRequest>>>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let runtime = Runtime::new(
        &Config::default(),
        &PathBuf::from("."),
        Box::new(RecordingBackend::new(responses, Arc::clone(&requests))),
        default_registry(PathBuf::from(".")),
    );
    (runtime, requests)
}

pub fn collect_events(runtime: &mut Runtime, request: RuntimeRequest) -> Vec<RuntimeEvent> {
    let mut events = Vec::new();
    runtime.handle(request, &mut |e| events.push(e));
    events
}

pub fn init_git_repo(root: &std::path::Path) {
    let status = std::process::Command::new("git")
        .args(["init"])
        .current_dir(root)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .unwrap();
    assert!(status.success(), "git init must succeed");
}

pub fn has_failed(events: &[RuntimeEvent]) -> bool {
    events
        .iter()
        .any(|e| matches!(e, RuntimeEvent::Failed { .. }))
}

pub fn failed_message(events: &[RuntimeEvent]) -> Option<String> {
    events.iter().find_map(|e| {
        if let RuntimeEvent::Failed { message } = e {
            Some(message.clone())
        } else {
            None
        }
    })
}
