use std::path::PathBuf;

use crate::app::config::Config;
use crate::app::paths::AppPaths;

use super::types::{Activity, RuntimeEvent, RuntimeRequest};

#[derive(Debug)]
pub struct Runtime {
    app_name: String,
    workspace_root: PathBuf,
}

impl Runtime {
    pub fn new(config: &Config, paths: &AppPaths) -> Self {
        Self {
            app_name: config.app.name.clone(),
            workspace_root: paths.root_dir.clone(),
        }
    }

    pub fn handle(&mut self, request: RuntimeRequest) -> Vec<RuntimeEvent> {
        match request {
            RuntimeRequest::Submit { text } => self.handle_submit(text),
        }
    }

    fn handle_submit(&mut self, text: String) -> Vec<RuntimeEvent> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return vec![RuntimeEvent::Failed {
                message: "Cannot submit an empty prompt.".to_string(),
            }];
        }

        let reply = self.compose_reply(trimmed);
        vec![
            RuntimeEvent::ActivityChanged(Activity::Processing),
            RuntimeEvent::AssistantMessageStarted,
            RuntimeEvent::ActivityChanged(Activity::Responding),
            RuntimeEvent::AssistantMessageChunk(reply),
            RuntimeEvent::AssistantMessageFinished,
            RuntimeEvent::ActivityChanged(Activity::Idle),
        ]
    }

    fn compose_reply(&self, prompt: &str) -> String {
        match prompt {
            "help" | "Help" => format!(
                "{} is running with the new runtime boundary in place. The next layer is wiring a real model backend into this request/event flow.",
                self.app_name
            ),
            _ => format!(
                "Runtime received: \"{prompt}\".\n\nThis reply is coming from src/runtime/engine.rs, so the TUI is now only rendering state and forwarding requests. Workspace root: {}.",
                self.workspace_root.display()
            ),
        }
    }
}
