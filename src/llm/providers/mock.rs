use crate::app::Result;
use crate::llm::backend::{
    BackendCapabilities, BackendEvent, BackendStatus, GenerateRequest, ModelBackend, Role,
};

pub struct MockBackend {
    app_name: String,
}

impl MockBackend {
    pub fn new(app_name: String) -> Self {
        Self { app_name }
    }

    fn build_reply(&self, request: &GenerateRequest) -> String {
        let prompt = request
            .messages
            .iter()
            .rev()
            .find(|message| message.role == Role::User)
            .map(|message| message.content.trim())
            .unwrap_or("...");

        match prompt {
            "help" | "Help" => format!(
                "{} is now talking through the new llm/runtime boundary. The next step is swapping this mock provider for a real model provider without changing the TUI or runtime contract.",
                self.app_name
            ),
            _ => format!(
                "Mock backend received: \"{prompt}\".\n\nThis response is being produced by src/llm/providers/mock.rs through the ModelBackend trait, so the new runtime no longer owns assistant text generation directly."
            ),
        }
    }
}

impl ModelBackend for MockBackend {
    fn name(&self) -> &str {
        "mock"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            context_window_tokens: None,
            max_output_tokens: None,
        }
    }

    fn generate(
        &mut self,
        request: GenerateRequest,
        on_event: &mut dyn FnMut(BackendEvent),
    ) -> Result<()> {
        on_event(BackendEvent::StatusChanged(BackendStatus::Generating));
        let reply = self.build_reply(&request);
        for chunk in chunk_text(&reply, 28) {
            on_event(BackendEvent::TextDelta(chunk));
        }
        on_event(BackendEvent::Finished);
        Ok(())
    }
}

fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    if max_chars == 0 || text.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut current_len = 0usize;

    for ch in text.chars() {
        current.push(ch);
        current_len += 1;

        if current_len >= max_chars && (ch.is_whitespace() || matches!(ch, '.' | ',' | ';')) {
            chunks.push(std::mem::take(&mut current));
            current_len = 0;
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}
