use crate::error::Result;
use crate::events::InferenceEvent;
use serde::Serialize;
use std::sync::mpsc::Sender;

#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: content.to_string(),
        }
    }
    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: content.to_string(),
        }
    }
    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.to_string(),
        }
    }
}

/// Base system prompt
pub const SYSTEM_PROMPT_BASE: &str = "\
You are params, a local AI coding assistant. \
Be concise and precise. \
Prefer code over lengthy explanation. \
When showing code, use markdown code blocks with the language specified. \
If you are unsure about something, say so.";

/// Returns the full system prompt with tool descriptions appended
pub fn system_prompt_with_tools(tool_descriptions: &str) -> String {
    format!("{}\n\n{}", SYSTEM_PROMPT_BASE, tool_descriptions)
}

/// For backward compatibility
pub const SYSTEM_PROMPT: &str = SYSTEM_PROMPT_BASE;

pub trait InferenceBackend: Send {
    fn generate(&self, messages: &[Message], tx: Sender<InferenceEvent>) -> Result<()>;

    fn name(&self) -> String;
}
