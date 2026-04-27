use crate::app::Result;

/// Role of a message within a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    // Converts the Role enum into its corresponding string representation for prompt formatting.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// A single message in the conversation history passed to the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// Input to a model generation call.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub messages: Vec<Message>,
}

impl GenerateRequest {
    pub fn new(messages: Vec<Message>) -> Self {
        Self { messages }
    }
}

/// High-level status updates emitted by the backend.
#[derive(Debug, Clone)]
pub enum BackendStatus {
    LoadingModel,
    Generating,
}

/// Events streamed from the model during generation.
#[derive(Debug, Clone)]
pub enum BackendEvent {
    StatusChanged(BackendStatus),
    TextDelta(String),
    Finished,
    /// Advisory timing event — emitted by backends at key internal stages.
    /// Consumers may route this to logging; it must not affect control flow.
    Timing {
        stage: &'static str,
        elapsed_ms: u64,
    },
}

/// Static capabilities exposed by a backend so callers can make informed decisions
/// without calling generate(). All fields are optional — `None` means unknown or
/// unconfigured; callers must not assume a bound when a field is absent.
#[derive(Debug, Clone, Copy)]
pub struct BackendCapabilities {
    /// Maximum tokens the backend can receive as input in a single call.
    /// `None` if unknown (e.g. mock) or if the backend defers to the model's
    /// trained context window (llama.cpp with context_tokens = 0).
    pub context_window_tokens: Option<u32>,

    /// Maximum tokens the backend will generate per call.
    /// `None` if unknown or unconfigured.
    pub max_output_tokens: Option<usize>,
}

/// Defines the abstraction over a language model backend.
/// This is responsible for receiving a structured generation request, streaming output via events, and
/// hiding backend-specific implementation details from the rest of the application.
pub trait ModelBackend: Send {
    fn name(&self) -> &str;

    /// Returns static capability information for this backend.
    /// Called at construction time or on-demand; never during generation.
    fn capabilities(&self) -> BackendCapabilities;

    fn generate(
        &mut self,
        request: GenerateRequest,
        on_event: &mut dyn FnMut(BackendEvent),
    ) -> Result<()>;
}
