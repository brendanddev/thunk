use crate::app::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub messages: Vec<Message>,
}

impl GenerateRequest {
    pub fn new(messages: Vec<Message>) -> Self {
        Self { messages }
    }
}

#[derive(Debug, Clone)]
pub enum BackendStatus {
    LoadingModel,
    Generating,
}

#[derive(Debug, Clone)]
pub enum BackendEvent {
    StatusChanged(BackendStatus),
    TextDelta(String),
    Finished,
}

pub trait ModelBackend: Send {
    fn name(&self) -> &str;

    fn generate(
        &mut self,
        request: GenerateRequest,
        on_event: &mut dyn FnMut(BackendEvent),
    ) -> Result<()>;
}
