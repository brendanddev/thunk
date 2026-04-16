#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activity {
    Idle,
    Processing,
    Responding,
}

impl Activity {
    pub fn label(self) -> &'static str {
        match self {
            Self::Idle => "ready",
            Self::Processing => "processing",
            Self::Responding => "responding",
        }
    }
}

#[derive(Debug, Clone)]
pub enum RuntimeRequest {
    Submit { text: String },
}

#[derive(Debug, Clone)]
pub enum RuntimeEvent {
    ActivityChanged(Activity),
    AssistantMessageStarted,
    AssistantMessageChunk(String),
    AssistantMessageFinished,
    Failed { message: String },
}
