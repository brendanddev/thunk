#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activity {
    Idle,
    Processing,
    LoadingModel,
    Generating,
    Responding,
    ExecutingTools,
}

impl Activity {
    pub fn label(self) -> &'static str {
        match self {
            Self::Idle => "ready",
            Self::Processing => "processing",
            Self::LoadingModel => "loading model",
            Self::Generating => "generating",
            Self::Responding => "responding",
            Self::ExecutingTools => "running tools",
        }
    }
}

/// Describes why the tool loop terminated and how the final answer was reached.
#[derive(Debug, Clone)]
pub enum AnswerSource {
    /// Model produced a final answer without using any tools.
    Direct,
    /// Model produced a final answer after one or more tool rounds.
    ToolAssisted { rounds: usize },
    /// Loop was cut off at the tool round limit before a final answer.
    ToolLimitReached,
}

#[derive(Debug, Clone)]
pub enum RuntimeRequest {
    Submit { text: String },
    /// Clears conversation history and resets to a fresh session.
    Reset,
}

#[derive(Debug, Clone)]
pub enum RuntimeEvent {
    ActivityChanged(Activity),
    AssistantMessageStarted,
    AssistantMessageChunk(String),
    AssistantMessageFinished,
    ToolCallStarted { name: String },
    ToolCallFinished { name: String, success: bool },
    AnswerReady(AnswerSource),
    Failed { message: String },
}
