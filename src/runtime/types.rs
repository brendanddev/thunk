use crate::tools::PendingAction;

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
    /// Runtime produced a deterministic terminal answer without model synthesis.
    RuntimeTerminal {
        reason: RuntimeTerminalReason,
        rounds: usize,
    },
    /// Loop was cut off at the tool round limit before a final answer.
    ToolLimitReached,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeTerminalReason {
    RejectedMutation,
    ReadFileFailed,
    /// Search was attempted but all results were empty and no file was read.
    /// The runtime emits the answer directly rather than letting the model speculate.
    InsufficientEvidence,
}

#[derive(Debug, Clone)]
pub enum RuntimeRequest {
    Submit {
        text: String,
    },
    /// Clears conversation history and resets to a fresh session.
    Reset,
    /// Confirms a pending tool action, allowing execute_approved() to run.
    Approve,
    /// Cancels a pending tool action without executing it.
    Reject,
}

#[derive(Debug, Clone)]
pub enum RuntimeEvent {
    ActivityChanged(Activity),
    AssistantMessageStarted,
    AssistantMessageChunk(String),
    AssistantMessageFinished,
    ToolCallStarted {
        name: String,
    },
    /// Fired when a tool completes. `summary` is a compact one-line render of the
    /// result for TUI display. `None` means the tool failed.
    ToolCallFinished {
        name: String,
        summary: Option<String>,
    },
    /// Fired when a mutating tool requires user approval before execution.
    /// The turn is paused until RuntimeRequest::Approve or Reject is received.
    ApprovalRequired(PendingAction),
    AnswerReady(AnswerSource),
    Failed {
        message: String,
    },
    /// Advisory timing event routed from the backend. Consumed by the logging layer only;
    /// must not be forwarded to the TUI or drive any control flow.
    BackendTiming {
        stage: &'static str,
        elapsed_ms: u64,
    },
}
