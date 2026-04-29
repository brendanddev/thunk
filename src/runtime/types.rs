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
    /// A final answer was produced after one or more tool rounds.
    ToolAssisted { rounds: usize },
    /// Runtime produced a deterministic terminal answer without model synthesis.
    RuntimeTerminal {
        reason: RuntimeTerminalReason,
        rounds: usize,
    },
    /// Loop was cut off at the tool round limit before a final answer.
    ToolLimitReached,
}

/// Runtime-owned terminal outcomes. These are policy decisions, not model output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeTerminalReason {
    RejectedMutation,
    ReadFileFailed,
    /// A mutation tool call was rejected at resolver level (e.g. path escapes project root).
    /// Distinct from RejectedMutation, which is a user-initiated cancellation of an approved action.
    MutationFailed,
    RepeatedDisallowedTool,
    RepeatedToolAfterEvidenceReady,
    RepeatedWeakSearchQuery,
    /// Model attempted further tool use after the turn's artifact was already acquired.
    RepeatedToolAfterAnswerPhase,
    /// Search was attempted but all results were empty and no file was read.
    /// The runtime emits the answer directly rather than letting the model speculate.
    InsufficientEvidence,
}

/// External inputs the runtime accepts from the app/TUI layer.
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
    /// Read-only query: returns the last assistant message as an InfoMessage event.
    /// Does not mutate conversation state or trigger session save.
    QueryLast,
    /// Read-only query: returns current anchor state as an InfoMessage event.
    /// Does not mutate any state or trigger session save.
    QueryAnchors,
    /// Read-only query: returns bounded recent conversation history as an InfoMessage event.
    /// Does not mutate any state or trigger session save.
    QueryHistory,
    /// Command-triggered read_file invocation. Goes through CommandTool allowlist.
    /// Does not mutate conversation or trigger session save.
    ReadFile {
        path: String,
    },
    /// Command-triggered search_code invocation. Goes through CommandTool allowlist.
    /// Does not mutate conversation or trigger session save.
    SearchCode {
        query: String,
    },
}

/// Events emitted by the runtime for UI rendering, logging, and lifecycle handling.
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
    /// Informational output from a read-only runtime query (e.g. /last).
    /// Rendered by the TUI as a system message; never added to conversation state.
    InfoMessage(String),
    /// Advisory timing event routed from the backend. Consumed by the logging layer only;
    /// must not be forwarded to the TUI or drive any control flow.
    BackendTiming {
        stage: &'static str,
        elapsed_ms: u64,
    },
    /// Advisory runtime decision trace. Consumed by the application logging layer only;
    /// must not be forwarded to the TUI or drive any control flow.
    RuntimeTrace(String),
}
