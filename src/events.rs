// src/events.rs
//
// Shared event types for communication between the model thread and the UI thread.

#[derive(Debug, Clone)]
pub enum PendingActionKind {
    ShellCommand,
}

#[derive(Debug, Clone)]
pub struct PendingAction {
    pub id: u64,
    pub kind: PendingActionKind,
    pub title: String,
    pub preview: String,
}

pub enum InferenceEvent {
    /// Model/backend loaded and ready.
    Ready,
    /// Active backend name for the sidebar.
    BackendName(String),
    /// A generated token — append to current response.
    Token(String),
    /// Tool calls being executed — shown in sidebar.
    ToolCall(String),
    /// A mutating action needs user approval before execution.
    PendingAction(PendingAction),
    /// Generation finished.
    Done,
    /// An error occurred.
    Error(String),
}
