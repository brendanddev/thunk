// src/events.rs
//
// Shared event types for communication between the model thread and the UI thread.

pub enum InferenceEvent {
    /// Model/backend loaded and ready.
    Ready,
    /// Active backend name for the sidebar.
    BackendName(String),
    /// A generated token — append to current response.
    Token(String),
    /// Tool calls being executed — shown in sidebar.
    ToolCall(String),
    /// Generation finished.
    Done,
    /// An error occurred.
    Error(String),
}