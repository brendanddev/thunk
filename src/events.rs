// src/events.rs
//
// Shared event types for communication between the model thread and the UI thread.

#[derive(Debug, Clone)]
pub enum PendingActionKind {
    ShellCommand,
    FileWrite,
}

#[derive(Debug, Clone)]
pub struct PendingAction {
    pub id: u64,
    pub kind: PendingActionKind,
    pub title: String,
    pub preview: String,
}

#[derive(Debug, Clone)]
pub struct BudgetUpdate {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub estimated_cost_usd: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct CacheUpdate {
    pub last_hit: bool,
    pub hits: usize,
    pub misses: usize,
    pub tokens_saved: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressStatus {
    Started,
    Updated,
    Finished,
    Failed,
}

#[derive(Debug, Clone)]
pub struct ProgressTrace {
    pub status: ProgressStatus,
    pub label: String,
    pub persist: bool,
}

pub enum InferenceEvent {
    /// Model/backend loaded and ready.
    Ready,
    /// A previous session was restored. Carries display-friendly (role, content) pairs
    /// and the unix timestamp when the session was saved.
    SessionRestored {
        display_messages: Vec<(String, String)>,
        saved_at: u64,
    },
    /// Active backend name for the sidebar.
    BackendName(String),
    /// A generation phase started.
    GenerationStarted {
        label: String,
        show_placeholder: bool,
    },
    /// A transient or durable progress-trace update.
    Trace(ProgressTrace),
    /// A generated token — append to current response.
    Token(String),
    /// Tool calls being executed — shown in sidebar.
    ToolCall(String),
    /// A local context/result message should be added to the chat.
    ContextMessage(String),
    /// Reflection mode changed for the current session.
    ReflectionEnabled(bool),
    /// Eco mode changed for the current session.
    EcoEnabled(bool),
    /// Separate content debug logging changed for the current session.
    DebugLoggingEnabled(bool),
    /// Updated token/cost estimates for the current session.
    Budget(BudgetUpdate),
    /// Updated cache stats for the current session.
    Cache(CacheUpdate),
    /// A mutating action needs user approval before execution.
    PendingAction(PendingAction),
    /// Generation finished.
    Done,
    /// An error occurred.
    Error(String),
}
