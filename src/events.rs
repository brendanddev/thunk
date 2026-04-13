use crate::safety::InspectionReport;

#[derive(Debug, Clone)]
pub enum PendingActionKind {
    ShellCommand,
    FileWrite,
    FileEdit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactProvenance {
    Legacy,
    Verified,
}

#[derive(Debug, Clone)]
pub struct MemoryFactView {
    pub content: String,
    pub provenance: FactProvenance,
}

#[derive(Debug, Clone)]
pub struct MemorySessionExcerptView {
    pub session_label: String,
    pub role: String,
    pub excerpt: String,
}

#[derive(Debug, Clone)]
pub struct MemorySkippedReasonCount {
    pub reason: String,
    pub count: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryUpdateReport {
    pub accepted_facts: Vec<MemoryFactView>,
    pub skipped_reasons: Vec<MemorySkippedReasonCount>,
    pub duplicate_count: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryConsolidationView {
    pub ttl_pruned: usize,
    pub dedup_removed: usize,
    pub cap_removed: usize,
}

#[derive(Debug, Clone, Default)]
pub struct MemorySnapshot {
    pub loaded_facts: Vec<MemoryFactView>,
    pub last_summary_paths: Vec<String>,
    pub last_retrieval_query: Option<String>,
    pub last_selected_facts: Vec<MemoryFactView>,
    pub last_selected_session_excerpts: Vec<MemorySessionExcerptView>,
    pub last_update: Option<MemoryUpdateReport>,
    pub last_consolidation: Option<MemoryConsolidationView>,
}

#[derive(Debug, Clone)]
pub struct PendingAction {
    pub id: u64,
    pub kind: PendingActionKind,
    pub title: String,
    pub preview: String,
    pub inspection: InspectionReport,
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

#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub id: String,
    pub name: Option<String>,
    pub message_count: usize,
}

pub enum InferenceEvent {
    /// Model/backend loaded and ready
    Ready,

    /// A session was loaded into the active conversation. Carries display-friendly
    /// (role, content) pairs and optional saved timestamp metadata
    SessionLoaded {
        session: SessionInfo,
        display_messages: Vec<(String, String)>,
        saved_at: Option<u64>,
    },

    /// The active session metadata changed
    SessionStatus(SessionInfo),

    /// Active backend name for the sidebar
    BackendName(String),

    /// A generation phase started
    GenerationStarted {
        label: String,
        show_placeholder: bool,
    },

    /// A transient or durable progress-trace update
    Trace(ProgressTrace),

    /// A generated token, append to current response
    Token(String),

    /// Tool calls being executed, shown in sidebar
    ToolCall(String),

    /// A local context/result message should be added to the chat
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

    /// A system/status message should be added to the chat.
    SystemMessage(String),

    /// The runtime memory snapshot changed.
    MemoryState(MemorySnapshot),

    /// Generation finished.
    Done,

    /// An error occurred.
    Error(String),
}
