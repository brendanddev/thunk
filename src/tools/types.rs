use thiserror::Error;

use super::pending::{PendingAction, RiskLevel};

// Input

/// Typed input for each supported tool. The runtime's tool-call parser converts
/// raw model output into one of these variants before dispatch.
/// Nothing outside the tools module should parse these back out of strings.
#[derive(Debug, Clone)]
pub enum ToolInput {
    ReadFile {
        /// Path relative to the project root, or absolute.
        path: String,
    },
    ListDir {
        /// Directory to list. Defaults to project root if empty.
        path: String,
    },
    SearchCode {
        /// Literal string or pattern to search for.
        query: String,
        /// Optional sub-path to restrict the search. Searches entire tree if None.
        path: Option<String>,
    },
    EditFile {
        /// Path relative to the project root, or absolute.
        path: String,
        /// Exact text to find in the file.
        search: String,
        /// Replacement text.
        replace: String,
    },
    WriteFile {
        /// Path relative to the project root, or absolute.
        path: String,
        /// Full content to write.
        content: String,
    },
}

impl ToolInput {
    /// Returns the canonical tool name for this input variant.
    /// Used by ToolRegistry::dispatch to look up the right implementation.
    pub fn tool_name(&self) -> &'static str {
        match self {
            ToolInput::ReadFile { .. } => "read_file",
            ToolInput::ListDir { .. } => "list_dir",
            ToolInput::SearchCode { .. } => "search_code",
            ToolInput::EditFile { .. } => "edit_file",
            ToolInput::WriteFile { .. } => "write_file",
        }
    }
}

// Output

/// Structured output from a completed tool execution. Callers consume the typed
/// data; rendering into prompt text happens in the tool loop.
#[derive(Debug, Clone)]
pub enum ToolOutput {
    FileContents(FileContentsOutput),
    DirectoryListing(DirectoryListingOutput),
    SearchResults(SearchResultsOutput),
    EditFile(EditFileOutput),
    WriteFile(WriteFileOutput),
}

#[derive(Debug, Clone)]
pub struct FileContentsOutput {
    pub path: String,
    /// The (possibly truncated) file content injected into the conversation.
    pub contents: String,
    /// Total lines in the file, before any truncation.
    pub total_lines: usize,
    /// True when the file exceeded the line cap and contents was cut off.
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub struct DirectoryListingOutput {
    pub path: String,
    pub entries: Vec<DirEntry>,
}

#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub kind: EntryKind,
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    File,
    Dir,
    Symlink,
}

#[derive(Debug, Clone)]
pub struct SearchResultsOutput {
    pub query: String,
    pub matches: Vec<SearchMatch>,
    /// True when the result set was cut at the match limit.
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub struct SearchMatch {
    pub file: String,
    pub line_number: usize,
    pub line: String,
}

#[derive(Debug, Clone)]
pub struct EditFileOutput {
    pub path: String,
    /// Number of lines in the search text that was replaced.
    pub lines_replaced: usize,
}

#[derive(Debug, Clone)]
pub struct WriteFileOutput {
    pub path: String,
    pub bytes_written: usize,
    /// True when the file was newly created; false when an existing file was overwritten.
    pub created: bool,
}

// Run result

/// The outcome of dispatching a tool. Read-only tools always return Immediate.
/// Mutating tools return Approval, pausing the turn until the user responds.
#[derive(Debug, Clone)]
pub enum ToolRunResult {
    Immediate(ToolOutput),
    Approval(PendingAction),
}

// Spec

/// Whether a tool runs immediately or requires explicit user approval before mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionKind {
    /// Tool completes immediately, no approval step.
    Immediate,
    /// Tool proposes a mutation that must be approved before executing.
    RequiresApproval,
}

/// Static metadata describing a tool. Used to build the system prompt and to
/// consult tool behavior (approval, risk) without running the tool.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub input_hint: &'static str,
    /// Whether this tool requires an approval round before executing.
    pub execution_kind: ExecutionKind,
    /// Baseline risk for approval-required tools. `None` for immediate tools.
    pub default_risk: Option<RiskLevel>,
}

// Error

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("tool not found: {name}")]
    NotFound { name: String },

    #[error("IO error in tool: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid tool input: {0}")]
    InvalidInput(String),
}

impl From<ToolError> for crate::app::AppError {
    fn from(e: ToolError) -> Self {
        crate::app::AppError::Tool(e.to_string())
    }
}
