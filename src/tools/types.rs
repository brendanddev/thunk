use thiserror::Error;

// ── Input ─────────────────────────────────────────────────────────────────────

/// Typed input for each supported tool. The runtime's tool-call parser (Phase 2)
/// converts raw model output into one of these variants before dispatch.
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
}

impl ToolInput {
    /// Returns the canonical tool name for this input variant.
    /// Used by ToolRegistry::dispatch to look up the right implementation.
    pub fn tool_name(&self) -> &'static str {
        match self {
            ToolInput::ReadFile { .. } => "read_file",
            ToolInput::ListDir { .. } => "list_dir",
            ToolInput::SearchCode { .. } => "search_code",
        }
    }
}

// ── Output ────────────────────────────────────────────────────────────────────

/// Structured output for each tool. Callers consume the typed data; rendering
/// into prompt text happens separately in the tool loop (Phase 2).
#[derive(Debug, Clone)]
pub enum ToolOutput {
    FileContents(FileContentsOutput),
    DirectoryListing(DirectoryListingOutput),
    SearchResults(SearchResultsOutput),
}

#[derive(Debug, Clone)]
pub struct FileContentsOutput {
    pub path: String,
    pub contents: String,
    pub line_count: usize,
    /// True when the file was larger than the read limit and was cut off.
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

// ── Call / Result ─────────────────────────────────────────────────────────────

/// A dispatched tool invocation.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub input: ToolInput,
}

/// The complete outcome of a tool invocation: the original call paired with its output.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub call: ToolCall,
    pub output: ToolOutput,
}

// ── Spec ──────────────────────────────────────────────────────────────────────

/// Static metadata describing a tool. Used by Phase 2 to build tool descriptions
/// for the system prompt.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub input_hint: &'static str,
}

// ── Error ─────────────────────────────────────────────────────────────────────

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
