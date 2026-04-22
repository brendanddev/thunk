pub mod context;
mod edit_file;
mod git_diff;
mod git_status;
mod list_dir;
mod pending;
mod read_file;
mod registry;
mod search_code;
pub mod types;
mod write_file;

use std::path::PathBuf;

use edit_file::EditFileTool;
use git_diff::GitDiffTool;
use git_status::GitStatusTool;
use list_dir::ListDirTool;
use read_file::ReadFileTool;
use search_code::SearchCodeTool;
use write_file::WriteFileTool;

pub use context::ToolContext;
pub use pending::{PendingAction, RiskLevel};
pub use registry::ToolRegistry;
pub use types::{
    EntryKind, ExecutionKind, ToolError, ToolInput, ToolOutput, ToolRunResult, ToolSpec,
};

/// The core tool trait. Each implementation handles exactly one ToolInput variant.
///
/// Read-only tools implement only run(). Mutating tools implement both run() and
/// execute_approved(). The two-phase design ensures mutations never happen without
/// explicit approval.
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;

    /// Phase 1 of execution: validate input and return either an immediate result
    /// or a PendingAction describing the proposed mutation.
    fn run(&self, input: &ToolInput) -> Result<ToolRunResult, ToolError>;

    /// Phase 2 of execution: apply a previously approved mutation and return the
    /// result. Only mutating tools implement this — read-only tools never produce
    /// Approval outcomes and this method is never called on them.
    fn execute_approved(&self, _payload: &str) -> Result<ToolOutput, ToolError> {
        Err(ToolError::InvalidInput(format!(
            "tool '{}' does not support approved execution",
            self.spec().name
        )))
    }
}

/// Builds a ToolRegistry pre-loaded with all tools.
/// Each tool receives a ToolContext so it can resolve relative paths against
/// the project root rather than the process working directory.
pub fn default_registry(root: PathBuf) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(ReadFileTool::new(ToolContext::new(root.clone())));
    registry.register(ListDirTool::new(ToolContext::new(root.clone())));
    registry.register(SearchCodeTool::new(ToolContext::new(root.clone())));
    registry.register(GitStatusTool::new(ToolContext::new(root.clone())));
    registry.register(GitDiffTool::new(ToolContext::new(root.clone())));
    registry.register(EditFileTool::new(ToolContext::new(root.clone())));
    registry.register(WriteFileTool::new(ToolContext::new(root)));
    registry
}
