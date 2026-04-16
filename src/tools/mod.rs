pub mod context;
mod list_dir;
mod pending;
mod read_file;
mod registry;
mod search_code;
pub mod types;

use std::path::PathBuf;

use list_dir::ListDirTool;
use read_file::ReadFileTool;
use search_code::SearchCodeTool;

pub use context::ToolContext;
pub use pending::{PendingAction, RiskLevel};
pub use registry::ToolRegistry;
pub use types::{
    EntryKind, ToolError, ToolInput, ToolOutput, ToolRunResult, ToolSpec,
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
        unimplemented!(
            "tool '{}' does not support approved execution — implement execute_approved",
            self.spec().name
        )
    }
}

/// Builds a ToolRegistry pre-loaded with all available read-only tools.
/// Each tool receives a ToolContext so it can resolve relative paths against
/// the project root rather than the process working directory.
/// Mutating tools (edit_file, write_file) are registered here once implemented.
pub fn default_registry(root: PathBuf) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(ReadFileTool::new(ToolContext::new(root.clone())));
    registry.register(ListDirTool::new(ToolContext::new(root.clone())));
    registry.register(SearchCodeTool::new(ToolContext::new(root)));
    registry
}
