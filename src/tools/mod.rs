pub mod context;
mod list_dir;
mod read_file;
mod registry;
mod search_code;
pub mod types;

pub use context::ToolContext;
pub use list_dir::ListDirTool;
pub use read_file::ReadFileTool;
pub use registry::ToolRegistry;
pub use search_code::SearchCodeTool;
pub use types::{EntryKind, FileContentsOutput, ToolError, ToolInput, ToolOutput, ToolSpec};

/// The core tool trait. Each implementation handles exactly one ToolInput variant.
/// Returns structured ToolOutput — never a formatted string intended for logic consumers.
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    fn run(&self, input: &ToolInput) -> Result<ToolOutput, ToolError>;
}

/// Builds a ToolRegistry pre-loaded with all available read-only tools.
/// Each tool receives a ToolContext so it can resolve relative paths against
/// the project root rather than the process working directory.
/// Mutating tools (bash, edit_file, write_file) are Phase 5 additions.
pub fn default_registry(root: std::path::PathBuf) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(ReadFileTool::new(ToolContext::new(root.clone())));
    registry.register(ListDirTool::new(ToolContext::new(root.clone())));
    registry.register(SearchCodeTool::new(ToolContext::new(root)));
    registry
}
