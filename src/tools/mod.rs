mod list_dir;
mod read_file;
mod registry;
mod search_code;
pub mod types;

pub use list_dir::ListDirTool;
pub use read_file::ReadFileTool;
pub use registry::ToolRegistry;
pub use search_code::SearchCodeTool;
pub use types::{EntryKind, ToolError, ToolInput, ToolOutput, ToolSpec};

/// The core tool trait. Each implementation handles exactly one ToolInput variant.
/// Returns structured ToolOutput — never a formatted string intended for logic consumers.
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    fn run(&self, input: &ToolInput) -> Result<ToolOutput, ToolError>;
}

/// Builds a ToolRegistry pre-loaded with all available read-only tools.
/// Mutating tools (bash, edit_file, write_file) are Phase 4 additions.
pub fn default_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(ReadFileTool);
    registry.register(ListDirTool);
    registry.register(SearchCodeTool);
    registry
}
