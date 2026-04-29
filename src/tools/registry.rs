use std::collections::HashMap;
use std::path::PathBuf;

use crate::runtime::ResolvedToolInput;

use super::edit_file::EditFileTool;
use super::git_diff::GitDiffTool;
use super::git_log::GitLogTool;
use super::git_status::GitStatusTool;
use super::pending::PendingAction;
use super::search_code::SearchCodeTool;
use super::types::{ExecutionKind, ToolError, ToolOutput, ToolRunResult, ToolSpec};
use super::write_file::WriteFileTool;
use super::Tool;

/// Owns all registered tools. Responsibilities: registration, spec enumeration, dispatch.
/// This type does NOT parse model output, format results, or truncate content —
/// those concerns belong in the tool loop and the individual tools respectively.
pub struct ToolRegistry {
    tools: HashMap<&'static str, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Registers a tool. The tool's name (from its spec) is used as the lookup key.
    /// Overwrites any existing registration with the same name.
    pub fn register(&mut self, tool: impl Tool + 'static) {
        let name = tool.spec().name;
        self.tools.insert(name, Box::new(tool));
    }

    /// Registers the tools that need the runtime-owned project root.
    pub fn with_project_root(mut self, root: PathBuf) -> Self {
        self.register(SearchCodeTool::new(root.clone()));
        self.register(GitStatusTool::new(root.clone()));
        self.register(GitDiffTool::new(root.clone()));
        self.register(GitLogTool::new(root.clone()));
        self.register(EditFileTool::new(root.clone()));
        self.register(WriteFileTool::new(root));
        self
    }

    /// Dispatches a typed input to the correct tool and returns the run result.
    /// Returns ToolError::NotFound if no tool is registered for the input's tool_name.
    pub fn dispatch(&self, input: ResolvedToolInput) -> Result<ToolRunResult, ToolError> {
        let name = input.tool_name();
        let tool = self.tools.get(name).ok_or_else(|| ToolError::NotFound {
            name: name.to_string(),
        })?;
        tool.run(&input)
    }

    /// Applies a previously approved mutation by delegating to the correct tool's
    /// execute_approved() method. Returns ToolError::NotFound for unknown tools.
    pub fn execute_approved(&self, pending: &PendingAction) -> Result<ToolOutput, ToolError> {
        let tool =
            self.tools
                .get(pending.tool_name.as_str())
                .ok_or_else(|| ToolError::NotFound {
                    name: pending.tool_name.clone(),
                })?;
        tool.execute_approved(&pending.payload)
    }

    /// Returns the spec for every registered tool. Used to build the system prompt.
    pub fn specs(&self) -> Vec<ToolSpec> {
        let mut specs: Vec<ToolSpec> = self.tools.values().map(|t| t.spec()).collect();
        specs.sort_by_key(|s| s.name);
        specs
    }

    /// Returns the spec for a specific tool by name, or `None` if not registered.
    pub fn spec_for(&self, name: &str) -> Option<ToolSpec> {
        self.tools.get(name).map(|t| t.spec())
    }

    /// Returns true if the named tool requires approval before executing.
    /// Returns false for unknown tools (safe default — caller sees no Approval outcome anyway).
    pub fn is_approval_required(&self, name: &str) -> bool {
        self.spec_for(name)
            .map(|s| s.execution_kind == ExecutionKind::RequiresApproval)
            .unwrap_or(false)
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::runtime::{ProjectPath, ProjectRoot, ProjectScope};
    use crate::tools::list_dir::ListDirTool;
    use crate::tools::read_file::ReadFileTool;
    use crate::tools::types::{ToolOutput, ToolRunResult};

    fn resolved_root_path() -> ProjectPath {
        let root = ProjectRoot::new(PathBuf::from(".")).unwrap();
        ProjectPath::from_trusted(root.path().to_path_buf(), ".".to_string())
    }

    fn resolved_root_scope() -> ProjectScope {
        ProjectScope::from_trusted_path(resolved_root_path())
    }

    #[test]
    fn specs_are_sorted_by_name() {
        let mut registry = ToolRegistry::new();
        registry.register(ReadFileTool::new());
        registry.register(ListDirTool::new());

        let specs = registry.specs();
        let names: Vec<_> = specs.iter().map(|s| s.name).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[test]
    fn dispatch_returns_not_found_for_unregistered_tool() {
        let registry = ToolRegistry::new();
        let err = registry
            .dispatch(ResolvedToolInput::ReadFile {
                path: ProjectPath::from_trusted(PathBuf::from("/tmp/any"), "any".into()),
            })
            .unwrap_err();
        assert!(matches!(err, ToolError::NotFound { .. }));
    }

    #[test]
    fn dispatch_routes_to_correct_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(ListDirTool::new());

        let result = registry.dispatch(ResolvedToolInput::ListDir {
            path: resolved_root_scope(),
        });
        assert!(result.is_ok());
        let ToolRunResult::Immediate(ToolOutput::DirectoryListing(_)) = result.unwrap() else {
            panic!("expected Immediate(DirectoryListing)");
        };
    }

    #[test]
    fn spec_for_returns_spec_for_registered_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(ReadFileTool::new());

        let spec = registry.spec_for("read_file");
        assert!(spec.is_some());
        assert_eq!(spec.unwrap().name, "read_file");
    }

    #[test]
    fn spec_for_returns_none_for_unknown_tool() {
        let registry = ToolRegistry::new();
        assert!(registry.spec_for("nonexistent").is_none());
    }

    #[test]
    fn is_approval_required_true_for_mutating_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(EditFileTool::new(PathBuf::from(".")));
        registry.register(WriteFileTool::new(PathBuf::from(".")));

        assert!(registry.is_approval_required("edit_file"));
        assert!(registry.is_approval_required("write_file"));
    }

    #[test]
    fn is_approval_required_false_for_read_only_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(ReadFileTool::new());
        registry.register(ListDirTool::new());

        assert!(!registry.is_approval_required("read_file"));
        assert!(!registry.is_approval_required("list_dir"));
    }

    #[test]
    fn is_approval_required_false_for_unknown_tool() {
        let registry = ToolRegistry::new();
        assert!(!registry.is_approval_required("unknown"));
    }
}
