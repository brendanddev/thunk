use std::collections::HashMap;

use super::types::{ToolCall, ToolError, ToolInput, ToolResult, ToolSpec};
use super::Tool;

/// Owns all registered tools. Responsibilities: registration, spec enumeration, dispatch.
/// This type does NOT parse model output, format results, or truncate content —
/// those concerns belong in the tool loop (Phase 2) and the individual tools respectively.
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

    /// Dispatches a typed input to the correct tool and returns the result.
    /// Returns ToolError::NotFound if no tool is registered for the input's tool_name.
    pub fn dispatch(&self, input: ToolInput) -> Result<ToolResult, ToolError> {
        let name = input.tool_name();
        let tool = self.tools.get(name).ok_or_else(|| ToolError::NotFound {
            name: name.to_string(),
        })?;
        let output = tool.run(&input)?;
        Ok(ToolResult {
            call: ToolCall { input },
            output,
        })
    }

    /// Returns the spec for every registered tool. Used by Phase 2 to build the system prompt.
    pub fn specs(&self) -> Vec<ToolSpec> {
        let mut specs: Vec<ToolSpec> = self.tools.values().map(|t| t.spec()).collect();
        // Stable ordering so the system prompt is deterministic.
        specs.sort_by_key(|s| s.name);
        specs
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
    use crate::tools::context::ToolContext;
    use crate::tools::list_dir::ListDirTool;
    use crate::tools::read_file::ReadFileTool;
    use crate::tools::types::{ToolInput, ToolOutput};

    fn ctx() -> ToolContext {
        ToolContext::new(PathBuf::from("."))
    }

    #[test]
    fn specs_are_sorted_by_name() {
        let mut registry = ToolRegistry::new();
        registry.register(ReadFileTool::new(ctx()));
        registry.register(ListDirTool::new(ctx()));

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
            .dispatch(ToolInput::ReadFile { path: "any".into() })
            .unwrap_err();
        assert!(matches!(err, ToolError::NotFound { .. }));
    }

    #[test]
    fn dispatch_routes_to_correct_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(ListDirTool::new(ctx()));

        // list_dir on a valid path should succeed (use cwd, which always exists)
        let result = registry.dispatch(ToolInput::ListDir { path: ".".into() });
        assert!(result.is_ok());
        let ToolOutput::DirectoryListing(_) = result.unwrap().output else {
            panic!("expected DirectoryListing output");
        };
    }
}
