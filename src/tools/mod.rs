// src/tools/mod.rs
//
// The tool system for params-cli.
//
// Tools let the model interact with your filesystem — read files, list
// directories, search code. The model signals it wants to use a tool by
// including a special tag in its response:
//
//   [read_file: src/main.rs]
//   [list_dir: src/]
//   [search: fn main]
//
// After generation completes, params scans the response for these tags,
// runs the tools, and injects the results back as a new user message so
// the model can reason about the actual file contents.

mod fs;
mod search;

pub use fs::{ListDir, ReadFile};
pub use search::SearchCode;

use crate::error::Result;

/// The contract every tool must fulfill.
/// Tools are simple — they take a string argument and return a string result.
pub trait Tool: Send + Sync {
    /// The name used in tool tags, e.g. "read_file" for [read_file: path]
    fn name(&self) -> &str;

    /// A short description shown in the system prompt so the model knows
    /// the tool exists and how to use it.
    fn description(&self) -> &str;

    /// Run the tool with the given argument string.
    /// Returns the result as a string to be injected back into the conversation.
    fn run(&self, arg: &str) -> Result<String>;
}

/// The global tool registry — holds all available tools.
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a registry with the default built-in tools.
    pub fn default() -> Self {
        Self {
            tools: vec![
                Box::new(ReadFile),
                Box::new(ListDir),
                Box::new(SearchCode),
            ],
        }
    }

    /// Returns a string describing all available tools, injected into the
    /// system prompt so the model knows what it can use.
    pub fn tool_descriptions(&self) -> String {
        let mut desc = String::from(
            "You have access to the following tools.\n\
             When you need one, respond with only the tool call tags, one per line, \
             using the exact syntax `[tool_name: argument]`.\n\
             After tool results are returned, continue in a follow-up response.\n\n"
        );
        for tool in &self.tools {
            desc.push_str(&format!(
                "  {}: {}\n  Usage: [{}: <argument>]\n\n",
                tool.name(),
                tool.description(),
                tool.name(),
            ));
        }
        desc.push_str(
            "Only use tools when you actually need to read or search files.\n\
             Do not use tools for questions that don't require file access."
        );
        desc
    }

    /// Scan a response string for tool calls and execute them all.
    /// Returns a list of (tool_name, argument, result) tuples.
    pub fn execute_tool_calls(&self, response: &str) -> Vec<ToolResult> {
        let mut results = Vec::new();

        for tool in &self.tools {
            let tag = format!("[{}:", tool.name());
            let mut search_from = 0;

            while let Some(start) = response[search_from..].find(&tag) {
                let abs_start = search_from + start;
                let after_tag = abs_start + tag.len();

                // Find the closing ]
                if let Some(end_offset) = response[after_tag..].find(']') {
                    let arg = response[after_tag..after_tag + end_offset].trim().to_string();

                    let result = match tool.run(&arg) {
                        Ok(output) => output,
                        Err(e) => format!("Error: {e}"),
                    };

                    results.push(ToolResult {
                        tool_name: tool.name().to_string(),
                        argument: arg,
                        output: result,
                    });

                    search_from = after_tag + end_offset + 1;
                } else {
                    break;
                }
            }
        }

        results
    }

    /// Format tool results into a message to inject back into the conversation.
    /// Returns None if there were no tool calls.
    pub fn format_results(results: &[ToolResult]) -> Option<String> {
        if results.is_empty() {
            return None;
        }

        let mut msg = String::from("Tool results:\n\n");
        for r in results {
            msg.push_str(&format!(
                "--- {}({}) ---\n{}\n\n",
                r.tool_name, r.argument, r.output
            ));
        }

        Some(msg)
    }
}

/// The result of running a single tool call.
pub struct ToolResult {
    pub tool_name: String,
    pub argument: String,
    pub output: String,
}
