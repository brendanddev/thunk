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

mod bash;
mod fs;
mod git;
mod search;

pub use bash::BashTool;
pub use fs::{ListDir, ReadFile};
pub use git::GitTool;
pub use search::SearchCode;

use crate::error::{ParamsError, Result};

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
    fn run(&self, arg: &str) -> Result<ToolRunResult>;

    /// Run the tool after the user has approved it.
    fn run_approved(&self, _arg: &str) -> Result<String> {
        Err(ParamsError::Config(format!(
            "Tool {} does not support approval-based execution",
            self.name()
        )))
    }
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
                Box::new(GitTool),
                Box::new(BashTool),
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
            "Only use tools when you actually need repo or file context.\n\
             Do not use tools for questions that don't require file access."
        );
        desc
    }

    /// Scan a response string for tool calls and execute them all.
    /// Stops early if a tool requires approval.
    pub fn execute_tool_calls(&self, response: &str) -> ToolExecution {
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

                    let tool_run = match tool.run(&arg) {
                        Ok(output) => output,
                        Err(e) => ToolRunResult::Immediate(format!("Error: {e}")),
                    };

                    match tool_run {
                        ToolRunResult::Immediate(output) => {
                            results.push(ToolResult {
                                tool_name: tool.name().to_string(),
                                argument: arg,
                                output,
                            });
                        }
                        ToolRunResult::RequiresApproval(pending) => {
                            return ToolExecution {
                                results,
                                pending: Some(pending),
                            };
                        }
                    }

                    search_from = after_tag + end_offset + 1;
                } else {
                    break;
                }
            }
        }

        ToolExecution {
            results,
            pending: None,
        }
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

    pub fn execute_pending_action(&self, pending: &PendingToolAction) -> ToolResult {
        let output = self
            .tools
            .iter()
            .find(|tool| tool.name() == pending.tool_name)
            .map(|tool| tool.run_approved(&pending.argument))
            .unwrap_or_else(|| {
                Err(ParamsError::Config(format!(
                    "Unknown pending tool: {}",
                    pending.tool_name
                )))
            });

        ToolResult {
            tool_name: pending.tool_name.clone(),
            argument: pending.argument.clone(),
            output: match output {
                Ok(output) => output,
                Err(e) => format!("Error: {e}"),
            },
        }
    }
}

/// The result of running a single tool call.
pub struct ToolResult {
    pub tool_name: String,
    pub argument: String,
    pub output: String,
}

pub enum ToolRunResult {
    Immediate(String),
    RequiresApproval(PendingToolAction),
}

pub struct ToolExecution {
    pub results: Vec<ToolResult>,
    pub pending: Option<PendingToolAction>,
}

#[derive(Debug, Clone)]
pub struct PendingToolAction {
    pub tool_name: String,
    pub argument: String,
    pub title: String,
    pub preview: String,
}
