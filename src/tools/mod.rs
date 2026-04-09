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
mod edit;
mod fs;
mod git;
mod lsp;
mod search;
mod web;
mod write;

pub use bash::BashTool;
pub use edit::{build_pending_edit_request, EditFileTool};
pub use fs::{ListDir, ReadFile};
pub use git::GitTool;
pub use lsp::{rust_lsp_health_report, LspDefinitionTool, LspDiagnosticsTool, LspHoverTool};
pub use search::SearchCode;
pub use web::FetchUrlTool;
pub use write::{build_pending_write_request, WriteFileTool};

use crate::error::{ParamsError, Result};
use crate::events::PendingActionKind;
use crate::safety::InspectionReport;
#[cfg(test)]
use crate::safety::{InspectionDecision, RiskLevel};

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

    /// Run the tool with access to the text that follows the tool call tag.
    fn run_with_context(&self, arg: &str, _following_text: &str) -> Result<ToolRunResult> {
        self.run(arg)
    }

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

fn is_read_only_tool_name(name: &str) -> bool {
    matches!(
        name,
        "read_file"
            | "list_dir"
            | "search"
            | "git"
            | "lsp_diagnostics"
            | "lsp_hover"
            | "lsp_definition"
            | "fetch_url"
    )
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
                Box::new(LspDiagnosticsTool),
                Box::new(LspHoverTool),
                Box::new(LspDefinitionTool),
                Box::new(FetchUrlTool),
                Box::new(BashTool),
                Box::new(EditFileTool),
                Box::new(WriteFileTool),
            ],
        }
    }

    /// Returns a string describing all available tools, injected into the
    /// system prompt so the model knows what it can use.
    pub fn tool_descriptions(&self) -> String {
        let mut desc = String::from(
            "You have access to the following tools.\n\
             When you need one, respond with only tool call content.\n\
             For normal tools, use one tag per line with the exact syntax `[tool_name: argument]`.\n\
             For `write_file`, put the tag on its own line and immediately follow it with a \
             ```params-file fenced block containing the full new file contents.\n\
             For `edit_file`, put the tag on its own line and immediately follow it with a \
             ```params-edit fenced block containing SEARCH/REPLACE sections.\n\
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
             Do not use tools for questions that don't require file access.",
        );
        desc
    }

    pub fn compact_tool_descriptions(&self) -> String {
        let mut desc = String::from(
            "You have tool access.\n\
             Use only the tools you need.\n\
             Use exact tags like `[tool_name: argument]`.\n\
             For `write_file`, place `[write_file: path]` on its own line and follow it with a \
             ```params-file block containing the full file contents.\n\
             For `edit_file`, place `[edit_file: path]` on its own line and follow it with a \
             ```params-edit block using SEARCH/REPLACE sections.\n\
             After tool results are returned, continue in a follow-up response.\n\n",
        );
        for tool in &self.tools {
            desc.push_str(&format!("  {}: {}\n", tool.name(), tool.description()));
        }
        desc.push_str("\nKeep tool use minimal.");
        desc
    }

    pub fn read_only_tool_descriptions(&self) -> String {
        let mut desc = String::from(
            "You have access to read-only repo inspection tools.\n\
             When you need one, respond with only tool call content.\n\
             Use one tag per line with the exact syntax `[tool_name: argument]`.\n\
             Search before guessing, read candidate files, expand only as needed, and answer with file/line evidence once you have enough information.\n\
             Do not call mutating tools in this mode.\n\n",
        );
        for tool in &self.tools {
            if is_read_only_tool_name(tool.name()) {
                desc.push_str(&format!(
                    "  {}: {}\n  Usage: [{}: <argument>]\n\n",
                    tool.name(),
                    tool.description(),
                    tool.name(),
                ));
            }
        }
        desc.push_str(
            "If the current evidence is still insufficient after inspecting the repo, say so briefly instead of inventing details.",
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
                    let arg = response[after_tag..after_tag + end_offset]
                        .trim()
                        .to_string();

                    let tool_run = match tool
                        .run_with_context(&arg, &response[after_tag + end_offset + 1..])
                    {
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

    pub fn execute_read_only_tool_calls(&self, response: &str) -> ReadOnlyToolExecution {
        let mut results = Vec::new();
        let mut disallowed_calls = Vec::new();

        for tool in &self.tools {
            let tag = format!("[{}:", tool.name());
            let mut search_from = 0;
            let is_read_only = is_read_only_tool_name(tool.name());

            while let Some(start) = response[search_from..].find(&tag) {
                let abs_start = search_from + start;
                let after_tag = abs_start + tag.len();

                if let Some(end_offset) = response[after_tag..].find(']') {
                    let arg = response[after_tag..after_tag + end_offset]
                        .trim()
                        .to_string();

                    if is_read_only {
                        match tool.run_with_context(&arg, &response[after_tag + end_offset + 1..]) {
                            Ok(ToolRunResult::Immediate(output)) => {
                                results.push(ToolResult {
                                    tool_name: tool.name().to_string(),
                                    argument: arg,
                                    output,
                                });
                            }
                            Ok(ToolRunResult::RequiresApproval(_)) => {
                                disallowed_calls.push(tool.name().to_string());
                            }
                            Err(e) => {
                                results.push(ToolResult {
                                    tool_name: tool.name().to_string(),
                                    argument: arg,
                                    output: format!("Error: {e}"),
                                });
                            }
                        }
                    } else {
                        disallowed_calls.push(tool.name().to_string());
                    }

                    search_from = after_tag + end_offset + 1;
                } else {
                    break;
                }
            }
        }

        disallowed_calls.sort();
        disallowed_calls.dedup();
        ReadOnlyToolExecution {
            results,
            disallowed_calls,
        }
    }

    /// Format tool results into a message to inject back into the conversation.
    /// Returns None if there were no tool calls.
    pub fn format_results_with_limit(
        results: &[ToolResult],
        max_chars_per_result: Option<usize>,
    ) -> Option<String> {
        if results.is_empty() {
            return None;
        }

        let mut msg = String::from("Tool results:\n\n");
        for r in results {
            let output = truncate_tool_output(&r.output, max_chars_per_result);
            msg.push_str(&format!(
                "--- {}({}) ---\n{}\n\n",
                r.tool_name, r.argument, output
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
            argument: pending.display_argument.clone(),
            output: match output {
                Ok(output) => output,
                Err(e) => format!("Error: {e}"),
            },
        }
    }
}

/// The result of running a single tool call.
#[derive(Clone, Debug, PartialEq, Eq)]
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

pub struct ReadOnlyToolExecution {
    pub results: Vec<ToolResult>,
    pub disallowed_calls: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PendingToolAction {
    pub kind: PendingActionKind,
    pub tool_name: String,
    pub argument: String,
    pub display_argument: String,
    pub title: String,
    pub preview: String,
    pub inspection: InspectionReport,
}

fn truncate_tool_output(output: &str, max_chars: Option<usize>) -> String {
    let Some(limit) = max_chars else {
        return output.to_string();
    };

    let total = output.chars().count();
    if total <= limit {
        return output.to_string();
    }

    let keep = limit.saturating_sub(80);
    let truncated: String = output.chars().take(keep).collect();
    format!(
        "{truncated}\n\n[truncated {} chars for eco mode]",
        total.saturating_sub(keep)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ContextTool;

    impl Tool for ContextTool {
        fn name(&self) -> &str {
            "context_tool"
        }

        fn description(&self) -> &str {
            "test tool"
        }

        fn run(&self, _arg: &str) -> Result<ToolRunResult> {
            Ok(ToolRunResult::Immediate("unexpected".to_string()))
        }

        fn run_with_context(&self, arg: &str, following_text: &str) -> Result<ToolRunResult> {
            let next_line = following_text
                .trim_start_matches(['\n', '\r'])
                .lines()
                .next()
                .unwrap_or("");
            Ok(ToolRunResult::Immediate(format!(
                "arg={arg};next={next_line}"
            )))
        }
    }

    struct ImmediateTool;

    impl Tool for ImmediateTool {
        fn name(&self) -> &str {
            "immediate"
        }

        fn description(&self) -> &str {
            "test tool"
        }

        fn run(&self, arg: &str) -> Result<ToolRunResult> {
            Ok(ToolRunResult::Immediate(format!("immediate:{arg}")))
        }
    }

    struct PendingTool;

    impl Tool for PendingTool {
        fn name(&self) -> &str {
            "pending"
        }

        fn description(&self) -> &str {
            "test tool"
        }

        fn run(&self, arg: &str) -> Result<ToolRunResult> {
            Ok(ToolRunResult::RequiresApproval(PendingToolAction {
                kind: PendingActionKind::FileWrite,
                tool_name: self.name().to_string(),
                argument: format!("raw:{arg}"),
                display_argument: format!("display:{arg}"),
                title: "Approve test action".to_string(),
                preview: "preview".to_string(),
                inspection: InspectionReport {
                    operation: "pending".to_string(),
                    decision: InspectionDecision::NeedsApproval,
                    risk: RiskLevel::Low,
                    summary: "test".to_string(),
                    reasons: Vec::new(),
                    targets: vec!["display".to_string()],
                    segments: Vec::new(),
                    network_targets: Vec::new(),
                },
            }))
        }
    }

    #[test]
    fn execute_tool_calls_passes_following_text_to_tool() {
        let registry = ToolRegistry {
            tools: vec![Box::new(ContextTool)],
        };

        let execution =
            registry.execute_tool_calls("[context_tool: src/main.rs]\n```params-file\nhello\n```");

        assert!(execution.pending.is_none());
        assert_eq!(execution.results.len(), 1);
        assert_eq!(
            execution.results[0].output,
            "arg=src/main.rs;next=```params-file"
        );
    }

    #[test]
    fn format_results_can_truncate_for_eco_mode() {
        let results = vec![ToolResult {
            tool_name: "search".to_string(),
            argument: "foo".to_string(),
            output: "a".repeat(200),
        }];

        let formatted = ToolRegistry::format_results_with_limit(&results, Some(80)).unwrap();

        assert!(formatted.contains("[truncated"));
        assert!(formatted.contains("--- search(foo) ---"));
        assert!(formatted.len() < 220);
    }

    #[test]
    fn execute_tool_calls_keeps_immediate_results_before_pending_action() {
        let registry = ToolRegistry {
            tools: vec![Box::new(ImmediateTool), Box::new(PendingTool)],
        };

        let execution =
            registry.execute_tool_calls("[immediate: first]\n[pending: second]\nignored text");

        assert_eq!(execution.results.len(), 1);
        assert_eq!(execution.results[0].output, "immediate:first");

        let pending = execution.pending.expect("expected pending action");
        assert_eq!(pending.display_argument, "display:second");
        assert_eq!(pending.argument, "raw:second");
    }

    #[test]
    fn read_only_tool_descriptions_exclude_mutating_tools() {
        let registry = ToolRegistry::default();
        let descriptions = registry.read_only_tool_descriptions();

        assert!(descriptions.contains("read_file"));
        assert!(descriptions.contains("search"));
        assert!(!descriptions.contains("write_file"));
        assert!(!descriptions.contains("edit_file"));
        assert!(!descriptions.contains("bash"));
    }

    #[test]
    fn execute_read_only_tool_calls_flags_disallowed_tools() {
        let registry = ToolRegistry {
            tools: vec![Box::new(ImmediateTool), Box::new(PendingTool)],
        };

        let execution =
            registry.execute_read_only_tool_calls("[immediate: first]\n[pending: second]");

        assert_eq!(execution.results.len(), 0);
        assert_eq!(
            execution.disallowed_calls,
            vec!["immediate".to_string(), "pending".to_string()]
        );
    }
}
