// src/tools/bash.rs
//
// Approval-driven shell execution tool.

use std::process::Command;
use tracing::info;

use crate::error::{ParamsError, Result};
use crate::events::PendingActionKind;
use super::{PendingToolAction, Tool, ToolRunResult};

pub struct BashTool;

impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Propose a shell command for user approval. Usage: [bash: cargo check]"
    }

    fn run(&self, arg: &str) -> Result<ToolRunResult> {
        info!(tool = "bash", phase = "proposal", "tool called");
        let command = arg.trim();
        if command.is_empty() {
            return Err(ParamsError::Config("Shell command cannot be empty".into()));
        }
        if command.contains('\n') || command.contains('\r') {
            return Err(ParamsError::Config(
                "Multiline shell commands are not supported".into()
            ));
        }

        Ok(ToolRunResult::RequiresApproval(PendingToolAction {
            kind: PendingActionKind::ShellCommand,
            tool_name: self.name().to_string(),
            argument: command.to_string(),
            display_argument: command.to_string(),
            title: "Approve shell command".to_string(),
            preview: command.to_string(),
        }))
    }

    fn run_approved(&self, arg: &str) -> Result<String> {
        info!(tool = "bash", phase = "execute", "approved tool executing");
        let output = Command::new("/bin/zsh")
            .arg("-lc")
            .arg(arg)
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let status = output.status.code().unwrap_or(-1);
        info!(tool = "bash", exit_code = status, "approved tool finished");

        let mut result = format!("Command: {arg}\nExit code: {status}\n");
        if !stdout.trim().is_empty() {
            result.push_str("\nstdout:\n");
            result.push_str(&truncate_output(&stdout, 12_000));
        }
        if !stderr.trim().is_empty() {
            result.push_str("\nstderr:\n");
            result.push_str(&truncate_output(&stderr, 8_000));
        }

        Ok(result)
    }
}

fn truncate_output(output: &str, max_chars: usize) -> String {
    let total = output.chars().count();
    if total <= max_chars {
        return output.to_string();
    }

    let truncated: String = output.chars().take(max_chars).collect();
    format!("{truncated}\n[truncated {} chars]\n", total - max_chars)
}
