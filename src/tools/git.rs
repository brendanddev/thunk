use std::process::Command;
use tracing::info;

use super::{Tool, ToolRunResult};
use crate::error::{ParamsError, Result};
use crate::safety;

pub struct GitTool;

impl Tool for GitTool {
    fn name(&self) -> &str {
        "git"
    }

    fn description(&self) -> &str {
        "Read git repo context. Usage examples: [git: status], [git: diff], [git: log 5]"
    }

    fn run(&self, arg: &str) -> Result<ToolRunResult> {
        info!(tool = "git", "tool called");
        let trimmed = arg.trim();
        let subcommand = if trimmed.is_empty() {
            "status"
        } else {
            trimmed
        };
        let mut parts = subcommand.split_whitespace();
        let head = parts.next().unwrap_or("status");

        match head {
            "status" => run_git(subcommand, &["status", "--short", "--branch"]),
            "diff" => run_git(subcommand, &["diff", "--stat", "--patch", "--minimal"]),
            "log" => {
                let count = parts
                    .next()
                    .and_then(|n| n.parse::<usize>().ok())
                    .unwrap_or(5)
                    .clamp(1, 20);
                run_git(
                    subcommand,
                    &["log", &format!("-n{count}"), "--oneline", "--decorate"],
                )
            }
            other => Err(ParamsError::Config(format!(
                "Unsupported git command: {other}. Use status, diff, or log [n]."
            ))),
        }
    }
}

fn run_git(subcommand: &str, args: &[&str]) -> Result<ToolRunResult> {
    let _ = safety::inspect_git_operation(subcommand)?;
    let root = safety::project_root()?;
    let output = Command::new("git").args(args).current_dir(root).output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(ParamsError::Config(if stderr.is_empty() {
            "git command failed".to_string()
        } else {
            stderr
        }));
    }

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    Ok(ToolRunResult::Immediate(truncate_output(&stdout, 16_000)))
}

fn truncate_output(output: &str, max_chars: usize) -> String {
    let total = output.chars().count();
    if total <= max_chars {
        return output.to_string();
    }

    let truncated: String = output.chars().take(max_chars).collect();
    format!("{truncated}\n\n[truncated {} chars]", total - max_chars)
}
