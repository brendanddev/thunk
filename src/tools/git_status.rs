use std::io;
use std::process::Command;

use super::context::ToolContext;
use super::types::{
    ExecutionKind, GitStatusEntry, GitStatusOutput, ToolError, ToolInput, ToolOutput,
    ToolRunResult, ToolSpec,
};
use super::Tool;

const MAX_STATUS_ENTRIES: usize = 100;
const MAX_STATUS_PATH_CHARS: usize = 240;

pub struct GitStatusTool {
    context: ToolContext,
}

impl GitStatusTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

impl Tool for GitStatusTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "git_status",
            description: "Show read-only git working tree status for the project.",
            input_hint: "",
            execution_kind: ExecutionKind::Immediate,
            default_risk: None,
        }
    }

    fn run(&self, input: &ToolInput) -> Result<ToolRunResult, ToolError> {
        let ToolInput::GitStatus = input else {
            return Err(ToolError::InvalidInput(
                "git_status received wrong input variant".into(),
            ));
        };

        let output = Command::new("git")
            .args(["status", "--short", "--branch"])
            .current_dir(&self.context.root)
            .output()
            .map_err(git_command_error)?;

        if !output.status.success() {
            return Err(git_status_error(&output.stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(ToolRunResult::Immediate(ToolOutput::GitStatus(
            parse_git_status_output(&stdout),
        )))
    }
}

fn git_command_error(error: io::Error) -> ToolError {
    if error.kind() == io::ErrorKind::NotFound {
        ToolError::InvalidInput("git_status failed: git executable unavailable".into())
    } else {
        ToolError::Io(error)
    }
}

fn git_status_error(stderr: &[u8]) -> ToolError {
    let stderr = String::from_utf8_lossy(stderr);
    if stderr.contains("not a git repository") {
        ToolError::InvalidInput("git_status failed: not a Git repository".into())
    } else {
        ToolError::InvalidInput("git_status failed".into())
    }
}

fn parse_git_status_output(stdout: &str) -> GitStatusOutput {
    let mut lines = stdout.lines();
    let (branch, upstream, ahead, behind) = lines
        .next()
        .and_then(parse_branch_line)
        .unwrap_or((None, None, None, None));

    let mut entries = Vec::new();
    let mut total_entries = 0usize;
    for line in lines {
        if line.is_empty() {
            continue;
        }
        total_entries += 1;
        if entries.len() < MAX_STATUS_ENTRIES {
            entries.push(parse_status_entry(line));
        }
    }

    GitStatusOutput {
        branch,
        upstream,
        ahead,
        behind,
        entries,
        total_entries,
        truncated: total_entries > MAX_STATUS_ENTRIES,
    }
}

fn parse_branch_line(
    line: &str,
) -> Option<(Option<String>, Option<String>, Option<u32>, Option<u32>)> {
    let rest = line.strip_prefix("## ")?.trim();
    let (branch_part, state_part) = rest
        .split_once(" [")
        .map(|(branch, state)| (branch.trim(), Some(state.trim_end_matches(']'))))
        .unwrap_or((rest, None));
    let (branch, upstream) = branch_part
        .split_once("...")
        .map(|(branch, upstream)| (empty_to_none(branch), empty_to_none(upstream)))
        .unwrap_or((empty_to_none(branch_part), None));
    let (ahead, behind) = state_part.map(parse_ahead_behind).unwrap_or((None, None));
    Some((branch, upstream, ahead, behind))
}

fn parse_ahead_behind(state: &str) -> (Option<u32>, Option<u32>) {
    let mut ahead = None;
    let mut behind = None;
    for part in state.split(',') {
        let part = part.trim();
        if let Some(value) = part.strip_prefix("ahead ") {
            ahead = value.parse().ok();
        } else if let Some(value) = part.strip_prefix("behind ") {
            behind = value.parse().ok();
        }
    }
    (ahead, behind)
}

fn empty_to_none(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn parse_status_entry(line: &str) -> GitStatusEntry {
    let (xy, path) = if line.len() >= 3 {
        (&line[..2], line[3..].trim())
    } else {
        (line, "")
    };
    let (path, path_truncated) = truncate_path(path);
    GitStatusEntry {
        xy: xy.to_string(),
        path,
        path_truncated,
    }
}

fn truncate_path(path: &str) -> (String, bool) {
    let char_count = path.chars().count();
    if char_count <= MAX_STATUS_PATH_CHARS {
        return (path.to_string(), false);
    }
    let mut truncated = path
        .chars()
        .take(MAX_STATUS_PATH_CHARS.saturating_sub(3))
        .collect::<String>();
    truncated.push_str("...");
    (truncated, true)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};

    use tempfile::TempDir;

    use super::*;

    fn init_git_repo(path: &Path) {
        let status = Command::new("git")
            .args(["init"])
            .current_dir(path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "git init must succeed");
    }

    fn run_status(path: &Path) -> Result<ToolRunResult, ToolError> {
        GitStatusTool::new(ToolContext::new(PathBuf::from(path))).run(&ToolInput::GitStatus)
    }

    #[test]
    fn spec_is_immediate() {
        let tool = GitStatusTool::new(ToolContext::new(PathBuf::from(".")));
        let spec = tool.spec();
        assert_eq!(spec.name, "git_status");
        assert_eq!(spec.execution_kind, ExecutionKind::Immediate);
        assert!(spec.default_risk.is_none());
    }

    #[test]
    fn returns_git_status_output() {
        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());
        fs::write(tmp.path().join("changed.txt"), "changed\n").unwrap();

        let out = run_status(tmp.path()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::GitStatus(status)) = out else {
            panic!("expected Immediate(GitStatus)");
        };
        assert!(status.branch.is_some());
        assert_eq!(status.total_entries, 1);
        assert_eq!(status.entries[0].xy, "??");
        assert_eq!(status.entries[0].path, "changed.txt");
        assert!(!status.truncated);
    }

    #[test]
    fn default_registry_dispatches_git_status() {
        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());
        let registry = crate::tools::default_registry(tmp.path().to_path_buf());

        let out = registry.dispatch(ToolInput::GitStatus).unwrap();
        assert!(matches!(
            out,
            ToolRunResult::Immediate(ToolOutput::GitStatus(_))
        ));
    }

    #[test]
    fn non_git_directory_returns_deterministic_error() {
        let tmp = TempDir::new().unwrap();
        let err = run_status(tmp.path()).unwrap_err();
        assert!(matches!(
            err,
            ToolError::InvalidInput(message)
                if message == "git_status failed: not a Git repository"
        ));
    }

    #[test]
    fn truncates_many_status_entries() {
        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());
        for i in 0..105 {
            fs::write(tmp.path().join(format!("file_{i:03}.txt")), "changed\n").unwrap();
        }

        let out = run_status(tmp.path()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::GitStatus(status)) = out else {
            panic!("expected Immediate(GitStatus)");
        };
        assert_eq!(status.total_entries, 105);
        assert_eq!(status.entries.len(), MAX_STATUS_ENTRIES);
        assert!(status.truncated);
    }

    #[test]
    fn parses_branch_tracking_state() {
        let parsed = parse_branch_line("## main...origin/main [ahead 2, behind 3]").unwrap();
        assert_eq!(parsed.0.as_deref(), Some("main"));
        assert_eq!(parsed.1.as_deref(), Some("origin/main"));
        assert_eq!(parsed.2, Some(2));
        assert_eq!(parsed.3, Some(3));
    }
}
