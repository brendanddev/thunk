use std::fs;
use std::path::Path;

use super::context::ToolContext;
use super::pending::{PendingAction, RiskLevel};
use super::types::{ExecutionKind, ToolError, ToolInput, ToolOutput, ToolRunResult, ToolSpec, WriteFileOutput};
use super::Tool;

pub struct WriteFileTool {
    context: ToolContext,
}

impl WriteFileTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

const SEP: char = '\x00';

fn encode_payload(path: &str, content: &str) -> String {
    format!("{}{SEP}{}", path, content)
}

fn decode_payload(payload: &str) -> Option<(String, String)> {
    let mut parts = payload.splitn(2, SEP);
    Some((parts.next()?.to_string(), parts.next()?.to_string()))
}

fn check_path_safety(path: &str, root: &Path) -> Result<(), ToolError> {
    if Path::new(path)
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(ToolError::InvalidInput(
            "path must not contain '..' components".into(),
        ));
    }
    if Path::new(path).is_absolute() && !Path::new(path).starts_with(root) {
        return Err(ToolError::InvalidInput(
            "absolute path must be within project root".into(),
        ));
    }
    Ok(())
}

impl Tool for WriteFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file",
            description: "Create a new file or overwrite an existing file with the given content.",
            input_hint: "path: path/to/file.rs",
            execution_kind: ExecutionKind::RequiresApproval,
            default_risk: Some(RiskLevel::Medium),
        }
    }

    fn run(&self, input: &ToolInput) -> Result<ToolRunResult, ToolError> {
        let ToolInput::WriteFile { path, content } = input else {
            return Err(ToolError::InvalidInput(
                "write_file received wrong input variant".into(),
            ));
        };

        if path.is_empty() {
            return Err(ToolError::InvalidInput("path must not be empty".into()));
        }

        check_path_safety(path, &self.context.root)?;

        let resolved = self.context.resolve(path);
        let file_exists = resolved.exists();
        let line_count = content.lines().count();

        let (summary, risk) = if file_exists {
            (
                format!("overwrite {path} ({line_count} lines)"),
                RiskLevel::High,
            )
        } else {
            (
                format!("create {path} ({line_count} lines)"),
                RiskLevel::Medium,
            )
        };

        let payload = encode_payload(path, content);

        Ok(ToolRunResult::Approval(PendingAction {
            tool_name: "write_file".to_string(),
            summary,
            risk,
            payload,
        }))
    }

    fn execute_approved(&self, payload: &str) -> Result<ToolOutput, ToolError> {
        let (path, content) = decode_payload(payload)
            .ok_or_else(|| ToolError::InvalidInput("malformed write_file payload".into()))?;

        let resolved = self.context.resolve(&path);

        // Parent must exist — we don't create intermediate directories.
        if let Some(parent) = resolved.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                return Err(ToolError::InvalidInput(format!(
                    "parent directory does not exist: {}",
                    parent.display()
                )));
            }
        }

        // Check existence before writing so created reflects the actual outcome.
        let created = !resolved.exists();
        let bytes_written = content.len();
        fs::write(&resolved, &content)?;

        Ok(ToolOutput::WriteFile(WriteFileOutput {
            path,
            bytes_written,
            created,
        }))
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    fn tool_in(dir: &TempDir) -> WriteFileTool {
        WriteFileTool::new(ToolContext::new(dir.path().to_path_buf()))
    }

    fn run_write(
        tool: &WriteFileTool,
        path: &str,
        content: &str,
    ) -> Result<ToolRunResult, ToolError> {
        tool.run(&ToolInput::WriteFile {
            path: path.to_string(),
            content: content.to_string(),
        })
    }

    // ── run() ────────────────────────────────────────────────────────────────

    #[test]
    fn run_returns_approval_for_new_file() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        let result = run_write(&tool, "new.rs", "pub fn hello() {}").unwrap();
        assert!(matches!(result, ToolRunResult::Approval(_)));
    }

    #[test]
    fn run_sets_medium_risk_for_new_file() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        let ToolRunResult::Approval(pa) = run_write(&tool, "new.rs", "content").unwrap() else {
            panic!("expected Approval");
        };
        assert_eq!(pa.risk, RiskLevel::Medium);
        assert!(pa.summary.contains("create"));
    }

    #[test]
    fn run_sets_high_risk_for_overwrite() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("existing.rs"), "old content").unwrap();
        let tool = tool_in(&dir);
        let ToolRunResult::Approval(pa) = run_write(&tool, "existing.rs", "new content").unwrap() else {
            panic!("expected Approval");
        };
        assert_eq!(pa.risk, RiskLevel::High);
        assert!(pa.summary.contains("overwrite"));
    }

    #[test]
    fn run_summary_includes_path_and_line_count() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        let ToolRunResult::Approval(pa) = run_write(&tool, "out.rs", "line1\nline2\nline3").unwrap() else {
            panic!("expected Approval");
        };
        assert!(pa.summary.contains("out.rs"));
        assert!(pa.summary.contains("3 lines"));
    }

    #[test]
    fn run_fails_for_empty_path() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        let err = run_write(&tool, "", "content").unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn run_rejects_parent_dir_traversal() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        let err = run_write(&tool, "../escape.rs", "content").unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn run_rejects_absolute_path_outside_root() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        let err = run_write(&tool, "/etc/hosts", "evil").unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn run_wrong_input_variant_returns_error() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        let err = tool
            .run(&ToolInput::ReadFile { path: "f.rs".into() })
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    // execute_approved()

    #[test]
    fn execute_approved_creates_new_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("new.rs");
        assert!(!path.exists());

        let tool = tool_in(&dir);
        let ToolRunResult::Approval(pa) = run_write(&tool, "new.rs", "pub fn hello() {}").unwrap() else {
            panic!("expected Approval");
        };

        let ToolOutput::WriteFile(wf) = tool.execute_approved(&pa.payload).unwrap() else {
            panic!("expected WriteFile output");
        };
        assert!(wf.created);
        assert_eq!(wf.bytes_written, "pub fn hello() {}".len());
        assert!(path.exists());
        assert_eq!(fs::read_to_string(&path).unwrap(), "pub fn hello() {}");
    }

    #[test]
    fn execute_approved_overwrites_existing_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("f.rs");
        fs::write(&path, "old content").unwrap();

        let tool = tool_in(&dir);
        let ToolRunResult::Approval(pa) = run_write(&tool, "f.rs", "new content").unwrap() else {
            panic!("expected Approval");
        };

        let ToolOutput::WriteFile(wf) = tool.execute_approved(&pa.payload).unwrap() else {
            panic!("expected WriteFile output");
        };
        assert!(!wf.created);
        assert_eq!(fs::read_to_string(&path).unwrap(), "new content");
    }

    #[test]
    fn execute_approved_created_flag_reflects_actual_state_at_execution_time() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("new.rs");

        let tool = tool_in(&dir);
        // Propose as new file (doesn't exist yet).
        let ToolRunResult::Approval(pa) = run_write(&tool, "new.rs", "content").unwrap() else {
            panic!("expected Approval");
        };
        assert!(pa.summary.contains("create"));

        // File is created externally before approval is processed.
        fs::write(&path, "something else").unwrap();

        // execute_approved checks existence at execution time, not at proposal time.
        let ToolOutput::WriteFile(wf) = tool.execute_approved(&pa.payload).unwrap() else {
            panic!("expected WriteFile output");
        };
        assert!(!wf.created); // file already existed when execute_approved ran
    }

    #[test]
    fn execute_approved_fails_when_parent_dir_missing() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        // Payload for a path inside a nonexistent subdirectory.
        let payload = encode_payload("nonexistent_dir/file.rs", "content");
        let err = tool.execute_approved(&payload).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn execute_approved_malformed_payload_returns_error() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        // Payload missing the separator entirely (splitn(2, SEP) can't produce path+content)
        let err = tool.execute_approved("").unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn execute_approved_accepts_absolute_path_within_root() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("out.rs");
        let abs = path.to_str().unwrap();

        let tool = WriteFileTool::new(ToolContext::new("/".into()));
        let payload = encode_payload(abs, "content");
        tool.execute_approved(&payload).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "content");
    }
}
