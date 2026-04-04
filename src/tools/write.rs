// src/tools/write.rs
//
// Approval-driven whole-file writer with diff preview.

use std::fs;
use std::io::ErrorKind;
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tracing::info;

use super::{PendingToolAction, Tool, ToolRunResult};
use crate::error::{ParamsError, Result};
use crate::events::PendingActionKind;
use crate::safety::{self, ProjectPathKind};

pub struct WriteFileTool;

#[derive(Debug, Serialize, Deserialize)]
struct WriteFilePayload {
    path: String,
    display_path: String,
    content: String,
}

impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Propose a full-file write with approval. Usage: [write_file: path] followed by a ```params-file fenced block."
    }

    fn run(&self, arg: &str) -> Result<ToolRunResult> {
        if arg.trim().is_empty() {
            return Err(ParamsError::Config(
                "write_file path cannot be empty".into(),
            ));
        }
        Err(ParamsError::Config(
            "write_file requires a following ```params-file fenced block".into(),
        ))
    }

    fn run_with_context(&self, arg: &str, following_text: &str) -> Result<ToolRunResult> {
        info!(tool = "write_file", phase = "proposal", "tool called");
        build_pending_write_from_block(arg, following_text)
    }

    fn run_approved(&self, arg: &str) -> Result<String> {
        info!(
            tool = "write_file",
            phase = "execute",
            "approved tool executing"
        );
        let payload: WriteFilePayload = serde_json::from_str(arg)
            .map_err(|e| ParamsError::Config(format!("Invalid write payload: {e}")))?;

        let path = Path::new(&payload.path);
        if path.exists() && path.is_dir() {
            return Err(ParamsError::Config(format!(
                "Cannot write file contents to directory: {}",
                payload.display_path
            )));
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(path, payload.content)?;
        info!(
            tool = "write_file",
            phase = "execute",
            "approved tool finished"
        );
        Ok(format!("Wrote file: {}", payload.display_path))
    }
}

pub fn build_pending_write_request(path: &str, content: &str) -> Result<PendingToolAction> {
    match build_pending_write(path, content)? {
        Some(pending) => Ok(pending),
        None => Err(ParamsError::Config(format!(
            "No changes needed for {}",
            path.trim()
        ))),
    }
}

fn build_pending_write_from_block(arg: &str, following_text: &str) -> Result<ToolRunResult> {
    let new_content = extract_file_block(following_text)?;
    match build_pending_write(arg, &new_content)? {
        Some(pending) => Ok(ToolRunResult::RequiresApproval(pending)),
        None => Ok(ToolRunResult::Immediate(format!(
            "No changes needed for {}",
            arg.trim()
        ))),
    }
}

fn build_pending_write(
    requested_path: &str,
    new_content: &str,
) -> Result<Option<PendingToolAction>> {
    let requested_path = requested_path.trim();
    if requested_path.is_empty() {
        return Err(ParamsError::Config(
            "write_file path cannot be empty".into(),
        ));
    }

    let path_info =
        safety::inspect_project_path("write_file", requested_path, ProjectPathKind::File, true)?;
    let resolved_path = path_info.resolved_path;
    let display_path = path_info.display_path;

    let existing = read_existing_text_file(&resolved_path, &display_path)?;

    if existing == new_content {
        return Ok(None);
    }

    let diff = build_diff(&display_path, &existing, &new_content)?;
    let payload = WriteFilePayload {
        path: resolved_path.to_string_lossy().to_string(),
        display_path: display_path.clone(),
        content: new_content.to_string(),
    };
    let inspection = safety::inspect_write_target(&display_path, path_info.exists)?;

    Ok(Some(PendingToolAction {
        kind: PendingActionKind::FileWrite,
        tool_name: "write_file".to_string(),
        argument: serde_json::to_string(&payload)
            .map_err(|e| ParamsError::Config(e.to_string()))?,
        display_argument: display_path.clone(),
        title: format!("Approve file write: {display_path}"),
        preview: diff,
        inspection,
    }))
}

fn extract_file_block(text: &str) -> Result<String> {
    let marker = "```params-file";
    let start = text.find(marker).ok_or_else(|| {
        ParamsError::Config(
            "write_file requires a following ```params-file fenced block".to_string(),
        )
    })?;

    let after_marker = &text[start + marker.len()..];
    let after_newline = after_marker
        .strip_prefix("\r\n")
        .or_else(|| after_marker.strip_prefix('\n'))
        .ok_or_else(|| ParamsError::Config("Expected newline after ```params-file".to_string()))?;

    let end = after_newline
        .find("```")
        .ok_or_else(|| ParamsError::Config("Unclosed ```params-file fenced block".to_string()))?;

    Ok(after_newline[..end].to_string())
}

fn read_existing_text_file(path: &Path, display_path: &str) -> Result<String> {
    match fs::read_to_string(path) {
        Ok(contents) => Ok(contents),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(String::new()),
        Err(e) if e.kind() == ErrorKind::InvalidData => Err(ParamsError::Config(format!(
            "Cannot diff non-text file: {display_path}"
        ))),
        Err(e) => Err(ParamsError::Io(e)),
    }
}

fn build_diff(path: &str, old_content: &str, new_content: &str) -> Result<String> {
    let temp_dir = std::env::temp_dir();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| ParamsError::Config(e.to_string()))?
        .as_nanos();
    let old_path = temp_dir.join(format!("params-old-{nonce}.tmp"));
    let new_path = temp_dir.join(format!("params-new-{nonce}.tmp"));

    fs::write(&old_path, old_content)?;
    fs::write(&new_path, new_content)?;

    let output = Command::new("diff")
        .args([
            "-u",
            "--label",
            &format!("a/{path}"),
            "--label",
            &format!("b/{path}"),
            old_path.to_string_lossy().as_ref(),
            new_path.to_string_lossy().as_ref(),
        ])
        .output()?;

    let _ = fs::remove_file(&old_path);
    let _ = fs::remove_file(&new_path);

    if output.status.success() || output.status.code() == Some(1) {
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        if stdout.trim().is_empty() {
            Ok(format!("No textual changes detected for {path}"))
        } else {
            Ok(truncate_output(&stdout, 20_000))
        }
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        Err(ParamsError::Config(if stderr.is_empty() {
            "Failed to generate diff preview".to_string()
        } else {
            stderr
        }))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_project_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("params-write-test-{label}-{nonce}"));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn extracts_params_file_block() {
        let content =
            extract_file_block("Some lead-in\n```params-file\nfn main() {}\n```\ntrailing")
                .expect("extract block");

        assert_eq!(content, "fn main() {}\n");
    }

    #[test]
    fn rejects_paths_outside_project() {
        let _guard = crate::safety::test_cwd_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let root = temp_project_dir("escape");
        let original = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&root).expect("set cwd");
        let result = safety::inspect_project_path(
            "write_file",
            "../outside.txt",
            ProjectPathKind::File,
            true,
        );
        std::env::set_current_dir(original).expect("restore cwd");

        assert!(result.is_err());
    }

    #[test]
    fn no_op_write_returns_immediate_result() {
        let _guard = crate::safety::test_cwd_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let root = temp_project_dir("noop");
        let file_path = root.join("src").join("main.rs");
        fs::create_dir_all(file_path.parent().expect("parent")).expect("mkdir");
        fs::write(&file_path, "fn main() {}\n").expect("write file");
        let original = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&root).expect("set cwd");

        let result =
            build_pending_write_from_block("src/main.rs", "```params-file\nfn main() {}\n```")
                .expect("build write");
        std::env::set_current_dir(original).expect("restore cwd");

        match result {
            ToolRunResult::Immediate(message) => {
                assert!(message.contains("No changes needed for src/main.rs"));
            }
            ToolRunResult::RequiresApproval(_) => panic!("expected no-op immediate result"),
        }
    }

    #[test]
    fn write_payload_uses_display_path_and_writes_file() {
        let _guard = crate::safety::test_cwd_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let root = temp_project_dir("approve");
        let original = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&root).expect("set cwd");
        let result = build_pending_write_from_block(
            "src/lib.rs",
            "```params-file\npub fn value() -> i32 { 42 }\n```",
        )
        .expect("build write");
        std::env::set_current_dir(original.clone()).expect("restore cwd");

        let pending = match result {
            ToolRunResult::RequiresApproval(pending) => pending,
            ToolRunResult::Immediate(_) => panic!("expected approval"),
        };

        assert_eq!(pending.display_argument, "src/lib.rs");
        assert!(pending.preview.contains("src/lib.rs"));

        let tool = WriteFileTool;
        let output = tool
            .run_approved(&pending.argument)
            .expect("write approved");
        assert!(output.contains("Wrote file: src/lib.rs"));

        let written = fs::read_to_string(root.join("src/lib.rs")).expect("read written file");
        assert_eq!(written, "pub fn value() -> i32 { 42 }\n");
    }

    #[test]
    fn direct_write_request_builds_pending_action() {
        let _guard = crate::safety::test_cwd_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let root = temp_project_dir("direct");
        let original = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&root).expect("set cwd");

        let pending = build_pending_write_request("scratch/demo.txt", "hello\nfrom\nparams\n")
            .expect("pending write");

        std::env::set_current_dir(original).expect("restore cwd");

        assert_eq!(pending.display_argument, "scratch/demo.txt");
        assert!(pending.preview.contains("scratch/demo.txt"));
    }
}
