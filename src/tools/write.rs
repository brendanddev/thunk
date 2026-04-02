// src/tools/write.rs
//
// Approval-driven whole-file writer with diff preview.

use std::fs;
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::error::{ParamsError, Result};
use crate::events::PendingActionKind;
use super::{PendingToolAction, Tool, ToolRunResult};

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
            return Err(ParamsError::Config("write_file path cannot be empty".into()));
        }
        Err(ParamsError::Config(
            "write_file requires a following ```params-file fenced block".into()
        ))
    }

    fn run_with_context(&self, arg: &str, following_text: &str) -> Result<ToolRunResult> {
        info!(tool = "write_file", phase = "proposal", "tool called");
        let root = std::env::current_dir()?;
        build_pending_write(&root, arg, following_text)
    }

    fn run_approved(&self, arg: &str) -> Result<String> {
        info!(tool = "write_file", phase = "execute", "approved tool executing");
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
        info!(tool = "write_file", phase = "execute", "approved tool finished");
        Ok(format!("Wrote file: {}", payload.display_path))
    }
}

fn build_pending_write(root: &Path, arg: &str, following_text: &str) -> Result<ToolRunResult> {
    let requested_path = arg.trim();
    if requested_path.is_empty() {
        return Err(ParamsError::Config("write_file path cannot be empty".into()));
    }

    let resolved_path = resolve_target_path(root, requested_path)?;
    let display_path = resolved_path
        .strip_prefix(root)
        .ok()
        .and_then(|p| p.to_str())
        .map(|p| p.to_string())
        .unwrap_or_else(|| requested_path.to_string());

    let new_content = extract_file_block(following_text)?;
    let existing = read_existing_text_file(&resolved_path, &display_path)?;

    if existing == new_content {
        return Ok(ToolRunResult::Immediate(format!(
            "No changes needed for {}",
            display_path
        )));
    }

    let diff = build_diff(&display_path, &existing, &new_content)?;
    let payload = WriteFilePayload {
        path: resolved_path.to_string_lossy().to_string(),
        display_path: display_path.clone(),
        content: new_content,
    };

    Ok(ToolRunResult::RequiresApproval(PendingToolAction {
        kind: PendingActionKind::FileWrite,
        tool_name: "write_file".to_string(),
        argument: serde_json::to_string(&payload)
            .map_err(|e| ParamsError::Config(e.to_string()))?,
        display_argument: display_path.clone(),
        title: format!("Approve file write: {display_path}"),
        preview: diff,
    }))
}

fn extract_file_block(text: &str) -> Result<String> {
    let marker = "```params-file";
    let start = text.find(marker).ok_or_else(|| {
        ParamsError::Config(
            "write_file requires a following ```params-file fenced block".to_string()
        )
    })?;

    let after_marker = &text[start + marker.len()..];
    let after_newline = after_marker
        .strip_prefix("\r\n")
        .or_else(|| after_marker.strip_prefix('\n'))
        .ok_or_else(|| {
            ParamsError::Config(
                "Expected newline after ```params-file".to_string()
            )
        })?;

    let end = after_newline.find("```").ok_or_else(|| {
        ParamsError::Config("Unclosed ```params-file fenced block".to_string())
    })?;

    Ok(after_newline[..end].to_string())
}

fn resolve_target_path(root: &Path, requested: &str) -> Result<PathBuf> {
    let requested = Path::new(requested);
    let candidate = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        root.join(requested)
    };

    let normalized = normalize_path(candidate);
    if !normalized.starts_with(root) {
        return Err(ParamsError::Config(
            "write_file path must stay within the current project".to_string()
        ));
    }

    Ok(normalized)
}

fn normalize_path(path: PathBuf) -> PathBuf {
    let mut normalized = PathBuf::new();

    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(part) => normalized.push(part),
            Component::RootDir => normalized.push(Path::new("/")),
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
        }
    }

    normalized
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
        let content = extract_file_block(
            "Some lead-in\n```params-file\nfn main() {}\n```\ntrailing",
        )
        .expect("extract block");

        assert_eq!(content, "fn main() {}\n");
    }

    #[test]
    fn rejects_paths_outside_project() {
        let root = temp_project_dir("escape");
        let result = resolve_target_path(&root, "../outside.txt");

        assert!(result.is_err());
    }

    #[test]
    fn no_op_write_returns_immediate_result() {
        let root = temp_project_dir("noop");
        let file_path = root.join("src").join("main.rs");
        fs::create_dir_all(file_path.parent().expect("parent")).expect("mkdir");
        fs::write(&file_path, "fn main() {}\n").expect("write file");

        let result = build_pending_write(
            &root,
            "src/main.rs",
            "```params-file\nfn main() {}\n```",
        )
        .expect("build write");

        match result {
            ToolRunResult::Immediate(message) => {
                assert!(message.contains("No changes needed for src/main.rs"));
            }
            ToolRunResult::RequiresApproval(_) => panic!("expected no-op immediate result"),
        }
    }

    #[test]
    fn write_payload_uses_display_path_and_writes_file() {
        let root = temp_project_dir("approve");
        let result = build_pending_write(
            &root,
            "src/lib.rs",
            "```params-file\npub fn value() -> i32 { 42 }\n```",
        )
        .expect("build write");

        let pending = match result {
            ToolRunResult::RequiresApproval(pending) => pending,
            ToolRunResult::Immediate(_) => panic!("expected approval"),
        };

        assert_eq!(pending.display_argument, "src/lib.rs");
        assert!(pending.preview.contains("src/lib.rs"));

        let tool = WriteFileTool;
        let output = tool.run_approved(&pending.argument).expect("write approved");
        assert!(output.contains("Wrote file: src/lib.rs"));

        let written = fs::read_to_string(root.join("src/lib.rs")).expect("read written file");
        assert_eq!(written, "pub fn value() -> i32 { 42 }\n");
    }
}
