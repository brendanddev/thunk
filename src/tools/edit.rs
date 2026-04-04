// src/tools/edit.rs
//
// Approval-driven targeted file editor using exact search/replace blocks.

use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::info;

use super::write::build_diff;
use super::{PendingToolAction, Tool, ToolRunResult};
use crate::error::{ParamsError, Result};
use crate::events::PendingActionKind;
use crate::safety::{self, ProjectPathKind};

pub struct EditFileTool;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct EditBlock {
    search: String,
    replace: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
enum LineEndingStyle {
    Lf,
    Crlf,
}

#[derive(Debug, Serialize, Deserialize)]
struct EditFilePayload {
    path: String,
    display_path: String,
    edits: Vec<EditBlock>,
    original_hash: String,
    final_content: String,
    replacement_count: usize,
    line_ending: LineEndingStyle,
}

impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Propose a targeted file edit with approval. Usage: [edit_file: path] followed by a ```params-edit fenced block with SEARCH/REPLACE sections."
    }

    fn run(&self, arg: &str) -> Result<ToolRunResult> {
        if arg.trim().is_empty() {
            return Err(ParamsError::Config("edit_file path cannot be empty".into()));
        }
        Err(ParamsError::Config(
            "edit_file requires a following ```params-edit fenced block".into(),
        ))
    }

    fn run_with_context(&self, arg: &str, following_text: &str) -> Result<ToolRunResult> {
        info!(tool = "edit_file", phase = "proposal", "tool called");
        build_pending_edit_from_block(arg, following_text)
    }

    fn run_approved(&self, arg: &str) -> Result<String> {
        info!(
            tool = "edit_file",
            phase = "execute",
            "approved tool executing"
        );
        let payload: EditFilePayload = serde_json::from_str(arg)
            .map_err(|e| ParamsError::Config(format!("Invalid edit payload: {e}")))?;

        let path = Path::new(&payload.path);
        let current = fs::read_to_string(path).map_err(|e| match e.kind() {
            ErrorKind::NotFound => ParamsError::Config(format!(
                "Target file no longer exists: {}. Use write_file to create new files.",
                payload.display_path
            )),
            ErrorKind::InvalidData => ParamsError::Config(format!(
                "Cannot edit non-text file: {}",
                payload.display_path
            )),
            _ => ParamsError::Io(e),
        })?;

        let (normalized_current, _) = normalize_for_matching(&current);
        let current_hash = hash_normalized(&normalized_current);
        if current_hash != payload.original_hash {
            return Err(ParamsError::Config(
                "File changed since edit was proposed; regenerate the edit".to_string(),
            ));
        }

        let output = match payload.line_ending {
            LineEndingStyle::Lf => payload.final_content.clone(),
            LineEndingStyle::Crlf => payload.final_content.replace('\n', "\r\n"),
        };
        fs::write(path, output)?;
        info!(
            tool = "edit_file",
            phase = "execute",
            "approved tool finished"
        );
        Ok(format!(
            "Edited file: {} ({} replacements)",
            payload.display_path, payload.replacement_count
        ))
    }
}

pub fn build_pending_edit_request(path: &str, body: &str) -> Result<PendingToolAction> {
    let edit_body = if body.contains("```params-edit") {
        extract_edit_block(body)?
    } else {
        body.to_string()
    };

    match build_pending_edit(path, &edit_body)? {
        Some(pending) => Ok(pending),
        None => Err(ParamsError::Config(format!(
            "No changes needed for {}",
            path.trim()
        ))),
    }
}

fn build_pending_edit_from_block(arg: &str, following_text: &str) -> Result<ToolRunResult> {
    let edit_body = extract_edit_block(following_text)?;
    match build_pending_edit(arg, &edit_body)? {
        Some(pending) => Ok(ToolRunResult::RequiresApproval(pending)),
        None => Ok(ToolRunResult::Immediate(format!(
            "No changes needed for {}",
            arg.trim()
        ))),
    }
}

fn build_pending_edit(requested_path: &str, edit_body: &str) -> Result<Option<PendingToolAction>> {
    let requested_path = requested_path.trim();
    if requested_path.is_empty() {
        return Err(ParamsError::Config("edit_file path cannot be empty".into()));
    }

    let path_info =
        safety::inspect_project_path("edit_file", requested_path, ProjectPathKind::File, true)?;
    if !path_info.exists {
        return Err(ParamsError::Config(format!(
            "Target file does not exist: {}. Use write_file to create new files.",
            path_info.display_path
        )));
    }

    let existing = fs::read_to_string(&path_info.resolved_path).map_err(|e| match e.kind() {
        ErrorKind::InvalidData => ParamsError::Config(format!(
            "Cannot edit non-text file: {}",
            path_info.display_path
        )),
        _ => ParamsError::Io(e),
    })?;

    let (normalized_existing, line_ending) = normalize_for_matching(&existing);
    let edits = parse_edit_blocks(edit_body)?;
    let (final_content, replacement_count) = apply_edit_blocks(&normalized_existing, &edits)?;

    if final_content == normalized_existing {
        return Ok(None);
    }

    let preview = build_diff(
        &path_info.display_path,
        &normalized_existing,
        &final_content,
    )?;
    let inspection = safety::inspect_edit_target(&path_info.display_path, replacement_count)?;
    let payload = EditFilePayload {
        path: path_info.resolved_path.to_string_lossy().to_string(),
        display_path: path_info.display_path.clone(),
        edits,
        original_hash: hash_normalized(&normalized_existing),
        final_content,
        replacement_count,
        line_ending,
    };

    Ok(Some(PendingToolAction {
        kind: PendingActionKind::FileEdit,
        tool_name: "edit_file".to_string(),
        argument: serde_json::to_string(&payload)
            .map_err(|e| ParamsError::Config(e.to_string()))?,
        display_argument: path_info.display_path.clone(),
        title: format!("Approve file edit: {}", path_info.display_path),
        preview,
        inspection,
    }))
}

fn extract_edit_block(text: &str) -> Result<String> {
    let marker = "```params-edit";
    let start = text.find(marker).ok_or_else(|| {
        ParamsError::Config(
            "edit_file requires a following ```params-edit fenced block".to_string(),
        )
    })?;

    let after_marker = &text[start + marker.len()..];
    let after_newline = after_marker
        .strip_prefix("\r\n")
        .or_else(|| after_marker.strip_prefix('\n'))
        .ok_or_else(|| ParamsError::Config("Expected newline after ```params-edit".to_string()))?;

    let end = after_newline.find("```");

    Ok(match end {
        Some(end) => after_newline[..end].to_string(),
        None => after_newline.to_string(),
    })
}

fn parse_edit_blocks(body: &str) -> Result<Vec<EditBlock>> {
    let normalized = normalize_newlines(body);
    let mut rest = normalized.as_str();
    let mut blocks = Vec::new();

    while !rest.trim().is_empty() {
        rest = rest.trim_start_matches('\n');
        if rest.is_empty() {
            break;
        }
        let Some(after_search) = rest.strip_prefix("<<<<<<< SEARCH\n") else {
            return Err(ParamsError::Config(
                "Malformed params-edit block: expected `<<<<<<< SEARCH`".to_string(),
            ));
        };
        if after_search.starts_with("=======\n") {
            return Err(ParamsError::Config(
                "edit_file SEARCH block cannot be empty".to_string(),
            ));
        }
        let Some(separator_idx) = after_search.find("\n=======\n") else {
            return Err(ParamsError::Config(
                "Malformed params-edit block: missing `=======`".to_string(),
            ));
        };
        let search = &after_search[..separator_idx];
        if search.is_empty() {
            return Err(ParamsError::Config(
                "edit_file SEARCH block cannot be empty".to_string(),
            ));
        }

        let after_separator = &after_search[separator_idx + "\n=======\n".len()..];
        let Some(replace_end_idx) = after_separator.find("\n>>>>>>> REPLACE") else {
            return Err(ParamsError::Config(
                "Malformed params-edit block: missing `>>>>>>> REPLACE`".to_string(),
            ));
        };
        let replace = &after_separator[..replace_end_idx];
        let after_replace = &after_separator[replace_end_idx + "\n>>>>>>> REPLACE".len()..];
        rest = after_replace.strip_prefix('\n').unwrap_or(after_replace);

        blocks.push(EditBlock {
            search: search.to_string(),
            replace: replace.to_string(),
        });
    }

    if blocks.is_empty() {
        return Err(ParamsError::Config(
            "edit_file requires at least one SEARCH/REPLACE block".to_string(),
        ));
    }

    Ok(blocks)
}

fn apply_edit_blocks(original: &str, edits: &[EditBlock]) -> Result<(String, usize)> {
    let mut current = original.to_string();
    for (idx, edit) in edits.iter().enumerate() {
        let matches = current.matches(&edit.search).count();
        if matches == 0 {
            return Err(ParamsError::Config(format!(
                "edit_file SEARCH block {} did not match the file content exactly once",
                idx + 1
            )));
        }
        if matches > 1 {
            return Err(ParamsError::Config(format!(
                "edit_file SEARCH block {} matched multiple locations; make it more specific",
                idx + 1
            )));
        }
        current = current.replacen(&edit.search, &edit.replace, 1);
    }
    Ok((current, edits.len()))
}

fn normalize_newlines(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

fn normalize_for_matching(text: &str) -> (String, LineEndingStyle) {
    let style = if uses_consistent_crlf(text) {
        LineEndingStyle::Crlf
    } else {
        LineEndingStyle::Lf
    };
    (normalize_newlines(text), style)
}

fn uses_consistent_crlf(text: &str) -> bool {
    if !text.contains("\r\n") {
        return false;
    }

    let bytes = text.as_bytes();
    let mut idx = 0usize;
    while idx < bytes.len() {
        match bytes[idx] {
            b'\r' => {
                if idx + 1 >= bytes.len() || bytes[idx + 1] != b'\n' {
                    return false;
                }
                idx += 2;
            }
            b'\n' => return false,
            _ => idx += 1,
        }
    }

    true
}

fn hash_normalized(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temp_project_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("params-edit-test-{label}-{nonce}"));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn parses_single_edit_block() {
        let blocks = parse_edit_blocks("<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n")
            .expect("blocks");
        assert_eq!(
            blocks,
            vec![EditBlock {
                search: "old".to_string(),
                replace: "new".to_string(),
            }]
        );
    }

    #[test]
    fn parses_multiple_edit_blocks() {
        let blocks = parse_edit_blocks(
            "<<<<<<< SEARCH\none\n=======\n1\n>>>>>>> REPLACE\n\n<<<<<<< SEARCH\ntwo\n=======\n2\n>>>>>>> REPLACE\n",
        )
        .expect("blocks");
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn rejects_empty_search_block() {
        let error =
            parse_edit_blocks("<<<<<<< SEARCH\n=======\nnew\n>>>>>>> REPLACE\n").unwrap_err();
        assert!(error.to_string().contains("SEARCH block cannot be empty"));
    }

    #[test]
    fn applies_multiple_blocks_sequentially() {
        let edits = vec![
            EditBlock {
                search: "alpha".to_string(),
                replace: "beta".to_string(),
            },
            EditBlock {
                search: "beta".to_string(),
                replace: "gamma".to_string(),
            },
        ];

        let (result, count) = apply_edit_blocks("alpha", &edits).expect("apply");
        assert_eq!(result, "gamma");
        assert_eq!(count, 2);
    }

    #[test]
    fn rejects_ambiguous_search_match() {
        let edits = vec![EditBlock {
            search: "x".to_string(),
            replace: "y".to_string(),
        }];
        let error = apply_edit_blocks("x x", &edits).unwrap_err();
        assert!(error.to_string().contains("matched multiple locations"));
    }

    #[test]
    fn normalizes_and_restores_crlf() {
        let (normalized, style) = normalize_for_matching("one\r\ntwo\r\n");
        assert_eq!(normalized, "one\ntwo\n");
        assert_eq!(style, LineEndingStyle::Crlf);
    }

    #[test]
    fn missing_file_is_rejected_for_edit_requests() {
        let _guard = crate::safety::test_cwd_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let root = temp_project_dir("missing");
        let original = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&root).expect("set cwd");

        let error = build_pending_edit_request(
            "src/missing.rs",
            "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n",
        )
        .unwrap_err();
        std::env::set_current_dir(original).expect("restore cwd");

        assert!(error
            .to_string()
            .contains("Use write_file to create new files"));
    }

    #[test]
    fn no_op_edit_returns_immediate_result() {
        let _guard = crate::safety::test_cwd_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let root = temp_project_dir("noop");
        let file_path = root.join("src").join("main.rs");
        fs::create_dir_all(file_path.parent().expect("parent")).expect("mkdir");
        fs::write(&file_path, "fn main() {}\n").expect("write file");
        let original = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&root).expect("set cwd");

        let result = build_pending_edit_from_block(
            "src/main.rs",
            "```params-edit\n<<<<<<< SEARCH\nfn main() {}\n=======\nfn main() {}\n>>>>>>> REPLACE\n```",
        )
        .expect("build edit");
        std::env::set_current_dir(original).expect("restore cwd");

        match result {
            ToolRunResult::Immediate(message) => {
                assert!(message.contains("No changes needed for src/main.rs"));
            }
            ToolRunResult::RequiresApproval(_) => panic!("expected no-op immediate result"),
        }
    }

    #[test]
    fn approved_edit_writes_file_and_reports_count() {
        let _guard = crate::safety::test_cwd_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let root = temp_project_dir("approve");
        let file_path = root.join("src").join("lib.rs");
        fs::create_dir_all(file_path.parent().expect("parent")).expect("mkdir");
        fs::write(&file_path, "pub fn value() -> i32 { 1 }\n").expect("write file");
        let original = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&root).expect("set cwd");

        let pending = build_pending_edit_request(
            "src/lib.rs",
            "<<<<<<< SEARCH\npub fn value() -> i32 { 1 }\n=======\npub fn value() -> i32 { 2 }\n>>>>>>> REPLACE\n",
        )
        .expect("pending edit");
        std::env::set_current_dir(original).expect("restore cwd");

        let tool = EditFileTool;
        let output = tool.run_approved(&pending.argument).expect("approve edit");
        assert!(output.contains("Edited file: src/lib.rs (1 replacements)"));

        let written = fs::read_to_string(root.join("src/lib.rs")).expect("read file");
        assert_eq!(written, "pub fn value() -> i32 { 2 }\n");
    }

    #[test]
    fn stale_edit_is_rejected_after_file_changes() {
        let _guard = crate::safety::test_cwd_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let root = temp_project_dir("stale");
        let file_path = root.join("src").join("lib.rs");
        fs::create_dir_all(file_path.parent().expect("parent")).expect("mkdir");
        fs::write(&file_path, "pub fn value() -> i32 { 1 }\n").expect("write file");
        let original = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&root).expect("set cwd");

        let pending = build_pending_edit_request(
            "src/lib.rs",
            "<<<<<<< SEARCH\npub fn value() -> i32 { 1 }\n=======\npub fn value() -> i32 { 2 }\n>>>>>>> REPLACE\n",
        )
        .expect("pending edit");
        std::env::set_current_dir(original).expect("restore cwd");

        fs::write(&file_path, "pub fn value() -> i32 { 5 }\n").expect("mutate file");

        let tool = EditFileTool;
        let error = tool.run_approved(&pending.argument).unwrap_err();
        assert!(error
            .to_string()
            .contains("File changed since edit was proposed"));
    }

    #[test]
    fn direct_edit_request_accepts_fenced_block() {
        let _guard = crate::safety::test_cwd_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let root = temp_project_dir("direct-fenced");
        let file_path = root.join("scratch").join("edit-demo.txt");
        fs::create_dir_all(file_path.parent().expect("parent")).expect("mkdir");
        fs::write(&file_path, "hello\nworld\n").expect("write file");
        let original = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&root).expect("set cwd");

        let pending = build_pending_edit_request(
            "scratch/edit-demo.txt",
            "```params-edit\n<<<<<<< SEARCH\nhello\n=======\nhello from edit_file\n>>>>>>> REPLACE\n```",
        )
        .expect("pending edit");

        std::env::set_current_dir(original).expect("restore cwd");

        assert_eq!(pending.display_argument, "scratch/edit-demo.txt");
        assert!(pending.preview.contains("hello from edit_file"));
    }

    #[test]
    fn direct_edit_request_accepts_unclosed_fenced_block() {
        let _guard = crate::safety::test_cwd_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let root = temp_project_dir("direct-unclosed");
        let file_path = root.join("scratch").join("edit-demo.txt");
        fs::create_dir_all(file_path.parent().expect("parent")).expect("mkdir");
        fs::write(&file_path, "hello\nworld\n").expect("write file");
        let original = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&root).expect("set cwd");

        let pending = build_pending_edit_request(
            "scratch/edit-demo.txt",
            "```params-edit\n<<<<<<< SEARCH\nhello\n=======\nhello from edit_file\n>>>>>>> REPLACE\n",
        )
        .expect("pending edit");

        std::env::set_current_dir(original).expect("restore cwd");

        assert_eq!(pending.display_argument, "scratch/edit-demo.txt");
        assert!(pending.preview.contains("hello from edit_file"));
    }
}
