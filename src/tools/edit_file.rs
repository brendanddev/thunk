use std::fs;
use std::path::Path;

use crate::runtime::ResolvedToolInput;

use super::context::ToolContext;
use super::pending::{PendingAction, RiskLevel};
use super::types::{
    EditFileOutput, ExecutionKind, ToolError, ToolInput, ToolOutput, ToolRunResult, ToolSpec,
};
use super::Tool;

pub struct EditFileTool {
    context: ToolContext,
}

impl EditFileTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }

    fn run_legacy(&self, input: &ToolInput) -> Result<ToolRunResult, ToolError> {
        let ToolInput::EditFile {
            path,
            search,
            replace,
        } = input
        else {
            return Err(ToolError::InvalidInput(
                "edit_file received wrong input variant".into(),
            ));
        };

        if path.is_empty() {
            return Err(ToolError::InvalidInput("path must not be empty".into()));
        }
        if search.is_empty() {
            return Err(ToolError::InvalidInput(
                "missing ---search--- section. The [edit_file] block requires both \
                 ---search--- (the exact text to find) and ---replace--- (the replacement). \
                 Re-emit the [edit_file] block with both sections included."
                    .into(),
            ));
        }

        check_path_safety(path, &self.context.root)?;

        let resolved = self.context.resolve(path);
        let contents = fs::read_to_string(&resolved)?;

        if !contents.contains(search.as_str()) {
            return Err(ToolError::InvalidInput(format!(
                "search text not found in {path}"
            )));
        }

        let lines_in_search = search.lines().count().max(1);
        let summary = format!("edit {path}: replace {lines_in_search} line(s)");
        let payload = encode_payload(path, search, replace);

        Ok(ToolRunResult::Approval(PendingAction {
            tool_name: "edit_file".to_string(),
            summary,
            risk: RiskLevel::Medium,
            payload,
        }))
    }
}

// Null byte: safe separator for paths and code text, which never contain \x00.
const SEP: char = '\x00';

fn encode_payload(path: &str, search: &str, replace: &str) -> String {
    format!("{}{SEP}{}{SEP}{}", path, search, replace)
}

fn decode_payload(payload: &str) -> Option<(String, String, String)> {
    let mut parts = payload.splitn(3, SEP);
    Some((
        parts.next()?.to_string(),
        parts.next()?.to_string(),
        parts.next()?.to_string(),
    ))
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

impl Tool for EditFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "edit_file",
            description: "Replace an exact block of text in an existing file. The search text must match exactly, including whitespace.",
            input_hint: "path: path/to/file.rs",
            execution_kind: ExecutionKind::RequiresApproval,
            default_risk: Some(RiskLevel::Medium),
        }
    }

    fn run(&self, input: &ResolvedToolInput) -> Result<ToolRunResult, ToolError> {
        // Temporary Slice 15.3.3 shim: keep legacy edit_file behavior unchanged
        // until the resolved-input-native migration lands in 15.3.4.
        let legacy = match input {
            ResolvedToolInput::EditFile {
                path,
                search,
                replace,
            } => ToolInput::EditFile {
                path: path.display().to_string(),
                search: search.clone(),
                replace: replace.clone(),
            },
            _ => {
                return Err(ToolError::InvalidInput(
                    "edit_file received wrong input variant".into(),
                ))
            }
        };

        self.run_legacy(&legacy)
    }

    fn execute_approved(&self, payload: &str) -> Result<ToolOutput, ToolError> {
        let (path, search, replace) = decode_payload(payload)
            .ok_or_else(|| ToolError::InvalidInput("malformed edit_file payload".into()))?;

        let resolved = self.context.resolve(&path);
        let contents = fs::read_to_string(&resolved)?;

        // Staleness check: the search text must still be present in the file.
        // If the file was modified between proposal and approval, this catches it.
        if !contents.contains(search.as_str()) {
            return Err(ToolError::InvalidInput(
                "search text no longer found in file — it may have changed since the edit was proposed".to_string(),
            ));
        }

        // Replace only the first occurrence so the model controls specificity via
        // the search string rather than having all occurrences silently changed.
        let new_contents = contents.replacen(&search, &replace, 1);
        fs::write(&resolved, new_contents)?;

        let lines_replaced = search.lines().count().max(1);
        Ok(ToolOutput::EditFile(EditFileOutput {
            path,
            lines_replaced,
        }))
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    fn tool_in(dir: &TempDir) -> EditFileTool {
        EditFileTool::new(ToolContext::new(dir.path().to_path_buf()))
    }

    fn run_edit(
        tool: &EditFileTool,
        path: &str,
        search: &str,
        replace: &str,
    ) -> Result<ToolRunResult, ToolError> {
        tool.run_legacy(&ToolInput::EditFile {
            path: path.to_string(),
            search: search.to_string(),
            replace: replace.to_string(),
        })
    }

    // run()

    #[test]
    fn run_returns_approval_for_valid_input() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("src.rs");
        fs::write(&file, "fn old() {}").unwrap();

        let tool = tool_in(&dir);
        let result = run_edit(&tool, "src.rs", "fn old() {}", "fn new() {}").unwrap();
        assert!(matches!(result, ToolRunResult::Approval(_)));
    }

    #[test]
    fn run_summary_describes_path_and_line_count() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("lib.rs");
        fs::write(&file, "fn a() {}\nfn b() {}").unwrap();

        let tool = tool_in(&dir);
        let ToolRunResult::Approval(pa) =
            run_edit(&tool, "lib.rs", "fn a() {}\nfn b() {}", "fn c() {}").unwrap()
        else {
            panic!("expected Approval");
        };
        assert!(pa.summary.contains("lib.rs"));
        assert!(pa.summary.contains("2 line(s)"));
    }

    #[test]
    fn run_risk_is_medium() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("f.rs"), "old").unwrap();
        let tool = tool_in(&dir);
        let ToolRunResult::Approval(pa) = run_edit(&tool, "f.rs", "old", "new").unwrap() else {
            panic!("expected Approval");
        };
        assert_eq!(pa.risk, RiskLevel::Medium);
    }

    #[test]
    fn run_fails_for_empty_path() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        let err = run_edit(&tool, "", "search", "replace").unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn run_fails_for_empty_search() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("f.rs"), "content").unwrap();
        let tool = tool_in(&dir);
        let err = run_edit(&tool, "f.rs", "", "replace").unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn run_fails_when_search_not_found() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("f.rs"), "actual content").unwrap();
        let tool = tool_in(&dir);
        let err = run_edit(&tool, "f.rs", "not present", "replace").unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn run_fails_for_missing_file() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        let err = run_edit(&tool, "nonexistent.rs", "search", "replace").unwrap_err();
        assert!(matches!(err, ToolError::Io(_)));
    }

    #[test]
    fn run_rejects_parent_dir_traversal() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        let err = run_edit(&tool, "../escape.rs", "old", "new").unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn run_rejects_absolute_path_outside_root() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        let err = run_edit(&tool, "/etc/passwd", "root", "evil").unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    // ── execute_approved() ────────────────────────────────────────────────────

    #[test]
    fn execute_approved_applies_edit_correctly() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("f.rs");
        fs::write(&path, "fn old() {}\n").unwrap();

        let tool = tool_in(&dir);
        let ToolRunResult::Approval(pa) =
            run_edit(&tool, "f.rs", "fn old() {}", "fn new() {}").unwrap()
        else {
            panic!("expected Approval");
        };

        let out = tool.execute_approved(&pa.payload).unwrap();
        let ToolOutput::EditFile(ef) = out else {
            panic!("expected EditFile output");
        };
        assert_eq!(ef.lines_replaced, 1);

        let written = fs::read_to_string(&path).unwrap();
        assert!(written.contains("fn new() {}"));
        assert!(!written.contains("fn old() {}"));
    }

    #[test]
    fn execute_approved_staleness_check_rejects_changed_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("f.rs");
        fs::write(&path, "fn original() {}").unwrap();

        let tool = tool_in(&dir);
        let ToolRunResult::Approval(pa) =
            run_edit(&tool, "f.rs", "fn original() {}", "fn new() {}").unwrap()
        else {
            panic!("expected Approval");
        };

        // Simulate external modification between proposal and approval.
        fs::write(&path, "fn completely_different() {}").unwrap();

        let err = tool.execute_approved(&pa.payload).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn execute_approved_replaces_first_occurrence_only() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("f.rs");
        fs::write(&path, "foo\nfoo\nbar\n").unwrap();

        let tool = tool_in(&dir);
        let ToolRunResult::Approval(pa) = run_edit(&tool, "f.rs", "foo", "baz").unwrap() else {
            panic!("expected Approval");
        };

        tool.execute_approved(&pa.payload).unwrap();
        let written = fs::read_to_string(&path).unwrap();
        assert_eq!(written, "baz\nfoo\nbar\n");
    }

    #[test]
    fn execute_approved_multiline_edit() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("f.rs");
        fs::write(&path, "fn a() {\n    let x = 1;\n}\n").unwrap();

        let tool = tool_in(&dir);
        let search = "fn a() {\n    let x = 1;\n}";
        let replace = "fn a() {\n    let x = 42;\n}";
        let ToolRunResult::Approval(pa) = run_edit(&tool, "f.rs", search, replace).unwrap() else {
            panic!("expected Approval");
        };
        assert!(matches!(pa.risk, RiskLevel::Medium));

        tool.execute_approved(&pa.payload).unwrap();
        let written = fs::read_to_string(&path).unwrap();
        assert!(written.contains("let x = 42;"));
        assert!(!written.contains("let x = 1;"));
    }

    #[test]
    fn run_wrong_input_variant_returns_error() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        let err = tool
            .run_legacy(&ToolInput::ReadFile {
                path: "f.rs".into(),
            })
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn execute_approved_malformed_payload_returns_error() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        // Payload missing both separators
        let err = tool.execute_approved("no-separators-at-all").unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    // ── NamedTempFile: absolute path within root is accepted ─────────────────

    #[test]
    fn run_accepts_absolute_path_within_root() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("inside.rs");
        fs::write(&path, "old content").unwrap();

        // Use a tool whose root is "/" so the absolute path is within root.
        let tool = EditFileTool::new(ToolContext::new("/".into()));
        let abs_path = path.to_str().unwrap();
        let result = run_edit(&tool, abs_path, "old content", "new content");
        assert!(result.is_ok());
    }
}
