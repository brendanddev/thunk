use std::fs;
use std::path::{Path, PathBuf};

use crate::runtime::{ProjectPath, ResolvedToolInput};

use super::pending::{PendingAction, RiskLevel};
use super::types::{EditFileOutput, ExecutionKind, ToolError, ToolOutput, ToolRunResult, ToolSpec};
use super::Tool;

pub struct EditFileTool {
    root: PathBuf,
}

impl EditFileTool {
    pub fn new(root: PathBuf) -> Self {
        let root = root.canonicalize().unwrap_or(root);
        Self { root }
    }
}

// Null byte: safe separator for paths and code text, which never contain \x00.
const SEP: char = '\x00';
const PAYLOAD_V2: &str = "v2";

fn encode_payload(path: &ProjectPath, search: &str, replace: &str) -> String {
    format!(
        "{PAYLOAD_V2}{SEP}{}{SEP}{}{SEP}{}{SEP}{}",
        path.absolute().display(),
        path.display(),
        search,
        replace
    )
}

struct ApprovedEditPayload {
    absolute: PathBuf,
    display: String,
    search: String,
    replace: String,
}

fn decode_payload(payload: &str) -> Option<ApprovedEditPayload> {
    let mut versioned = payload.splitn(5, SEP);
    let first = versioned.next()?;
    if first == PAYLOAD_V2 {
        return Some(ApprovedEditPayload {
            absolute: PathBuf::from(versioned.next()?),
            display: versioned.next()?.to_string(),
            search: versioned.next()?.to_string(),
            replace: versioned.next()?.to_string(),
        });
    }

    let mut legacy = payload.splitn(3, SEP);
    let path = legacy.next()?.to_string();
    let search = legacy.next()?.to_string();
    let replace = legacy.next()?.to_string();
    let absolute = PathBuf::from(&path);
    if !absolute.is_absolute() {
        return None;
    }

    Some(ApprovedEditPayload {
        absolute,
        display: path,
        search,
        replace,
    })
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
        let ResolvedToolInput::EditFile {
            path,
            search,
            replace,
        } = input
        else {
            return Err(ToolError::InvalidInput(
                "edit_file received wrong input variant".into(),
            ));
        };

        if search.is_empty() {
            return Err(ToolError::InvalidInput(
                "missing ---search--- section. The [edit_file] block requires both \
                 ---search--- (the exact text to find) and ---replace--- (the replacement). \
                 Re-emit the [edit_file] block with both sections included."
                    .into(),
            ));
        }

        let contents = fs::read_to_string(path.absolute())?;

        if !contents.contains(search.as_str()) {
            return Err(ToolError::InvalidInput(format!(
                "search text not found in {}",
                path.display()
            )));
        }

        let lines_in_search = search.lines().count().max(1);
        let summary = format!("edit {}: replace {lines_in_search} line(s)", path.display());
        let payload = encode_payload(path, search, replace);

        Ok(ToolRunResult::Approval(PendingAction {
            tool_name: "edit_file".to_string(),
            summary,
            risk: RiskLevel::Medium,
            payload,
        }))
    }

    fn execute_approved(&self, payload: &str) -> Result<ToolOutput, ToolError> {
        let ApprovedEditPayload {
            absolute,
            display,
            search,
            replace,
        } = decode_payload(payload)
            .ok_or_else(|| ToolError::InvalidInput("malformed edit_file payload".into()))?;

        validate_approved_path(&self.root, &absolute)?;

        let contents = fs::read_to_string(&absolute)?;

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
        fs::write(&absolute, new_contents)?;

        let lines_replaced = search.lines().count().max(1);
        Ok(ToolOutput::EditFile(EditFileOutput {
            path: display,
            lines_replaced,
        }))
    }
}

fn validate_approved_path(root: &Path, absolute: &Path) -> Result<(), ToolError> {
    let normalized = normalized_approved_path(absolute)?;
    if !normalized.starts_with(root) {
        return Err(ToolError::InvalidInput(
            "approved path must be within project root".into(),
        ));
    }
    Ok(())
}

fn normalized_approved_path(absolute: &Path) -> Result<PathBuf, ToolError> {
    if absolute.exists() {
        return fs::canonicalize(absolute).map_err(ToolError::Io);
    }

    let mut existing = absolute;
    let mut missing = Vec::new();

    while !existing.exists() {
        let Some(name) = existing.file_name() else {
            return Err(ToolError::InvalidInput(
                "approved path must be absolute".into(),
            ));
        };
        missing.push(name.to_os_string());
        existing = existing
            .parent()
            .ok_or_else(|| ToolError::InvalidInput("approved path must be absolute".into()))?;
    }

    let mut normalized = fs::canonicalize(existing)?;
    for component in missing.iter().rev() {
        normalized.push(component);
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tempfile::TempDir;

    use super::*;
    use crate::runtime::{resolve, PathResolutionError, ProjectPath, ProjectRoot};
    use crate::tools::ToolInput;

    #[cfg(unix)]
    fn symlink_file(src: &Path, dst: &Path) {
        std::os::unix::fs::symlink(src, dst).unwrap();
    }

    #[cfg(unix)]
    fn symlink_dir(src: &Path, dst: &Path) {
        std::os::unix::fs::symlink(src, dst).unwrap();
    }

    #[cfg(windows)]
    fn symlink_file(src: &Path, dst: &Path) {
        std::os::windows::fs::symlink_file(src, dst).unwrap();
    }

    #[cfg(windows)]
    fn symlink_dir(src: &Path, dst: &Path) {
        std::os::windows::fs::symlink_dir(src, dst).unwrap();
    }

    fn tool_in(dir: &TempDir) -> EditFileTool {
        EditFileTool::new(dir.path().to_path_buf())
    }

    fn resolved_path(root: &TempDir, relative: &str) -> ProjectPath {
        let absolute = root.path().canonicalize().unwrap().join(relative);
        ProjectPath::from_trusted(absolute, relative.to_string())
    }

    fn project_root(root: &TempDir) -> ProjectRoot {
        ProjectRoot::new(root.path().to_path_buf()).unwrap()
    }

    fn run_edit(
        tool: &EditFileTool,
        path: ProjectPath,
        search: &str,
        replace: &str,
    ) -> Result<ToolRunResult, ToolError> {
        tool.run(&ResolvedToolInput::EditFile {
            path,
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
        let result = run_edit(
            &tool,
            resolved_path(&dir, "src.rs"),
            "fn old() {}",
            "fn new() {}",
        )
        .unwrap();
        assert!(matches!(result, ToolRunResult::Approval(_)));
    }

    #[test]
    fn run_summary_describes_path_and_line_count() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("lib.rs");
        fs::write(&file, "fn a() {}\nfn b() {}").unwrap();

        let tool = tool_in(&dir);
        let ToolRunResult::Approval(pa) = run_edit(
            &tool,
            resolved_path(&dir, "lib.rs"),
            "fn a() {}\nfn b() {}",
            "fn c() {}",
        )
        .unwrap() else {
            panic!("expected Approval");
        };
        let root_display = dir.path().canonicalize().unwrap().display().to_string();
        assert!(pa.summary.contains("lib.rs"));
        assert!(pa.summary.contains("2 line(s)"));
        assert!(
            !pa.summary.contains(&root_display),
            "approval summary must not contain absolute root: {}",
            pa.summary
        );
    }

    #[test]
    fn run_risk_is_medium() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("f.rs"), "old").unwrap();
        let tool = tool_in(&dir);
        let ToolRunResult::Approval(pa) =
            run_edit(&tool, resolved_path(&dir, "f.rs"), "old", "new").unwrap()
        else {
            panic!("expected Approval");
        };
        assert_eq!(pa.risk, RiskLevel::Medium);
    }

    #[test]
    fn run_fails_for_empty_search() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("f.rs"), "content").unwrap();
        let tool = tool_in(&dir);
        let err = run_edit(&tool, resolved_path(&dir, "f.rs"), "", "replace").unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn run_fails_when_search_not_found() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("f.rs"), "actual content").unwrap();
        let tool = tool_in(&dir);
        let err =
            run_edit(&tool, resolved_path(&dir, "f.rs"), "not present", "replace").unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn run_fails_for_missing_file() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        let err = run_edit(
            &tool,
            resolved_path(&dir, "nonexistent.rs"),
            "search",
            "replace",
        )
        .unwrap_err();
        assert!(matches!(err, ToolError::Io(_)));
    }

    #[test]
    fn edit_path_outside_root_fails_before_tool_execution() {
        let dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let raw = outside.path().join("escape.rs").display().to_string();
        let err = resolve(
            &project_root(&dir),
            &ToolInput::EditFile {
                path: raw.clone(),
                search: "old".into(),
                replace: "new".into(),
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            PathResolutionError::EscapesRoot { raw: actual, .. } if actual == raw
        ));
    }

    // ── execute_approved() ────────────────────────────────────────────────────

    #[test]
    fn execute_approved_applies_edit_correctly() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("f.rs");
        fs::write(&path, "fn old() {}\n").unwrap();

        let tool = tool_in(&dir);
        let ToolRunResult::Approval(pa) = run_edit(
            &tool,
            resolved_path(&dir, "f.rs"),
            "fn old() {}",
            "fn new() {}",
        )
        .unwrap() else {
            panic!("expected Approval");
        };

        let out = tool.execute_approved(&pa.payload).unwrap();
        let ToolOutput::EditFile(ef) = out else {
            panic!("expected EditFile output");
        };
        let root_display = dir.path().canonicalize().unwrap().display().to_string();
        assert_eq!(ef.path, "f.rs");
        assert!(
            !ef.path.contains(&root_display),
            "normal edit output path must not contain absolute root: {}",
            ef.path
        );
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
        let ToolRunResult::Approval(pa) = run_edit(
            &tool,
            resolved_path(&dir, "f.rs"),
            "fn original() {}",
            "fn new() {}",
        )
        .unwrap() else {
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
        let ToolRunResult::Approval(pa) =
            run_edit(&tool, resolved_path(&dir, "f.rs"), "foo", "baz").unwrap()
        else {
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
        let ToolRunResult::Approval(pa) =
            run_edit(&tool, resolved_path(&dir, "f.rs"), search, replace).unwrap()
        else {
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
            .run(&ResolvedToolInput::ReadFile {
                path: resolved_path(&dir, "f.rs"),
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

    #[test]
    fn edit_symlink_parent_path_fails_before_tool_execution() {
        let dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        fs::create_dir_all(outside.path().join("real")).unwrap();
        symlink_dir(&outside.path().join("real"), &dir.path().join("linked"));

        let err = resolve(
            &project_root(&dir),
            &ToolInput::EditFile {
                path: "linked/file.txt".into(),
                search: "old".into(),
                replace: "new".into(),
            },
        )
        .unwrap_err();
        assert!(matches!(err, PathResolutionError::SymlinkParent { .. }));
    }

    #[test]
    fn execute_approved_accepts_legacy_absolute_payload() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("inside.rs");
        fs::write(&path, "old content").unwrap();

        let tool = tool_in(&dir);
        let payload = format!("{}\x00old content\x00new content", path.display());
        let ToolOutput::EditFile(ef) = tool.execute_approved(&payload).unwrap() else {
            panic!("expected EditFile output");
        };
        assert_eq!(ef.path, path.display().to_string());
        assert_eq!(fs::read_to_string(&path).unwrap(), "new content");
    }

    #[test]
    fn edit_target_symlink_fails_before_tool_execution() {
        let dir = TempDir::new().unwrap();
        let real = dir.path().join("real.txt");
        let link = dir.path().join("link.txt");
        fs::write(&real, "old").unwrap();
        symlink_file(&real, &link);

        let err = resolve(
            &project_root(&dir),
            &ToolInput::EditFile {
                path: "link.txt".into(),
                search: "old".into(),
                replace: "new".into(),
            },
        )
        .unwrap_err();
        assert!(matches!(err, PathResolutionError::SymlinkTarget { .. }));
    }

    #[test]
    fn execute_approved_rejects_payload_path_outside_root() {
        let dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let outside_path = outside.path().join("evil.rs");
        fs::write(&outside_path, "old").unwrap();

        let tool = tool_in(&dir);
        let payload = format!(
            "v2{SEP}{}{SEP}evil.rs{SEP}old{SEP}new",
            outside_path.display()
        );
        let err = tool.execute_approved(&payload).unwrap_err();

        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert_eq!(fs::read_to_string(&outside_path).unwrap(), "old");
    }

    #[test]
    fn execute_approved_rejects_payload_from_another_root() {
        let source_root = TempDir::new().unwrap();
        let target_root = TempDir::new().unwrap();
        let source_file = source_root.path().join("shared.rs");
        fs::write(&source_file, "old").unwrap();
        let source_path = ProjectPath::from_trusted(source_file.clone(), "shared.rs".into());
        let payload = encode_payload(&source_path, "old", "new");

        let tool = tool_in(&target_root);
        let err = tool.execute_approved(&payload).unwrap_err();

        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert_eq!(fs::read_to_string(&source_file).unwrap(), "old");
        assert!(!target_root.path().join("shared.rs").exists());
    }
}
