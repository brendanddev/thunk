use std::fs;
use std::path::{Path, PathBuf};

use crate::runtime::{ProjectPath, ResolvedToolInput};

use super::context::ToolContext;
use super::pending::{PendingAction, RiskLevel};
use super::types::{
    ExecutionKind, ToolError, ToolOutput, ToolRunResult, ToolSpec, WriteFileOutput,
};
use super::Tool;

pub struct WriteFileTool {
    root: PathBuf,
}

impl WriteFileTool {
    pub fn new(context: ToolContext) -> Self {
        let root = context.root.canonicalize().unwrap_or(context.root);
        Self { root }
    }
}

const SEP: char = '\x00';
const PAYLOAD_V2: &str = "v2";

fn encode_payload(path: &ProjectPath, content: &str) -> String {
    format!(
        "{PAYLOAD_V2}{SEP}{}{SEP}{}{SEP}{}",
        path.absolute().display(),
        path.display(),
        content
    )
}

struct ApprovedWritePayload {
    absolute: PathBuf,
    display: String,
    content: String,
}

fn decode_payload(payload: &str) -> Option<ApprovedWritePayload> {
    let mut versioned = payload.splitn(4, SEP);
    let first = versioned.next()?;
    if first == PAYLOAD_V2 {
        return Some(ApprovedWritePayload {
            absolute: PathBuf::from(versioned.next()?),
            display: versioned.next()?.to_string(),
            content: versioned.next()?.to_string(),
        });
    }

    let mut legacy = payload.splitn(2, SEP);
    let path = legacy.next()?.to_string();
    let content = legacy.next()?.to_string();
    let absolute = PathBuf::from(&path);
    if !absolute.is_absolute() {
        return None;
    }

    Some(ApprovedWritePayload {
        absolute,
        display: path,
        content,
    })
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

    fn run(&self, input: &ResolvedToolInput) -> Result<ToolRunResult, ToolError> {
        let ResolvedToolInput::WriteFile { path, content } = input else {
            return Err(ToolError::InvalidInput(
                "write_file received wrong input variant".into(),
            ));
        };

        let file_exists = path.absolute().exists();
        let line_count = content.lines().count();

        let (summary, risk) = if file_exists {
            (
                format!("overwrite {} ({line_count} lines)", path.display()),
                RiskLevel::High,
            )
        } else {
            (
                format!("create {} ({line_count} lines)", path.display()),
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
        let ApprovedWritePayload {
            absolute,
            display,
            content,
        } = decode_payload(payload)
            .ok_or_else(|| ToolError::InvalidInput("malformed write_file payload".into()))?;

        validate_approved_path(&self.root, &absolute)?;

        // Parent must exist — we don't create intermediate directories.
        if let Some(parent) = absolute.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                return Err(ToolError::InvalidInput(format!(
                    "parent directory does not exist: {}",
                    parent.display()
                )));
            }
        }

        // Check existence before writing so created reflects the actual outcome.
        let created = !absolute.exists();
        let bytes_written = content.len();
        fs::write(&absolute, &content)?;

        Ok(ToolOutput::WriteFile(WriteFileOutput {
            path: display,
            bytes_written,
            created,
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

    fn tool_in(dir: &TempDir) -> WriteFileTool {
        WriteFileTool::new(ToolContext::new(dir.path().to_path_buf()))
    }

    fn resolved_path(root: &TempDir, relative: &str) -> ProjectPath {
        let absolute = root.path().canonicalize().unwrap().join(relative);
        ProjectPath::from_trusted(absolute, relative.to_string())
    }

    fn project_root(root: &TempDir) -> ProjectRoot {
        ProjectRoot::new(root.path().to_path_buf()).unwrap()
    }

    fn run_write(
        tool: &WriteFileTool,
        path: ProjectPath,
        content: &str,
    ) -> Result<ToolRunResult, ToolError> {
        tool.run(&ResolvedToolInput::WriteFile {
            path,
            content: content.to_string(),
        })
    }

    // run()

    #[test]
    fn run_returns_approval_for_new_file() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        let result = run_write(&tool, resolved_path(&dir, "new.rs"), "pub fn hello() {}").unwrap();
        assert!(matches!(result, ToolRunResult::Approval(_)));
    }

    #[test]
    fn run_sets_medium_risk_for_new_file() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        let ToolRunResult::Approval(pa) =
            run_write(&tool, resolved_path(&dir, "new.rs"), "content").unwrap()
        else {
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
        let ToolRunResult::Approval(pa) =
            run_write(&tool, resolved_path(&dir, "existing.rs"), "new content").unwrap()
        else {
            panic!("expected Approval");
        };
        assert_eq!(pa.risk, RiskLevel::High);
        assert!(pa.summary.contains("overwrite"));
    }

    #[test]
    fn run_summary_includes_path_and_line_count() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        let ToolRunResult::Approval(pa) =
            run_write(&tool, resolved_path(&dir, "out.rs"), "line1\nline2\nline3").unwrap()
        else {
            panic!("expected Approval");
        };
        assert!(pa.summary.contains("out.rs"));
        assert!(pa.summary.contains("3 lines"));
    }

    #[test]
    fn write_path_outside_root_fails_before_tool_execution() {
        let dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let raw = outside.path().join("escape.rs").display().to_string();
        let err = resolve(
            &project_root(&dir),
            &ToolInput::WriteFile {
                path: raw.clone(),
                content: "content".into(),
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            PathResolutionError::EscapesRoot { raw: actual, .. } if actual == raw
        ));
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
    fn write_symlink_parent_path_fails_before_tool_execution() {
        let dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        fs::create_dir_all(outside.path().join("real")).unwrap();
        symlink_dir(&outside.path().join("real"), &dir.path().join("linked"));

        let err = resolve(
            &project_root(&dir),
            &ToolInput::WriteFile {
                path: "linked/file.txt".into(),
                content: "content".into(),
            },
        )
        .unwrap_err();
        assert!(matches!(err, PathResolutionError::SymlinkParent { .. }));
    }

    #[test]
    fn write_target_symlink_fails_before_tool_execution() {
        let dir = TempDir::new().unwrap();
        let real = dir.path().join("real.txt");
        let link = dir.path().join("link.txt");
        fs::write(&real, "hello").unwrap();
        symlink_file(&real, &link);

        let err = resolve(
            &project_root(&dir),
            &ToolInput::WriteFile {
                path: "link.txt".into(),
                content: "content".into(),
            },
        )
        .unwrap_err();
        assert!(matches!(err, PathResolutionError::SymlinkTarget { .. }));
    }

    // execute_approved()

    #[test]
    fn execute_approved_creates_new_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("new.rs");
        assert!(!path.exists());

        let tool = tool_in(&dir);
        let ToolRunResult::Approval(pa) =
            run_write(&tool, resolved_path(&dir, "new.rs"), "pub fn hello() {}").unwrap()
        else {
            panic!("expected Approval");
        };

        let ToolOutput::WriteFile(wf) = tool.execute_approved(&pa.payload).unwrap() else {
            panic!("expected WriteFile output");
        };
        assert_eq!(wf.path, "new.rs");
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
        let ToolRunResult::Approval(pa) =
            run_write(&tool, resolved_path(&dir, "f.rs"), "new content").unwrap()
        else {
            panic!("expected Approval");
        };

        let ToolOutput::WriteFile(wf) = tool.execute_approved(&pa.payload).unwrap() else {
            panic!("expected WriteFile output");
        };
        assert_eq!(wf.path, "f.rs");
        assert!(!wf.created);
        assert_eq!(fs::read_to_string(&path).unwrap(), "new content");
    }

    #[test]
    fn execute_approved_created_flag_reflects_actual_state_at_execution_time() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("new.rs");

        let tool = tool_in(&dir);
        // Propose as new file (doesn't exist yet).
        let ToolRunResult::Approval(pa) =
            run_write(&tool, resolved_path(&dir, "new.rs"), "content").unwrap()
        else {
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
        let payload = encode_payload(
            &ProjectPath::from_trusted(
                dir.path()
                    .canonicalize()
                    .unwrap()
                    .join("nonexistent_dir/file.rs"),
                "nonexistent_dir/file.rs".into(),
            ),
            "content",
        );
        let err = tool.execute_approved(&payload).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn execute_approved_malformed_payload_returns_error() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        // Payload missing the separators entirely.
        let err = tool.execute_approved("").unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn execute_approved_accepts_legacy_absolute_payload() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("out.rs");
        let abs = path.to_str().unwrap();

        let tool = tool_in(&dir);
        let payload = format!("{abs}{SEP}content");
        tool.execute_approved(&payload).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "content");
    }

    #[test]
    fn execute_approved_rejects_payload_path_outside_root() {
        let dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let outside_path = outside.path().join("evil.rs");

        let tool = tool_in(&dir);
        let payload = format!("v2{SEP}{}{SEP}evil.rs{SEP}content", outside_path.display());
        let err = tool.execute_approved(&payload).unwrap_err();

        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(!outside_path.exists());
    }

    #[test]
    fn execute_approved_rejects_payload_from_another_root() {
        let source_root = TempDir::new().unwrap();
        let target_root = TempDir::new().unwrap();
        let source_path =
            ProjectPath::from_trusted(source_root.path().join("shared.rs"), "shared.rs".into());
        let payload = encode_payload(&source_path, "content");

        let tool = tool_in(&target_root);
        let err = tool.execute_approved(&payload).unwrap_err();

        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(!source_root.path().join("shared.rs").exists());
        assert!(!target_root.path().join("shared.rs").exists());
    }
}
