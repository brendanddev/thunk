use std::fs;

use crate::runtime::ResolvedToolInput;

use super::context::ToolContext;
use super::types::{
    ExecutionKind, FileContentsOutput, ToolError, ToolOutput, ToolRunResult, ToolSpec,
};
use super::Tool;

/// Maximum lines of file content injected into the conversation per read.
/// Files with more lines are truncated; the metadata line reports total vs shown.
const MAX_LINES: usize = 200;

pub struct ReadFileTool;

impl ReadFileTool {
    pub fn new(_context: ToolContext) -> Self {
        Self
    }
}

impl Tool for ReadFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file",
            description: "Read the contents of a file at the given path.",
            input_hint: "path/to/file.rs",
            execution_kind: ExecutionKind::Immediate,
            default_risk: None,
        }
    }

    fn run(&self, input: &ResolvedToolInput) -> Result<ToolRunResult, ToolError> {
        let ResolvedToolInput::ReadFile { path } = input else {
            return Err(ToolError::InvalidInput(
                "read_file received wrong input variant".into(),
            ));
        };

        let raw = fs::read(path.absolute())?;
        let full = String::from_utf8_lossy(&raw).into_owned();
        let total_lines = full.lines().count();

        let (contents, truncated) = if total_lines > MAX_LINES {
            let shown = full.lines().take(MAX_LINES).collect::<Vec<_>>().join("\n");
            (shown, true)
        } else {
            (full, false)
        };

        Ok(ToolRunResult::Immediate(ToolOutput::FileContents(
            FileContentsOutput {
                path: path.display().to_string(),
                contents,
                total_lines,
                truncated,
            },
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::ProjectPath;
    use std::fs;
    use tempfile::TempDir;

    fn resolved_path(root: &TempDir, relative: &str) -> ProjectPath {
        let absolute = root.path().canonicalize().unwrap().join(relative);
        ProjectPath::from_trusted(absolute, relative.to_string())
    }

    fn read(root: &TempDir, relative: &str) -> Result<ToolRunResult, ToolError> {
        ReadFileTool::new(ToolContext::new(root.path().to_path_buf())).run(
            &ResolvedToolInput::ReadFile {
                path: resolved_path(root, relative),
            },
        )
    }

    #[test]
    fn reads_file_contents() {
        let root = TempDir::new().unwrap();
        fs::write(root.path().join("notes.txt"), "line one\nline two\n").unwrap();

        let out = read(&root, "notes.txt").unwrap();
        let ToolRunResult::Immediate(ToolOutput::FileContents(fc)) = out else {
            panic!("expected Immediate(FileContents)")
        };
        assert_eq!(fc.path, "notes.txt");
        assert!(fc.contents.contains("line one"));
        assert_eq!(fc.total_lines, 2);
        assert!(!fc.truncated);
    }

    #[test]
    fn truncates_at_line_cap_and_reports_total() {
        let root = TempDir::new().unwrap();
        let contents = (0..205).map(|i| format!("line {i}\n")).collect::<String>();
        fs::write(root.path().join("big.txt"), contents).unwrap();

        let out = read(&root, "big.txt").unwrap();
        let ToolRunResult::Immediate(ToolOutput::FileContents(fc)) = out else {
            panic!("expected Immediate(FileContents)")
        };
        assert_eq!(fc.path, "big.txt");
        assert!(fc.truncated);
        assert_eq!(fc.total_lines, 205);
        assert_eq!(fc.contents.lines().count(), MAX_LINES);
        assert!(fc.contents.contains("line 0"));
        assert!(!fc.contents.contains("line 200"));
    }

    #[test]
    fn returns_io_error_for_missing_file() {
        let root = TempDir::new().unwrap();
        let err = read(&root, "missing.rs").unwrap_err();
        assert!(matches!(err, ToolError::Io(_)));
    }
}
