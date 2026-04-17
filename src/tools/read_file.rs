use std::fs;

use super::context::ToolContext;
use super::types::{FileContentsOutput, ToolError, ToolInput, ToolOutput, ToolRunResult, ToolSpec};
use super::Tool;

/// Maximum lines of file content injected into the conversation per read.
/// Files with more lines are truncated; the metadata line reports total vs shown.
const MAX_LINES: usize = 200;

pub struct ReadFileTool {
    context: ToolContext,
}

impl ReadFileTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

impl Tool for ReadFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file",
            description: "Read the contents of a file at the given path.",
            input_hint: "path/to/file.rs",
        }
    }

    fn run(&self, input: &ToolInput) -> Result<ToolRunResult, ToolError> {
        let ToolInput::ReadFile { path } = input else {
            return Err(ToolError::InvalidInput(
                "read_file received wrong input variant".into(),
            ));
        };

        let path = self.context.resolve(path);
        let raw = fs::read(&path)?;
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
                path: path.to_string_lossy().into_owned(),
                contents,
                total_lines,
                truncated,
            },
        )))
    }
}


#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn read(path: &str) -> Result<ToolRunResult, ToolError> {
        ReadFileTool::new(ToolContext::new(PathBuf::from(".")))
            .run(&ToolInput::ReadFile { path: path.to_string() })
    }

    #[test]
    fn reads_file_contents() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "line one").unwrap();
        writeln!(f, "line two").unwrap();
        let out = read(f.path().to_str().unwrap()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::FileContents(fc)) = out else { panic!("expected Immediate(FileContents)") };
        assert!(fc.contents.contains("line one"));
        assert_eq!(fc.total_lines, 2);
        assert!(!fc.truncated);
    }

    #[test]
    fn truncates_at_line_cap_and_reports_total() {
        let mut f = NamedTempFile::new().unwrap();
        // Write MAX_LINES + 5 lines (205 total)
        for i in 0..205 {
            writeln!(f, "line {i}").unwrap();
        }
        let out = read(f.path().to_str().unwrap()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::FileContents(fc)) = out else { panic!("expected Immediate(FileContents)") };
        assert!(fc.truncated);
        assert_eq!(fc.total_lines, 205);
        // contents must have exactly MAX_LINES lines
        assert_eq!(fc.contents.lines().count(), MAX_LINES);
        assert!(fc.contents.contains("line 0"));
        assert!(!fc.contents.contains("line 200")); // line 200 is the 201st line, beyond cap
    }

    #[test]
    fn returns_io_error_for_missing_file() {
        let err = read("/nonexistent/path/file.rs").unwrap_err();
        assert!(matches!(err, ToolError::Io(_)));
    }
}
