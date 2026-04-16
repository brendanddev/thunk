use std::fs;
use std::path::Path;

use super::types::{
    FileContentsOutput, ToolError, ToolInput, ToolOutput, ToolSpec,
};
use super::Tool;

/// Maximum bytes read from a single file before truncation.
/// Prevents large files from flooding the model context.
const MAX_BYTES: usize = 100_000;

pub struct ReadFileTool;

impl Tool for ReadFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file",
            description: "Read the contents of a file at the given path.",
            input_hint: "path/to/file.rs",
        }
    }

    fn run(&self, input: &ToolInput) -> Result<ToolOutput, ToolError> {
        let ToolInput::ReadFile { path } = input else {
            return Err(ToolError::InvalidInput(
                "read_file received wrong input variant".into(),
            ));
        };

        let path = Path::new(path);
        let raw = fs::read(path)?;

        let (contents, truncated) = if raw.len() > MAX_BYTES {
            let sliced = &raw[..MAX_BYTES];
            // Back off to the last valid UTF-8 boundary so we don't split a codepoint.
            let boundary = last_utf8_boundary(sliced);
            (String::from_utf8_lossy(&raw[..boundary]).into_owned(), true)
        } else {
            (String::from_utf8_lossy(&raw).into_owned(), false)
        };

        let line_count = contents.lines().count();

        Ok(ToolOutput::FileContents(FileContentsOutput {
            path: path.to_string_lossy().into_owned(),
            contents,
            line_count,
            truncated,
        }))
    }
}

/// Returns the largest index <= limit that sits on a UTF-8 character boundary.
fn last_utf8_boundary(bytes: &[u8]) -> usize {
    let mut i = bytes.len();
    while i > 0 && !is_char_boundary(bytes[i - 1]) {
        i -= 1;
    }
    i
}

/// A byte is a UTF-8 continuation byte if its top two bits are 10.
fn is_char_boundary(b: u8) -> bool {
    (b & 0xC0) != 0x80
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn read(path: &str) -> Result<ToolOutput, ToolError> {
        ReadFileTool.run(&ToolInput::ReadFile { path: path.to_string() })
    }

    #[test]
    fn reads_file_contents() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "line one").unwrap();
        writeln!(f, "line two").unwrap();
        let out = read(f.path().to_str().unwrap()).unwrap();
        let ToolOutput::FileContents(fc) = out else { panic!("wrong variant") };
        assert!(fc.contents.contains("line one"));
        assert_eq!(fc.line_count, 2);
        assert!(!fc.truncated);
    }

    #[test]
    fn returns_io_error_for_missing_file() {
        let err = read("/nonexistent/path/file.rs").unwrap_err();
        assert!(matches!(err, ToolError::Io(_)));
    }
}
