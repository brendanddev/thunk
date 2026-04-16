use std::fs;

use super::context::ToolContext;
use super::types::{DirEntry, DirectoryListingOutput, EntryKind, ToolError, ToolInput, ToolOutput, ToolRunResult, ToolSpec};
use super::Tool;

pub struct ListDirTool {
    context: ToolContext,
}

impl ListDirTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

impl Tool for ListDirTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_dir",
            description: "List the immediate contents of a directory.",
            input_hint: "path/to/dir",
        }
    }

    fn run(&self, input: &ToolInput) -> Result<ToolRunResult, ToolError> {
        let ToolInput::ListDir { path } = input else {
            return Err(ToolError::InvalidInput(
                "list_dir received wrong input variant".into(),
            ));
        };

        let dir = self.context.resolve(path);
        let read = fs::read_dir(&dir)?;

        let mut entries: Vec<DirEntry> = read
            .filter_map(|entry| entry.ok())
            .map(|entry| {
                let meta = entry.metadata().ok();
                let kind = if let Some(ref m) = meta {
                    if m.is_symlink() {
                        EntryKind::Symlink
                    } else if m.is_dir() {
                        EntryKind::Dir
                    } else {
                        EntryKind::File
                    }
                } else {
                    EntryKind::File
                };
                let size_bytes = meta.as_ref().filter(|m| m.is_file()).map(|m| m.len());
                DirEntry {
                    name: entry.file_name().to_string_lossy().into_owned(),
                    kind,
                    size_bytes,
                }
            })
            .collect();

        // Directories first, then files; alphabetical within each group.
        entries.sort_by(|a, b| {
            let a_is_dir = a.kind == EntryKind::Dir;
            let b_is_dir = b.kind == EntryKind::Dir;
            b_is_dir
                .cmp(&a_is_dir)
                .then_with(|| a.name.cmp(&b.name))
        });

        Ok(ToolRunResult::Immediate(ToolOutput::DirectoryListing(
            DirectoryListingOutput {
                path: dir.to_string_lossy().into_owned(),
                entries,
            },
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn list(path: &str) -> Result<ToolRunResult, ToolError> {
        ListDirTool::new(ToolContext::new(PathBuf::from("."))).run(&ToolInput::ListDir {
            path: path.to_string(),
        })
    }

    #[test]
    fn lists_files_and_dirs() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("a.rs"), "").unwrap();
        fs::create_dir(tmp.path().join("subdir")).unwrap();

        let result = list(tmp.path().to_str().unwrap()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::DirectoryListing(dl)) = result else {
            panic!("expected Immediate(DirectoryListing)")
        };

        assert_eq!(dl.entries.len(), 2);
        // Directories come first
        assert_eq!(dl.entries[0].name, "subdir");
        assert_eq!(dl.entries[0].kind, EntryKind::Dir);
        assert_eq!(dl.entries[1].name, "a.rs");
        assert_eq!(dl.entries[1].kind, EntryKind::File);
    }

    #[test]
    fn returns_io_error_for_missing_dir() {
        let err = list("/nonexistent/path/dir").unwrap_err();
        assert!(matches!(err, ToolError::Io(_)));
    }
}
