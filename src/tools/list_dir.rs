use std::fs;

use crate::runtime::ResolvedToolInput;

use super::types::{
    DirEntry, DirectoryListingOutput, EntryKind, ExecutionKind, ToolError, ToolOutput,
    ToolRunResult, ToolSpec,
};
use super::Tool;

pub struct ListDirTool;

impl ListDirTool {
    pub fn new() -> Self {
        Self
    }
}

impl Tool for ListDirTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_dir",
            description: "List the immediate contents of a directory.",
            input_hint: "path/to/dir",
            execution_kind: ExecutionKind::Immediate,
            default_risk: None,
        }
    }

    fn run(&self, input: &ResolvedToolInput) -> Result<ToolRunResult, ToolError> {
        let ResolvedToolInput::ListDir { path } = input else {
            return Err(ToolError::InvalidInput(
                "list_dir received wrong input variant".into(),
            ));
        };

        let read = fs::read_dir(path.absolute())?;

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
            b_is_dir.cmp(&a_is_dir).then_with(|| a.name.cmp(&b.name))
        });

        Ok(ToolRunResult::Immediate(ToolOutput::DirectoryListing(
            DirectoryListingOutput {
                path: path.display().to_string(),
                entries,
            },
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{ProjectPath, ProjectScope};
    use std::fs;
    use tempfile::TempDir;

    fn resolved_scope(root: &TempDir, relative: &str) -> ProjectScope {
        let root_absolute = root.path().canonicalize().unwrap();
        let absolute = if relative == "." {
            root_absolute
        } else {
            root_absolute.join(relative)
        };
        let path = ProjectPath::from_trusted(absolute, relative.to_string());
        ProjectScope::from_trusted_path(path)
    }

    fn list(root: &TempDir, relative: &str) -> Result<ToolRunResult, ToolError> {
        ListDirTool::new().run(&ResolvedToolInput::ListDir {
            path: resolved_scope(root, relative),
        })
    }

    #[test]
    fn lists_files_and_dirs() {
        let root = TempDir::new().unwrap();
        fs::write(root.path().join("a.rs"), "").unwrap();
        fs::create_dir(root.path().join("subdir")).unwrap();

        let result = list(&root, ".").unwrap();
        let ToolRunResult::Immediate(ToolOutput::DirectoryListing(dl)) = result else {
            panic!("expected Immediate(DirectoryListing)")
        };

        assert_eq!(dl.path, ".");
        assert_eq!(dl.entries.len(), 2);
        // Directories come first
        assert_eq!(dl.entries[0].name, "subdir");
        assert_eq!(dl.entries[0].kind, EntryKind::Dir);
        assert_eq!(dl.entries[1].name, "a.rs");
        assert_eq!(dl.entries[1].kind, EntryKind::File);
    }

    #[test]
    fn returns_io_error_for_missing_dir() {
        let root = TempDir::new().unwrap();
        let err = list(&root, "missing").unwrap_err();
        assert!(matches!(err, ToolError::Io(_)));
    }
}
