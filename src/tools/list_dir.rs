use std::fs;

use crate::dirs::DEFAULT_SKIP_DIRS;
use crate::runtime::ResolvedToolInput;

use super::types::{
    DirEntry, DirectoryListingOutput, EntryKind, ExecutionKind, ToolError, ToolOutput,
    ToolRunResult, ToolSpec,
};
use super::Tool;

const MAX_ENTRIES: usize = 200;

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
            .filter(|e| !(e.kind == EntryKind::Dir && DEFAULT_SKIP_DIRS.contains(&e.name.as_str())))
            .collect();

        // Directories first, then files; alphabetical within each group.
        entries.sort_by(|a, b| {
            let a_is_dir = a.kind == EntryKind::Dir;
            let b_is_dir = b.kind == EntryKind::Dir;
            b_is_dir.cmp(&a_is_dir).then_with(|| a.name.cmp(&b.name))
        });

        let total_entries = entries.len();
        let truncated = total_entries > MAX_ENTRIES;
        if truncated {
            entries.truncate(MAX_ENTRIES);
        }

        Ok(ToolRunResult::Immediate(ToolOutput::DirectoryListing(
            DirectoryListingOutput {
                path: path.display().to_string(),
                entries,
                truncated,
                total_entries,
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

    #[test]
    fn small_directory_returns_full_output() {
        let root = TempDir::new().unwrap();
        for i in 0..10 {
            fs::write(root.path().join(format!("file{i}.txt")), "").unwrap();
        }

        let result = list(&root, ".").unwrap();
        let ToolRunResult::Immediate(ToolOutput::DirectoryListing(dl)) = result else {
            panic!("expected Immediate(DirectoryListing)")
        };

        assert_eq!(dl.entries.len(), 10);
        assert_eq!(dl.total_entries, 10);
        assert!(!dl.truncated);
    }

    #[test]
    fn large_directory_is_capped_at_max_entries() {
        let root = TempDir::new().unwrap();
        for i in 0..=MAX_ENTRIES {
            fs::write(root.path().join(format!("file{i:04}.txt")), "").unwrap();
        }

        let result = list(&root, ".").unwrap();
        let ToolRunResult::Immediate(ToolOutput::DirectoryListing(dl)) = result else {
            panic!("expected Immediate(DirectoryListing)")
        };

        assert!(dl.truncated);
        assert_eq!(dl.entries.len(), MAX_ENTRIES);
        assert_eq!(dl.total_entries, MAX_ENTRIES + 1);
    }

    #[test]
    fn skips_noisy_directories() {
        let root = TempDir::new().unwrap();
        fs::create_dir(root.path().join("node_modules")).unwrap();
        fs::create_dir(root.path().join("target")).unwrap();
        fs::create_dir(root.path().join("src")).unwrap();
        fs::write(root.path().join("Cargo.toml"), "").unwrap();

        let result = list(&root, ".").unwrap();
        let ToolRunResult::Immediate(ToolOutput::DirectoryListing(dl)) = result else {
            panic!("expected Immediate(DirectoryListing)")
        };

        let names: Vec<&str> = dl.entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"src"));
        assert!(names.contains(&"Cargo.toml"));
        assert!(!names.contains(&"node_modules"));
        assert!(!names.contains(&"target"));
    }

    #[test]
    fn capped_output_is_deterministic() {
        let root = TempDir::new().unwrap();
        for i in 0..=MAX_ENTRIES {
            fs::write(root.path().join(format!("file{i:04}.txt")), "").unwrap();
        }

        let r1 = list(&root, ".").unwrap();
        let r2 = list(&root, ".").unwrap();

        let ToolRunResult::Immediate(ToolOutput::DirectoryListing(dl1)) = r1 else {
            panic!()
        };
        let ToolRunResult::Immediate(ToolOutput::DirectoryListing(dl2)) = r2 else {
            panic!()
        };

        let names1: Vec<&str> = dl1.entries.iter().map(|e| e.name.as_str()).collect();
        let names2: Vec<&str> = dl2.entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names1, names2);
    }
}
