// Phase 15.6.1: bounded structure builder only. Runtime integration lands later.
#![allow(dead_code)]

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::project_path::relative_display;
use super::ProjectRoot;
use crate::dirs::DEFAULT_SKIP_DIRS;

pub(crate) const MAX_SNAPSHOT_DEPTH: u8 = 2;
pub(crate) const MAX_SNAPSHOT_NODES: usize = 40;
const IMPORTANT_TOP_LEVEL_FILES: &[&str] = &[
    "Cargo.toml",
    "README",
    "README.md",
    "README.txt",
    "README.rst",
    "package.json",
    "pyproject.toml",
    "go.mod",
    "config.toml",
    "tsconfig.json",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProjectStructureSnapshot {
    pub entries: Vec<ProjectStructureEntry>,
    pub important_files: Vec<String>,
    pub max_depth: u8,
    pub max_nodes: usize,
    pub truncated: bool,
}

impl ProjectStructureSnapshot {
    pub(crate) fn build(root: &ProjectRoot) -> io::Result<Self> {
        build_snapshot(root.path())
    }
}

#[derive(Debug, Default)]
pub(crate) struct ProjectStructureSnapshotCache {
    snapshot: Option<ProjectStructureSnapshot>,
}

impl ProjectStructureSnapshotCache {
    pub(crate) fn get_or_build(
        &mut self,
        root: &ProjectRoot,
    ) -> io::Result<&ProjectStructureSnapshot> {
        if self.snapshot.is_none() {
            self.snapshot = Some(ProjectStructureSnapshot::build(root)?);
        }
        Ok(self
            .snapshot
            .as_ref()
            .expect("snapshot cache must be populated after build"))
    }

    pub(crate) fn invalidate(&mut self) {
        self.snapshot = None;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProjectStructureEntry {
    pub path: String,
    pub depth: u8,
    pub kind: ProjectStructureEntryKind,
    pub important: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProjectStructureEntryKind {
    File,
    Dir,
    Symlink,
}

#[derive(Debug, Clone)]
struct CandidateEntry {
    absolute: PathBuf,
    path: String,
    kind: ProjectStructureEntryKind,
    important: bool,
}

impl CandidateEntry {
    fn into_snapshot_entry(self, depth: u8) -> ProjectStructureEntry {
        ProjectStructureEntry {
            path: self.path,
            depth,
            kind: self.kind,
            important: self.important,
        }
    }
}

fn build_snapshot(root: &Path) -> io::Result<ProjectStructureSnapshot> {
    let top_level = read_entries(root, root, 1)?;
    let important_files = top_level
        .iter()
        .filter(|entry| entry.important)
        .map(|entry| entry.path.clone())
        .collect();

    let mut entries = Vec::new();
    let mut truncated = false;

    for entry in &top_level {
        if entries.len() == MAX_SNAPSHOT_NODES {
            truncated = true;
            break;
        }
        entries.push(entry.clone().into_snapshot_entry(1));
    }

    if !truncated {
        'dirs: for entry in &top_level {
            if entry.kind != ProjectStructureEntryKind::Dir {
                continue;
            }

            let children = read_entries(entry.absolute.as_path(), root, 2)?;
            for child in children {
                if entries.len() == MAX_SNAPSHOT_NODES {
                    truncated = true;
                    break 'dirs;
                }
                entries.push(child.into_snapshot_entry(2));
            }
        }
    }

    Ok(ProjectStructureSnapshot {
        entries,
        important_files,
        max_depth: MAX_SNAPSHOT_DEPTH,
        max_nodes: MAX_SNAPSHOT_NODES,
        truncated,
    })
}

fn read_entries(dir: &Path, root: &Path, depth: u8) -> io::Result<Vec<CandidateEntry>> {
    let read = fs::read_dir(dir)?;
    let mut entries = Vec::new();

    for item in read {
        let item = match item {
            Ok(item) => item,
            Err(_) => continue,
        };

        let file_type = match item.file_type() {
            Ok(file_type) => file_type,
            Err(_) => continue,
        };

        let kind = if file_type.is_symlink() {
            ProjectStructureEntryKind::Symlink
        } else if file_type.is_dir() {
            ProjectStructureEntryKind::Dir
        } else {
            ProjectStructureEntryKind::File
        };

        let name = item.file_name().to_string_lossy().into_owned();
        if matches!(kind, ProjectStructureEntryKind::Dir)
            && DEFAULT_SKIP_DIRS.contains(&name.as_str())
        {
            continue;
        }

        let absolute = item.path();
        let Some(path) = relative_display(&absolute, root) else {
            continue;
        };

        entries.push(CandidateEntry {
            absolute,
            path,
            kind,
            important: depth == 1
                && matches!(kind, ProjectStructureEntryKind::File)
                && is_important_top_level_file(&name),
        });
    }

    entries.sort_by(|a, b| {
        entry_kind_rank(a.kind)
            .cmp(&entry_kind_rank(b.kind))
            .then_with(|| a.path.cmp(&b.path))
    });

    Ok(entries)
}

fn entry_kind_rank(kind: ProjectStructureEntryKind) -> u8 {
    match kind {
        ProjectStructureEntryKind::Dir => 0,
        ProjectStructureEntryKind::File => 1,
        ProjectStructureEntryKind::Symlink => 2,
    }
}

fn is_important_top_level_file(name: &str) -> bool {
    IMPORTANT_TOP_LEVEL_FILES.contains(&name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn build_in(dir: &TempDir) -> ProjectStructureSnapshot {
        let root = ProjectRoot::new(dir.path().to_path_buf()).unwrap();
        ProjectStructureSnapshot::build(&root).unwrap()
    }

    fn entry_paths(snapshot: &ProjectStructureSnapshot) -> Vec<&str> {
        snapshot
            .entries
            .iter()
            .map(|entry| entry.path.as_str())
            .collect()
    }

    #[test]
    fn snapshot_includes_top_level_files_and_directories() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"demo\"\n",
        )
        .unwrap();
        fs::write(dir.path().join("notes.txt"), "hello\n").unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::create_dir_all(dir.path().join("docs")).unwrap();
        fs::write(dir.path().join("src").join("lib.rs"), "pub fn demo() {}\n").unwrap();
        fs::write(dir.path().join("docs").join("guide.md"), "# Guide\n").unwrap();

        let snapshot = build_in(&dir);
        let paths = entry_paths(&snapshot);

        assert!(paths.contains(&"Cargo.toml"));
        assert!(paths.contains(&"notes.txt"));
        assert!(paths.contains(&"src"));
        assert!(paths.contains(&"docs"));
        assert!(paths.contains(&"src/lib.rs"));
        assert!(paths.contains(&"docs/guide.md"));
        assert!(
            snapshot
                .entries
                .iter()
                .all(|entry| !entry.path.starts_with('/')),
            "snapshot paths must be project-relative: {:?}",
            snapshot.entries
        );
    }

    #[test]
    fn snapshot_respects_depth_bound() {
        let dir = TempDir::new().unwrap();
        fs::create_dir_all(dir.path().join("src/nested/deeper")).unwrap();
        fs::write(dir.path().join("src").join("lib.rs"), "pub fn demo() {}\n").unwrap();
        fs::write(
            dir.path().join("src/nested/deeper").join("file.rs"),
            "pub fn hidden() {}\n",
        )
        .unwrap();

        let snapshot = build_in(&dir);
        let paths = entry_paths(&snapshot);

        assert!(snapshot.entries.iter().all(|entry| entry.depth <= 2));
        assert!(paths.contains(&"src"));
        assert!(paths.contains(&"src/lib.rs"));
        assert!(paths.contains(&"src/nested"));
        assert!(!paths.contains(&"src/nested/deeper"));
        assert!(!paths.contains(&"src/nested/deeper/file.rs"));
    }

    #[test]
    fn snapshot_respects_node_cap() {
        let dir = TempDir::new().unwrap();
        for i in 0..45 {
            let path = dir.path().join(format!("file_{i:02}.txt"));
            fs::write(path, "x\n").unwrap();
        }

        let snapshot = build_in(&dir);
        let paths = entry_paths(&snapshot);

        assert_eq!(snapshot.entries.len(), MAX_SNAPSHOT_NODES);
        assert!(snapshot.truncated);
        assert!(paths.contains(&"file_00.txt"));
        assert!(paths.contains(&"file_39.txt"));
        assert!(!paths.contains(&"file_44.txt"));
    }

    #[test]
    fn snapshot_ordering_is_deterministic() {
        let dir = TempDir::new().unwrap();
        fs::create_dir_all(dir.path().join("z_dir")).unwrap();
        fs::create_dir_all(dir.path().join("a_dir")).unwrap();
        fs::write(dir.path().join("b.txt"), "b\n").unwrap();
        fs::write(dir.path().join("a.txt"), "a\n").unwrap();
        fs::write(dir.path().join("a_dir").join("z.log"), "z\n").unwrap();
        fs::write(dir.path().join("a_dir").join("a.log"), "a\n").unwrap();
        fs::write(dir.path().join("z_dir").join("z.log"), "z\n").unwrap();
        fs::write(dir.path().join("z_dir").join("a.log"), "a\n").unwrap();

        let first = build_in(&dir);
        let second = build_in(&dir);
        let first_paths = entry_paths(&first);

        assert_eq!(first, second);
        assert_eq!(
            first_paths,
            vec![
                "a_dir",
                "z_dir",
                "a.txt",
                "b.txt",
                "a_dir/a.log",
                "a_dir/z.log",
                "z_dir/a.log",
                "z_dir/z.log",
            ]
        );
    }

    #[test]
    fn snapshot_detects_important_files() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"demo\"\n",
        )
        .unwrap();
        fs::write(dir.path().join("README.md"), "# Demo\n").unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src").join("lib.rs"), "pub fn demo() {}\n").unwrap();

        let snapshot = build_in(&dir);

        assert_eq!(snapshot.important_files, vec!["Cargo.toml", "README.md"]);
        assert!(snapshot
            .entries
            .iter()
            .find(|entry| entry.path == "Cargo.toml")
            .is_some_and(|entry| entry.important));
        assert!(snapshot
            .entries
            .iter()
            .find(|entry| entry.path == "README.md")
            .is_some_and(|entry| entry.important));
        assert!(snapshot
            .entries
            .iter()
            .find(|entry| entry.path == "src")
            .is_some_and(|entry| !entry.important));
    }

    #[test]
    fn snapshot_ignores_noisy_directories() {
        let dir = TempDir::new().unwrap();
        fs::create_dir_all(dir.path().join(".git")).unwrap();
        fs::create_dir_all(dir.path().join("target")).unwrap();
        fs::create_dir_all(dir.path().join("node_modules")).unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(dir.path().join(".git").join("config"), "[core]\n").unwrap();
        fs::write(dir.path().join("target").join("build.log"), "done\n").unwrap();
        fs::write(
            dir.path().join("node_modules").join("package.json"),
            "{ }\n",
        )
        .unwrap();
        fs::write(dir.path().join("src").join("lib.rs"), "pub fn demo() {}\n").unwrap();

        let snapshot = build_in(&dir);
        let paths = entry_paths(&snapshot);

        assert!(paths.contains(&"src"));
        assert!(paths.contains(&"src/lib.rs"));
        assert!(!paths.contains(&".git"));
        assert!(!paths.contains(&".git/config"));
        assert!(!paths.contains(&"target"));
        assert!(!paths.contains(&"target/build.log"));
        assert!(!paths.contains(&"node_modules"));
        assert!(!paths.contains(&"node_modules/package.json"));
    }
}
