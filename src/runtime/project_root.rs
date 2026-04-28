use std::fmt;
use std::path::{Path, PathBuf};

/// A validated, canonical, immutable project root directory.
///
/// Invariants upheld at construction:
/// - path is absolute
/// - path is canonical (no `.`, `..`, or unresolved symlinks)
/// - path refers to an existing directory
///
/// The runtime holds one `ProjectRoot` for the lifetime of a session. No tool
/// may infer or override it.
#[derive(Debug, Clone)]
pub struct ProjectRoot {
    path: PathBuf,
}

#[derive(Debug)]
pub enum ProjectRootError {
    NotADirectory(PathBuf),
    CanonicalizeFailed(PathBuf, std::io::Error),
}

impl fmt::Display for ProjectRootError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotADirectory(p) => {
                write!(f, "project root is not a directory: {}", p.display())
            }
            Self::CanonicalizeFailed(p, e) => {
                write!(f, "cannot canonicalize {}: {e}", p.display())
            }
        }
    }
}

impl std::error::Error for ProjectRootError {}

impl ProjectRoot {
    /// Validates and canonicalizes `path`.
    ///
    /// Fails if the path cannot be canonicalized (covers non-existent paths,
    /// broken symlinks, and permission errors) or if the canonical path is not
    /// a directory.
    pub fn new(path: PathBuf) -> Result<Self, ProjectRootError> {
        let canonical = std::fs::canonicalize(&path)
            .map_err(|e| ProjectRootError::CanonicalizeFailed(path.clone(), e))?;

        if !canonical.is_dir() {
            return Err(ProjectRootError::NotADirectory(canonical));
        }

        Ok(Self { path: canonical })
    }

    /// Returns the canonical absolute path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns an owned clone of the canonical path.
    ///
    /// Use only where ownership is required (e.g., constructing `ToolContext`).
    pub fn as_path_buf(&self) -> PathBuf {
        self.path.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn temp_dir() -> TempDir {
        TempDir::new().unwrap()
    }

    #[test]
    fn valid_directory_constructs_successfully() {
        let dir = temp_dir();
        assert!(ProjectRoot::new(dir.path().to_path_buf()).is_ok());
    }

    #[test]
    fn constructed_path_is_absolute() {
        let dir = temp_dir();
        let root = ProjectRoot::new(dir.path().to_path_buf()).unwrap();
        assert!(root.path().is_absolute());
    }

    #[test]
    fn dot_components_are_resolved() {
        let dir = temp_dir();
        let with_dot = dir.path().join(".").join(".");
        let root = ProjectRoot::new(with_dot).unwrap();
        let s = root.path().display().to_string();
        assert!(!s.contains("/./"), "dot component not resolved: {s}");
    }

    #[test]
    fn path_to_file_is_rejected() {
        let dir = temp_dir();
        let file = dir.path().join("file.txt");
        fs::write(&file, "x").unwrap();
        assert!(matches!(
            ProjectRoot::new(file).unwrap_err(),
            ProjectRootError::NotADirectory(_)
        ));
    }

    #[test]
    fn nonexistent_path_is_rejected() {
        let dir = temp_dir();
        let missing = dir.path().join("does_not_exist");
        assert!(matches!(
            ProjectRoot::new(missing).unwrap_err(),
            ProjectRootError::CanonicalizeFailed(_, _)
        ));
    }

    #[test]
    fn as_path_buf_matches_path() {
        let dir = temp_dir();
        let root = ProjectRoot::new(dir.path().to_path_buf()).unwrap();
        assert_eq!(root.path(), root.as_path_buf().as_path());
    }

    #[test]
    fn relative_dot_path_resolves_to_existing_directory() {
        let root = ProjectRoot::new(PathBuf::from(".")).unwrap();
        assert!(root.path().is_absolute());
        assert!(root.path().is_dir());
    }
}
