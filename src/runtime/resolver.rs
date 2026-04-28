#![allow(dead_code)]

use std::ffi::OsString;
use std::fs;
use std::path::{Component, Path, PathBuf};

use thiserror::Error;

use crate::tools::{ToolError, ToolInput};

use super::{
    project_path::relative_display, ProjectPath, ProjectRoot, ProjectScope, ResolvedToolInput,
};

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PathResolutionError {
    #[error("path '{raw}' escapes project root {}", root.display())]
    EscapesRoot { raw: String, root: PathBuf },

    #[error("path not found: '{raw}'")]
    NotFound { raw: String },

    #[error("path is not a directory: '{raw}'")]
    NotADirectory { raw: String },

    #[error("path '{raw}' uses symlink parent '{component}'")]
    SymlinkParent { raw: String, component: String },

    #[error("path '{raw}' resolves to symlink target {}", target.display())]
    SymlinkTarget { raw: String, target: PathBuf },

    #[error("invalid path '{raw}': {reason}")]
    InvalidPath { raw: String, reason: String },
}

impl From<PathResolutionError> for ToolError {
    fn from(error: PathResolutionError) -> Self {
        match error {
            PathResolutionError::EscapesRoot { raw, root } => ToolError::InvalidInput(format!(
                "path escapes project root: '{raw}' is outside {}",
                root.display()
            )),
            PathResolutionError::NotFound { raw } => {
                ToolError::InvalidInput(format!("path not found: '{raw}'"))
            }
            PathResolutionError::NotADirectory { raw } => {
                ToolError::InvalidInput(format!("path is not a directory: '{raw}'"))
            }
            PathResolutionError::SymlinkParent { raw, component } => ToolError::InvalidInput(
                format!("path uses symlink parent: '{raw}' via '{component}'"),
            ),
            PathResolutionError::SymlinkTarget { raw, target } => ToolError::InvalidInput(format!(
                "path resolves to symlink target: '{raw}' -> {}",
                target.display()
            )),
            PathResolutionError::InvalidPath { raw, reason } => {
                ToolError::InvalidInput(format!("invalid path: '{raw}': {reason}"))
            }
        }
    }
}

pub fn resolve(
    root: &ProjectRoot,
    input: &ToolInput,
) -> Result<ResolvedToolInput, PathResolutionError> {
    match input {
        ToolInput::ReadFile { path } => Ok(ResolvedToolInput::ReadFile {
            path: resolve_read_path(root, path)?,
        }),
        ToolInput::ListDir { path } => Ok(ResolvedToolInput::ListDir {
            path: resolve_scope(root, path)?,
        }),
        ToolInput::SearchCode { query, path } => Ok(ResolvedToolInput::SearchCode {
            query: query.clone(),
            scope: path
                .as_deref()
                .map(|raw| resolve_scope(root, raw))
                .transpose()?,
        }),
        ToolInput::WriteFile { path, content } => Ok(ResolvedToolInput::WriteFile {
            path: resolve_write_path(root, path)?,
            content: content.clone(),
        }),
        ToolInput::EditFile {
            path,
            search,
            replace,
        } => Ok(ResolvedToolInput::EditFile {
            path: resolve_write_path(root, path)?,
            search: search.clone(),
            replace: replace.clone(),
        }),
        ToolInput::GitStatus => Ok(ResolvedToolInput::GitStatus),
        ToolInput::GitDiff => Ok(ResolvedToolInput::GitDiff { path: None }),
        ToolInput::GitLog => Ok(ResolvedToolInput::GitLog),
    }
}

fn resolve_read_path(root: &ProjectRoot, raw: &str) -> Result<ProjectPath, PathResolutionError> {
    let raw_path = Path::new(raw);
    let candidate = if raw_path.is_absolute() {
        raw_path.to_path_buf()
    } else {
        root.path().join(raw_path)
    };

    let canonical = fs::canonicalize(&candidate).map_err(|_| PathResolutionError::NotFound {
        raw: raw.to_string(),
    })?;

    project_path_from_absolute(root, raw, canonical)
}

fn resolve_write_path(root: &ProjectRoot, raw: &str) -> Result<ProjectPath, PathResolutionError> {
    let normalized = normalize_write_path(root, raw)?;
    let relative =
        normalized
            .strip_prefix(root.path())
            .map_err(|_| PathResolutionError::EscapesRoot {
                raw: raw.to_string(),
                root: root.path().to_path_buf(),
            })?;

    let components = relative_components(relative, raw)?;
    let final_path = rebuild_write_target(root, raw, &components)?;

    if !final_path.starts_with(root.path()) {
        return Err(PathResolutionError::EscapesRoot {
            raw: raw.to_string(),
            root: root.path().to_path_buf(),
        });
    }

    match fs::symlink_metadata(&final_path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(PathResolutionError::SymlinkTarget {
                raw: raw.to_string(),
                target: final_path,
            });
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(PathResolutionError::InvalidPath {
                raw: raw.to_string(),
                reason: format!("cannot inspect target {}: {error}", final_path.display()),
            });
        }
    }

    project_path_from_absolute(root, raw, final_path)
}

fn resolve_scope(root: &ProjectRoot, raw: &str) -> Result<ProjectScope, PathResolutionError> {
    let path = resolve_read_path(root, raw)?;
    if !path.absolute().is_dir() {
        return Err(PathResolutionError::NotADirectory {
            raw: raw.to_string(),
        });
    }
    Ok(ProjectScope::from_trusted_path(path))
}

fn project_path_from_absolute(
    root: &ProjectRoot,
    raw: &str,
    absolute: PathBuf,
) -> Result<ProjectPath, PathResolutionError> {
    let relative = relative_display(&absolute, root.path()).ok_or_else(|| {
        PathResolutionError::EscapesRoot {
            raw: raw.to_string(),
            root: root.path().to_path_buf(),
        }
    })?;

    Ok(ProjectPath::from_trusted(absolute, relative))
}

fn normalize_write_path(root: &ProjectRoot, raw: &str) -> Result<PathBuf, PathResolutionError> {
    let raw_path = Path::new(raw);
    if raw_path.is_absolute() {
        normalize_absolute_path(raw_path, raw)
    } else {
        normalize_relative_path(root, raw_path, raw)
    }
}

fn normalize_relative_path(
    root: &ProjectRoot,
    raw_path: &Path,
    raw: &str,
) -> Result<PathBuf, PathResolutionError> {
    let mut normalized = root.path().to_path_buf();
    let boundary = root.path().components().count();

    for component in raw_path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => normalized.push(part),
            Component::ParentDir => {
                if normalized.components().count() == boundary {
                    return Err(PathResolutionError::EscapesRoot {
                        raw: raw.to_string(),
                        root: root.path().to_path_buf(),
                    });
                }
                normalized.pop();
            }
            Component::Prefix(_) | Component::RootDir => {
                return Err(PathResolutionError::InvalidPath {
                    raw: raw.to_string(),
                    reason: "unexpected absolute component in relative path".to_string(),
                });
            }
        }
    }

    if !normalized.starts_with(root.path()) {
        return Err(PathResolutionError::EscapesRoot {
            raw: raw.to_string(),
            root: root.path().to_path_buf(),
        });
    }

    Ok(normalized)
}

fn normalize_absolute_path(path: &Path, raw: &str) -> Result<PathBuf, PathResolutionError> {
    let mut normalized = PathBuf::new();

    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::Normal(part) => normalized.push(part),
            Component::ParentDir => {
                if !normalized.pop() {
                    return Err(PathResolutionError::InvalidPath {
                        raw: raw.to_string(),
                        reason: "path traverses above filesystem root".to_string(),
                    });
                }
            }
        }
    }

    Ok(normalized)
}

fn relative_components(relative: &Path, raw: &str) -> Result<Vec<OsString>, PathResolutionError> {
    let mut components = Vec::new();

    for component in relative.components() {
        match component {
            Component::Normal(part) => components.push(part.to_os_string()),
            Component::CurDir => {}
            other => {
                return Err(PathResolutionError::InvalidPath {
                    raw: raw.to_string(),
                    reason: format!(
                        "unexpected normalized component: {}",
                        other.as_os_str().to_string_lossy()
                    ),
                });
            }
        }
    }

    Ok(components)
}

fn rebuild_write_target(
    root: &ProjectRoot,
    raw: &str,
    components: &[OsString],
) -> Result<PathBuf, PathResolutionError> {
    if components.is_empty() {
        return Ok(root.path().to_path_buf());
    }

    let parent_component_count = components.len().saturating_sub(1);
    let mut current = root.path().to_path_buf();
    let mut first_missing_parent = parent_component_count;

    for (index, component) in components.iter().take(parent_component_count).enumerate() {
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) => {
                let display = relative_display(&current, root.path())
                    .unwrap_or_else(|| component.to_string_lossy().into_owned());

                if metadata.file_type().is_symlink() {
                    return Err(PathResolutionError::SymlinkParent {
                        raw: raw.to_string(),
                        component: display,
                    });
                }

                if !metadata.is_dir() {
                    return Err(PathResolutionError::InvalidPath {
                        raw: raw.to_string(),
                        reason: format!("parent is not a directory: {display}"),
                    });
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                current.pop();
                first_missing_parent = index;
                break;
            }
            Err(error) => {
                return Err(PathResolutionError::InvalidPath {
                    raw: raw.to_string(),
                    reason: format!("cannot inspect parent {}: {error}", current.display()),
                });
            }
        }
    }

    let canonical_parent =
        fs::canonicalize(&current).map_err(|error| PathResolutionError::InvalidPath {
            raw: raw.to_string(),
            reason: format!(
                "cannot canonicalize existing parent {}: {error}",
                current.display()
            ),
        })?;

    if !canonical_parent.starts_with(root.path()) {
        return Err(PathResolutionError::EscapesRoot {
            raw: raw.to_string(),
            root: root.path().to_path_buf(),
        });
    }

    let mut final_path = canonical_parent;
    let remaining_components: Vec<&OsString> = if first_missing_parent < parent_component_count {
        components[first_missing_parent..].iter().collect()
    } else {
        vec![components.last().expect("components is non-empty")]
    };

    for component in remaining_components {
        final_path.push(component);
    }

    Ok(final_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

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

    fn temp_dir() -> TempDir {
        TempDir::new().unwrap()
    }

    fn make_root() -> (TempDir, ProjectRoot) {
        let dir = temp_dir();
        let root = ProjectRoot::new(dir.path().to_path_buf()).unwrap();
        (dir, root)
    }

    fn write_file(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    #[test]
    fn read_relative_path_inside_root() {
        let (_dir, root) = make_root();
        write_file(&root.path().join("src/main.rs"), "fn main() {}\n");

        let resolved = resolve_read_path(&root, "src/main.rs").unwrap();

        assert_eq!(resolved.absolute(), root.path().join("src/main.rs"));
        assert_eq!(resolved.display(), "src/main.rs");
    }

    #[test]
    fn read_absolute_path_inside_root() {
        let (_dir, root) = make_root();
        let file = root.path().join("README.md");
        write_file(&file, "hello\n");

        let resolved = resolve_read_path(&root, file.to_str().unwrap()).unwrap();

        assert_eq!(resolved.absolute(), file);
        assert_eq!(resolved.display(), "README.md");
    }

    #[test]
    fn read_absolute_path_outside_root_is_rejected() {
        let (_dir, root) = make_root();
        let outside = temp_dir();
        let outside_file = outside.path().join("outside.txt");
        write_file(&outside_file, "outside\n");
        let raw = outside_file.display().to_string();

        let err = resolve_read_path(&root, &raw).unwrap_err();

        assert!(matches!(
            err,
            PathResolutionError::EscapesRoot { raw: actual, .. } if actual == raw
        ));
    }

    #[test]
    fn read_parent_escape_is_rejected() {
        let (_dir, root) = make_root();
        let outside_file = root.path().parent().unwrap().join("outside.txt");
        write_file(&outside_file, "outside\n");

        let err = resolve_read_path(&root, "../outside.txt").unwrap_err();

        assert!(matches!(err, PathResolutionError::EscapesRoot { .. }));
        fs::remove_file(outside_file).unwrap();
    }

    #[test]
    fn read_nonexistent_path_is_not_found() {
        let (_dir, root) = make_root();

        let err = resolve_read_path(&root, "missing.txt").unwrap_err();

        assert!(matches!(err, PathResolutionError::NotFound { .. }));
    }

    #[test]
    fn read_symlink_pointing_outside_root_is_rejected() {
        let (_dir, root) = make_root();
        let outside = temp_dir();
        let outside_file = outside.path().join("outside.txt");
        write_file(&outside_file, "outside\n");
        symlink_file(&outside_file, &root.path().join("link.txt"));

        let err = resolve_read_path(&root, "link.txt").unwrap_err();

        assert!(matches!(err, PathResolutionError::EscapesRoot { .. }));
    }

    #[test]
    fn scope_valid_directory() {
        let (_dir, root) = make_root();
        fs::create_dir_all(root.path().join("src/runtime")).unwrap();

        let scope = resolve_scope(&root, "src").unwrap();

        assert_eq!(scope.absolute(), root.path().join("src"));
        assert_eq!(scope.display(), "src");
    }

    #[test]
    fn scope_file_is_not_a_directory() {
        let (_dir, root) = make_root();
        write_file(&root.path().join("notes.txt"), "notes\n");

        let err = resolve_scope(&root, "notes.txt").unwrap_err();

        assert!(matches!(err, PathResolutionError::NotADirectory { .. }));
    }

    #[test]
    fn write_new_file_inside_root() {
        let (_dir, root) = make_root();

        let resolved = resolve_write_path(&root, "new.txt").unwrap();

        assert_eq!(resolved.absolute(), root.path().join("new.txt"));
        assert_eq!(resolved.display(), "new.txt");
    }

    #[test]
    fn write_nested_file_inside_root() {
        let (_dir, root) = make_root();
        fs::create_dir_all(root.path().join("src/bin")).unwrap();

        let resolved = resolve_write_path(&root, "src/bin/tool.rs").unwrap();

        assert_eq!(resolved.absolute(), root.path().join("src/bin/tool.rs"));
        assert_eq!(resolved.display(), "src/bin/tool.rs");
    }

    #[test]
    fn write_parent_escape_is_rejected() {
        let (_dir, root) = make_root();

        let err = resolve_write_path(&root, "../escape.txt").unwrap_err();

        assert!(matches!(err, PathResolutionError::EscapesRoot { .. }));
    }

    #[test]
    fn write_absolute_outside_root_is_rejected() {
        let (_dir, root) = make_root();
        let outside = temp_dir();
        let raw = outside.path().join("outside.txt").display().to_string();

        let err = resolve_write_path(&root, &raw).unwrap_err();

        assert!(matches!(
            err,
            PathResolutionError::EscapesRoot { raw: actual, .. } if actual == raw
        ));
    }

    #[test]
    fn write_parent_symlink_is_rejected() {
        let (_dir, root) = make_root();
        let outside = temp_dir();
        fs::create_dir_all(outside.path().join("real")).unwrap();
        symlink_dir(&outside.path().join("real"), &root.path().join("linked"));

        let err = resolve_write_path(&root, "linked/file.txt").unwrap_err();

        assert!(matches!(err, PathResolutionError::SymlinkParent { .. }));
    }

    #[test]
    fn write_existing_target_symlink_is_rejected() {
        let (_dir, root) = make_root();
        let real = root.path().join("real.txt");
        let link = root.path().join("link.txt");
        write_file(&real, "hello\n");
        symlink_file(&real, &link);

        let err = resolve_write_path(&root, "link.txt").unwrap_err();

        assert!(matches!(err, PathResolutionError::SymlinkTarget { .. }));
    }

    #[test]
    fn write_existing_real_file_is_allowed() {
        let (_dir, root) = make_root();
        let existing = root.path().join("existing.txt");
        write_file(&existing, "hello\n");

        let resolved = resolve_write_path(&root, "existing.txt").unwrap();

        assert_eq!(resolved.absolute(), existing);
        assert_eq!(resolved.display(), "existing.txt");
    }

    #[test]
    fn write_deep_path_normalization() {
        let (_dir, root) = make_root();

        let resolved = resolve_write_path(&root, "./a/./b/../c/../file.txt").unwrap();

        assert_eq!(resolved.absolute(), root.path().join("a/file.txt"));
        assert_eq!(resolved.display(), "a/file.txt");
    }

    #[test]
    fn path_resolution_error_maps_to_structured_tool_error() {
        let tool_error: crate::tools::ToolError = PathResolutionError::EscapesRoot {
            raw: "../secret.txt".into(),
            root: PathBuf::from("/project"),
        }
        .into();

        assert_eq!(
            tool_error.to_string(),
            "invalid tool input: path escapes project root: '../secret.txt' is outside /project"
        );
    }
}
