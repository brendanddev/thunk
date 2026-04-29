// Phase 15.2: vocabulary only. Constructors and callers are added in Phase 15.3.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// A path within the project root, carrying both an execution representation and a
/// display representation.
///
/// ## Invariants
///
/// - `absolute` is canonical (no `.`, `..`, or unresolved symlinks)
/// - `absolute` is within the project root (component-wise, not string-prefix)
/// - `relative` is `absolute` with the root prefix stripped, using `/` separators
/// - `relative` is `"."` when `absolute == root`
/// - No file existence is implied — write targets are representable
///
/// ## Construction in Phase 15.2
///
/// Only `from_trusted` is available. Public constructors that accept raw model-emitted
/// input (with canonicalization and within-root verification) are added in Phase 15.3.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectPath {
    absolute: PathBuf,
    relative: String,
}

impl ProjectPath {
    /// Constructs a `ProjectPath` from pre-validated parts.
    ///
    /// The caller is responsible for upholding all invariants. Use `relative_display`
    /// to compute the `relative` field from a canonical absolute path and root.
    pub(crate) fn from_trusted(absolute: PathBuf, relative: String) -> Self {
        Self { absolute, relative }
    }

    /// Returns the canonical absolute path for execution-layer use (filesystem ops, tool dispatch).
    pub fn absolute(&self) -> &Path {
        &self.absolute
    }

    /// Returns the root-relative display path for model-facing output.
    ///
    /// Uses `/` separators on all platforms. Has no leading `./` or `/`.
    pub fn display(&self) -> &str {
        &self.relative
    }

    /// Consumes this path and returns the owned absolute `PathBuf`.
    pub fn into_path_buf(self) -> PathBuf {
        self.absolute
    }
}

/// A directory scope within the project root, bounding search and listing operations.
///
/// All `ProjectPath` invariants apply, plus:
/// - The path refers to a directory (enforced by Phase 15.3 constructors)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectScope {
    path: ProjectPath,
}

impl ProjectScope {
    /// Constructs a `ProjectScope` from a pre-validated `ProjectPath`.
    ///
    /// The caller is responsible for ensuring `path.absolute()` is a directory.
    pub(crate) fn from_trusted_path(path: ProjectPath) -> Self {
        Self { path }
    }

    /// Returns the underlying `ProjectPath`.
    pub fn as_project_path(&self) -> &ProjectPath {
        &self.path
    }

    /// Returns the root-relative display path for model-facing output.
    pub fn display(&self) -> &str {
        self.path.display()
    }

    /// Returns the canonical absolute path for execution-layer use.
    pub fn absolute(&self) -> &Path {
        self.path.absolute()
    }

    /// Returns true if `path` is equal to or nested within this scope.
    ///
    /// Uses component-aware prefix matching to avoid false positives from paths
    /// that share a string prefix but not a component boundary (e.g., `src_extra`
    /// does not match scope `src`).
    pub fn contains(&self, path: &ProjectPath) -> bool {
        path.absolute().starts_with(self.absolute())
    }
}

/// Computes the root-relative display string for a canonical absolute path.
///
/// Returns `None` if `absolute` is not within `root`.
/// Returns `"."` if `absolute == root`.
///
/// The result always uses `/` separators and has no leading `./`. This is the shared
/// normalization step that Phase 15.3 constructors call after canonicalization and
/// within-root verification.
pub(crate) fn relative_display(absolute: &Path, root: &Path) -> Option<String> {
    let rel = absolute.strip_prefix(root).ok()?;
    if rel == Path::new("") {
        return Some(".".to_string());
    }
    Some(
        rel.components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── relative_display ─────────────────────────────────────────────────────

    #[cfg(unix)]
    #[test]
    fn relative_display_returns_root_relative_path() {
        assert_eq!(
            relative_display(Path::new("/project/src/main.rs"), Path::new("/project")).as_deref(),
            Some("src/main.rs")
        );
    }

    #[cfg(unix)]
    #[test]
    fn relative_display_returns_dot_for_root_itself() {
        assert_eq!(
            relative_display(Path::new("/project"), Path::new("/project")).as_deref(),
            Some(".")
        );
    }

    #[cfg(unix)]
    #[test]
    fn relative_display_returns_none_outside_root() {
        assert!(relative_display(Path::new("/other/file.rs"), Path::new("/project")).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn relative_display_handles_deep_nesting() {
        assert_eq!(
            relative_display(Path::new("/project/a/b/c/d.rs"), Path::new("/project")).as_deref(),
            Some("a/b/c/d.rs")
        );
    }

    #[cfg(unix)]
    #[test]
    fn relative_display_uses_forward_slashes() {
        let result = relative_display(
            Path::new("/project/src/runtime/engine.rs"),
            Path::new("/project"),
        )
        .unwrap();
        assert!(
            !result.contains('\\'),
            "must not contain backslashes: {result}"
        );
        assert!(result.contains('/'));
    }

    // ── ProjectPath ──────────────────────────────────────────────────────────

    #[cfg(unix)]
    fn make_path(abs: &str, rel: &str) -> ProjectPath {
        ProjectPath::from_trusted(PathBuf::from(abs), rel.to_string())
    }

    #[cfg(unix)]
    #[test]
    fn project_path_absolute_returns_stored_value() {
        let p = make_path("/project/src/main.rs", "src/main.rs");
        assert_eq!(p.absolute(), Path::new("/project/src/main.rs"));
    }

    #[cfg(unix)]
    #[test]
    fn project_path_display_returns_relative_string() {
        let p = make_path("/project/src/main.rs", "src/main.rs");
        assert_eq!(p.display(), "src/main.rs");
    }

    #[cfg(unix)]
    #[test]
    fn project_path_into_path_buf_returns_absolute() {
        let abs = PathBuf::from("/project/src/main.rs");
        let p = make_path("/project/src/main.rs", "src/main.rs");
        assert_eq!(p.into_path_buf(), abs);
    }

    #[cfg(unix)]
    #[test]
    fn project_path_equality_on_same_parts() {
        let a = make_path("/project/src/main.rs", "src/main.rs");
        let b = make_path("/project/src/main.rs", "src/main.rs");
        assert_eq!(a, b);
    }

    #[cfg(unix)]
    #[test]
    fn project_path_inequality_on_different_absolute() {
        let a = make_path("/project/src/main.rs", "src/main.rs");
        let b = make_path("/project/src/other.rs", "src/other.rs");
        assert_ne!(a, b);
    }

    // ── ProjectScope ─────────────────────────────────────────────────────────

    #[cfg(unix)]
    fn make_scope(abs: &str, rel: &str) -> ProjectScope {
        ProjectScope::from_trusted_path(make_path(abs, rel))
    }

    #[cfg(unix)]
    #[test]
    fn scope_contains_exact_match() {
        let s = make_scope("/project/src", "src");
        let p = make_path("/project/src", "src");
        assert!(s.contains(&p));
    }

    #[cfg(unix)]
    #[test]
    fn scope_contains_direct_child() {
        let s = make_scope("/project/src", "src");
        let p = make_path("/project/src/main.rs", "src/main.rs");
        assert!(s.contains(&p));
    }

    #[cfg(unix)]
    #[test]
    fn scope_contains_deeply_nested_child() {
        let s = make_scope("/project/src", "src");
        let p = make_path("/project/src/runtime/engine.rs", "src/runtime/engine.rs");
        assert!(s.contains(&p));
    }

    #[cfg(unix)]
    #[test]
    fn scope_does_not_contain_sibling() {
        let s = make_scope("/project/src", "src");
        let p = make_path("/project/tests/main.rs", "tests/main.rs");
        assert!(!s.contains(&p));
    }

    #[cfg(unix)]
    #[test]
    fn scope_does_not_contain_parent() {
        let s = make_scope("/project/src", "src");
        let p = make_path("/project", ".");
        assert!(!s.contains(&p));
    }

    #[cfg(unix)]
    #[test]
    fn scope_boundary_guard_prevents_prefix_collision() {
        // "src_extra" shares the string prefix "src" but is not within scope "src".
        let s = make_scope("/project/src", "src");
        let p = make_path("/project/src_extra/main.rs", "src_extra/main.rs");
        assert!(!s.contains(&p));
    }

    #[cfg(unix)]
    #[test]
    fn scope_display_and_absolute_delegate_to_inner_path() {
        let s = make_scope("/project/src", "src");
        assert_eq!(s.display(), "src");
        assert_eq!(s.absolute(), Path::new("/project/src"));
    }

    #[cfg(unix)]
    #[test]
    fn scope_as_project_path_returns_inner() {
        let s = make_scope("/project/src", "src");
        assert_eq!(s.as_project_path().display(), "src");
        assert_eq!(s.as_project_path().absolute(), Path::new("/project/src"));
    }
}
