/// Directory names excluded from all tool output: snapshots, searches, and directory listings.
/// Exact name match only — no pattern matching, no recursion changes.
pub(crate) const DEFAULT_SKIP_DIRS: &[&str] =
    &[".git", ".hg", "build", "dist", "node_modules", "target"];
