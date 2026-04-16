use std::path::{Path, PathBuf};

/// Carries project-level context into the tool layer.
/// Tools use this to resolve relative paths against the project root
/// rather than against the process working directory.
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub root: PathBuf,
}

impl ToolContext {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Resolves a path argument from the model: relative paths are joined
    /// against the project root; absolute paths pass through unchanged.
    pub fn resolve(&self, path: &str) -> PathBuf {
        let p = Path::new(path);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.root.join(p)
        }
    }
}
