use std::path::{Path, PathBuf};

use crate::error::{ParamsError, Result};
use crate::safety::{self, ProjectPathKind};

use super::types::HoverInput;

pub(super) fn resolve_input_path(requested: &str) -> Result<PathBuf> {
    Ok(
        safety::inspect_project_path("lsp_file", requested, ProjectPathKind::File, false)?
            .resolved_path,
    )
}

pub(super) fn validate_rust_file(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Err(ParamsError::Config(format!(
            "{} is not a file",
            path.display()
        )));
    }

    if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
        return Err(ParamsError::Config(
            "The first LSP slice currently supports Rust `.rs` files only".to_string(),
        ));
    }

    Ok(())
}

pub(super) fn parse_hover_input(requested: &str) -> Result<HoverInput> {
    let Some((path_part, line_part, col_part)) = split_path_line_col(requested) else {
        return Err(ParamsError::Config(
            "Expected <file>:<line>:<col>, for example src/main.rs:12:8".to_string(),
        ));
    };

    let line = line_part
        .parse::<usize>()
        .map_err(|_| ParamsError::Config("Hover line must be a positive integer".to_string()))?;
    let column = col_part
        .parse::<usize>()
        .map_err(|_| ParamsError::Config("Hover column must be a positive integer".to_string()))?;

    if line == 0 || column == 0 {
        return Err(ParamsError::Config(
            "Hover line and column are 1-based and must be greater than 0".to_string(),
        ));
    }

    Ok(HoverInput {
        path: resolve_input_path(path_part)?,
        line,
        column,
    })
}

fn split_path_line_col(input: &str) -> Option<(&str, &str, &str)> {
    let (path_and_line, col) = input.rsplit_once(':')?;
    let (path, line) = path_and_line.rsplit_once(':')?;
    Some((path, line, col))
}

pub(super) fn display_path(cwd: &Path, path: &Path) -> String {
    path.strip_prefix(cwd)
        .ok()
        .and_then(|p| p.to_str())
        .map(|p| p.to_string())
        .unwrap_or_else(|| path.display().to_string())
}

pub(super) fn find_rust_project_root(path: &Path) -> Result<PathBuf> {
    for ancestor in path.ancestors() {
        if ancestor.join("Cargo.toml").exists() {
            return Ok(ancestor.to_path_buf());
        }
    }

    Err(ParamsError::Config(format!(
        "No Cargo.toml found above {}. LSP diagnostics currently expect a Rust project.",
        path.display()
    )))
}

pub(super) fn path_to_file_uri(path: &Path) -> String {
    let path = path.to_string_lossy();
    let escaped = path
        .replace('%', "%25")
        .replace(' ', "%20")
        .replace('#', "%23")
        .replace('?', "%3F");
    format!("file://{escaped}")
}

pub(super) fn file_uri_to_path(uri: &str) -> Option<PathBuf> {
    let path = uri.strip_prefix("file://")?;
    let decoded = path
        .replace("%20", " ")
        .replace("%23", "#")
        .replace("%3F", "?")
        .replace("%25", "%");
    Some(PathBuf::from(decoded))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn builds_file_uri() {
        let uri = path_to_file_uri(Path::new("/tmp/hello world.rs"));
        assert_eq!(uri, "file:///tmp/hello%20world.rs");
    }

    #[test]
    fn splits_path_line_column_triplet() {
        let parts = split_path_line_col("src/main.rs:12:8");
        assert_eq!(parts, Some(("src/main.rs", "12", "8")));
    }
}
