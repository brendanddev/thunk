// src/tools/search.rs
//
// Search tool — find text across all files in the current directory.
//
// This is what makes the model useful for navigating unfamiliar codebases.
// Instead of you having to manually find which file contains a function,
// the model can search for it directly.
//
// Usage in model response: [search: fn authenticate]

use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;

use super::{Tool, ToolRunResult};
use crate::error::{ParamsError, Result};
use crate::safety;

// File extensions we'll search — avoids binary files and build artifacts
const SEARCHABLE_EXTENSIONS: &[&str] = &[
    "rs", "py", "ts", "tsx", "js", "jsx", "go", "c", "cpp", "h", "java", "kt", "swift", "rb",
    "php", "cs", "toml", "yaml", "yml", "json", "md", "txt", "sh", "env", "sql",
];

pub struct SearchCode;

impl Tool for SearchCode {
    fn name(&self) -> &str {
        "search"
    }

    fn description(&self) -> &str {
        "Search for text across all source files in the current directory. \
         Returns matching lines with file names and line numbers."
    }

    fn run(&self, arg: &str) -> Result<ToolRunResult> {
        info!(tool = "search", "tool called");
        let query = arg.trim();

        if query.is_empty() {
            return Err(ParamsError::Config("Search query cannot be empty".into()));
        }

        let current_dir = safety::project_root()?;
        let _ = safety::inspect_search_scope()?;
        let mut matches: Vec<SearchMatch> = Vec::new();

        // Walk the directory tree
        walk_and_search(&current_dir, query, &mut matches)?;

        if matches.is_empty() {
            return Ok(ToolRunResult::Immediate(format!(
                "No results found for: {query}"
            )));
        }

        // Cap results to avoid flooding context
        let total = matches.len();
        matches.truncate(50);

        let mut output = format!(
            "Search results for '{}' ({} matches{}):\n\n",
            query,
            total,
            if total > 50 { ", showing first 50" } else { "" }
        );

        // Group by file
        let mut current_file = String::new();
        for m in &matches {
            let file_str = m.path.display().to_string();
            if file_str != current_file {
                output.push_str(&format!("\n{}:\n", file_str));
                current_file = file_str;
            }
            output.push_str(&format!(
                "  {:4}: {}\n",
                m.line_number,
                m.line_content.trim()
            ));
        }

        Ok(ToolRunResult::Immediate(output))
    }
}

struct SearchMatch {
    path: PathBuf,
    line_number: usize,
    line_content: String,
}

fn walk_and_search(dir: &Path, query: &str, matches: &mut Vec<SearchMatch>) -> Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Ok(()), // Skip unreadable dirs silently
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();

        // Skip hidden dirs and build artifacts
        if name.starts_with('.') {
            continue;
        }
        if matches!(name.as_str(), "target" | "node_modules" | "__pycache__") {
            continue;
        }

        if path.is_dir() {
            walk_and_search(&path, query, matches)?;
        } else if is_searchable(&path) {
            search_file(&path, query, matches);
        }
    }

    Ok(())
}

fn is_searchable(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| SEARCHABLE_EXTENSIONS.contains(&ext))
        .unwrap_or(false)
}

fn search_file(path: &Path, query: &str, matches: &mut Vec<SearchMatch>) {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return, // Skip unreadable files silently
    };

    let query_lower = query.to_lowercase();

    for (i, line) in content.lines().enumerate() {
        if line.to_lowercase().contains(&query_lower) {
            matches.push(SearchMatch {
                path: path.to_path_buf(),
                line_number: i + 1,
                line_content: line.to_string(),
            });
        }
    }
}
