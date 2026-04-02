// src/tools/fs.rs
//
// Filesystem tools — read files and list directories.
//
// These are the most fundamental coding assistant tools. Without them
// the model can only work with code you paste directly into the chat.
// With them, you can say "look at src/auth.rs and tell me what's wrong"
// and the model will actually read the file.

use std::path::Path;
use std::fs;
use tracing::info;

use crate::error::{ParamsError, Result};
use super::{Tool, ToolRunResult};

/// Reads the contents of a file and returns them as a string.
///
/// Usage in model response: [read_file: src/main.rs]
pub struct ReadFile;

impl Tool for ReadFile {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Provide a relative or absolute path."
    }

    fn run(&self, arg: &str) -> Result<ToolRunResult> {
        info!(tool = "read_file", "tool called");
        let path = Path::new(arg.trim());

        if !path.exists() {
            return Err(ParamsError::Config(format!(
                "File not found: {}", path.display()
            )));
        }

        if !path.is_file() {
            return Err(ParamsError::Config(format!(
                "{} is a directory, not a file. Use list_dir instead.",
                path.display()
            )));
        }

        // Cap file size to avoid feeding massive files into context
        let metadata = fs::metadata(path)?;
        let max_bytes = 100_000; // ~100KB
        if metadata.len() > max_bytes {
            return Err(ParamsError::Config(format!(
                "File too large ({} bytes). Max is {}KB. Consider reading a specific section.",
                metadata.len(),
                max_bytes / 1000
            )));
        }

        let content = fs::read_to_string(path)?;
        let line_count = content.lines().count();

        // Include the path and line count as a header so the model has context
        Ok(ToolRunResult::Immediate(format!(
            "File: {}\nLines: {line_count}\n\n```\n{content}\n```",
            path.display()
        )))
    }
}

/// Lists the contents of a directory.
///
/// Usage in model response: [list_dir: src/]
pub struct ListDir;

impl Tool for ListDir {
    fn name(&self) -> &str {
        "list_dir"
    }

    fn description(&self) -> &str {
        "List files and directories at a path. Use '.' for current directory."
    }

    fn run(&self, arg: &str) -> Result<ToolRunResult> {
        info!(tool = "list_dir", "tool called");
        let path = Path::new(arg.trim());

        // Default to current directory if empty
        let path = if arg.trim().is_empty() {
            Path::new(".")
        } else {
            path
        };

        if !path.exists() {
            return Err(ParamsError::Config(format!(
                "Path not found: {}", path.display()
            )));
        }

        if !path.is_dir() {
            return Err(ParamsError::Config(format!(
                "{} is a file, not a directory. Use read_file instead.",
                path.display()
            )));
        }

        let mut entries: Vec<String> = Vec::new();

        // Read directory entries and sort them — dirs first, then files
        let mut dirs: Vec<String> = Vec::new();
        let mut files: Vec<String> = Vec::new();

        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();

            // Skip hidden files and common noise
            if name.starts_with('.') {
                continue;
            }
            if matches!(name.as_str(), "target" | "node_modules" | "__pycache__" | ".git") {
                continue;
            }

            if entry.path().is_dir() {
                dirs.push(format!("{name}/"));
            } else {
                files.push(name);
            }
        }

        dirs.sort();
        files.sort();

        entries.extend(dirs);
        entries.extend(files);

        if entries.is_empty() {
            return Ok(ToolRunResult::Immediate(format!(
                "Directory {} is empty.",
                path.display()
            )));
        }

        Ok(ToolRunResult::Immediate(format!(
            "Directory: {}\n\n{}",
            path.display(),
            entries.join("\n")
        )))
    }
}
