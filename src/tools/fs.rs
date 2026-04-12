use std::fs;
use tracing::info;

use super::{Tool, ToolRunResult};
use crate::error::{ParamsError, Result};
use crate::safety::{self, ProjectPathKind};

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
        let inspected =
            safety::inspect_project_path("read_file", arg.trim(), ProjectPathKind::File, false)?;
        let path = &inspected.resolved_path;

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
            inspected.display_path
        )))
    }
}

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
        let inspected = safety::inspect_project_path(
            "list_dir",
            if arg.trim().is_empty() {
                "."
            } else {
                arg.trim()
            },
            ProjectPathKind::Directory,
            false,
        )?;
        let path = &inspected.resolved_path;

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
            if matches!(
                name.as_str(),
                "target" | "node_modules" | "__pycache__" | ".git"
            ) {
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
                inspected.display_path
            )));
        }

        Ok(ToolRunResult::Immediate(format!(
            "Directory: {}\n\n{}",
            inspected.display_path,
            entries.join("\n")
        )))
    }
}
