#![allow(dead_code)]

use crate::tools::ToolInput;

use super::{ProjectPath, ProjectScope};

/// Runtime-owned tool input after path resolution and scope validation.
///
/// This type is intentionally separate from `tools::ToolInput`: the raw tool
/// vocabulary carries model-emitted strings, while the runtime owns the job of
/// resolving those strings into validated project-local paths and scopes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedToolInput {
    ReadFile {
        path: ProjectPath,
    },
    ListDir {
        path: ProjectScope,
    },
    SearchCode {
        query: String,
        scope: Option<ProjectScope>,
    },
    WriteFile {
        path: ProjectPath,
        content: String,
    },
    EditFile {
        path: ProjectPath,
        search: String,
        replace: String,
    },
    GitStatus,
    GitDiff {
        path: Option<ProjectPath>,
    },
    GitLog,
}

impl ResolvedToolInput {
    pub fn tool_name(&self) -> &'static str {
        match self {
            Self::ReadFile { .. } => "read_file",
            Self::ListDir { .. } => "list_dir",
            Self::SearchCode { .. } => "search_code",
            Self::WriteFile { .. } => "write_file",
            Self::EditFile { .. } => "edit_file",
            Self::GitStatus => "git_status",
            Self::GitDiff { .. } => "git_diff",
            Self::GitLog => "git_log",
        }
    }
}

impl From<ResolvedToolInput> for ToolInput {
    fn from(input: ResolvedToolInput) -> Self {
        match input {
            // Temporary Phase 15.3.2 adapter: reconstruct legacy raw-tool inputs only
            // from trusted runtime-owned values. All path strings here come from
            // `ProjectPath::display()` / `ProjectScope::display()`, never from the
            // original model-emitted raw input.
            ResolvedToolInput::ReadFile { path } => ToolInput::ReadFile {
                path: path.display().to_string(),
            },
            ResolvedToolInput::ListDir { path } => ToolInput::ListDir {
                path: path.display().to_string(),
            },
            ResolvedToolInput::SearchCode { query, scope } => ToolInput::SearchCode {
                query,
                path: scope.map(|scope| scope.display().to_string()),
            },
            ResolvedToolInput::WriteFile { path, content } => ToolInput::WriteFile {
                path: path.display().to_string(),
                content,
            },
            ResolvedToolInput::EditFile {
                path,
                search,
                replace,
            } => ToolInput::EditFile {
                path: path.display().to_string(),
                search,
                replace,
            },
            ResolvedToolInput::GitStatus => ToolInput::GitStatus,
            // The legacy `ToolInput::GitDiff` carries no optional path yet, so this
            // temporary adapter cannot forward a resolved path until the later tool
            // migration slice updates the raw/legacy tool boundary.
            ResolvedToolInput::GitDiff { .. } => ToolInput::GitDiff,
            ResolvedToolInput::GitLog => ToolInput::GitLog,
        }
    }
}
