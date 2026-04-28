#![allow(dead_code)]

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
