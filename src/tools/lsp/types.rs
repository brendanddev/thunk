use std::path::PathBuf;

#[derive(Debug, Clone)]
pub(super) struct LspDiagnostic {
    pub severity: String,
    pub line: usize,
    pub column: usize,
    pub message: String,
    pub source: Option<String>,
}

#[derive(Debug, Clone)]
pub(super) struct LspCommandSpec {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub display: String,
}

#[derive(Debug, Clone)]
pub(super) struct LspProbe {
    pub spec: LspCommandSpec,
    pub status: LspProbeStatus,
}

#[derive(Debug, Clone)]
pub(super) enum LspProbeStatus {
    Ready(String),
    Failed(String),
}

#[derive(Debug, Clone)]
pub(super) struct HoverInput {
    pub path: PathBuf,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct HoverPosition {
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone)]
pub(super) struct LspResponseError {
    pub code: i64,
    pub message: String,
    pub data: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DefinitionLocation {
    pub path: PathBuf,
    pub line: usize,
    pub column: usize,
}

pub(super) enum HoverResponse {
    Hover(String),
    NoInfo,
    RetryableError(String),
}

pub(super) enum DefinitionResponse {
    Definitions(Vec<DefinitionLocation>),
    NoInfo,
    RetryableError(String),
}
