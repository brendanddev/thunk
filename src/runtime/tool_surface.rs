use crate::tools::ToolInput;

use super::prompt_analysis::normalized_prompt_tokens;

/// Runtime-owned per-turn tool surface.
///
/// A surface defines which read-only tool family is available for the current
/// turn. This is policy enforced by the runtime before dispatch; tools and
/// tool_codec must not own or interpret surface rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ToolSurface {
    RetrievalFirst,
    GitReadOnly,
}

/// Canonical registry entry for a tool surface.
///
/// Keeping the surface name and allowed tools together prevents hint rendering
/// and enforcement from drifting apart.
struct ToolSurfaceDefinition {
    surface: ToolSurface,
    name: &'static str,
    tools: &'static [SurfaceTool],
}

/// Runtime policy view of model-callable tools.
///
/// Mutation tools are intentionally excluded from surfaces because approval and
/// mutation permission are governed by a separate lifecycle path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SurfaceTool {
    SearchCode,
    ReadFile,
    ListDir,
    GitStatus,
    GitDiff,
    GitLog,
}

const RETRIEVAL_FIRST_TOOLS: &[SurfaceTool] = &[
    SurfaceTool::SearchCode,
    SurfaceTool::ReadFile,
    SurfaceTool::ListDir,
];
const GIT_READ_ONLY_TOOLS: &[SurfaceTool] = &[
    SurfaceTool::GitStatus,
    SurfaceTool::GitDiff,
    SurfaceTool::GitLog,
];
const TOOL_SURFACE_DEFINITIONS: &[ToolSurfaceDefinition] = &[
    ToolSurfaceDefinition {
        surface: ToolSurface::RetrievalFirst,
        name: "RetrievalFirst",
        tools: RETRIEVAL_FIRST_TOOLS,
    },
    ToolSurfaceDefinition {
        surface: ToolSurface::GitReadOnly,
        name: "GitReadOnly",
        tools: GIT_READ_ONLY_TOOLS,
    },
];

impl SurfaceTool {
    pub(super) fn from_input(input: &ToolInput) -> Option<Self> {
        match input {
            ToolInput::SearchCode { .. } => Some(Self::SearchCode),
            ToolInput::ReadFile { .. } => Some(Self::ReadFile),
            ToolInput::ListDir { .. } => Some(Self::ListDir),
            ToolInput::GitStatus => Some(Self::GitStatus),
            ToolInput::GitDiff => Some(Self::GitDiff),
            ToolInput::GitLog => Some(Self::GitLog),
            ToolInput::EditFile { .. } | ToolInput::WriteFile { .. } => None,
        }
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::SearchCode => "search_code",
            Self::ReadFile => "read_file",
            Self::ListDir => "list_dir",
            Self::GitStatus => "git_status",
            Self::GitDiff => "git_diff",
            Self::GitLog => "git_log",
        }
    }
}

impl ToolSurface {
    fn definition(self) -> &'static ToolSurfaceDefinition {
        TOOL_SURFACE_DEFINITIONS
            .iter()
            .find(|definition| definition.surface == self)
            .expect("tool surface definition must exist")
    }

    pub(super) fn as_str(self) -> &'static str {
        self.definition().name
    }

    pub(super) fn tools(self) -> &'static [SurfaceTool] {
        self.definition().tools
    }

    pub(super) fn allowed_tool_names(self) -> impl Iterator<Item = &'static str> {
        self.tools().iter().copied().map(SurfaceTool::name)
    }
}

/// Selects the tool surface for a user turn using explicit structural phrasing only.
///
/// This intentionally avoids fuzzy matching so ambiguous prompts stay on the
/// safer RetrievalFirst surface.
pub(super) fn select_tool_surface(prompt: &str) -> ToolSurface {
    if is_explicit_git_tooling_prompt(prompt) {
        ToolSurface::GitReadOnly
    } else {
        ToolSurface::RetrievalFirst
    }
}

/// Returns true only for explicit Git read-only requests.
///
/// The accepted phrases are narrow by design; code-investigation prompts that
/// merely mention "git" should remain RetrievalFirst.
fn is_explicit_git_tooling_prompt(prompt: &str) -> bool {
    let tokens = normalized_prompt_tokens(prompt);
    starts_with_token_phrase(&tokens, &["show", "git", "status"])
        || starts_with_token_phrase(&tokens, &["show", "git", "diff"])
        || starts_with_token_phrase(&tokens, &["show", "git", "log"])
        || starts_with_token_phrase(&tokens, &["git", "status"])
        || starts_with_token_phrase(&tokens, &["git", "diff"])
        || starts_with_token_phrase(&tokens, &["git", "log"])
        || starts_with_token_phrase(&tokens, &["show", "working", "tree"])
        || starts_with_token_phrase(&tokens, &["show", "recent", "commits"])
        || starts_with_token_phrase(&tokens, &["show", "latest", "commits"])
        || starts_with_token_phrase(&tokens, &["show", "recent", "git", "status"])
        || starts_with_token_phrase(&tokens, &["show", "recent", "git", "diff"])
        || starts_with_token_phrase(&tokens, &["show", "recent", "git", "log"])
        || starts_with_token_phrase(&tokens, &["show", "latest", "git", "status"])
        || starts_with_token_phrase(&tokens, &["show", "latest", "git", "diff"])
        || starts_with_token_phrase(&tokens, &["show", "latest", "git", "log"])
}

fn starts_with_token_phrase(tokens: &[String], phrase: &[&str]) -> bool {
    tokens.len() >= phrase.len()
        && tokens
            .iter()
            .take(phrase.len())
            .map(String::as_str)
            .eq(phrase.iter().copied())
}

/// Enforces whether a tool call is available on the active surface.
///
/// Mutation calls return true here because they are checked by the separate
/// approval/mutation policy, not by read-only surface enforcement.
pub(super) fn tool_allowed_for_surface(input: &ToolInput, surface: ToolSurface) -> bool {
    if let Some(tool) = SurfaceTool::from_input(input) {
        tool_surface_for_tool(tool) == Some(surface)
    } else {
        // Mutation permission remains separate from tool-surface policy.
        true
    }
}

/// Identifies Git read-only tool calls for Git acquisition/finalization logic.
pub(super) fn is_git_read_only_tool_input(input: &ToolInput) -> bool {
    matches!(
        SurfaceTool::from_input(input).and_then(tool_surface_for_tool),
        Some(ToolSurface::GitReadOnly)
    )
}

fn tool_surface_for_tool(tool: SurfaceTool) -> Option<ToolSurface> {
    TOOL_SURFACE_DEFINITIONS
        .iter()
        .find(|definition| definition.tools.contains(&tool))
        .map(|definition| definition.surface)
}
