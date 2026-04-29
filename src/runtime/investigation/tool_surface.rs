use crate::tools::ToolInput;

use super::prompt_analysis::normalized_prompt_tokens;

/// Runtime-owned per-turn tool surface.
///
/// A surface defines which read-only tool family is available for the current
/// turn. This is policy enforced by the runtime before dispatch; tools and
/// tool_codec must not own or interpret surface rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolSurface {
    RetrievalFirst,
    GitReadOnly,
    /// Synthesis-only surface: no tools offered.
    /// Used for answer-phase generations after evidence is accepted or a read completes,
    /// to prevent the model from attempting tool calls and triggering a correction round.
    AnswerOnly,
    /// Read tools plus mutation tools (edit_file, write_file) visible in the per-turn hint.
    /// Selected when the prompt requests a mutation so the model knows those tools are
    /// available this turn. Enforcement for mutation calls remains the same as RetrievalFirst:
    /// they bypass surface checks via the approval path.
    MutationEnabled,
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
pub(crate) enum SurfaceTool {
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
const ANSWER_ONLY_TOOLS: &[SurfaceTool] = &[];
// MutationEnabled has the same read tools as RetrievalFirst. Mutation tools (edit_file,
// write_file) are not SurfaceTool variants — they bypass surface enforcement and are
// exposed to the model only via the mutation_tool_names() hint extension.
const MUTATION_ENABLED_TOOLS: &[SurfaceTool] = &[
    SurfaceTool::SearchCode,
    SurfaceTool::ReadFile,
    SurfaceTool::ListDir,
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
    ToolSurfaceDefinition {
        surface: ToolSurface::AnswerOnly,
        name: "AnswerOnly",
        tools: ANSWER_ONLY_TOOLS,
    },
    ToolSurfaceDefinition {
        surface: ToolSurface::MutationEnabled,
        name: "MutationEnabled",
        tools: MUTATION_ENABLED_TOOLS,
    },
];

impl SurfaceTool {
    pub(crate) fn from_input(input: &ToolInput) -> Option<Self> {
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

    pub(crate) fn name(self) -> &'static str {
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

    pub(crate) fn as_str(self) -> &'static str {
        self.definition().name
    }

    pub(crate) fn tools(self) -> &'static [SurfaceTool] {
        self.definition().tools
    }

    pub(crate) fn allowed_tool_names(self) -> impl Iterator<Item = &'static str> {
        self.tools().iter().copied().map(SurfaceTool::name)
    }

    /// Returns the mutation tool names that should be appended to the per-turn hint
    /// when this surface is active. Empty for all surfaces except MutationEnabled.
    pub(crate) fn mutation_tool_names(self) -> &'static [&'static str] {
        match self {
            Self::MutationEnabled => &["edit_file", "write_file"],
            _ => &[],
        }
    }

    pub(crate) fn includes_project_snapshot_hint(self) -> bool {
        matches!(self, Self::RetrievalFirst | Self::MutationEnabled)
    }
}

pub(crate) fn select_tool_surface(
    prompt: &str,
    investigation_required: bool,
    mutation_allowed: bool,
    has_direct_read: bool,
) -> ToolSurface {
    if is_explicit_git_tooling_prompt(prompt) {
        ToolSurface::GitReadOnly
    } else if mutation_allowed {
        ToolSurface::MutationEnabled
    } else if investigation_required
        || has_direct_read
        || prompt_requests_directory_navigation(prompt)
    {
        ToolSurface::RetrievalFirst
    } else {
        ToolSurface::AnswerOnly
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

fn prompt_requests_directory_navigation(prompt: &str) -> bool {
    let tokens = normalized_prompt_tokens(prompt);
    const NAV_VERBS: &[&str] = &["list", "show", "display", "tree", "explore"];
    const STRUCTURAL_KEYWORDS: &[&str] = &[
        "files",
        "file",
        "directory",
        "dir",
        "dirs",
        "structure",
        "contents",
        "folders",
        "folder",
    ];
    let has_nav_verb = tokens.iter().any(|t| NAV_VERBS.contains(&t.as_str()));
    if !has_nav_verb {
        return false;
    }
    let has_structural_keyword = tokens
        .iter()
        .any(|t| STRUCTURAL_KEYWORDS.contains(&t.as_str()));
    let has_path_token = prompt.split_whitespace().any(|t| t.contains('/'));
    has_structural_keyword || has_path_token
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
pub(crate) fn tool_allowed_for_surface(input: &ToolInput, surface: ToolSurface) -> bool {
    if let Some(tool) = SurfaceTool::from_input(input) {
        // Direct membership check: is this read-only tool in the surface's canonical set?
        // Using direct lookup avoids ambiguity when multiple surfaces share the same tools
        // (e.g., MutationEnabled and RetrievalFirst both carry search/read/list).
        surface.tools().contains(&tool)
    } else {
        // Mutation permission remains separate from tool-surface policy.
        true
    }
}

/// Identifies Git read-only tool calls for Git acquisition/finalization logic.
pub(crate) fn is_git_read_only_tool_input(input: &ToolInput) -> bool {
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
