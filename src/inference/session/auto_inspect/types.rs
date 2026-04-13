#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AutoInspectIntent {
    RepoOverview,
    DirectoryOverview,
    WhereIsImplementation,
    FeatureTrace,
    ConfigLocate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AutoInspectStep {
    pub(crate) label: String,
    pub(crate) tool_name: &'static str,
    pub(crate) argument: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AutoInspectPlan {
    pub(crate) intent: AutoInspectIntent,
    pub(crate) thinking: &'static str,
    pub(crate) status_label: &'static str,
    pub(crate) context_label: &'static str,
    pub(crate) query: Option<String>,
    pub(crate) steps: Vec<AutoInspectStep>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AutoInspectBudget {
    pub(crate) total_chars: usize,
    pub(crate) top_level_entries: usize,
    pub(crate) code_entries: usize,
    pub(crate) readme_chars: usize,
    pub(crate) manifest_chars: usize,
    pub(crate) entrypoint_chars: usize,
    pub(crate) search_files: usize,
    pub(crate) read_files: usize,
    pub(crate) key_hits_per_file: usize,
    pub(crate) workflow_summary_chars: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SearchLineHit {
    pub(crate) line_number: usize,
    pub(crate) line_content: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SearchFileHit {
    pub(crate) path: String,
    pub(crate) hits: Vec<SearchLineHit>,
}
