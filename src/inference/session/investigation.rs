use crate::inference::tool_loop::ToolLoopIntent;
use crate::inference::Message;
use crate::tools::ToolResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InvestigationLatencyPolicy {
    FastConvergence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum InvestigationAnchor {
    File(String),
    Directory(String),
    Query(String),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InjectedContextMetadata {
    pub file_path: Option<String>,
    pub directory_path: Option<String>,
    pub search_query: Option<String>,
    pub tool_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InvestigationResolution {
    pub(crate) intent: ToolLoopIntent,
    pub(crate) anchor: Option<InvestigationAnchor>,
    pub(crate) latency_policy: InvestigationLatencyPolicy,
    pub(crate) anchored_file: Option<String>,
    pub(crate) anchored_directory: Option<String>,
    pub(crate) anchored_query: Option<String>,
    pub(crate) prefer_answer_from_anchor: bool,
}

#[allow(dead_code)]
pub(crate) type TechnicalTurnResolution = InvestigationResolution;

impl InvestigationResolution {
    fn new(intent: ToolLoopIntent) -> Self {
        Self {
            intent,
            anchor: None,
            latency_policy: InvestigationLatencyPolicy::FastConvergence,
            anchored_file: None,
            anchored_directory: None,
            anchored_query: None,
            prefer_answer_from_anchor: false,
        }
    }

    fn anchored_to_file(path: Option<String>, query: Option<String>) -> Option<Self> {
        path.map(|anchored_file| Self {
            intent: ToolLoopIntent::CodeNavigation,
            anchor: Some(InvestigationAnchor::File(anchored_file.clone())),
            latency_policy: InvestigationLatencyPolicy::FastConvergence,
            anchored_file: Some(anchored_file),
            anchored_directory: None,
            anchored_query: query,
            prefer_answer_from_anchor: true,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct InvestigationState {
    last_intent: Option<ToolLoopIntent>,
    recent_loaded_files: Vec<String>,
    recent_directory_paths: Vec<String>,
    recent_search_queries: Vec<String>,
    last_anchor: Option<String>,
}

impl InvestigationState {
    pub(super) fn clear(&mut self) {
        self.last_intent = None;
        self.recent_loaded_files.clear();
        self.recent_directory_paths.clear();
        self.recent_search_queries.clear();
        self.last_anchor = None;
    }

    pub(super) fn has_recent_repo_context(&self) -> bool {
        self.last_intent.is_some()
            || !self.recent_loaded_files.is_empty()
            || !self.recent_directory_paths.is_empty()
            || !self.recent_search_queries.is_empty()
    }

    pub(super) fn apply_injected_context(
        &mut self,
        metadata: Option<&InjectedContextMetadata>,
        content: &str,
    ) {
        if let Some(metadata) = metadata {
            if let Some(path) = metadata.file_path.as_deref() {
                push_recent(&mut self.recent_loaded_files, path);
                self.last_anchor = Some(path.to_string());
            }
            if let Some(path) = metadata.directory_path.as_deref() {
                push_recent(&mut self.recent_directory_paths, path);
                self.last_anchor = Some(path.to_string());
            }
            if let Some(query) = metadata.search_query.as_deref() {
                push_recent(&mut self.recent_search_queries, query);
                self.last_anchor = Some(query.to_string());
            }
        }

        if let Some(path) = extract_line_value(content, "File: ") {
            push_recent(&mut self.recent_loaded_files, &path);
            self.last_anchor = Some(path);
        }
        if let Some(path) = extract_line_value(content, "Directory: ") {
            push_recent(&mut self.recent_directory_paths, &path);
            self.last_anchor = Some(path);
        }
        if let Some(query) = content
            .lines()
            .find_map(|line| line.strip_prefix("Search results for '"))
            .and_then(|rest| {
                rest.split_once('\'')
                    .map(|(query, _)| query.trim().to_string())
            })
        {
            push_recent(&mut self.recent_search_queries, &query);
            self.last_anchor = Some(query);
        }
    }

    pub(super) fn note_tool_loop_outcome(
        &mut self,
        intent: ToolLoopIntent,
        prompt: &str,
        results: &[ToolResult],
    ) {
        self.last_intent = Some(intent);
        self.last_anchor = Some(prompt.trim().to_string());
        for result in results {
            match result.tool_name.as_str() {
                "read_file" => push_recent(&mut self.recent_loaded_files, &result.argument),
                "list_dir" => push_recent(&mut self.recent_directory_paths, &result.argument),
                "search" => push_recent(&mut self.recent_search_queries, &result.argument),
                _ => {}
            }
        }
    }

    pub(super) fn compression_context(&self) -> Option<StructuredCompressionContext> {
        if !self.has_recent_repo_context() {
            return None;
        }

        Some(StructuredCompressionContext {
            active_investigation: self.last_intent.map(intent_label),
            recent_files: self.recent_loaded_files.clone(),
            recent_directories: self.recent_directory_paths.clone(),
            recent_searches: self.recent_search_queries.clone(),
            top_anchor: self.last_anchor.clone(),
        })
    }

    pub(super) fn follow_up_resolution(&self, prompt: &str) -> Option<InvestigationResolution> {
        if !self.has_recent_repo_context() {
            return None;
        }
        let normalized = normalize_follow_up_text(prompt);
        if normalized.is_empty() {
            return None;
        }

        if is_referential_file_prompt(prompt) {
            return InvestigationResolution::anchored_to_file(
                self.recent_loaded_files.last().cloned(),
                self.recent_search_queries.last().cloned(),
            );
        }

        let direct_follow_up = [
            "can you tell me now",
            "can you tell me more",
            "tell me now",
            "tell me more",
            "what about this file",
            "what about that file",
            "what about the logging path",
            "trace that",
            "trace it",
            "trace this",
            "explain that",
            "explain it",
            "describe that",
            "describe it",
            "what does this do",
            "what does that do",
            "what about that",
            "what about this",
        ];
        if direct_follow_up.contains(&normalized.as_str()) {
            if let Some(resolution) = InvestigationResolution::anchored_to_file(
                self.recent_loaded_files.last().cloned(),
                self.recent_search_queries.last().cloned(),
            ) {
                return Some(resolution);
            }
            return Some(self.fallback_resolution());
        }

        if normalized.contains("this file") || normalized.contains("that file") {
            return InvestigationResolution::anchored_to_file(
                self.recent_loaded_files.last().cloned(),
                self.recent_search_queries.last().cloned(),
            )
            .or_else(|| Some(self.fallback_resolution()));
        }

        if normalized.starts_with("can you tell me")
            || normalized.starts_with("tell me")
            || normalized.starts_with("what about ")
        {
            if let Some(resolution) = InvestigationResolution::anchored_to_file(
                self.recent_loaded_files.last().cloned(),
                self.recent_search_queries.last().cloned(),
            ) {
                return Some(resolution);
            }
            return Some(self.fallback_resolution());
        }

        None
    }

    pub(super) fn summary_message(&self) -> Option<Message> {
        let context = self.compression_context()?;
        Some(Message::user(&context.render()))
    }

    fn fallback_resolution(&self) -> InvestigationResolution {
        let anchored_file = self.recent_loaded_files.last().cloned();
        let anchored_directory = self.recent_directory_paths.last().cloned();
        let anchored_query = self.recent_search_queries.last().cloned();
        let anchor = anchored_file
            .as_ref()
            .map(|path| InvestigationAnchor::File(path.clone()))
            .or_else(|| {
                anchored_directory
                    .as_ref()
                    .map(|path| InvestigationAnchor::Directory(path.clone()))
            })
            .or_else(|| {
                anchored_query
                    .as_ref()
                    .map(|query| InvestigationAnchor::Query(query.clone()))
            });
        InvestigationResolution {
            intent: self.last_intent.unwrap_or(ToolLoopIntent::RepoOverview),
            anchor,
            latency_policy: InvestigationLatencyPolicy::FastConvergence,
            anchored_file,
            anchored_directory,
            anchored_query,
            prefer_answer_from_anchor: false,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StructuredCompressionContext {
    pub(crate) active_investigation: Option<String>,
    pub(crate) recent_files: Vec<String>,
    pub(crate) recent_directories: Vec<String>,
    pub(crate) recent_searches: Vec<String>,
    pub(crate) top_anchor: Option<String>,
}

impl StructuredCompressionContext {
    pub(crate) fn render(&self) -> String {
        let mut lines = vec!["Structured investigation context:".to_string()];
        if let Some(intent) = &self.active_investigation {
            lines.push(format!("Active investigation: {intent}"));
        }
        if !self.recent_files.is_empty() {
            lines.push(format!(
                "Recent loaded files: {}",
                self.recent_files
                    .iter()
                    .take(4)
                    .map(|path| format!("`{path}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if !self.recent_directories.is_empty() {
            lines.push(format!(
                "Recent listed directories: {}",
                self.recent_directories
                    .iter()
                    .take(3)
                    .map(|path| format!("`{path}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if !self.recent_searches.is_empty() {
            lines.push(format!(
                "Recent searches: {}",
                self.recent_searches
                    .iter()
                    .take(4)
                    .map(|query| format!("`{query}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if let Some(anchor) = &self.top_anchor {
            lines.push(format!("Top anchor: `{anchor}`"));
        }
        lines.join("\n")
    }
}

pub(super) fn resolve_agentic_repo_turn(
    prompt: &str,
    prior_investigation: &InvestigationState,
) -> Option<InvestigationResolution> {
    if let Some(resolution) = prior_investigation.follow_up_resolution(prompt) {
        return Some(resolution);
    }

    if let Some(intent) = crate::inference::tool_loop::detect_tool_loop_intent(prompt) {
        return Some(InvestigationResolution::new(intent));
    }

    let normalized = normalize_follow_up_text(prompt);
    let tokens = normalized.split_whitespace().collect::<Vec<_>>();
    let has_any = |needles: &[&str]| needles.iter().any(|needle| tokens.contains(needle));

    let looks_like_repo_overview = has_any(&["repo", "project", "codebase"])
        && has_any(&[
            "see",
            "inspect",
            "understand",
            "tell",
            "describe",
            "explain",
            "overview",
            "summarize",
            "kind",
        ]);
    if looks_like_repo_overview {
        return Some(InvestigationResolution::new(ToolLoopIntent::RepoOverview));
    }

    let looks_like_directory_overview = has_any(&["directory", "folder", "here"])
        && has_any(&[
            "see",
            "inspect",
            "understand",
            "tell",
            "describe",
            "summarize",
            "what",
        ]);
    if looks_like_directory_overview {
        return Some(InvestigationResolution::new(
            ToolLoopIntent::DirectoryOverview,
        ));
    }

    detect_broad_technical_intent(&normalized).map(InvestigationResolution::new)
}

fn detect_broad_technical_intent(normalized: &str) -> Option<ToolLoopIntent> {
    if normalized.is_empty() {
        return None;
    }

    let tokens = normalized.split_whitespace().collect::<Vec<_>>();
    let has_any = |needles: &[&str]| needles.iter().any(|needle| tokens.contains(needle));
    let contains_any = |needles: &[&str]| needles.iter().any(|needle| normalized.contains(needle));

    if contains_any(&[
        " src/",
        ".rs",
        ".ts",
        ".tsx",
        ".js",
        ".py",
        "cargo toml",
        "readme md",
    ]) {
        return Some(ToolLoopIntent::CodeNavigation);
    }

    if contains_any(&[" this file", " that file", " current file", " loaded file"])
        && contains_any(&["what", "explain", "describe", "summarize", "tell"])
    {
        return Some(ToolLoopIntent::CodeNavigation);
    }

    if has_any(&["repo", "project", "codebase"]) {
        return Some(ToolLoopIntent::RepoOverview);
    }

    if has_any(&["directory", "folder"]) {
        return Some(ToolLoopIntent::DirectoryOverview);
    }

    if has_any(&["trace", "flow", "walk"])
        || contains_any(&["how does", "explain how", "xplain how", "describe how"])
    {
        return Some(ToolLoopIntent::FlowTrace);
    }

    if has_any(&["call", "calls"]) {
        return Some(ToolLoopIntent::CallSiteLookup);
    }

    if has_any(&["use", "uses"]) {
        return Some(ToolLoopIntent::UsageLookup);
    }

    if has_any(&["config", "configured", "setting", "settings"]) {
        return Some(ToolLoopIntent::ConfigLocate);
    }

    if has_any(&[
        "file", "function", "method", "module", "struct", "enum", "trait", "impl", "type",
        "symbol", "code",
    ]) || contains_any(&[
        "where is",
        "what does",
        "where does",
        "implemented",
        "defined",
        "entrypoint",
        "entry point",
    ]) {
        return Some(ToolLoopIntent::CodeNavigation);
    }

    None
}

fn intent_label(intent: ToolLoopIntent) -> String {
    match intent {
        ToolLoopIntent::RepoOverview => "repo overview".to_string(),
        ToolLoopIntent::DirectoryOverview => "directory overview".to_string(),
        ToolLoopIntent::CodeNavigation => "implementation lookup".to_string(),
        ToolLoopIntent::ConfigLocate => "config lookup".to_string(),
        ToolLoopIntent::CallSiteLookup => "call-site lookup".to_string(),
        ToolLoopIntent::UsageLookup => "usage lookup".to_string(),
        ToolLoopIntent::FlowTrace => "flow trace".to_string(),
    }
}

fn normalize_follow_up_text(prompt: &str) -> String {
    prompt
        .to_ascii_lowercase()
        .replace(['\'', '’'], "")
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '/' {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_referential_file_prompt(prompt: &str) -> bool {
    matches!(
        normalize_follow_up_text(prompt).as_str(),
        "what does this file do"
            | "what does the current file do"
            | "what does the loaded file do"
            | "what is this file for"
            | "explain this file"
            | "describe this file"
            | "summarize this file"
    )
}

fn extract_line_value(content: &str, prefix: &str) -> Option<String> {
    content.lines().find_map(|line| {
        line.strip_prefix(prefix)
            .map(|value| value.trim().to_string())
    })
}

fn push_recent(values: &mut Vec<String>, value: &str) {
    if value.trim().is_empty() {
        return;
    }
    values.retain(|existing| existing != value);
    values.push(value.to_string());
    if values.len() > 6 {
        values.drain(0..values.len() - 6);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broad_repo_prompt_routes_to_repo_overview() {
        let state = InvestigationState::default();
        assert_eq!(
            resolve_agentic_repo_turn("Can you see my project?", &state)
                .map(|resolution| resolution.intent),
            Some(ToolLoopIntent::RepoOverview)
        );
        assert_eq!(
            resolve_agentic_repo_turn("Do you understand this repo?", &state)
                .map(|resolution| resolution.intent),
            Some(ToolLoopIntent::RepoOverview)
        );
    }

    #[test]
    fn typoed_explain_how_prompt_routes_to_flow_trace() {
        let state = InvestigationState::default();
        assert_eq!(
            resolve_agentic_repo_turn("xplain how session restore works", &state)
                .map(|resolution| resolution.intent),
            Some(ToolLoopIntent::FlowTrace)
        );
    }

    #[test]
    fn follow_up_prompt_reuses_recent_investigation_intent() {
        let mut state = InvestigationState::default();
        state.note_tool_loop_outcome(
            ToolLoopIntent::CodeNavigation,
            "Where is session restore implemented?",
            &[ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/session/mod.rs".to_string(),
                output: String::new(),
            }],
        );

        let resolution =
            resolve_agentic_repo_turn("Can you tell me now?", &state).expect("resolution");
        assert_eq!(resolution.intent, ToolLoopIntent::CodeNavigation);
        assert_eq!(
            resolution.anchored_file.as_deref(),
            Some("src/session/mod.rs")
        );
        assert!(resolution.prefer_answer_from_anchor);
    }

    #[test]
    fn injected_context_metadata_is_recorded() {
        let mut state = InvestigationState::default();
        state.apply_injected_context(
            Some(&InjectedContextMetadata {
                file_path: Some("src/main.rs".to_string()),
                directory_path: None,
                search_query: None,
                tool_name: Some("read_file".to_string()),
            }),
            "I've loaded this file for context:\n\nFile: src/main.rs",
        );

        let rendered = state.compression_context().expect("context").render();
        assert!(rendered.contains("src/main.rs"));
    }
}
