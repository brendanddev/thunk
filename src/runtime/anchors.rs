use crate::tools::ToolOutput;

/// Runtime-owned anchor state for explicit, structural multi-turn continuity.
///
/// Anchors are updated only from successful typed tool outputs and are never
/// inferred from model text or reconstructed from conversation history.
///
/// Supported anchors:
/// - last_read_file: updated from successful `read_file`
/// - last_search: updated from successful `search_code`
///
/// Anchors are intentionally:
/// - exact-match only (no semantic / pronoun / ordinal resolution)
/// - in-memory only (cleared on reset)
/// - not coupled to tool dispatch or conversation mutation
#[derive(Debug, Clone, Default)]
pub(super) struct AnchorState {
    last_read_file: Option<String>,
    last_search_query: Option<String>,
    last_search_scope: Option<String>,
}

impl AnchorState {
    /// Clears all anchor state (called on runtime reset).
    pub(super) fn clear(&mut self) {
        self.last_read_file = None;
        self.last_search_query = None;
        self.last_search_scope = None;
    }

    /// Records last-read-file anchor from a successful typed `read_file` output.
    /// Returns the resolved path if updated.
    ///
    /// Does not update on failed reads or non-file outputs.
    pub(super) fn record_successful_read(&mut self, output: &ToolOutput) -> Option<String> {
        if let ToolOutput::FileContents(file) = output {
            let path = file.path.clone();
            self.last_read_file = Some(path.clone());
            return Some(path);
        }
        None
    }

    /// Records last-search anchor from a successful typed `search_code` output.
    ///
    /// Stores the effective runtime-dispatched query and scope (post simplification
    /// and path-scope clamp).
    ///
    /// Does not update on failed searches.
    pub(super) fn record_successful_search(
        &mut self,
        output: &ToolOutput,
        query: String,
        scope: Option<String>,
    ) -> Option<(String, Option<String>)> {
        if matches!(output, ToolOutput::SearchResults(_)) {
            self.last_search_query = Some(query.clone());
            self.last_search_scope = scope.clone();
            return Some((query, scope));
        }
        None
    }

    /// Returns the last successfully read file path, if any.
    pub(super) fn last_read_file(&self) -> Option<&str> {
        self.last_read_file.as_deref()
    }

    /// Returns the last successful search (query + scope), if any.
    pub(super) fn last_search(&self) -> Option<(String, Option<String>)> {
        self.last_search_query
            .clone()
            .map(|query| (query, self.last_search_scope.clone()))
    }

    #[cfg(test)]
    pub(super) fn last_search_query(&self) -> Option<&str> {
        self.last_search_query.as_deref()
    }

    #[cfg(test)]
    pub(super) fn last_search_scope(&self) -> Option<&str> {
        self.last_search_scope.as_deref()
    }
}

/// Returns true if the input matches a supported last-read-file anchor prompt.
///
/// Matching is strictly structural and exact after normalization:
/// - no semantic interpretation
/// - no pronoun resolution
/// - no fuzzy matching
pub(super) fn is_last_read_file_anchor_prompt(text: &str) -> bool {
    let normalized = normalize_anchor_prompt(text);
    matches!(
        normalized.as_str(),
        "read that file"
            | "read that file again"
            | "read the last file"
            | "open that file"
            | "open that file again"
            | "open the last file"
    )
}

/// Returns true if the input matches a supported last-search anchor prompt.
///
/// Only exact replay phrases are supported; does not interpret query intent.
pub(super) fn is_last_search_anchor_prompt(text: &str) -> bool {
    let normalized = normalize_anchor_prompt(text);
    matches!(
        normalized.as_str(),
        "search that again"
            | "repeat that search"
            | "repeat the last search"
            | "run that search again"
            | "run the last search again"
            | "search the last query"
            | "search the last query again"
    )
}

/// Normalizes anchor prompts by:
/// - collapsing whitespace
/// - trimming trailing punctuation
/// - lowercasing
///
/// This ensures stable exact-match behavior for anchor detection.
fn normalize_anchor_prompt(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|c: char| matches!(c, '.' | '?' | '!' | ',' | ';' | ':'))
        .to_ascii_lowercase()
}
