use super::paths::normalize_evidence_path;

/// Determines whether a prompt should enter investigation mode.
///
/// Uses structural signals first (identifier-like tokens), then falls back to
/// constrained natural-language lookup detection. This must remain conservative
/// to avoid over-triggering investigation on general questions.
pub(super) fn prompt_requires_investigation(text: &str) -> bool {
    for raw in text.split(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                ',' | '.'
                    | '?'
                    | '!'
                    | ';'
                    | ':'
                    | '"'
                    | '\''
                    | '`'
                    | '('
                    | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
            )
    }) {
        let token = raw.trim();
        if token.is_empty() {
            continue;
        }
        if is_snake_case_identifier(token) || is_pascal_case_identifier(token) {
            return true;
        }
    }

    natural_language_code_lookup_requires_investigation(text)
}

/// Detects investigation intent from natural-language lookup phrasing.
///
/// Requires both a lookup verb (find/where/locate/search) and a secondary
/// condition indicating code-related intent, except for "search" which is
/// treated as an explicit tool request.
fn natural_language_code_lookup_requires_investigation(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let has_lookup_verb = contains_word(&lower, "find")
        || contains_word(&lower, "where")
        || contains_word(&lower, "locate")
        || contains_word(&lower, "search");
    if !has_lookup_verb {
        return false;
    }

    // "search" is a self-sufficient trigger — it is an explicit request to run the search tool.
    // "find/where/locate" still require a secondary condition to avoid false positives on
    // conversational phrasing like "find a good approach".
    if contains_word(&lower, "search") {
        return true;
    }

    [
        "defined",
        "implemented",
        "initialize",
        "initialized",
        "initialization",
        "initialised",
        "configured",
        "create",
        "created",
        "creation",
        "register",
        "registered",
        "registration",
        "load",
        "loaded",
        "loading",
        "save",
        "saved",
        "saving",
        "stored",
        "handled",
        "called",
        "used",
        // occurrence/appearance phrasing: "find all occurrences of X", "where it appears"
        "occur",
        "occurs",
        "occurrence",
        "occurrences",
        "appear",
        "appears",
        "filtered",
        "rendered",
    ]
    .iter()
    .any(|term| contains_word(&lower, term))
}

/// Checks for exact token matches within a normalized text stream.
///
/// Avoids substring matching to prevent false positives (e.g., "find" in "finder").
fn contains_word(text: &str, needle: &str) -> bool {
    text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .any(|token| token == needle)
}

/// Produces a normalized token stream for prompt analysis.
///
/// Lowercases and splits on non-identifier characters. Shared by multiple
/// classification helpers to ensure consistent tokenization.
pub(super) fn normalized_prompt_tokens(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect()
}

/// Detects whether the user is requesting a mutation operation.
///
/// Uses a strict keyword list to avoid accidental triggering from
/// conversational language.
pub(super) fn user_requested_mutation(text: &str) -> bool {
    text.split(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                ',' | '.'
                    | '?'
                    | '!'
                    | ';'
                    | ':'
                    | '"'
                    | '\''
                    | '`'
                    | '('
                    | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
                    | '/'
                    | '\\'
            )
    })
    .any(|token| {
        matches!(
            token.to_ascii_lowercase().as_str(),
            "add"
                | "change"
                | "create"
                | "delete"
                | "edit"
                | "modify"
                | "overwrite"
                | "replace"
                | "update"
                | "write"
        )
    })
}

/// Extracts a single relative path scope from an investigation prompt.
///
/// Fires only on the conservative pattern `in <token>` / `within <token>`, with
/// an optional `the` before the token, where the token contains `/`, has no
/// whitespace, and is not a URL. Trailing punctuation
/// that is not part of a path is stripped. Returns `None` when the pattern is absent
/// or ambiguous (multiple qualifying tokens, empty token after stripping, etc.).
///
/// Examples that match:
///   "Where is TaskStatus handled in sandbox/cli/"     → Some("sandbox/cli/")
///   "Find logging in sandbox/services/"               → Some("sandbox/services/")
///   "Find where database is configured in the sandbox/ folder" → Some("sandbox/")
///
/// Examples that do not match:
///   "Find X in the application"  → None  (no `/` in token)
///   "Find X in context"          → None  (no `/`)
///   "Find X in https://…"        → None  (URL rejected)
pub(super) fn extract_investigation_path_scope(text: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    let words: Vec<&str> = text.split_whitespace().collect();
    let lower_words: Vec<&str> = lower.split_whitespace().collect();

    let mut found: Option<String> = None;

    for (i, lw) in lower_words.iter().enumerate() {
        if (*lw == "in" || *lw == "within") && i + 1 < words.len() {
            let next = i + 1;
            let path_index = if lower_words[next] == "the" && next + 1 < words.len() {
                next + 1
            } else {
                next
            };
            let raw = words[path_index];
            // Strip trailing punctuation that cannot be part of a relative path.
            let stripped = raw.trim_end_matches(|c: char| {
                matches!(
                    c,
                    '.' | ',' | '?' | '!' | ';' | ':' | ')' | ']' | '}' | '"' | '\''
                )
            });
            if stripped.is_empty() {
                continue;
            }
            // Require at least one `/` — distinguishes paths from plain words.
            if !stripped.contains('/') {
                continue;
            }
            // Reject URLs.
            if stripped.starts_with("http://") || stripped.starts_with("https://") {
                continue;
            }
            // Reject anything with embedded whitespace (shouldn't happen after split, but be safe).
            if stripped.contains(|c: char| c.is_whitespace()) {
                continue;
            }
            // More than one qualifying token → ambiguous; return None.
            if found.is_some() {
                return None;
            }
            found = Some(normalize_evidence_path(stripped));
        }
    }

    found
}

/// Extracts a direct-read file path from a prompt starting with "read".
///
/// Accepts "read <path>" and "read file <path>" forms. Returns None if the
/// structure does not match or the candidate does not resemble a file path.
pub(super) fn requested_read_path(text: &str) -> Option<String> {
    let mut tokens = text.split_whitespace();
    let first = tokens.next()?;
    if !first.eq_ignore_ascii_case("read") {
        return None;
    }

    let mut candidate = tokens.next()?;
    if candidate.eq_ignore_ascii_case("file") {
        candidate = tokens.next()?;
    }

    let path = candidate.trim_matches(|c: char| {
        matches!(
            c,
            '`' | '"' | '\'' | ',' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}'
        )
    });
    if looks_like_file_path(path) {
        Some(path.to_string())
    } else {
        None
    }
}

/// Heuristic check for whether a token resembles a file path.
///
/// Allows common patterns (directories, extensions, README) without resolving
/// or validating against the filesystem.
pub(super) fn looks_like_file_path(path: &str) -> bool {
    !path.is_empty()
        && (path.contains('/')
            || path.contains('\\')
            || path.contains('.')
            || path.eq_ignore_ascii_case("README"))
}

/// snake_case: contains underscore, ≥2 segments, each segment ≥2 alphanumeric chars.
pub(super) fn is_snake_case_identifier(token: &str) -> bool {
    if !token.contains('_') {
        return false;
    }
    let segments: Vec<&str> = token.split('_').collect();
    segments.len() >= 2
        && segments
            .iter()
            .all(|s| s.len() >= 2 && s.bytes().all(|b| b.is_ascii_alphanumeric()))
}

/// Matches PascalCase/camelCase identifiers.
/// Note: also intentionally matches ALLCAPS tokens of sufficient length (e.g., DEBUG, README)
/// for Phase 8.4 structural detection.
pub(super) fn is_pascal_case_identifier(token: &str) -> bool {
    if token.len() < 5 {
        return false;
    }
    let mut chars = token.chars();
    match chars.next() {
        Some(c) if c.is_ascii_uppercase() => {}
        _ => return false,
    }
    token[1..].chars().any(|c| c.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snake_case_classifier_accepts_valid_identifiers() {
        assert!(is_snake_case_identifier("run_turns"));
        assert!(is_snake_case_identifier("search_code"));
        assert!(is_snake_case_identifier("read_file"));
        assert!(is_snake_case_identifier("tool_rounds"));
        assert!(is_snake_case_identifier("investigation_state"));
        assert!(is_snake_case_identifier("is_snake_case_identifier"));
    }

    #[test]
    fn snake_case_classifier_rejects_non_identifiers() {
        assert!(!is_snake_case_identifier("word"));
        assert!(!is_snake_case_identifier("a_b"));
        assert!(!is_snake_case_identifier("_leading"));
        assert!(!is_snake_case_identifier("trailing_"));
        assert!(!is_snake_case_identifier("has space"));
        assert!(!is_snake_case_identifier("run_turns()"));
    }

    #[test]
    fn pascal_case_classifier_accepts_valid_identifiers() {
        assert!(is_pascal_case_identifier("AnswerSource"));
        assert!(is_pascal_case_identifier("RuntimeTerminalReason"));
        assert!(is_pascal_case_identifier("InvestigationState"));
        assert!(is_pascal_case_identifier("ToolInput"));
        assert!(is_pascal_case_identifier("SearchBudget"));
    }

    #[test]
    fn pascal_case_classifier_rejects_non_identifiers() {
        assert!(!is_pascal_case_identifier("Hi"));
        assert!(!is_pascal_case_identifier("Short"));
        assert!(!is_pascal_case_identifier("allower"));
        assert!(!is_pascal_case_identifier("Done"));
    }

    #[test]
    fn prompt_requires_investigation_detects_snake_case() {
        assert!(prompt_requires_investigation("What does run_turns do?"));
        assert!(prompt_requires_investigation("Explain search_code to me."));
        assert!(prompt_requires_investigation("Where is read_file defined?"));
    }

    #[test]
    fn prompt_requires_investigation_detects_pascal_case() {
        assert!(prompt_requires_investigation("What is AnswerSource?"));
        assert!(prompt_requires_investigation("Explain InvestigationState"));
        assert!(prompt_requires_investigation(
            "How does RuntimeTerminalReason work?"
        ));
    }

    #[test]
    fn prompt_requires_investigation_detects_natural_language_lookup() {
        assert!(prompt_requires_investigation(
            "Find where logging is initialized"
        ));
        assert!(prompt_requires_investigation(
            "Find where sessions are saved"
        ));
        assert!(prompt_requires_investigation(
            "Where is configuration loaded?"
        ));
        assert!(prompt_requires_investigation("Where are tasks created?"));
        assert!(prompt_requires_investigation(
            "Find where session creation happens"
        ));
    }

    #[test]
    fn prompt_requires_investigation_detects_rendered_lookup() {
        assert!(prompt_requires_investigation(
            "where is git status rendered"
        ));
        assert!(prompt_requires_investigation(
            "Find where status is rendered"
        ));
    }

    #[test]
    fn prompt_requires_investigation_detects_search_verb() {
        assert!(prompt_requires_investigation(
            "Search for 'task' in sandbox/ and explain what parts of the system use it."
        ));
        assert!(prompt_requires_investigation("Search for task in sandbox/"));
        assert!(prompt_requires_investigation(
            "search the codebase for SessionLog"
        ));
    }

    #[test]
    fn prompt_requires_investigation_detects_occurrence_phrasing() {
        assert!(prompt_requires_investigation(
            "Find all occurrences of 'logging' in sandbox/ and summarize where it appears."
        ));
        assert!(prompt_requires_investigation(
            "Find all occurrences of logging in sandbox/"
        ));
        assert!(prompt_requires_investigation(
            "Where does TaskStatus appear in the codebase?"
        ));
        assert!(prompt_requires_investigation(
            "Find where the error occurs in this module."
        ));
        assert!(prompt_requires_investigation(
            "Where are completed tasks filtered in sandbox/"
        ));
    }

    #[test]
    fn prompt_requires_investigation_rejects_plain_questions() {
        assert!(!prompt_requires_investigation("How are you?"));
        assert!(!prompt_requires_investigation("What time is it?"));
        assert!(!prompt_requires_investigation(
            "Can you help me with something?"
        ));
        assert!(!prompt_requires_investigation(
            "What is the purpose of this project?"
        ));
        assert!(!prompt_requires_investigation(
            "Find a good approach to this problem."
        ));
        assert!(!prompt_requires_investigation(
            "Find the best way to structure this."
        ));
    }

    #[test]
    fn mutation_intent_classifier_ignores_tool_name_mentions() {
        assert!(!user_requested_mutation("Where is write_file implemented?"));
        assert!(!user_requested_mutation("How does edit_file recover?"));
        assert!(user_requested_mutation("Create a file named demo.txt"));
        assert!(user_requested_mutation(
            "Edit src/main.rs and change hello to hi"
        ));
    }

    #[test]
    fn requested_read_path_detects_explicit_file_reads() {
        assert_eq!(
            requested_read_path("Read missing_file_phase84x.rs").as_deref(),
            Some("missing_file_phase84x.rs")
        );
        assert_eq!(
            requested_read_path("read file `src/runtime/engine.rs`").as_deref(),
            Some("src/runtime/engine.rs")
        );
        assert_eq!(requested_read_path("Read about logging"), None);
    }

    #[test]
    fn extract_investigation_path_scope_detects_in_pattern() {
        assert_eq!(
            extract_investigation_path_scope("Where is TaskStatus handled in sandbox/cli/"),
            Some("sandbox/cli/".into())
        );
        assert_eq!(
            extract_investigation_path_scope(
                "Find where logging is initialized in sandbox/services/"
            ),
            Some("sandbox/services/".into())
        );
        assert_eq!(
            extract_investigation_path_scope("Where is TaskStatus used in sandbox/"),
            Some("sandbox/".into())
        );
    }

    #[test]
    fn extract_investigation_path_scope_detects_the_before_path() {
        assert_eq!(
            extract_investigation_path_scope(
                "Find where database is configured in the sandbox/ folder"
            ),
            Some("sandbox/".into())
        );
    }

    #[test]
    fn extract_investigation_path_scope_detects_within_pattern() {
        assert_eq!(
            extract_investigation_path_scope("Find TaskStatus within sandbox/cli/"),
            Some("sandbox/cli/".into())
        );
    }

    #[test]
    fn extract_investigation_path_scope_rejects_plain_words() {
        assert_eq!(
            extract_investigation_path_scope("Find X in the application"),
            None
        );
        assert_eq!(
            extract_investigation_path_scope("Where is X used in context"),
            None
        );
        assert_eq!(
            extract_investigation_path_scope("What does run_turns do?"),
            None
        );
    }

    #[test]
    fn extract_investigation_path_scope_rejects_urls() {
        assert_eq!(
            extract_investigation_path_scope("Find X in https://example.com/path"),
            None
        );
    }

    #[test]
    fn extract_investigation_path_scope_returns_none_for_ambiguous_multiple_paths() {
        assert_eq!(
            extract_investigation_path_scope("Find X in sandbox/a/ and in sandbox/b/"),
            None
        );
    }

    #[test]
    fn extract_investigation_path_scope_strips_trailing_punctuation() {
        assert_eq!(
            extract_investigation_path_scope("Where is TaskStatus in sandbox/cli?"),
            Some("sandbox/cli".into())
        );
        assert_eq!(
            extract_investigation_path_scope("Find X in sandbox/services/."),
            Some("sandbox/services/".into())
        );
    }
}
