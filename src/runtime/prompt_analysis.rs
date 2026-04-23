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
