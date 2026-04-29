use crate::tools::ToolInput;

/// Extracts a single strong token from a natural-language query.
///
/// Drops common stopwords and returns the first meaningful identifier-like
/// token. Falls back to the original query when no better token is found.
pub(crate) fn simplify_search_query(query: &str) -> String {
    const STOPWORDS: &[&str] = &[
        "a",
        "an",
        "and",
        "are",
        "fn",
        "for",
        "find",
        "how",
        "in",
        "initialize",
        "initialized",
        "initialization",
        "implemented",
        "is",
        "of",
        "or",
        "the",
        "to",
        "what",
        "where",
    ];

    let trimmed = query.trim();
    for raw in trimmed.split(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                '\\' | '(' | ')' | '[' | ']' | '{' | '}' | '.' | ',' | ';' | ':' | '"' | '\'' | '`'
            )
    }) {
        let token = raw.trim_matches(|c: char| !(c.is_ascii_alphanumeric() || c == '_'));
        if token.is_empty() {
            continue;
        }
        let lower = token.to_ascii_lowercase();
        if !STOPWORDS.contains(&lower.as_str()) {
            return token.to_string();
        }
    }

    trimmed.to_string()
}

/// Applies query simplification in-place for SearchCode inputs.
///
/// Ensures the runtime always sends a minimally useful query to the tool.
pub(crate) fn simplify_search_input(input: &mut ToolInput) {
    if let ToolInput::SearchCode { query, .. } = input {
        let simplified = simplify_search_query(query);
        if !simplified.is_empty() && simplified != *query {
            *query = simplified;
        }
    }
}

/// Classifies weak search queries for runtime guardrails.
///
/// Returns a reason when the query is too weak to be useful, allowing
/// deterministic correction/termination behavior.
pub(crate) fn weak_search_query_reason(query: &str) -> Option<&'static str> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return Some("empty");
    }
    if trimmed.chars().count() < 3 {
        return Some("too_short");
    }
    if trimmed.eq_ignore_ascii_case("git") {
        return Some("git");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weak_search_query_reason_rejects_only_approved_structural_cases() {
        assert_eq!(weak_search_query_reason(""), Some("empty"));
        assert_eq!(weak_search_query_reason("   "), Some("empty"));
        assert_eq!(weak_search_query_reason("g"), Some("too_short"));
        assert_eq!(weak_search_query_reason("gt"), Some("too_short"));
        assert_eq!(weak_search_query_reason("git"), Some("git"));
        assert_eq!(weak_search_query_reason("GIT"), Some("git"));

        for allowed in [
            "status",
            "diff",
            "log",
            "file",
            "code",
            "src",
            "main",
            "test",
            "tests",
            "git_status",
            "src/runtime",
        ] {
            assert_eq!(
                weak_search_query_reason(allowed),
                None,
                "query should not be rejected by the first-pass weak-query guard: {allowed}"
            );
        }
    }

    #[test]
    fn search_query_simplification_prefers_single_literal_keyword() {
        assert_eq!(simplify_search_query("logging initialization"), "logging");
        assert_eq!(simplify_search_query("logger initialization"), "logger");
        assert_eq!(simplify_search_query("write_file()"), "write_file");
        assert_eq!(simplify_search_query("sessions saved"), "sessions");
        assert_eq!(simplify_search_query("fn main"), "main");
        assert_eq!(simplify_search_query(r"logging\.init\(\)"), "logging");
    }
}
