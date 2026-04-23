use crate::tools::ToolInput;

pub(super) fn simplify_search_query(query: &str) -> String {
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

pub(super) fn simplify_search_input(input: &mut ToolInput) {
    if let ToolInput::SearchCode { query, .. } = input {
        let simplified = simplify_search_query(query);
        if !simplified.is_empty() && simplified != *query {
            *query = simplified;
        }
    }
}

pub(super) fn weak_search_query_reason(query: &str) -> Option<&'static str> {
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
