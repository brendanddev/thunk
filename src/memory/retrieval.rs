use std::collections::HashSet;

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "this", "to", "was", "what", "when", "where", "which", "with",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryTerm {
    pub token: String,
    pub weight: usize,
}

pub fn query_terms(query: &str) -> Vec<QueryTerm> {
    let mut seen = HashSet::new();
    let mut terms = Vec::new();

    for raw in split_terms(query) {
        let token = normalize_term(raw);
        if token.is_empty() || token.len() < 2 || STOPWORDS.contains(&token.as_str()) {
            continue;
        }
        if seen.insert(token.clone()) {
            terms.push(QueryTerm {
                weight: token_weight(&token, raw),
                token,
            });
        }
    }

    terms
}

pub fn score_text(query: &[QueryTerm], candidate: &str) -> usize {
    if query.is_empty() {
        return 0;
    }

    let candidate_lower = candidate.to_lowercase();
    query
        .iter()
        .filter(|term| candidate_lower.contains(term.token.as_str()))
        .map(|term| term.weight)
        .sum()
}

pub fn clip_excerpt(text: &str, max_chars: usize) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let total = normalized.chars().count();
    if total <= max_chars {
        return normalized;
    }

    let mut clipped = String::new();
    for ch in normalized.chars().take(max_chars.saturating_sub(1)) {
        clipped.push(ch);
    }
    clipped.push('…');
    clipped
}

fn split_terms(text: &str) -> impl Iterator<Item = &str> {
    text.split_whitespace()
}

fn normalize_term(raw: &str) -> String {
    raw.trim_matches(|c: char| {
        matches!(
            c,
            ',' | ';' | '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\'' | '`' | '!' | '?'
        )
    })
    .to_ascii_lowercase()
}

fn token_weight(normalized: &str, raw: &str) -> usize {
    if raw.contains("::")
        || raw.contains('/')
        || raw.contains('.')
        || raw.contains('_')
        || raw.contains('-')
    {
        3
    } else if normalized.len() >= 6 {
        2
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::{clip_excerpt, query_terms, score_text};

    #[test]
    fn query_terms_drop_stopwords_and_keep_symbols() {
        let terms = query_terms("what is src/main.rs cache invalidation");
        let tokens = terms
            .iter()
            .map(|term| term.token.as_str())
            .collect::<Vec<_>>();

        assert!(!tokens.contains(&"what"));
        assert!(tokens.contains(&"src/main.rs"));
        assert!(tokens.contains(&"cache"));
    }

    #[test]
    fn score_text_prefers_path_and_symbol_like_terms() {
        let query = query_terms("src/session/mod.rs session resume");
        let score_with_path = score_text(&query, "src/session/mod.rs handles session resume");
        let score_without_path = score_text(&query, "session resume flow");

        assert!(score_with_path > score_without_path);
    }

    #[test]
    fn clip_excerpt_normalizes_whitespace_and_clips() {
        let clipped = clip_excerpt("line one\nline two\tline three", 12);
        assert!(clipped.starts_with("line one"));
        assert!(clipped.ends_with('…'));
    }
}
