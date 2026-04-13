use std::collections::HashSet;

use super::{
    SkippedFactReason, TurnMemoryEvidence, GENERIC_ANCHOR_TOKENS, MAX_FACT_LEN,
    MAX_TOOL_EVIDENCE_CHARS, MIN_FACT_LEN,
};

pub(crate) fn validate_candidate_fact(
    fact: &str,
    anchors: &HashSet<String>,
) -> std::result::Result<(), SkippedFactReason> {
    if !is_quality_fact(fact) {
        return Err(SkippedFactReason::Quality);
    }
    if contains_hedged_language(fact) {
        return Err(SkippedFactReason::Hedged);
    }
    if contains_unresolved_language(fact) {
        return Err(SkippedFactReason::Unresolved);
    }
    if !contains_anchor(fact, anchors) {
        return Err(SkippedFactReason::Unanchored);
    }
    Ok(())
}

pub(crate) fn evidence_anchors(evidence: &TurnMemoryEvidence) -> HashSet<String> {
    let mut anchors = HashSet::new();
    collect_anchor_tokens(&evidence.user_prompt, &mut anchors);
    for (path, summary) in &evidence.summaries {
        anchors.insert(path.to_lowercase());
        collect_anchor_tokens(path, &mut anchors);
        collect_anchor_tokens(summary, &mut anchors);
    }
    for tool in &evidence.tool_results {
        collect_anchor_tokens(&tool.tool_name, &mut anchors);
        collect_anchor_tokens(&tool.argument, &mut anchors);
        collect_anchor_tokens(
            &clip_text(&tool.output, MAX_TOOL_EVIDENCE_CHARS),
            &mut anchors,
        );
    }
    anchors
}

pub(crate) fn collect_anchor_tokens(text: &str, anchors: &mut HashSet<String>) {
    for token in tokenize(text) {
        if token.len() < 4 {
            continue;
        }
        if is_project_anchor_token(&token) {
            anchors.insert(token.to_lowercase());
        }
    }
}

pub(crate) fn is_project_anchor_token(token: &str) -> bool {
    let lower = token.to_ascii_lowercase();
    if GENERIC_ANCHOR_TOKENS.contains(&lower.as_str()) {
        return false;
    }

    token.contains('/')
        || token.contains('.')
        || token.contains("::")
        || token.contains('_')
        || token.contains("://")
        || token.contains('-')
        || token.starts_with('/')
        || token.chars().skip(1).any(|c| c.is_ascii_uppercase())
        || (token.chars().any(|c| c.is_ascii_digit())
            && token.chars().any(|c| c.is_ascii_alphabetic()))
}

pub(crate) fn contains_anchor(fact: &str, anchors: &HashSet<String>) -> bool {
    let fact_lower = fact.to_lowercase();
    tokenize(fact).into_iter().any(|token| {
        let token = token.to_lowercase();
        anchors.contains(&token) || fact_lower.contains(&token) && anchors.contains(&token)
    })
}

pub(crate) fn contains_project_anchor_text(text: &str) -> bool {
    tokenize(text)
        .into_iter()
        .filter(|token| token.len() >= 4)
        .any(|token| is_project_anchor_token(&token))
}

pub(crate) fn is_retrievable_project_fact(fact: &str) -> bool {
    is_quality_fact(fact)
        && !contains_hedged_language(fact)
        && !contains_unresolved_language(fact)
        && contains_project_anchor_text(fact)
}

pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|token| {
            token
                .trim_matches(|c: char| {
                    matches!(
                        c,
                        ',' | '.'
                            | ':'
                            | ';'
                            | '('
                            | ')'
                            | '['
                            | ']'
                            | '{'
                            | '}'
                            | '"'
                            | '\''
                            | '`'
                    )
                })
                .to_string()
        })
        .filter(|token| !token.is_empty())
        .collect()
}

pub(crate) fn clip_text(text: &str, max_chars: usize) -> String {
    let total = text.chars().count();
    if total <= max_chars {
        return text.to_string();
    }
    let clipped: String = text.chars().take(max_chars).collect();
    format!("{clipped}…")
}

pub(crate) fn is_quality_fact(fact: &str) -> bool {
    if fact.len() < MIN_FACT_LEN || fact.len() > MAX_FACT_LEN {
        return false;
    }
    if fact.ends_with('?') {
        return false;
    }
    let first = fact.chars().next().unwrap_or('x');
    if first.is_ascii_digit() || matches!(first, '-' | '*') {
        return false;
    }
    let lower = fact.to_lowercase();
    let meta_prefixes = [
        "the user ",
        "the assistant ",
        "the conversation ",
        "the session ",
        "in this conversation",
        "in this session",
        "this conversation",
        "this session",
        "i was asked",
        "we discussed",
        "the code was",
        "the developer ",
    ];
    for prefix in meta_prefixes {
        if lower.starts_with(prefix) {
            return false;
        }
    }
    if looks_like_code_snippet(fact) {
        return false;
    }
    if looks_like_summary_boilerplate(fact) {
        return false;
    }
    let low_value_phrases = [
        "the contents of ",
        " are as follows",
        "typically contains",
        "for a rust project",
        "this file (`",
        "is the entry point of the rust project",
        "defines the main behavior of the cli tool",
        "smart pointers like",
        "other languages also have pointers",
        "managed by the rust compiler and runtime",
    ];
    for phrase in low_value_phrases {
        if lower.contains(phrase) {
            return false;
        }
    }
    true
}

pub(crate) fn looks_like_code_snippet(fact: &str) -> bool {
    let trimmed = fact.trim();
    let lower = trimmed.to_ascii_lowercase();

    if lower.starts_with("let ")
        || lower.starts_with("const ")
        || lower.starts_with("var ")
        || lower.starts_with("fn ")
    {
        return true;
    }

    let has_assignment = trimmed.contains('=');
    let has_semicolon = trimmed.contains(';');
    let has_comment = trimmed.contains("//");
    let has_slice = trimmed.contains("&") || trimmed.contains("::");
    let token_count = trimmed.split_whitespace().count();

    (has_assignment && has_semicolon && token_count <= 16)
        || (has_comment && has_assignment)
        || (has_slice && has_semicolon && token_count <= 12)
}

pub(crate) fn looks_like_summary_boilerplate(fact: &str) -> bool {
    let trimmed = fact.trim();
    let lower = trimmed.to_ascii_lowercase();

    if !(lower.contains(": describes ")
        || lower.starts_with("this document ")
        || lower.starts_with("this doc ")
        || lower.starts_with("this file "))
    {
        return false;
    }

    lower.contains("describes ")
        || lower.contains("documents ")
        || lower.contains("outlines ")
        || lower.contains("covers ")
}

pub(crate) fn contains_hedged_language(fact: &str) -> bool {
    let lower = fact.to_lowercase();
    ["maybe", "seems", "likely", "probably", "might", "could"]
        .iter()
        .any(|word| lower.contains(word))
}

pub(crate) fn contains_unresolved_language(fact: &str) -> bool {
    let lower = fact.to_lowercase();
    [
        "todo",
        "fixme",
        "should ",
        "needs ",
        "need to",
        "plan to",
        "consider",
        "want to",
        "later",
        "next step",
        "we can ",
        "can replace",
        "will allow us to",
    ]
    .iter()
    .any(|word| lower.contains(word))
}

pub(crate) fn are_near_duplicate(a: &str, b: &str) -> bool {
    let tokens_a: HashSet<&str> = a.split_whitespace().collect();
    let tokens_b: HashSet<&str> = b.split_whitespace().collect();

    if tokens_a.is_empty() || tokens_b.is_empty() {
        return false;
    }

    let intersection = tokens_a.intersection(&tokens_b).count();
    let union = tokens_a.union(&tokens_b).count();

    if union == 0 {
        return false;
    }

    let jaccard = intersection as f64 / union as f64;
    jaccard >= 0.70
}
