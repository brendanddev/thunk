use super::types::AutoInspectIntent;

pub(crate) fn detect_auto_inspect_intent(prompt: &str) -> Option<AutoInspectIntent> {
    let normalized = normalize_intent_text(prompt);
    let tokens = normalized
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    if normalized.starts_with('/') {
        return None;
    }

    let starts_with = |a: &str, b: &str| {
        tokens.first().map(|t| t == a).unwrap_or(false)
            && tokens.get(1).map(|t| t == b).unwrap_or(false)
    };
    let has_token = |value: &str| tokens.iter().any(|token| token == value);
    let has_prefix = |prefix: &str| tokens.iter().any(|token| token.starts_with(prefix));

    if (starts_with("what", "is") || starts_with("whats", "in") || starts_with("what", "does"))
        && (has_token("repo") || has_token("project") || has_token("codebase"))
    {
        return Some(AutoInspectIntent::RepoOverview);
    }

    if starts_with("summarize", "this")
        && (has_token("repo") || has_token("project") || has_token("codebase"))
    {
        return Some(AutoInspectIntent::RepoOverview);
    }

    if (starts_with("what", "is")
        || starts_with("whats", "in")
        || starts_with("whats", "here")
        || starts_with("what", "here"))
        && (has_token("directory") || has_token("folder") || has_token("here"))
    {
        return Some(AutoInspectIntent::DirectoryOverview);
    }

    if starts_with("where", "is")
        && (has_prefix("implement") || has_prefix("defin") || has_prefix("handl"))
    {
        return Some(AutoInspectIntent::WhereIsImplementation);
    }

    if normalized.starts_with("find ") || normalized.starts_with("which file has ") {
        return Some(AutoInspectIntent::WhereIsImplementation);
    }

    if normalized.starts_with("trace how ")
        || normalized.starts_with("how does ")
        || normalized.starts_with("what handles ")
        || normalized.starts_with("what writes to ")
    {
        return Some(AutoInspectIntent::FeatureTrace);
    }

    if starts_with("where", "is") && (has_prefix("config") || has_token("set")) {
        return Some(AutoInspectIntent::ConfigLocate);
    }

    if normalized.starts_with("which file configures ") {
        return Some(AutoInspectIntent::ConfigLocate);
    }

    None
}

pub(crate) fn normalize_intent_text(text: &str) -> String {
    let stripped = text.to_ascii_lowercase().replace(['\'', '’'], "");
    stripped
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

fn trim_query_noise(query: &str) -> String {
    let mut trimmed = query.trim().to_string();
    for prefix in ["the ", "a ", "an ", "this ", "that "] {
        if let Some(stripped) = trimmed.strip_prefix(prefix) {
            trimmed = stripped.trim().to_string();
        }
    }
    trimmed
}

fn trim_query_suffix<'a>(query: &'a str, suffixes: &[&str]) -> &'a str {
    for suffix in suffixes {
        if let Some(stripped) = query.strip_suffix(suffix) {
            return stripped.trim();
        }
    }
    query.trim()
}

fn singularize_token(token: &str) -> String {
    if token.len() > 4 && token.ends_with('s') {
        token[..token.len() - 1].to_string()
    } else {
        token.to_string()
    }
}

fn salient_search_token(phrase: &str, intent: AutoInspectIntent) -> Option<String> {
    let stopwords = match intent {
        AutoInspectIntent::WhereIsImplementation => &[
            "implemented",
            "define",
            "defined",
            "handle",
            "handled",
            "find",
            "which",
            "file",
            "has",
            "where",
            "is",
            "the",
            "project",
        ][..],
        AutoInspectIntent::FeatureTrace => &[
            "trace", "how", "does", "work", "works", "flow", "what", "handles", "writes", "to",
            "are", "saved", "save", "restored", "restore", "the",
        ][..],
        AutoInspectIntent::ConfigLocate => &[
            "configured",
            "configures",
            "set",
            "mode",
            "which",
            "file",
            "where",
            "is",
            "the",
        ][..],
        AutoInspectIntent::RepoOverview | AutoInspectIntent::DirectoryOverview => &[][..],
    };

    phrase
        .split_whitespace()
        .map(trim_query_noise)
        .map(|token| singularize_token(&token))
        .filter(|token| {
            !token.is_empty()
                && !stopwords.iter().any(|stop| stop == token)
                && token.chars().any(|ch| ch.is_ascii_alphanumeric())
        })
        .max_by(|a, b| {
            token_specificity_score(a)
                .cmp(&token_specificity_score(b))
                .then_with(|| a.len().cmp(&b.len()))
        })
}

fn token_specificity_score(token: &str) -> usize {
    let len = token.len();
    let alpha_bonus = if token.chars().all(|ch| ch.is_ascii_alphabetic()) {
        1
    } else {
        0
    };
    let suffix_bonus = if token.ends_with("ing")
        || token.ends_with("tion")
        || token.ends_with("ment")
        || token.ends_with("al")
    {
        2
    } else {
        0
    };

    len + alpha_bonus + suffix_bonus
}

pub(crate) fn extract_auto_inspect_query(
    prompt: &str,
    intent: AutoInspectIntent,
) -> Option<String> {
    let normalized = normalize_intent_text(prompt);
    if normalized.contains("session") {
        if normalized.contains("save") || normalized.contains("saved") {
            return Some("save_messages".to_string());
        }
        if normalized.contains("restore")
            || normalized.contains("restored")
            || normalized.contains("resume")
        {
            return Some("load_most_recent".to_string());
        }
    }
    if intent == AutoInspectIntent::ConfigLocate && normalized.contains("eco") {
        return Some("eco.enabled".to_string());
    }

    let extracted = match intent {
        AutoInspectIntent::WhereIsImplementation => {
            if let Some(rest) = normalized.strip_prefix("where is ") {
                trim_query_suffix(
                    rest,
                    &[
                        " implemented",
                        " defined",
                        " handled",
                        " configured",
                        " configged",
                    ],
                )
                .to_string()
            } else if let Some(rest) = normalized.strip_prefix("find ") {
                rest.trim().to_string()
            } else if let Some(rest) = normalized.strip_prefix("which file has ") {
                rest.trim().to_string()
            } else {
                String::new()
            }
        }
        AutoInspectIntent::FeatureTrace => {
            if let Some(rest) = normalized.strip_prefix("trace how ") {
                trim_query_suffix(rest, &[" works", " work", " flows", " flow"]).to_string()
            } else if let Some(rest) = normalized.strip_prefix("how does ") {
                trim_query_suffix(rest, &[" work", " works", " flow", " flow through"]).to_string()
            } else if let Some(rest) = normalized.strip_prefix("what handles ") {
                rest.trim().to_string()
            } else if let Some(rest) = normalized.strip_prefix("what writes to ") {
                rest.trim().to_string()
            } else {
                String::new()
            }
        }
        AutoInspectIntent::ConfigLocate => {
            if let Some(rest) = normalized.strip_prefix("where is ") {
                trim_query_suffix(rest, &[" configured", " configged", " set"]).to_string()
            } else if let Some(rest) = normalized.strip_prefix("which file configures ") {
                rest.trim().to_string()
            } else {
                String::new()
            }
        }
        AutoInspectIntent::RepoOverview | AutoInspectIntent::DirectoryOverview => String::new(),
    };

    let cleaned =
        salient_search_token(&extracted, intent).unwrap_or_else(|| trim_query_noise(&extracted));
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}
