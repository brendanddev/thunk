#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::inference) enum ToolLoopIntent {
    RepoOverview,
    DirectoryOverview,
    CodeNavigation,
    ConfigLocate,
    CallSiteLookup,
    UsageLookup,
    FlowTrace,
}

pub(super) fn normalize_intent_text(text: &str) -> String {
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

pub(super) fn is_referential_file_prompt(prompt: &str) -> bool {
    let normalized = normalize_intent_text(prompt);
    matches!(
        normalized.as_str(),
        "what does this file do"
            | "what does the current file do"
            | "what does the loaded file do"
            | "what is this file for"
            | "explain this file"
            | "describe this file"
            | "summarize this file"
    )
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

fn salient_search_token(phrase: &str, intent: ToolLoopIntent) -> Option<String> {
    let stopwords = match intent {
        ToolLoopIntent::CodeNavigation => &[
            "implemented",
            "implement",
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
            "trace",
            "how",
            "does",
            "work",
            "works",
            "flow",
            "what",
            "writes",
            "to",
            "are",
            "saved",
            "save",
            "restored",
            "restore",
            "explain",
            "describe",
            "show",
            "walk",
            "through",
            "call",
            "calls",
            "use",
            "uses",
            "do",
            "me",
            "who",
        ][..],
        ToolLoopIntent::ConfigLocate => &[
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
        ToolLoopIntent::FlowTrace => &[
            "trace", "explain", "describe", "show", "walk", "through", "how", "does", "work",
            "works", "flow", "what", "the", "is", "me",
        ][..],
        ToolLoopIntent::CallSiteLookup | ToolLoopIntent::UsageLookup => &[
            "what", "calls", "call", "who", "uses", "use", "the", "is", "where", "which", "file",
            "has", "how", "does", "find",
        ][..],
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview => &[][..],
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

pub(super) fn suggested_search_query(prompt: &str, intent: ToolLoopIntent) -> Option<String> {
    if is_referential_file_prompt(prompt) {
        return None;
    }

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
    if intent == ToolLoopIntent::ConfigLocate && normalized.contains("eco") {
        return Some("eco.enabled".to_string());
    }

    if matches!(
        intent,
        ToolLoopIntent::CallSiteLookup | ToolLoopIntent::UsageLookup
    ) {
        let lower = prompt.to_ascii_lowercase();
        let lower_trimmed = lower.trim();
        for prefix in ["what calls ", "who calls ", "what uses "] {
            if let Some(rest) = lower_trimmed.strip_prefix(prefix) {
                let subject = rest
                    .trim_end_matches(|c: char| c == '?' || c == '.' || c == '!')
                    .trim();
                if !subject.is_empty() {
                    return Some(subject.to_string());
                }
            }
        }
    }
    if intent == ToolLoopIntent::CodeNavigation {
        let lower = prompt.to_ascii_lowercase();
        let lower_trimmed = lower.trim();
        if let Some(rest) = lower_trimmed.strip_prefix("what does ") {
            let rest_clean = rest
                .trim_end_matches(|c: char| c == '?' || c == '.' || c == '!')
                .trim();
            if let Some(subject) = rest_clean.strip_suffix(" do") {
                let subject = subject.trim();
                if !subject.is_empty() {
                    return Some(subject.to_string());
                }
            }
        }
    }

    let extracted = match intent {
        ToolLoopIntent::CodeNavigation => {
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
            } else if let Some(rest) = normalized.strip_prefix("what handles ") {
                rest.trim().to_string()
            } else if let Some(rest) = normalized.strip_prefix("what writes to ") {
                rest.trim().to_string()
            } else if let Some(rest) = normalized.strip_prefix("what does ") {
                trim_query_suffix(rest, &[" do"]).to_string()
            } else {
                String::new()
            }
        }
        ToolLoopIntent::FlowTrace => {
            if let Some(rest) = normalized.strip_prefix("trace how ") {
                trim_query_suffix(rest, &[" works", " work", " flows", " flow"]).to_string()
            } else if let Some(rest) = normalized.strip_prefix("how does ") {
                trim_query_suffix(rest, &[" work", " works", " flow", " flow through"]).to_string()
            } else if let Some(rest) = normalized.strip_prefix("explain how ") {
                trim_query_suffix(rest, &[" works", " work", " flows", " flow"]).to_string()
            } else if let Some(rest) = normalized.strip_prefix("xplain how ") {
                trim_query_suffix(rest, &[" works", " work", " flows", " flow"]).to_string()
            } else if let Some(rest) = normalized.strip_prefix("describe how ") {
                trim_query_suffix(rest, &[" works", " work", " flows", " flow"]).to_string()
            } else if let Some(rest) = normalized.strip_prefix("show me how ") {
                trim_query_suffix(rest, &[" works", " work"]).to_string()
            } else if let Some(rest) = normalized.strip_prefix("show how ") {
                trim_query_suffix(rest, &[" works", " work"]).to_string()
            } else if let Some(rest) = normalized.strip_prefix("walk me through ") {
                rest.trim().to_string()
            } else if let Some(rest) = normalized.strip_prefix("walk through ") {
                rest.trim().to_string()
            } else {
                String::new()
            }
        }
        ToolLoopIntent::CallSiteLookup | ToolLoopIntent::UsageLookup => String::new(),
        ToolLoopIntent::ConfigLocate => {
            if let Some(rest) = normalized.strip_prefix("where is ") {
                trim_query_suffix(rest, &[" configured", " configged", " set"]).to_string()
            } else if let Some(rest) = normalized.strip_prefix("which file configures ") {
                rest.trim().to_string()
            } else {
                String::new()
            }
        }
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview => String::new(),
    };

    let cleaned =
        salient_search_token(&extracted, intent).unwrap_or_else(|| trim_query_noise(&extracted));
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

pub(super) fn detect_tool_loop_intent(prompt: &str) -> Option<ToolLoopIntent> {
    let normalized = normalize_intent_text(prompt);
    if normalized.starts_with('/') {
        return None;
    }

    let tokens = normalized
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let first_contains = |needle: &str| {
        tokens
            .first()
            .map(|token| token.contains(needle))
            .unwrap_or(false)
    };
    let second_is = |value: &str| tokens.get(1).map(|token| token == value).unwrap_or(false);
    let has_token = |value: &str| tokens.iter().any(|token| token == value);
    let has_prefix = |prefix: &str| tokens.iter().any(|token| token.starts_with(prefix));

    if ((first_contains("what") && second_is("is"))
        || (first_contains("whats") && (second_is("in") || second_is("here")))
        || (first_contains("summarize") && second_is("this")))
        && (has_token("repo")
            || has_token("project")
            || has_token("codebase")
            || has_token("directory")
            || has_token("folder")
            || has_token("here"))
    {
        if has_token("directory") || has_token("folder") || has_token("here") {
            return Some(ToolLoopIntent::DirectoryOverview);
        }
        return Some(ToolLoopIntent::RepoOverview);
    }

    if normalized.starts_with("what calls ") || normalized.starts_with("who calls ") {
        return Some(ToolLoopIntent::CallSiteLookup);
    }

    if normalized.starts_with("what uses ") {
        return Some(ToolLoopIntent::UsageLookup);
    }

    if normalized.starts_with("trace how ")
        || normalized.starts_with("how does ")
        || normalized.starts_with("explain how ")
        || normalized.starts_with("xplain how ")
        || normalized.starts_with("describe how ")
        || normalized.starts_with("show me how ")
        || normalized.starts_with("show how ")
        || normalized.starts_with("walk me through ")
        || normalized.starts_with("walk through ")
    {
        if has_prefix("config") {
            return Some(ToolLoopIntent::ConfigLocate);
        }
        return Some(ToolLoopIntent::FlowTrace);
    }

    if (first_contains("where") && second_is("is"))
        || normalized.starts_with("find ")
        || normalized.starts_with("which file has ")
        || normalized.starts_with("what handles ")
        || normalized.starts_with("what writes to ")
        || (normalized.starts_with("what does ") && normalized.ends_with(" do"))
    {
        if has_prefix("config") || has_token("set") {
            return Some(ToolLoopIntent::ConfigLocate);
        }
        if has_prefix("implement")
            || has_prefix("defin")
            || has_prefix("handl")
            || normalized.starts_with("what handles ")
            || normalized.starts_with("what writes to ")
            || normalized.starts_with("find ")
            || normalized.starts_with("which file has ")
            || (normalized.starts_with("what does ") && normalized.ends_with(" do"))
        {
            return Some(ToolLoopIntent::CodeNavigation);
        }
    }

    None
}
