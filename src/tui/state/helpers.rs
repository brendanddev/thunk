use crate::commands::CommandSuggestion;

use super::TranscriptPresentation;

pub(super) fn command_match_score(entry: &CommandSuggestion, query: &str) -> Option<(u8, usize)> {
    if query.is_empty() {
        return Some((0, 0));
    }

    let name = entry.name.to_lowercase();
    if name == query {
        return Some((0, 0));
    }
    if name.starts_with(query) {
        return Some((1, name.len()));
    }
    if entry
        .aliases
        .iter()
        .map(|alias| alias.to_lowercase())
        .any(|alias| alias == query)
    {
        return Some((2, 0));
    }
    if entry
        .aliases
        .iter()
        .map(|alias| alias.to_lowercase())
        .any(|alias| alias.starts_with(query))
    {
        return Some((3, 0));
    }
    if name.contains(query) {
        return Some((4, name.len()));
    }
    if entry.usage.to_lowercase().contains(query) {
        return Some((5, entry.usage.len()));
    }
    if entry.description.to_lowercase().contains(query) {
        return Some((6, entry.description.len()));
    }
    if entry.group.to_lowercase().contains(query) {
        return Some((7, entry.group.len()));
    }

    None
}

pub(super) fn group_rank(group: &str) -> u8 {
    match group {
        "context" => 0,
        "action" => 1,
        "session" => 2,
        "help" => 3,
        "custom" => 4,
        _ => 5,
    }
}

pub(super) fn describe_session_age(saved_at: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let age = now.saturating_sub(saved_at);
    if age < 120 {
        "moments ago".to_string()
    } else if age < 3600 {
        let m = age / 60;
        format!("{m}m ago")
    } else if age < 86400 {
        let h = age / 3600;
        format!("{h}h ago")
    } else {
        let d = age / 86400;
        format!("{d}d ago")
    }
}

pub(super) fn summarize_trace_steps(steps: &[String]) -> String {
    match steps {
        [] => String::new(),
        [only] => only.clone(),
        [first, second] => format!("{first} -> {second}"),
        [first, second, third] => format!("{first} -> {second} -> {third}"),
        _ => format!(
            "{} -> {} -> {} (+{})",
            steps[0],
            steps[1],
            steps[2],
            steps.len() - 3
        ),
    }
}

impl TranscriptPresentation {
    fn plain() -> Self {
        Self {
            collapsible: false,
            collapsed: false,
            summary: None,
            preview_lines: Vec::new(),
        }
    }
}

pub(super) fn transcript_presentation_for_content(content: &str) -> TranscriptPresentation {
    classify_collapsible_context(content).unwrap_or_else(TranscriptPresentation::plain)
}

fn classify_collapsible_context(content: &str) -> Option<TranscriptPresentation> {
    if content.starts_with("Tool results:\n") {
        let tool_count = content
            .lines()
            .filter(|line| line.starts_with("--- "))
            .count()
            .max(1);
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some(format!(
                "tool results • {tool_count} tool{}",
                if tool_count == 1 { "" } else { "s" }
            )),
            preview_lines: extract_preview_lines(content, &["Tool results:", ""], 2),
        });
    }

    if content.starts_with("I've loaded this file for context:") {
        let file_label = extract_value_after_label(content, "File:")
            .unwrap_or_else(|| "file context".to_string());
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some(format!("file context • {file_label}")),
            preview_lines: extract_preview_lines(
                content,
                &["I've loaded this file for context:", ""],
                2,
            ),
        });
    }

    if content.starts_with("Directory listing:") {
        let dir = extract_value_after_label(content, "Directory:")
            .unwrap_or_else(|| "directory".to_string());
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some(format!("directory listing • {dir}")),
            preview_lines: extract_preview_lines(
                content,
                &["Directory listing:", "", &format!("Directory: {dir}")],
                2,
            ),
        });
    }

    if content.starts_with("Search results:\n") {
        let query = content
            .lines()
            .find_map(|line| {
                line.strip_prefix("Search results for '")
                    .and_then(|rest| rest.split_once('\'').map(|(query, _)| query.to_string()))
            })
            .unwrap_or_else(|| "query output".to_string());
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some(format!("search results • {query}")),
            preview_lines: extract_preview_lines(content, &["Search results:", ""], 2),
        });
    }

    if content.starts_with("Git context (") {
        let mode = content
            .strip_prefix("Git context (")
            .and_then(|rest| rest.split_once("):").map(|(mode, _)| mode.to_string()))
            .unwrap_or_else(|| "status".to_string());
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some(format!("git context • {mode}")),
            preview_lines: extract_preview_lines(content, &[&format!("Git context ({mode}):")], 2),
        });
    }

    if content.starts_with("LSP diagnostics:\n") {
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some("diagnostics".to_string()),
            preview_lines: extract_preview_lines(content, &["LSP diagnostics:", ""], 2),
        });
    }

    if content.starts_with("LSP check:\n") {
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some("rust lsp check".to_string()),
            preview_lines: extract_preview_lines(content, &["LSP check:", ""], 2),
        });
    }

    if content.starts_with("LSP hover:") {
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some("hover".to_string()),
            preview_lines: extract_preview_lines(content, &["LSP hover:", ""], 2),
        });
    }

    if content.starts_with("LSP definition:") {
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some("definition".to_string()),
            preview_lines: extract_preview_lines(content, &["LSP definition:", ""], 2),
        });
    }

    if content.starts_with("Fetched web context:\n") {
        let url =
            extract_value_after_label(content, "Fetched URL:").unwrap_or_else(|| "web".to_string());
        let host = url
            .split("://")
            .nth(1)
            .and_then(|rest| rest.split('/').next())
            .unwrap_or(url.as_str())
            .to_string();
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some(format!("web context • {host}")),
            preview_lines: extract_preview_lines(content, &["Fetched web context:", ""], 2),
        });
    }

    None
}

fn extract_preview_lines(content: &str, skip_lines: &[&str], max_lines: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || skip_lines.iter().any(|skip| trimmed == *skip) {
            continue;
        }
        lines.push(trimmed.to_string());
        if lines.len() >= max_lines {
            break;
        }
    }
    if lines.is_empty() {
        vec!["(no preview)".to_string()]
    } else {
        lines
    }
}

fn extract_value_after_label(content: &str, label: &str) -> Option<String> {
    content
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix(label)
                .map(|value| value.trim().to_string())
        })
        .filter(|value| !value.is_empty())
}

pub(super) fn is_injected_context(content: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "Tool results:\n",
        "I've loaded this file for context:",
        "Directory listing:",
        "Search results:\n",
        "Git context (",
        "LSP diagnostics:\n",
        "LSP check:\n",
        "LSP hover:",
        "LSP definition:",
        "Fetched web context:\n",
        "User rejected proposed action:",
    ];
    PREFIXES.iter().any(|p| content.starts_with(p))
}
