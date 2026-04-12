use tracing::{debug, info};

use crate::inference::Message;
use crate::inference::StructuredCompressionContext;

/// Non-system messages above this count trigger compression (8 turns)
const COMPRESSION_THRESHOLD: usize = 16;
const ECO_COMPRESSION_THRESHOLD: usize = 10;

/// Message pairs to preserve at the tail (user + assistant each)
const KEEP_PAIRS: usize = 3;
const ECO_KEEP_PAIRS: usize = 2;

const STRUCTURED_CONTEXT_PREFIX: &str = "Structured conversation context:";

/// Compress `messages` in-place when history exceeds the threshold.
///
/// The compressed block is deterministic and structured so long sessions keep
/// the active goals, decisions, recent repo investigation state, and open
/// questions without flattening everything into a freeform narrative blob.
pub fn compress_history(
    messages: &mut Vec<Message>,
    eco_enabled: bool,
    investigation_context: Option<&StructuredCompressionContext>,
) {
    let non_system = messages.iter().filter(|m| m.role != "system").count();
    let threshold = if eco_enabled {
        ECO_COMPRESSION_THRESHOLD
    } else {
        COMPRESSION_THRESHOLD
    };

    if non_system <= threshold {
        return;
    }

    debug!(
        messages = non_system,
        threshold, eco_enabled, "history over threshold, compressing"
    );

    let history_start = if messages.first().map(|m| m.role.as_str()) == Some("system") {
        1
    } else {
        0
    };

    let keep_pairs = if eco_enabled {
        ECO_KEEP_PAIRS
    } else {
        KEEP_PAIRS
    };
    let keep_tail = keep_pairs * 2;

    if messages.len() <= history_start + keep_tail {
        return;
    }

    let compress_end = messages.len() - keep_tail;
    let to_summarize = &messages[history_start..compress_end];
    if to_summarize.is_empty() {
        return;
    }

    let summary = build_structured_context(to_summarize, investigation_context);
    if summary.trim().is_empty() {
        return;
    }

    let tail: Vec<Message> = messages.drain(compress_end..).collect();
    messages.truncate(history_start);
    messages.push(Message::user(&summary));
    messages.extend(tail);

    info!(
        original = non_system,
        compressed_to = messages.len().saturating_sub(history_start),
        "session history compressed"
    );
}

fn build_structured_context(
    messages: &[Message],
    investigation_context: Option<&StructuredCompressionContext>,
) -> String {
    let mut goals = Vec::new();
    let mut decisions = Vec::new();
    let mut open_questions = Vec::new();
    let mut grounded_facts = Vec::new();

    for message in messages {
        let content = message.content.trim();
        if content.is_empty() || is_structured_context_message(content) {
            continue;
        }

        match message.role.as_str() {
            "user" => {
                if !is_injected_context_message(content) {
                    if is_decision_like(content) {
                        push_unique(&mut decisions, clip_sentence(content));
                    } else {
                        push_unique(&mut goals, clip_sentence(content));
                    }
                    if content.ends_with('?') {
                        push_unique(&mut open_questions, clip_sentence(content));
                    }
                } else if let Some(fact) = grounded_fact_from_context(content) {
                    push_unique(&mut grounded_facts, fact);
                }
            }
            "assistant" => {
                if content.contains(':') && content.contains("src/") {
                    push_unique(&mut grounded_facts, clip_sentence(content));
                }
            }
            _ => {}
        }
    }

    let mut lines = vec![STRUCTURED_CONTEXT_PREFIX.to_string()];

    if !goals.is_empty() {
        lines.push("Active goals:".to_string());
        for goal in goals.into_iter().take(4) {
            lines.push(format!("- {goal}"));
        }
    }

    if !decisions.is_empty() {
        lines.push("Accepted decisions:".to_string());
        for decision in decisions.into_iter().take(4) {
            lines.push(format!("- {decision}"));
        }
    }

    if let Some(context) = investigation_context {
        lines.push("Current technical investigation state:".to_string());
        for line in context.render().lines().skip(1) {
            lines.push(format!("- {line}"));
        }
    }

    if !grounded_facts.is_empty() {
        lines.push("Grounded repo facts:".to_string());
        for fact in grounded_facts.into_iter().take(4) {
            lines.push(format!("- {fact}"));
        }
    }

    if !open_questions.is_empty() {
        lines.push("Pending open questions:".to_string());
        for question in open_questions.into_iter().take(3) {
            lines.push(format!("- {question}"));
        }
    }

    lines.join("\n")
}

fn is_structured_context_message(content: &str) -> bool {
    content.starts_with(STRUCTURED_CONTEXT_PREFIX)
        || content.starts_with("Structured investigation context:")
}

fn is_injected_context_message(content: &str) -> bool {
    content.starts_with("I've loaded this file for context:")
        || content.starts_with("Directory listing:")
        || content.starts_with("Search results:")
        || content.starts_with("Git context (")
        || content.starts_with("LSP diagnostics:")
        || content.starts_with("LSP hover:")
        || content.starts_with("LSP definition:")
        || content.starts_with("LSP check:")
        || content.starts_with("Fetched web context:")
}

fn is_decision_like(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    [
        "i want", "we want", "please", "prefer", "keep", "do not", "dont", "don't", "should",
        "need to",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn grounded_fact_from_context(content: &str) -> Option<String> {
    if let Some(path) = content
        .lines()
        .find_map(|line| line.strip_prefix("File: ").map(str::trim))
    {
        return Some(format!("Loaded `{path}` for context."));
    }
    if let Some(path) = content
        .lines()
        .find_map(|line| line.strip_prefix("Directory: ").map(str::trim))
    {
        return Some(format!("Listed directory `{path}`."));
    }
    if let Some(query) = content
        .lines()
        .find_map(|line| line.strip_prefix("Search results for '"))
        .and_then(|rest| {
            rest.split_once('\'')
                .map(|(query, _)| query.trim().to_string())
        })
    {
        return Some(format!("Searched for `{query}`."));
    }
    None
}

fn clip_sentence(text: &str) -> String {
    let text = text.replace('\n', " ");
    let clipped = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if clipped.chars().count() <= 140 {
        clipped
    } else {
        let truncated = clipped
            .chars()
            .take(139)
            .collect::<String>()
            .trim_end()
            .to_string();
        format!("{truncated}…")
    }
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if value.is_empty() || values.iter().any(|existing| existing == &value) {
        return;
    }
    values.push(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compression_emits_structured_context_sections() {
        let mut messages = vec![
            Message::system("system"),
            Message::user("Please make repo navigation more agentic."),
            Message::assistant("I inspected src/inference/tool_loop.rs:120 and found weak routing."),
            Message::user("I've loaded this file for context:\n\nFile: src/main.rs\n\n```rust\nfn main() {}\n```"),
            Message::assistant("Noted."),
            Message::user("Can you trace logging next?"),
            Message::assistant("Working on it."),
            Message::user("Do not use canned answers."),
            Message::assistant("Understood."),
            Message::user("Please keep the tool loop authoritative for repo questions."),
            Message::assistant("Agreed."),
            Message::user("Can you preserve the current slash commands?"),
            Message::assistant("Yes."),
            Message::user("What open questions are left?"),
            Message::assistant("Mostly routing and compression."),
            Message::user("Please avoid summary-first drift."),
            Message::assistant("I will."),
            Message::user("Can you keep repo grounding concise too?"),
        ];

        compress_history(
            &mut messages,
            false,
            Some(&StructuredCompressionContext {
                active_investigation: Some("flow trace".to_string()),
                recent_files: vec!["src/main.rs".to_string()],
                recent_directories: vec!["src".to_string()],
                recent_searches: vec!["init_logging".to_string()],
                top_anchor: Some("init_logging".to_string()),
            }),
        );

        let compressed = messages
            .iter()
            .find(|message| message.content.starts_with(STRUCTURED_CONTEXT_PREFIX))
            .expect("structured context should be inserted");
        assert!(compressed.content.contains("Active goals:"));
        assert!(compressed.content.contains("Accepted decisions:"));
        assert!(compressed
            .content
            .contains("Current technical investigation state:"));
        assert!(compressed.content.contains("Grounded repo facts:"));
    }

    #[test]
    fn compression_skips_existing_structured_context_messages() {
        let output = build_structured_context(
            &[
                Message::user("Structured conversation context:\nActive goals:\n- old"),
                Message::user("Please keep tool-first routing."),
            ],
            None,
        );

        assert!(!output.contains("old"));
        assert!(output.contains("Please keep tool-first routing."));
    }
}
