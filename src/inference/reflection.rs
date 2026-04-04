use std::path::Path;
use std::sync::mpsc::Sender;

use tracing::{info, warn};

use crate::cache::ExactCache;
use crate::config;
use crate::error::Result;
use crate::events::InferenceEvent;

use super::budget::{
    estimate_message_tokens, record_generation_budget, SessionBudget, SessionCacheStats,
};
use super::cache::{generate_with_cache, CacheMode};
use super::runtime::emit_generation_started;
use super::{InferenceBackend, Message};

pub(super) fn reflect_response(
    backend: &dyn InferenceBackend,
    cfg: &config::Config,
    project_root: &Path,
    budget: &mut SessionBudget,
    token_tx: &Sender<InferenceEvent>,
    exact_cache: Option<&ExactCache>,
    cache_stats: &mut SessionCacheStats,
    session_messages: &[Message],
    draft: &str,
) -> Result<String> {
    if draft.trim().is_empty() {
        return Ok(String::new());
    }

    info!(
        draft_chars = draft.chars().count(),
        "reflection pass started"
    );
    let reflection_messages = build_reflection_messages(session_messages, draft);
    let prompt_tokens = estimate_message_tokens(&reflection_messages);
    emit_generation_started(token_tx, "reflecting...", false);
    let reflected = generate_with_cache(
        backend,
        &reflection_messages,
        cfg,
        project_root,
        token_tx.clone(),
        true,
        exact_cache,
        cache_stats,
        CacheMode::ExactOnly,
    )?;
    if !reflected.hit {
        record_generation_budget(cfg, budget, token_tx, prompt_tokens, &reflected.text);
    }

    if reflected.text.trim().is_empty() || looks_like_reflection_meta(&reflected.text) {
        warn!("reflection returned empty or meta response, falling back to draft");
        Ok(draft.to_string())
    } else {
        info!(
            chars = reflected.text.chars().count(),
            "reflection pass completed"
        );
        Ok(reflected.text)
    }
}

fn build_reflection_messages(session_messages: &[Message], draft: &str) -> Vec<Message> {
    let mut messages = vec![Message::system(
        "You are a reflection pass for params-cli. Rewrite the assistant draft into the final user-facing answer. Fix correctness, safety, unsupported claims, missed repo/tool context, and clarity issues. Return only the final answer itself. Never talk about the draft, reflection, review, edits, or whether the answer was already good. Do not call tools. Keep good answers concise.",
    )];

    for message in session_messages {
        if message.role != "system" {
            messages.push(message.clone());
        }
    }

    messages.push(Message::assistant(draft));
    messages.push(Message::user(
        "Rewrite the assistant draft above as the final answer to the user. If the draft is already good, return the same answer text with only minimal edits. Do not add reviewer commentary. Return only the final answer text.",
    ));
    messages
}

fn looks_like_reflection_meta(text: &str) -> bool {
    let trimmed = text.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return false;
    }

    let meta_markers = [
        "the draft is already good",
        "the draft is good",
        "no further edits needed",
        "no changes needed",
        "no revisions needed",
        "the answer is already good",
        "minimal edits",
        "review the assistant draft",
        "reviewer commentary",
    ];

    meta_markers.iter().any(|marker| trimmed.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflection_messages_exclude_original_system_prompt() {
        let session_messages = vec![
            Message::system("tool-heavy system prompt"),
            Message::user("question"),
            Message::assistant("draft context"),
        ];

        let reflection = build_reflection_messages(&session_messages, "final draft");

        assert_eq!(reflection[0].role, "system");
        assert!(!reflection
            .iter()
            .skip(1)
            .any(|m| m.content == "tool-heavy system prompt"));
        assert_eq!(reflection[1].role, "user");
        assert_eq!(reflection[1].content, "question");
        assert_eq!(reflection[2].role, "assistant");
        assert_eq!(reflection[2].content, "draft context");
        assert_eq!(reflection[3].role, "assistant");
        assert_eq!(reflection[3].content, "final draft");
        assert_eq!(reflection[4].role, "user");
    }

    #[test]
    fn reflection_meta_text_falls_back_to_draft() {
        assert!(looks_like_reflection_meta(
            "The draft is already good. No further edits needed."
        ));
        assert!(looks_like_reflection_meta("No changes needed."));
        assert!(!looks_like_reflection_meta(
            "A pointer stores the memory address of another value."
        ));
    }
}
