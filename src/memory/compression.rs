// src/memory/compression.rs
//
// Level 1: session compression.
//
// When a conversation grows beyond COMPRESSION_THRESHOLD non-system messages
// (8 turns = 16 messages), we send the older portion to the backend for
// summarization and collapse it into a single context message. The last
// KEEP_PAIRS pairs (user + assistant) are always preserved verbatim so the
// model has immediate context for the current exchange.
//
// This function is called from inference/mod.rs inside model_thread(), where
// the backend reference is available. It cannot be called from state.rs
// (build_messages returns a plain Vec with no backend access) or from the
// UI thread (blocking generation would freeze the TUI).
//
// Compression is best-effort. If the backend call fails or the summary is
// empty, we log a warning and leave the message history unchanged.

use tracing::{debug, info, warn};

use crate::inference::{InferenceBackend, Message};
use super::run_prompt_sync;

/// Non-system messages above this count trigger compression (8 turns).
const COMPRESSION_THRESHOLD: usize = 16;

/// Message pairs to preserve at the tail (user + assistant each).
const KEEP_PAIRS: usize = 3;

/// Compress `messages` in-place when history exceeds the threshold.
///
/// After compression the Vec contains:
///   [system_prompt, summary_message, ...last KEEP_PAIRS pairs]
///
/// If there is no system prompt the function still works, treating index 0
/// as the first history message.
pub fn compress_history(messages: &mut Vec<Message>, backend: &dyn InferenceBackend) {
    let non_system = messages.iter().filter(|m| m.role != "system").count();

    if non_system <= COMPRESSION_THRESHOLD {
        return;
    }

    debug!(
        messages = non_system,
        threshold = COMPRESSION_THRESHOLD,
        "history over threshold, compressing"
    );

    // Index where real history starts (after optional system prompt).
    let history_start = if messages.first().map(|m| m.role.as_str()) == Some("system") {
        1
    } else {
        0
    };

    let keep_tail = KEEP_PAIRS * 2;

    // Guard: nothing meaningful left to summarize.
    if messages.len() <= history_start + keep_tail {
        return;
    }

    let compress_end = messages.len() - keep_tail;
    let to_summarize = &messages[history_start..compress_end];

    if to_summarize.is_empty() {
        return;
    }

    // Render the messages to be summarized as plain text.
    let conversation_text: String = to_summarize
        .iter()
        .map(|m| format!("{}: {}\n", m.role, m.content))
        .collect();

    let prompt = vec![
        Message::system(
            "You are a helpful assistant that writes concise conversation summaries.",
        ),
        Message::user(&format!(
            "Summarize the following conversation in 3-5 sentences. \
             Focus on what was discussed, which files were examined, and any \
             decisions or conclusions reached. Be concise.\n\n\
             {conversation_text}"
        )),
    ];

    match run_prompt_sync(backend, &prompt) {
        Ok(summary) if !summary.trim().is_empty() => {
            let summary_msg = Message::user(&format!(
                "Previous conversation summary: {}",
                summary.trim()
            ));

            // Drain the tail, truncate to history_start, re-attach.
            let tail: Vec<Message> = messages.drain(compress_end..).collect();
            messages.truncate(history_start);
            messages.push(summary_msg);
            messages.extend(tail);

            info!(
                original = non_system,
                compressed_to = messages.len().saturating_sub(history_start),
                "session history compressed"
            );
        }
        Ok(_) => {
            warn!("summarization returned empty response, keeping full history");
        }
        Err(e) => {
            warn!(error = %e, "session compression failed, keeping full history");
        }
    }
}
