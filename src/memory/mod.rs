// src/memory/mod.rs
//
// Three-level memory system.
//
// Level 1 (compression.rs) — session compression: when the conversation
//   grows past 8 turns, summarize old messages to stay within context limits.
//
// Level 2 (index.rs) — project index: store per-file summaries in SQLite at
//   .local/memory/{project_hash}.db for fast retrieval during sessions.
//
// Level 3 (facts.rs) — cross-session facts: extract key decisions and file
//   names from each session and inject them at the start of future sessions.
//
// All three levels degrade gracefully — errors are logged and the TUI keeps
// running. Nothing in this module should ever crash the application.

pub mod compression;
pub mod facts;
pub mod index;
pub mod retrieval;

use std::sync::mpsc;

use crate::error::Result;
use crate::events::InferenceEvent;
use crate::inference::{InferenceBackend, Message};

/// Run a prompt through the backend and collect the full response as a String
pub(crate) fn run_prompt_sync(
    backend: &dyn InferenceBackend,
    messages: &[Message],
) -> Result<String> {
    let (tx, rx) = mpsc::channel::<InferenceEvent>();
    backend.generate(messages, tx)?;

    let mut result = String::new();
    for event in rx.try_iter() {
        if let InferenceEvent::Token(t) = event {
            result.push_str(&t);
        }
    }

    Ok(result)
}
