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
