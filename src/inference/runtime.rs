use std::sync::mpsc::Sender;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use tracing::{debug, info, warn};

use crate::debug_log;
use crate::error::{ParamsError, Result};
use crate::events::{InferenceEvent, ProgressStatus, ProgressTrace};

pub(super) fn emit_generation_started(
    token_tx: &Sender<InferenceEvent>,
    label: &str,
    show_placeholder: bool,
) {
    let _ = token_tx.send(InferenceEvent::GenerationStarted {
        label: label.to_string(),
        show_placeholder,
    });
}

pub(super) fn emit_trace(
    token_tx: &Sender<InferenceEvent>,
    status: ProgressStatus,
    label: &str,
    persist: bool,
) {
    match status {
        ProgressStatus::Started => info!(label, "trace.started"),
        ProgressStatus::Updated => debug!(label, "trace.updated"),
        ProgressStatus::Finished => info!(label, "trace.finished"),
        ProgressStatus::Failed => warn!(label, "trace.failed"),
    }
    let _ = token_tx.send(InferenceEvent::Trace(ProgressTrace {
        status,
        label: label.to_string(),
        persist,
    }));
}

pub(super) fn log_debug_response(enabled: bool, text: &str, source: debug_log::ResponseSource) {
    if !enabled || text.trim().is_empty() {
        return;
    }

    if let Err(e) = debug_log::append_assistant_response(text, source) {
        warn!(error = %e, "debug assistant response logging failed");
    }
}

pub(super) fn effective_reflection(reflection_requested: bool, eco_enabled: bool) -> bool {
    reflection_requested && !eco_enabled
}

pub(super) fn summary_limit(eco_enabled: bool) -> usize {
    if eco_enabled {
        2
    } else {
        4
    }
}

pub(super) fn eco_tool_result_limit(eco_enabled: bool) -> Option<usize> {
    if eco_enabled {
        Some(1200)
    } else {
        None
    }
}

pub(super) fn emit_buffered_tokens(token_tx: &Sender<InferenceEvent>, text: &str) {
    if text.trim().is_empty() {
        return;
    }

    let mut chunk = String::new();
    let mut visible = 0usize;
    for ch in text.chars() {
        chunk.push(ch);
        visible += 1;
        let boundary = ch.is_whitespace() || matches!(ch, '.' | ',' | ';' | ':' | '!' | '?');
        if visible >= 32 && boundary {
            let _ = token_tx.send(InferenceEvent::Token(chunk.clone()));
            chunk.clear();
            visible = 0;
        }
    }

    if !chunk.is_empty() {
        let _ = token_tx.send(InferenceEvent::Token(chunk));
    }
}

pub(super) struct GuardedStreamRun {
    pub text: String,
    pub streamed: bool,
}

enum StreamPrefixDecision {
    Wait,
    Allow,
    Block,
}

fn is_identifier_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn looks_like_tool_tag_prefix(text: &str) -> StreamPrefixDecision {
    let trimmed = text.trim_start();
    let Some(rest) = trimmed.strip_prefix('[') else {
        return StreamPrefixDecision::Allow;
    };
    let mut saw_name = false;
    for ch in rest.chars() {
        if is_identifier_char(ch) {
            saw_name = true;
            continue;
        }
        return match ch {
            ':' if saw_name => StreamPrefixDecision::Block,
            ']' if saw_name => StreamPrefixDecision::Block,
            _ => StreamPrefixDecision::Allow,
        };
    }
    StreamPrefixDecision::Wait
}

fn guarded_stream_prefix_decision(text: &str) -> StreamPrefixDecision {
    let trimmed = text.trim_start();
    if trimmed.is_empty() {
        return StreamPrefixDecision::Wait;
    }

    if trimmed.starts_with("```") {
        return StreamPrefixDecision::Block;
    }
    if trimmed.starts_with('`') {
        return if trimmed.len() >= 2 && !trimmed[1..].starts_with('`') {
            StreamPrefixDecision::Allow
        } else {
            StreamPrefixDecision::Wait
        };
    }
    if trimmed.starts_with('[') {
        return looks_like_tool_tag_prefix(trimmed);
    }

    if trimmed.chars().count() >= 8 || trimmed.chars().any(char::is_whitespace) {
        StreamPrefixDecision::Allow
    } else {
        StreamPrefixDecision::Wait
    }
}

pub(super) fn run_and_collect_with_stream_guard(
    backend: &dyn super::InferenceBackend,
    messages: &[super::Message],
    token_tx: Sender<InferenceEvent>,
) -> Result<GuardedStreamRun> {
    debug!(
        message_count = messages.len(),
        "run_and_collect_with_stream_guard started"
    );

    let buffer = Arc::new(Mutex::new(String::new()));
    let streamed = Arc::new(AtomicBool::new(false));
    let buffer_clone = buffer.clone();
    let streamed_clone = streamed.clone();
    let (intercept_tx, intercept_rx) = std::sync::mpsc::channel::<InferenceEvent>();

    let relay_token_tx = token_tx.clone();
    let relay = std::thread::spawn(move || {
        let mut pending = String::new();
        let mut relay_enabled = false;

        while let Ok(event) = intercept_rx.recv() {
            if let InferenceEvent::Token(t) = &event {
                if let Ok(mut buf) = buffer_clone.lock() {
                    buf.push_str(t);
                }

                if relay_enabled {
                    let _ = relay_token_tx.send(event);
                    streamed_clone.store(true, Ordering::Relaxed);
                    continue;
                }

                pending.push_str(t);
                match guarded_stream_prefix_decision(&pending) {
                    StreamPrefixDecision::Wait => {}
                    StreamPrefixDecision::Allow => {
                        if !pending.is_empty() {
                            let _ = relay_token_tx.send(InferenceEvent::Token(pending.clone()));
                            streamed_clone.store(true, Ordering::Relaxed);
                            pending.clear();
                        }
                        relay_enabled = true;
                    }
                    StreamPrefixDecision::Block => {}
                }
            }
        }

        if !relay_enabled
            && !pending.is_empty()
            && matches!(
                guarded_stream_prefix_decision(&pending),
                StreamPrefixDecision::Allow
            )
        {
            let _ = relay_token_tx.send(InferenceEvent::Token(pending));
            streamed_clone.store(true, Ordering::Relaxed);
        }
    });

    backend.generate(messages, intercept_tx)?;
    relay
        .join()
        .map_err(|_| ParamsError::Inference("guarded token relay thread panicked".to_string()))?;

    let result = buffer.lock().map(|b| b.clone()).unwrap_or_default();

    debug!(
        chars = result.chars().count(),
        streamed = streamed.load(Ordering::Relaxed),
        "run_and_collect_with_stream_guard completed"
    );

    Ok(GuardedStreamRun {
        text: result,
        streamed: streamed.load(Ordering::Relaxed),
    })
}

pub(super) fn run_and_collect(
    backend: &dyn super::InferenceBackend,
    messages: &[super::Message],
    token_tx: Sender<InferenceEvent>,
    stream_tokens: bool,
) -> Result<String> {
    debug!(
        stream_tokens,
        message_count = messages.len(),
        "run_and_collect started"
    );

    let buffer = Arc::new(Mutex::new(String::new()));
    let buffer_clone = buffer.clone();
    let (intercept_tx, intercept_rx) = std::sync::mpsc::channel::<InferenceEvent>();

    let relay_token_tx = token_tx.clone();
    let relay = std::thread::spawn(move || {
        while let Ok(event) = intercept_rx.recv() {
            if let InferenceEvent::Token(t) = &event {
                if let Ok(mut buf) = buffer_clone.lock() {
                    buf.push_str(t);
                }
                if stream_tokens {
                    let _ = relay_token_tx.send(event);
                }
            }
        }
    });

    backend.generate(messages, intercept_tx)?;
    relay
        .join()
        .map_err(|_| ParamsError::Inference("token relay thread panicked".to_string()))?;

    let result = buffer.lock().map(|b| b.clone()).unwrap_or_default();

    debug!(
        chars = result.chars().count(),
        stream_tokens, "run_and_collect completed"
    );

    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::{self, Sender};

    use super::{
        effective_reflection, emit_buffered_tokens, guarded_stream_prefix_decision,
        run_and_collect_with_stream_guard, StreamPrefixDecision,
    };
    use crate::events::InferenceEvent;
    use crate::inference::{InferenceBackend, Message};
    use crate::Result;

    struct ChunkedBackend {
        chunks: Vec<&'static str>,
    }

    impl InferenceBackend for ChunkedBackend {
        fn generate(&self, _messages: &[Message], tx: Sender<InferenceEvent>) -> Result<()> {
            for chunk in &self.chunks {
                let _ = tx.send(InferenceEvent::Token((*chunk).to_string()));
            }
            Ok(())
        }

        fn name(&self) -> String {
            "chunked".to_string()
        }
    }

    #[test]
    fn eco_mode_disables_effective_reflection() {
        assert!(effective_reflection(true, false));
        assert!(!effective_reflection(true, true));
        assert!(!effective_reflection(false, false));
    }

    #[test]
    fn buffered_tokens_split_long_text_into_multiple_events() {
        let (tx, rx) = mpsc::channel();
        emit_buffered_tokens(
            &tx,
            "This is a fairly long tool-loop answer that should be replayed in chunks instead of one giant token.",
        );
        let chunks = rx
            .try_iter()
            .filter_map(|event| match event {
                InferenceEvent::Token(text) => Some(text),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert!(chunks.len() >= 2, "expected multiple buffered token events");
        assert_eq!(chunks.concat(), "This is a fairly long tool-loop answer that should be replayed in chunks instead of one giant token.");
    }

    #[test]
    fn guarded_stream_prefix_blocks_tool_tags_and_code_fences() {
        assert!(matches!(
            guarded_stream_prefix_decision("[read_file: src/main.rs]"),
            StreamPrefixDecision::Block
        ));
        assert!(matches!(
            guarded_stream_prefix_decision("```rust\nfn main() {}\n```"),
            StreamPrefixDecision::Block
        ));
        assert!(matches!(
            guarded_stream_prefix_decision("`src/main.rs` is the entrypoint."),
            StreamPrefixDecision::Allow
        ));
    }

    #[test]
    fn guarded_stream_forwards_normal_final_answer_tokens() {
        let backend = ChunkedBackend {
            chunks: vec!["This file ", "starts the CLI."],
        };
        let (tx, rx) = mpsc::channel();
        let run = run_and_collect_with_stream_guard(
            &backend,
            &[Message::user("What does this file do?")],
            tx,
        )
        .expect("guarded run");

        assert!(run.streamed);
        assert_eq!(run.text, "This file starts the CLI.");
        let streamed = rx
            .try_iter()
            .filter_map(|event| match event {
                InferenceEvent::Token(text) => Some(text),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(streamed, "This file starts the CLI.");
    }

    #[test]
    fn guarded_stream_suppresses_tool_tag_prefix() {
        let backend = ChunkedBackend {
            chunks: vec!["[read_file: src/main.rs]"],
        };
        let (tx, rx) = mpsc::channel();
        let run = run_and_collect_with_stream_guard(
            &backend,
            &[Message::user("What does this file do?")],
            tx,
        )
        .expect("guarded run");

        assert!(!run.streamed);
        assert_eq!(run.text, "[read_file: src/main.rs]");
        assert!(rx.try_iter().next().is_none());
    }
}
