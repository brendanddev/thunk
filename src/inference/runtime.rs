use std::sync::mpsc::Sender;

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

pub(super) fn run_and_collect(
    backend: &dyn super::InferenceBackend,
    messages: &[super::Message],
    token_tx: Sender<InferenceEvent>,
    stream_tokens: bool,
) -> Result<String> {
    use std::sync::{Arc, Mutex};

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
    use super::effective_reflection;

    #[test]
    fn eco_mode_disables_effective_reflection() {
        assert!(effective_reflection(true, false));
        assert!(!effective_reflection(true, true));
        assert!(!effective_reflection(false, false));
    }
}
