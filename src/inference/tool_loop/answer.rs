use std::sync::mpsc::Sender;

use tracing::info;

use crate::error::{ParamsError, Result};
use crate::events::InferenceEvent;
use crate::tools::ToolResult;

use super::super::investigation::InvestigationResolution;
use super::super::runtime::{emit_buffered_tokens, run_and_collect_with_stream_guard};
use super::super::{InferenceBackend, Message};
use super::evidence::{
    grounded_answer_guidance, render_structured_answer, validate_final_answer, StructuredEvidence,
};
use super::intent::ToolLoopIntent;
use super::prompting::is_referential_follow_up;

// ---------------------------------------------------------------------------
// Answer source types
// ---------------------------------------------------------------------------

/// Which code path produced the final answer delivered to the user.
///
/// This is carried on `ToolLoopOutcome` so callers can observe, log, and
/// eventually make decisions based on which path fired — without having to
/// infer it from the boolean `streamed_final_response` flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AnswerSource {
    /// The model produced a grounded prose answer via the synthesis pass and
    /// the answer passed evidence validation. Tokens were streamed to the
    /// client by `run_and_collect_with_stream_guard` inside `finalize_answer`.
    ModelSynthesis,
    /// Either (a) no synthesis guidance was available, (b) the synthesis pass
    /// produced junk / was blocked by the stream guard, (c) the model answer
    /// failed evidence validation, or (d) evidence collection was insufficient.
    /// In all these cases the answer text was emitted via `emit_buffered_tokens`.
    DeterministicFallback,
}

/// The result of a finalization attempt: the final answer text paired with
/// the source that produced it.
pub(super) struct AnswerAttempt {
    pub text: String,
    pub source: AnswerSource,
}

// ---------------------------------------------------------------------------
// Final answer generation
// ---------------------------------------------------------------------------

/// Attempt to produce the final structured answer for a completed investigation.
///
/// Tries model synthesis first (grounded prose via `run_and_collect_with_stream_guard`).
/// Falls back to deterministic template rendering if:
/// - no guidance is available for the intent
/// - the stream guard blocked the output (tool-tag prefix)
/// - synthesis produced junk (too short / starts with tool tag)
/// - evidence validation rejected the synthesized answer
/// - the synthesis call errored
///
/// **Token emission contract (unchanged from before Phase 2):**
/// - `ModelSynthesis`: tokens were already streamed to the user inside this
///   function via the stream guard. The caller must NOT emit them again.
/// - `DeterministicFallback`: tokens are emitted here via `emit_buffered_tokens`
///   before returning. The caller must NOT emit them again.
///
/// In both cases `ToolLoopOutcome::streamed_final_response` is set to `true`
/// so the session layer does not attempt a second emission.
#[allow(clippy::too_many_arguments)]
pub(super) fn finalize_answer(
    intent: ToolLoopIntent,
    prompt: &str,
    results: &[ToolResult],
    resolution: Option<&InvestigationResolution>,
    base_messages: &[Message],
    evidence: &StructuredEvidence,
    backend: &dyn InferenceBackend,
    token_tx: &Sender<InferenceEvent>,
) -> Result<AnswerAttempt> {
    // Primary path: model synthesis pass.
    if let Some(guidance) = grounded_answer_guidance(intent, prompt, resolution, results) {
        let synthesis_messages = build_synthesis_messages(prompt, &guidance, base_messages);
        match run_and_collect_with_stream_guard(backend, &synthesis_messages, token_tx.clone()) {
            Ok(run) if run.streamed && !is_synthesis_junk(&run.text) => {
                if validate_final_answer(&run.text, evidence, results) {
                    info!(
                        intent = ?intent,
                        chars = run.text.len(),
                        "answer finalized via model synthesis"
                    );
                    return Ok(AnswerAttempt {
                        text: run.text,
                        source: AnswerSource::ModelSynthesis,
                    });
                }
                // Synthesized answer failed evidence validation.
                info!(
                    intent = ?intent,
                    chars = run.text.len(),
                    "synthesis answer failed validation; falling back to deterministic template"
                );
            }
            Ok(_) => {
                // Stream guard blocked output, synthesis was not streamed, or answer was junk.
                info!(
                    intent = ?intent,
                    "synthesis blocked or produced junk; falling back to deterministic template"
                );
            }
            Err(e) => {
                info!(
                    intent = ?intent,
                    error = %e,
                    "synthesis pass errored; falling back to deterministic template"
                );
            }
        }
    } else {
        info!(
            intent = ?intent,
            "no synthesis guidance available; using deterministic template"
        );
    }

    // Fallback: deterministic template rendering emitted as simulated streaming.
    let text = render_structured_answer(prompt, evidence);
    if text.trim().is_empty() {
        return Err(ParamsError::Inference(
            "Structured evidence produced an empty final answer".to_string(),
        ));
    }
    info!(
        intent = ?intent,
        chars = text.len(),
        "answer finalized via deterministic fallback"
    );
    emit_buffered_tokens(token_tx, &text);
    Ok(AnswerAttempt {
        text,
        source: AnswerSource::DeterministicFallback,
    })
}

// ---------------------------------------------------------------------------
// Synthesis helpers
// ---------------------------------------------------------------------------

/// Build a tight message set for the synthesis pass.
///
/// For referential follow-ups ("Tell me more") the last two non-system messages
/// from the live session are included so the model can expand the prior answer.
/// For standalone questions no prior context is included.
pub(super) fn build_synthesis_messages(
    prompt: &str,
    guidance: &str,
    base_messages: &[Message],
) -> Vec<Message> {
    let is_follow_up = is_referential_follow_up(prompt);
    let system = if is_follow_up {
        "You are answering a code-navigation question from observed file evidence. \
         Write in natural language prose — do NOT emit tool tags or code fences. \
         Be concise: 2–5 sentences or a short structured list. \
         If a prior answer appears above, expand it with new detail rather than repeating it. \
         Ignore unrelated prior context and stay anchored to the observed evidence guidance. \
         Reuse prior assistant context only when it matches the current observed evidence guidance."
    } else {
        "You are answering a code-navigation question from observed file evidence. \
         Write in natural language prose — do NOT emit tool tags or code fences. \
         Be concise: 2–5 sentences or a short structured list. \
         Treat the current prompt as a standalone question. \
         Ignore unrelated prior conversation context and answer only from the observed evidence guidance. \
         Do not reuse earlier assistant claims unless they are restated in the current observed evidence guidance. \
         Do not invent behavior, logging, message counts, or helper steps that are not present in the evidence."
    };

    let mut messages = vec![Message::system(system)];

    let tail: Vec<_> = if is_follow_up {
        base_messages
            .iter()
            .filter(|m| m.role != "system")
            .rev()
            .take(2)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    } else {
        Vec::new()
    };

    let already_has_prompt = tail
        .last()
        .map(|m| m.role == "user" && m.content == prompt)
        .unwrap_or(false);

    messages.extend(tail);
    if !already_has_prompt {
        messages.push(Message::user(prompt));
    }

    messages.push(Message::user(guidance));
    messages
}

/// Returns true if the synthesis output looks like a tool tag or is too short
/// to be a real answer. Used to detect when the stream guard missed a bad output.
fn is_synthesis_junk(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.len() < 8 {
        return true;
    }
    // Looks like a tool tag prefix: `[toolname: ...]`
    if trimmed.starts_with('[') {
        let inner = &trimmed[1..];
        let colon_pos = inner.find(':').unwrap_or(usize::MAX);
        let name_end = inner
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .unwrap_or(usize::MAX);
        if colon_pos <= 32 && name_end == colon_pos {
            return true;
        }
    }
    false
}
