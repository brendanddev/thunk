mod answer;
mod evidence;
mod intent;
mod prompting;

use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc::Sender;
use tracing::info;

use crate::cache::ExactCache;
use crate::config;
use crate::error::Result;
use crate::events::{InferenceEvent, ProgressStatus};
use crate::tools::{ReadOnlyToolExecution, ToolRegistry, ToolResult};

use super::budget::{
    estimate_message_tokens, record_generation_budget, SessionBudget, SessionCacheStats,
};
use super::cache::{generate_with_cache, CacheMode};
use super::runtime::{emit_buffered_tokens, emit_generation_started, emit_trace};
use super::investigation::InvestigationResolution;
use super::{InferenceBackend, Message};
use evidence::{
    bootstrap_tool_results, format_tool_loop_results_with_limit, grounded_answer_guidance,
    investigation_outcome, InvestigationOutcome,
};
pub(super) use answer::AnswerSource;
pub(super) use intent::ToolLoopIntent;

use prompting::{
    build_tool_loop_seed_messages, build_tool_loop_system_prompt, initial_investigation_hint,
    initial_tool_only_followup, thinking_label, tool_loop_budget, tool_loop_result_limit,
    with_progress_heartbeat,
};

#[derive(Clone, Copy)]
struct ToolLoopBudget {
    max_iterations: usize,
    max_duplicate_calls: usize,
}

pub(super) struct ToolLoopOutcome {
    pub final_response: String,
    pub tool_results: Vec<ToolResult>,
    pub streamed_final_response: bool,
    pub answer_source: AnswerSource,
}

pub(super) fn detect_tool_loop_intent(prompt: &str) -> Option<ToolLoopIntent> {
    intent::detect_tool_loop_intent(prompt)
}

fn investigation_outcome_label(outcome: &InvestigationOutcome) -> &'static str {
    match outcome {
        InvestigationOutcome::NeedsMore { .. } => "needs_more",
        InvestigationOutcome::Ready { .. } => "ready",
        InvestigationOutcome::Insufficient { .. } => "insufficient",
    }
}

fn tool_result_summary(results: &[ToolResult]) -> String {
    results
        .iter()
        .map(|result| format!("{}({})", result.tool_name, result.argument))
        .collect::<Vec<_>>()
        .join(", ")
}

fn repeated_tool_calls(
    counts: &mut HashMap<String, usize>,
    results: &[ToolResult],
    max_duplicate_calls: usize,
) -> bool {
    let mut repeated = false;
    for result in results {
        let key = format!("{}:{}", result.tool_name, result.argument);
        let count = counts.entry(key).or_insert(0);
        *count += 1;
        if *count > max_duplicate_calls {
            repeated = true;
        }
    }
    repeated
}

pub(super) fn run_read_only_tool_loop(
    intent: ToolLoopIntent,
    prompt: &str,
    base_messages: &[Message],
    backend: &dyn InferenceBackend,
    tools: &ToolRegistry,
    cfg: &config::Config,
    project_root: &Path,
    token_tx: &Sender<InferenceEvent>,
    exact_cache: Option<&ExactCache>,
    cache_stats: &mut SessionCacheStats,
    budget: &mut SessionBudget,
    eco_enabled: bool,
    _reflection_enabled: bool,
) -> Result<ToolLoopOutcome> {
    run_read_only_tool_loop_with_resolution(
        intent,
        prompt,
        None,
        base_messages,
        backend,
        tools,
        cfg,
        project_root,
        token_tx,
        exact_cache,
        cache_stats,
        budget,
        eco_enabled,
        _reflection_enabled,
    )
}

pub(super) fn run_read_only_tool_loop_with_resolution(
    intent: ToolLoopIntent,
    prompt: &str,
    resolution: Option<&InvestigationResolution>,
    base_messages: &[Message],
    backend: &dyn InferenceBackend,
    tools: &ToolRegistry,
    cfg: &config::Config,
    project_root: &Path,
    token_tx: &Sender<InferenceEvent>,
    exact_cache: Option<&ExactCache>,
    cache_stats: &mut SessionCacheStats,
    budget: &mut SessionBudget,
    eco_enabled: bool,
    _reflection_enabled: bool,
) -> Result<ToolLoopOutcome> {
    let loop_budget = tool_loop_budget(eco_enabled);
    let result_limit = tool_loop_result_limit(&backend.name(), eco_enabled);
    let (thinking, status_label) = thinking_label(intent);
    info!(
        intent = ?intent,
        prompt,
        backend = backend.name(),
        anchored_file = resolution.and_then(|value| value.anchored_file.as_deref()).unwrap_or(""),
        "tool loop started"
    );
    emit_generation_started(token_tx, status_label, false);
    emit_trace(token_tx, ProgressStatus::Started, thinking, false);

    let system_prompt = build_tool_loop_system_prompt(tools, project_root, intent, eco_enabled);
    let mut loop_messages = build_tool_loop_seed_messages(base_messages, &system_prompt, prompt);
    let mut all_tool_results = bootstrap_tool_results(
        intent,
        prompt,
        resolution,
        base_messages,
        project_root,
        &backend.name(),
        tools,
        token_tx,
    );
    let mut tool_call_counts = HashMap::new();
    repeated_tool_calls(
        &mut tool_call_counts,
        &all_tool_results,
        loop_budget.max_duplicate_calls,
    );
    if !all_tool_results.is_empty() {
        info!(
            intent = ?intent,
            bootstrap_results = all_tool_results.len(),
            results = tool_result_summary(&all_tool_results),
            "tool loop bootstrap gathered evidence"
        );
    }

    if all_tool_results.is_empty() {
        if let Some(hint) = initial_investigation_hint(intent, prompt) {
            loop_messages.push(Message::user(&hint));
        }
    } else {
        push_evidence_messages(
            &mut loop_messages,
            intent,
            prompt,
            resolution,
            &all_tool_results,
            result_limit,
        );
        match investigation_outcome(intent, prompt, resolution, &all_tool_results) {
            InvestigationOutcome::Ready {
                evidence,
                stop_reason,
            } => {
                info!(
                    intent = ?intent,
                    stop_reason,
                    tool_results = all_tool_results.len(),
                    "tool loop reached ready outcome from bootstrap"
                );
                emit_trace(token_tx, ProgressStatus::Finished, stop_reason, false);
                let attempt = answer::finalize_answer(
                    intent,
                    prompt,
                    &all_tool_results,
                    resolution,
                    base_messages,
                    &evidence,
                    backend,
                    token_tx,
                )?;
                return Ok(ToolLoopOutcome {
                    final_response: attempt.text,
                    tool_results: all_tool_results,
                    streamed_final_response: true,
                    answer_source: attempt.source,
                });
            }
            InvestigationOutcome::Insufficient { reason } => {
                info!(
                    intent = ?intent,
                    reason,
                    tool_results = all_tool_results.len(),
                    "tool loop bootstrap ended insufficient"
                );
                emit_trace(
                    token_tx,
                    ProgressStatus::Failed,
                    "insufficient evidence",
                    false,
                );
                emit_buffered_tokens(token_tx, &reason);
                return Ok(ToolLoopOutcome {
                    final_response: reason,
                    tool_results: all_tool_results,
                    streamed_final_response: true,
                    answer_source: AnswerSource::DeterministicFallback,
                });
            }
            InvestigationOutcome::NeedsMore { .. } => {}
        }
    }

    for iteration in 0..loop_budget.max_iterations {
        info!(
            intent = ?intent,
            iteration,
            accumulated_results = all_tool_results.len(),
            "tool loop iteration started"
        );
        emit_trace(
            token_tx,
            ProgressStatus::Started,
            if iteration == 0 {
                "planning investigation..."
            } else {
                "continuing investigation..."
            },
            false,
        );
        let prompt_tokens = estimate_message_tokens(&loop_messages);
        let heartbeat_label = if iteration == 0 {
            "planning investigation..."
        } else {
            "continuing investigation..."
        };
        let draft = with_progress_heartbeat(token_tx, heartbeat_label, || {
            generate_with_cache(
                backend,
                &loop_messages,
                cfg,
                project_root,
                token_tx.clone(),
                false,
                exact_cache,
                cache_stats,
                CacheMode::PreferPromptLevel,
            )
        })?;

        if !draft.hit {
            record_generation_budget(cfg, budget, token_tx, prompt_tokens, &draft.text);
        }

        let ReadOnlyToolExecution {
            results,
            disallowed_calls,
        } = tools.execute_read_only_tool_calls(&draft.text);
        if !results.is_empty() || !disallowed_calls.is_empty() {
            info!(
                intent = ?intent,
                iteration,
                results = tool_result_summary(&results),
                disallowed = disallowed_calls.join(", "),
                "tool loop executed tool calls"
            );
        }

        if results.is_empty() && disallowed_calls.is_empty() {
            let outcome = investigation_outcome(intent, prompt, resolution, &all_tool_results);
            info!(
                intent = ?intent,
                iteration,
                outcome = investigation_outcome_label(&outcome),
                accumulated_results = all_tool_results.len(),
                "tool loop draft produced no tool calls"
            );
            if all_tool_results.is_empty() {
                emit_trace(
                    token_tx,
                    ProgressStatus::Updated,
                    "model replied without using a tool; requesting an actual tool call",
                    false,
                );
                loop_messages.push(Message::assistant(&draft.text));
                loop_messages.push(Message::user(&initial_tool_only_followup(intent, prompt)));
                continue;
            }
            match outcome {
                InvestigationOutcome::Ready {
                    evidence,
                    stop_reason,
                } => {
                    info!(
                        intent = ?intent,
                        iteration,
                        stop_reason,
                        tool_results = all_tool_results.len(),
                        "tool loop reached ready outcome"
                    );
                    emit_trace(token_tx, ProgressStatus::Finished, stop_reason, false);
                    let attempt = answer::finalize_answer(
                        intent,
                        prompt,
                        &all_tool_results,
                        resolution,
                        base_messages,
                        &evidence,
                        backend,
                        token_tx,
                    )?;
                    return Ok(ToolLoopOutcome {
                        final_response: attempt.text,
                        tool_results: all_tool_results,
                        streamed_final_response: true,
                        answer_source: attempt.source,
                    });
                }
                InvestigationOutcome::NeedsMore { required_next_step } => {
                    info!(
                        intent = ?intent,
                        iteration,
                        tool_results = all_tool_results.len(),
                        "tool loop requested another bounded investigation step"
                    );
                    emit_trace(
                        token_tx,
                        ProgressStatus::Updated,
                        "tool loop needs more evidence before answering",
                        false,
                    );
                    loop_messages.push(Message::assistant(&draft.text));
                    loop_messages.push(Message::user(&required_next_step));
                    continue;
                }
                InvestigationOutcome::Insufficient { reason } => {
                    info!(
                        intent = ?intent,
                        iteration,
                        reason,
                        tool_results = all_tool_results.len(),
                        "tool loop reached insufficient outcome"
                    );
                    emit_trace(
                        token_tx,
                        ProgressStatus::Failed,
                        "insufficient evidence",
                        false,
                    );
                    emit_buffered_tokens(token_tx, &reason);
                    return Ok(ToolLoopOutcome {
                        final_response: reason,
                        tool_results: all_tool_results,
                        streamed_final_response: true,
                        answer_source: AnswerSource::DeterministicFallback,
                    });
                }
            }
        }

        if !disallowed_calls.is_empty() {
            emit_trace(
                token_tx,
                ProgressStatus::Updated,
                "read-only tool loop rejected mutating tool request",
                false,
            );
            loop_messages.push(Message::assistant(&draft.text));
            loop_messages.push(Message::user(&format!(
                "Read-only repo navigation mode only allows read-only tools. \
                 Do not call {}. Continue by using read-only tools or answer from current evidence.",
                disallowed_calls
                    .iter()
                    .map(|name| format!("`{name}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
            continue;
        }

        for result in &results {
            emit_trace(
                token_tx,
                ProgressStatus::Finished,
                &format!("{} {}", result.tool_name, result.argument),
                false,
            );
        }

        let repeated = repeated_tool_calls(
            &mut tool_call_counts,
            &results,
            loop_budget.max_duplicate_calls,
        );
        all_tool_results.extend(results.clone());

        loop_messages.push(Message::assistant(&draft.text));
        push_evidence_messages(
            &mut loop_messages,
            intent,
            prompt,
            resolution,
            &results,
            result_limit,
        );

        match investigation_outcome(intent, prompt, resolution, &all_tool_results) {
            InvestigationOutcome::Ready {
                evidence,
                stop_reason,
            } => {
                info!(
                    intent = ?intent,
                    iteration,
                    stop_reason,
                    tool_results = all_tool_results.len(),
                    "tool loop reached ready outcome after tool execution"
                );
                emit_trace(token_tx, ProgressStatus::Finished, stop_reason, false);
                let attempt = answer::finalize_answer(
                    intent,
                    prompt,
                    &all_tool_results,
                    resolution,
                    base_messages,
                    &evidence,
                    backend,
                    token_tx,
                )?;
                return Ok(ToolLoopOutcome {
                    final_response: attempt.text,
                    tool_results: all_tool_results,
                    streamed_final_response: true,
                    answer_source: attempt.source,
                });
            }
            InvestigationOutcome::NeedsMore { required_next_step } => {
                info!(
                    intent = ?intent,
                    iteration,
                    tool_results = all_tool_results.len(),
                    "tool loop still needs more evidence after tool execution"
                );
                if repeated {
                    loop_messages.push(Message::user(
                        "You are repeating the same tool calls. Only continue if the next read materially improves the answer.",
                    ));
                }
                loop_messages.push(Message::user(&required_next_step));
            }
            InvestigationOutcome::Insufficient { reason } => {
                info!(
                    intent = ?intent,
                    iteration,
                    reason,
                    tool_results = all_tool_results.len(),
                    "tool loop reached insufficient outcome after tool execution"
                );
                emit_trace(
                    token_tx,
                    ProgressStatus::Failed,
                    "insufficient evidence",
                    false,
                );
                emit_buffered_tokens(token_tx, &reason);
                return Ok(ToolLoopOutcome {
                    final_response: reason,
                    tool_results: all_tool_results,
                    streamed_final_response: true,
                    answer_source: AnswerSource::DeterministicFallback,
                });
            }
        }
    }

    emit_trace(
        token_tx,
        ProgressStatus::Updated,
        "tool loop hit its iteration limit",
        false,
    );
    info!(
        intent = ?intent,
        tool_results = all_tool_results.len(),
        "tool loop hit iteration limit"
    );
    let attempt = match investigation_outcome(intent, prompt, resolution, &all_tool_results) {
        InvestigationOutcome::Ready {
            evidence,
            stop_reason,
        } => {
            emit_trace(token_tx, ProgressStatus::Finished, stop_reason, false);
            answer::finalize_answer(
                intent,
                prompt,
                &all_tool_results,
                resolution,
                base_messages,
                &evidence,
                backend,
                token_tx,
            )?
        }
        InvestigationOutcome::Insufficient { reason } => {
            emit_trace(
                token_tx,
                ProgressStatus::Failed,
                "insufficient evidence",
                false,
            );
            emit_buffered_tokens(token_tx, &reason);
            answer::AnswerAttempt {
                text: reason,
                source: AnswerSource::DeterministicFallback,
            }
        }
        InvestigationOutcome::NeedsMore { .. } => {
            let reason = format!(
                "I couldn't gather enough source evidence to answer {:?} within the current investigation budget.",
                intent
            );
            emit_trace(
                token_tx,
                ProgressStatus::Failed,
                "insufficient evidence",
                false,
            );
            emit_buffered_tokens(token_tx, &reason);
            answer::AnswerAttempt {
                text: reason,
                source: AnswerSource::DeterministicFallback,
            }
        }
    };

    Ok(ToolLoopOutcome {
        final_response: attempt.text,
        tool_results: all_tool_results,
        streamed_final_response: true,
        answer_source: attempt.source,
    })
}

fn push_evidence_messages(
    loop_messages: &mut Vec<Message>,
    intent: ToolLoopIntent,
    prompt: &str,
    resolution: Option<&InvestigationResolution>,
    results: &[ToolResult],
    result_limit: Option<usize>,
) {
    let result_message =
        format_tool_loop_results_with_limit(intent, prompt, resolution, results, result_limit)
            .unwrap_or_else(|| "Tool results:\n".to_string());
    loop_messages.push(Message::user(&result_message));
    if let Some(guidance) = grounded_answer_guidance(intent, prompt, resolution, results) {
        loop_messages.push(Message::user(&guidance));
    }
}

#[cfg(test)]
mod tests;
