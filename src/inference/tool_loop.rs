mod evidence;
mod intent;
mod prompting;

use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc::Sender;
use tracing::info;

use crate::cache::ExactCache;
use crate::config;
use crate::error::{ParamsError, Result};
use crate::events::{InferenceEvent, ProgressStatus};
use crate::tools::{ReadOnlyToolExecution, ToolRegistry, ToolResult};

use super::budget::{
    estimate_message_tokens, record_generation_budget, SessionBudget, SessionCacheStats,
};
use super::cache::{generate_with_cache, CacheMode};
use super::runtime::{
    emit_buffered_tokens, emit_generation_started, emit_trace, run_and_collect_with_stream_guard,
};
use super::session::investigation::InvestigationResolution;
use super::{InferenceBackend, Message};
use evidence::observe::observed_read_paths;
use evidence::{
    bootstrap_tool_results, format_tool_loop_results_with_limit, grounded_answer_guidance,
    investigation_outcome, render_structured_answer, InvestigationOutcome, StructuredEvidence,
};
pub(super) use intent::ToolLoopIntent;

fn validate_final_answer(
    answer: &str,
    evidence: &evidence::StructuredEvidence,
    results: &[ToolResult],
) -> bool {
    if answer.trim().len() < 12 {
        return false;
    }

    let read_paths_set: std::collections::HashSet<String> = observed_read_paths(results);

    match evidence {
        evidence::StructuredEvidence::FileSummary(fe) => {
            let all_paths: Vec<String> = read_paths_set.iter().cloned().collect();
            let mut valid = !fe.declarations().is_empty();
            for line in fe.declarations() {
                let path_str = line.path().to_string();
                let path_part = path_str.split(':').next().unwrap_or(&path_str).to_string();
                if !all_paths.contains(&path_part) {
                    valid = false;
                    break;
                }
            }
            valid
        }
        evidence::StructuredEvidence::Implementation(ie) => {
            let path_str = ie.primary().path().to_string();
            let path_part = path_str.split(':').next().unwrap_or(&path_str).to_string();
            read_paths_set.contains(&path_part) && ie.primary().line_number() > 0
        }
        evidence::StructuredEvidence::Config(ce) => {
            let mut valid = !ce.lines().is_empty();
            for line in ce.lines() {
                let path_str = line.path().to_string();
                if !path_str.contains("config") && !read_paths_set.contains(&path_str) {
                    valid = false;
                    break;
                }
            }
            valid
        }
        evidence::StructuredEvidence::CallSites(cse) => {
            let mut valid = !cse.sites().is_empty();
            for line in cse.sites() {
                let path_str = line.path().to_string();
                let path_part = path_str.split(':').next().unwrap_or(&path_str).to_string();
                if !read_paths_set.contains(&path_part) {
                    valid = false;
                    break;
                }
            }
            valid
        }
        evidence::StructuredEvidence::Usages(ue) => {
            let mut valid = !ue.usages().is_empty();
            for line in ue.usages() {
                let path_str = line.path().to_string();
                let path_part = path_str.split(':').next().unwrap_or(&path_str).to_string();
                if !read_paths_set.contains(&path_part) {
                    valid = false;
                    break;
                }
            }
            valid
        }
        evidence::StructuredEvidence::FlowTrace(fte) => {
            let mut valid = !fte.steps().is_empty();
            for step in fte.steps() {
                let path_str = step.path().to_string();
                if !read_paths_set.contains(&path_str) {
                    valid = false;
                    break;
                }
            }
            if valid {
                valid = validate_session_restore_trace(fte, results);
            }
            valid
        }
        evidence::StructuredEvidence::RepoOverview(_) => true,
    }
}

fn validate_session_restore_trace(
    fte: &evidence::FlowTraceEvidence,
    results: &[ToolResult],
) -> bool {
    let subject_lower = fte.subject().to_ascii_lowercase();
    if !subject_lower.contains("session") && !subject_lower.contains("restore") {
        return true;
    }

    let required_paths = [
        "src/inference/session/runtime/core.rs",
        "src/session/runtime/core.rs",
    ];
    let required_patterns = ["load_most_recent", "load_session_by_id"];
    let forbidden_patterns = ["message_count", "log_messages", "saved_messages"];

    let steps_paths: Vec<String> = fte.steps().iter().map(|s| s.path().to_string()).collect();
    let steps_text: String = steps_paths.join(" ");

    let has_core = steps_paths.iter().any(|p| p.contains("runtime/core.rs"));
    let has_load_most_recent = steps_text.contains("load_most_recent");
    let has_load_session_by_id = steps_text.contains("load_session_by_id");
    let has_no_session_branch = steps_text.contains("Ok(None)")
        || steps_text.contains("None")
        || steps_text.contains("no session");
    let has_handoff = steps_text.contains("load_session_by_id");

    let has_forbidden = forbidden_patterns
        .iter()
        .any(|p| steps_text.to_lowercase().contains(&p.to_lowercase()));

    if has_forbidden {
        return false;
    }

    has_core && has_load_most_recent && (has_no_session_branch || has_handoff)
}

use prompting::{
    build_tool_loop_seed_messages, build_tool_loop_system_prompt, initial_investigation_hint,
    initial_tool_only_followup, is_referential_follow_up, thinking_label, tool_loop_budget,
    tool_loop_result_limit, with_progress_heartbeat,
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
                let final_response = finalize_structured_answer(
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
                    final_response,
                    tool_results: all_tool_results,
                    streamed_final_response: true,
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
                    let final_response = finalize_structured_answer(
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
                        final_response,
                        tool_results: all_tool_results,
                        streamed_final_response: true,
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
                let final_response = finalize_structured_answer(
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
                    final_response,
                    tool_results: all_tool_results,
                    streamed_final_response: true,
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
    let final_response = match investigation_outcome(intent, prompt, resolution, &all_tool_results)
    {
        InvestigationOutcome::Ready {
            evidence,
            stop_reason,
        } => {
            emit_trace(token_tx, ProgressStatus::Finished, stop_reason, false);
            finalize_structured_answer(
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
            reason
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
            reason
        }
    };

    Ok(ToolLoopOutcome {
        final_response,
        tool_results: all_tool_results,
        streamed_final_response: true,
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

#[allow(clippy::too_many_arguments)]
fn finalize_structured_answer(
    intent: ToolLoopIntent,
    prompt: &str,
    results: &[ToolResult],
    resolution: Option<&InvestigationResolution>,
    base_messages: &[Message],
    evidence: &evidence::StructuredEvidence,
    backend: &dyn InferenceBackend,
    token_tx: &Sender<InferenceEvent>,
) -> Result<String> {
    // Primary path: model synthesis pass using compact evidence guidance.
    // This produces streamed, natural-prose answers grounded in real evidence.
    if let Some(guidance) = grounded_answer_guidance(intent, prompt, resolution, results) {
        let synthesis_messages = build_synthesis_messages(prompt, &guidance, base_messages);
        match run_and_collect_with_stream_guard(backend, &synthesis_messages, token_tx.clone()) {
            Ok(run) if run.streamed && !is_synthesis_junk(&run.text) => {
                // Validate the synthesized answer before returning it.
                // If validation fails, fall back to deterministic template.
                if validate_final_answer(&run.text, evidence, results) {
                    return Ok(run.text);
                }
                // Validation failed - fall through to deterministic fallback.
            }
            _ => {
                // Model started with a tool tag, produced junk, or errored.
                // Fall through to deterministic fallback below.
            }
        }
    }

    // Fallback: deterministic template rendering emitted as simulated streaming.
    let final_response = render_structured_answer(prompt, evidence);
    if final_response.trim().is_empty() {
        return Err(ParamsError::Inference(
            "Structured evidence produced an empty final answer".to_string(),
        ));
    }
    emit_buffered_tokens(token_tx, &final_response);
    Ok(final_response)
}

/// Build a tight synthesis message set for the final answer pass.
/// Includes the last two non-system base messages so referential follow-ups
/// like "Tell me more" have the previous answer as context.
fn build_synthesis_messages(
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
        // Include up to the last 2 non-system messages from the live conversation
        // so follow-ups like "Tell me more" see the previous answer as context.
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

fn final_answer_contains_tool_tags(text: &str, tools: &ToolRegistry) -> bool {
    let execution = tools.execute_read_only_tool_calls(text);
    !execution.results.is_empty() || !execution.disallowed_calls.is_empty()
}

/// Returns true if the synthesis output looks like a tool tag or is too short
/// to be a real answer. Used to trigger the deterministic fallback.
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

#[cfg(test)]
mod tests;
