mod evidence;
mod intent;
mod prompting;

use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc::Sender;

use crate::cache::ExactCache;
use crate::config;
use crate::error::{ParamsError, Result};
use crate::events::{InferenceEvent, ProgressStatus};
use crate::tools::{ReadOnlyToolExecution, ToolRegistry, ToolResult};

use super::budget::{
    estimate_message_tokens, record_generation_budget, SessionBudget, SessionCacheStats,
};
use super::cache::{generate_with_cache, CacheMode};
use super::reflection::reflect_response;
use super::runtime::{emit_generation_started, emit_trace};
use super::{InferenceBackend, Message};
use evidence::{
    bootstrap_tool_results, format_tool_loop_results_with_limit, grounded_answer_guidance,
    has_relevant_file_evidence, targeted_investigation_followup,
};
use intent::suggested_search_query;
pub(super) use intent::ToolLoopIntent;
use prompting::{
    build_tool_loop_seed_messages, build_tool_loop_system_prompt, initial_investigation_hint,
    initial_tool_only_followup, should_stream_tool_loop_generation, thinking_label,
    tool_loop_budget, tool_loop_result_limit, with_progress_heartbeat,
    with_progress_heartbeat_interval,
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
    reflection_enabled: bool,
) -> Result<ToolLoopOutcome> {
    let loop_budget = tool_loop_budget(eco_enabled);
    let result_limit = tool_loop_result_limit(&backend.name(), eco_enabled);
    let (thinking, status_label) = thinking_label(intent);
    let _ = token_tx.send(InferenceEvent::SystemMessage(thinking.to_string()));
    emit_generation_started(token_tx, status_label, false);

    let system_prompt = build_tool_loop_system_prompt(tools, project_root, intent, eco_enabled);
    let mut loop_messages = build_tool_loop_seed_messages(base_messages, &system_prompt, prompt);
    let mut all_tool_results = bootstrap_tool_results(
        intent,
        prompt,
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

    if all_tool_results.is_empty() {
        if let Some(hint) = initial_investigation_hint(intent, prompt) {
            loop_messages.push(Message::user(&hint));
        }
    } else {
        let result_message =
            format_tool_loop_results_with_limit(intent, prompt, &all_tool_results, result_limit)
                .unwrap_or_else(|| "Tool results:\n".to_string());
        loop_messages.push(Message::user(&result_message));
        if let Some(guidance) = grounded_answer_guidance(intent, prompt, &all_tool_results) {
            loop_messages.push(Message::user(&guidance));
        }
        loop_messages.push(Message::user(
            "Initial repo scan is ready. Continue investigating only if you still need more evidence. Otherwise answer now using the observed file and line evidence.",
        ));
    }

    for iteration in 0..loop_budget.max_iterations {
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
                should_stream_tool_loop_generation(
                    intent,
                    prompt,
                    &all_tool_results,
                    reflection_enabled,
                ),
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

        if results.is_empty() && disallowed_calls.is_empty() {
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
            if !has_relevant_file_evidence(intent, prompt, &all_tool_results) {
                emit_trace(
                    token_tx,
                    ProgressStatus::Updated,
                    "tool loop needs file-level evidence before answering",
                    false,
                );
                loop_messages.push(Message::assistant(&draft.text));
                let followup = targeted_investigation_followup(intent, prompt, &all_tool_results)
                    .unwrap_or_else(|| {
                        "You do not have enough file-level evidence yet. \
                         Do not answer from search hits alone. \
                         Read the most relevant candidate file or use an LSP read-only tool on the best location, then answer from that evidence."
                            .to_string()
                    });
                loop_messages.push(Message::user(&followup));
                continue;
            }
            let final_response = if reflection_enabled {
                emit_trace(
                    token_tx,
                    ProgressStatus::Updated,
                    "reflecting final answer...",
                    false,
                );
                reflect_response(
                    backend,
                    cfg,
                    project_root,
                    budget,
                    token_tx,
                    exact_cache,
                    cache_stats,
                    &loop_messages,
                    &draft.text,
                )?
            } else {
                draft.text
            };
            let streamed_final_response = !reflection_enabled
                && should_stream_tool_loop_generation(
                    intent,
                    prompt,
                    &all_tool_results,
                    reflection_enabled,
                );
            return Ok(ToolLoopOutcome {
                final_response,
                tool_results: all_tool_results,
                streamed_final_response,
            });
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
        let result_message = ToolRegistry::format_results_with_limit(&results, result_limit)
            .unwrap_or_else(|| "Tool results:\n".to_string());
        let result_message =
            format_tool_loop_results_with_limit(intent, prompt, &results, result_limit)
                .unwrap_or(result_message);
        loop_messages.push(Message::user(&result_message));
        if let Some(guidance) = grounded_answer_guidance(intent, prompt, &all_tool_results) {
            loop_messages.push(Message::user(&guidance));
        }

        if repeated {
            loop_messages.push(Message::user(
                "You are repeating the same tool calls. Answer from the gathered evidence or explain briefly what is still missing.",
            ));
        }
        if let Some(followup) = targeted_investigation_followup(intent, prompt, &all_tool_results) {
            loop_messages.push(Message::user(&followup));
        } else if !repeated {
            loop_messages.push(Message::user(
                "Continue investigating only if you still need more evidence. Otherwise answer now using the observed file and line evidence.",
            ));
        }
    }

    emit_trace(
        token_tx,
        ProgressStatus::Updated,
        "tool loop hit its iteration limit; answering from gathered evidence...",
        false,
    );
    let prompt_tokens = estimate_message_tokens(&loop_messages);
    let final_draft =
        with_progress_heartbeat(token_tx, "answering from gathered evidence...", || {
            generate_with_cache(
                backend,
                &loop_messages,
                cfg,
                project_root,
                token_tx.clone(),
                should_stream_tool_loop_generation(
                    intent,
                    prompt,
                    &all_tool_results,
                    reflection_enabled,
                ),
                exact_cache,
                cache_stats,
                CacheMode::PreferPromptLevel,
            )
        })?;
    if !final_draft.hit {
        record_generation_budget(cfg, budget, token_tx, prompt_tokens, &final_draft.text);
    }
    let final_response = if reflection_enabled {
        reflect_response(
            backend,
            cfg,
            project_root,
            budget,
            token_tx,
            exact_cache,
            cache_stats,
            &loop_messages,
            &final_draft.text,
        )?
    } else {
        final_draft.text
    };

    if final_response.trim().is_empty() {
        return Err(ParamsError::Inference(
            "Tool loop returned an empty final response".to_string(),
        ));
    }

    let streamed_final_response = !reflection_enabled
        && should_stream_tool_loop_generation(
            intent,
            prompt,
            &all_tool_results,
            reflection_enabled,
        );

    Ok(ToolLoopOutcome {
        final_response,
        tool_results: all_tool_results,
        streamed_final_response,
    })
}

#[cfg(test)]
mod tests;
