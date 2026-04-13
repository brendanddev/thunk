use tracing::{debug, info, warn};

use crate::debug_log;
use crate::events::{InferenceEvent, ProgressStatus};
use crate::hooks::HookEvent;
use crate::memory::{compression, facts::TurnMemoryEvidence};
use crate::tools::ToolRegistry;

use super::super::super::approval::handle_pending_action;
use super::super::super::budget::{estimate_message_tokens, record_generation_budget};
use super::super::super::cache::{
    generate_with_cache, store_exact_cache, store_prompt_level_cache, CacheMode,
};
use super::super::super::reflection::reflect_response;
use super::super::super::runtime::{
    eco_tool_result_limit, emit_buffered_tokens, emit_generation_started, emit_trace,
    log_debug_response,
};
use super::super::super::tool_loop::run_read_only_tool_loop_with_resolution;
use super::super::super::Message;
use super::super::investigation::resolve_agentic_repo_turn;
use super::super::memory::{
    apply_memory_update, clear_memory_retrieval, collect_retrieval_bundle, emit_memory_state,
    memory_fact_lines, retrieval_trace_label, set_memory_retrieval,
};
use super::super::support::save_session;
use super::state::{approval_context, RuntimeContext, RuntimeState};

pub(super) fn handle_injected_user_context(
    ctx: &RuntimeContext<'_>,
    state: &mut RuntimeState,
    content: String,
    metadata: Option<crate::inference::InjectedContextMetadata>,
) {
    info!(chars = content.chars().count(), "user context injected");
    state
        .investigation_state
        .apply_injected_context(metadata.as_ref(), &content);
    state.session_messages.push(Message::user(&content));
    let _ = ctx;
}

pub(super) fn handle_submit_user(
    ctx: &RuntimeContext<'_>,
    state: &mut RuntimeState,
    prompt: String,
) {
    info!(
        reflection_enabled = state.reflection_enabled,
        existing_messages = state.session_messages.len(),
        "user turn submitted"
    );
    if state.debug_logging_enabled {
        if let Err(e) = debug_log::append_user_prompt(&prompt) {
            warn!(error = %e, "debug user prompt logging failed");
        }
    }
    state.session_messages.push(Message::user(&prompt));

    let tool_loop_resolution = resolve_agentic_repo_turn(&prompt, &state.investigation_state);

    if let Some(resolution) = tool_loop_resolution {
        let intent = resolution.intent;
        clear_memory_retrieval(&mut state.memory_state);
        state.memory_state.last_update = None;
        emit_memory_state(ctx.token_tx, &state.memory_state);
        ctx.hooks
            .dispatch(HookEvent::MemorySummariesSelected { summary_count: 0 });
        state.set_chat_system_prompt(ctx, &prompt, &[], &[], &[]);
        compression::compress_history(
            &mut state.session_messages,
            state.eco_enabled,
            state.investigation_state.compression_context().as_ref(),
        );

        let mut tool_loop_messages = state.session_messages.clone();
        if let Some(summary) = state.investigation_state.summary_message() {
            let insert_at = tool_loop_messages.len().saturating_sub(1);
            tool_loop_messages.insert(insert_at, summary);
        }

        let mut turn_memory = TurnMemoryEvidence::new(prompt.clone(), Vec::new());
        ctx.hooks.dispatch(HookEvent::BeforeGeneration {
            backend: ctx.backend.name(),
            message_count: tool_loop_messages.len(),
            eco: state.eco_enabled,
            reflection: state.reflection_enabled,
        });

        match run_read_only_tool_loop_with_resolution(
            intent,
            &prompt,
            Some(&resolution),
            &tool_loop_messages,
            ctx.backend,
            ctx.tools,
            ctx.cfg,
            ctx.project_root,
            ctx.token_tx,
            None,
            &mut state.cache_stats,
            &mut state.budget,
            state.eco_enabled,
            state.reflection_enabled,
        ) {
            Ok(outcome) => {
                state.investigation_state.note_tool_loop_outcome(
                    intent,
                    &prompt,
                    &outcome.tool_results,
                );
                for result in &outcome.tool_results {
                    ctx.hooks.dispatch(HookEvent::ToolExecuted {
                        tool_name: result.tool_name.clone(),
                        argument_chars: result.argument.chars().count(),
                        result_chars: result.output.chars().count(),
                    });
                    turn_memory.record_tool_result(
                        result.tool_name.clone(),
                        result.argument.clone(),
                        result.output.clone(),
                        false,
                    );
                }
                turn_memory.set_final_response(outcome.final_response.clone());
                if !outcome.final_response.trim().is_empty() {
                    if !state.reflection_enabled && !outcome.streamed_final_response {
                        emit_buffered_tokens(ctx.token_tx, &outcome.final_response);
                    }
                    log_debug_response(
                        state.debug_logging_enabled,
                        &outcome.final_response,
                        debug_log::ResponseSource::Live,
                    );
                    state
                        .session_messages
                        .push(Message::assistant(&outcome.final_response));
                }
                ctx.hooks.dispatch(HookEvent::AfterGeneration {
                    backend: ctx.backend.name(),
                    response_chars: outcome.final_response.chars().count(),
                    from_cache: false,
                    elapsed_ms: 0,
                });
                emit_trace(
                    ctx.token_tx,
                    ProgressStatus::Finished,
                    "answer ready",
                    false,
                );
                if let Some(store) = ctx.fact_store {
                    let update =
                        store.verify_and_store_turn(ctx.project_name, &turn_memory, ctx.backend);
                    apply_memory_update(ctx.token_tx, ctx.hooks, &mut state.memory_state, update);
                }
                save_session(
                    ctx.session_store,
                    &mut state.active_session,
                    &state.session_messages,
                    &ctx.backend.name(),
                    ctx.token_tx,
                );
                let _ = ctx.token_tx.send(InferenceEvent::Done);
            }
            Err(e) => {
                emit_trace(
                    ctx.token_tx,
                    ProgressStatus::Failed,
                    "tool loop failed",
                    false,
                );
                let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
            }
        }
        return;
    }

    let retrieval = collect_retrieval_bundle(
        &prompt,
        state.eco_enabled,
        ctx.project_name,
        ctx.project_index,
        ctx.fact_store,
        ctx.session_store,
        state
            .active_session
            .as_ref()
            .map(|session| session.id.as_str()),
        &state.memory_state.loaded_facts,
    );
    let mut turn_memory = TurnMemoryEvidence::new(prompt.clone(), retrieval.summaries.clone());
    set_memory_retrieval(&mut state.memory_state, &prompt, &retrieval);
    state.memory_state.last_update = None;
    emit_memory_state(ctx.token_tx, &state.memory_state);
    ctx.hooks.dispatch(HookEvent::MemorySummariesSelected {
        summary_count: retrieval.summaries.len(),
    });
    if let Some(label) = retrieval_trace_label(&retrieval) {
        emit_trace(ctx.token_tx, ProgressStatus::Finished, &label, false);
    }

    state.set_chat_system_prompt(
        ctx,
        &prompt,
        &memory_fact_lines(&retrieval.facts),
        &retrieval.summaries,
        &retrieval.session_excerpts,
    );

    compression::compress_history(
        &mut state.session_messages,
        state.eco_enabled,
        state.investigation_state.compression_context().as_ref(),
    );

    let generation_messages = state.session_messages.clone();

    emit_generation_started(ctx.token_tx, "generating...", false);
    emit_trace(
        ctx.token_tx,
        ProgressStatus::Started,
        "drafting answer...",
        false,
    );
    ctx.hooks.dispatch(HookEvent::BeforeGeneration {
        backend: ctx.backend.name(),
        message_count: generation_messages.len(),
        eco: state.eco_enabled,
        reflection: state.reflection_enabled,
    });
    let response = generate_with_cache(
        ctx.backend,
        &generation_messages,
        ctx.cfg,
        ctx.project_root,
        ctx.token_tx.clone(),
        !state.reflection_enabled,
        ctx.exact_cache,
        &mut state.cache_stats,
        CacheMode::PreferPromptLevel,
    );
    debug!(
        reflection_enabled = state.reflection_enabled,
        message_count = generation_messages.len(),
        "generation started"
    );
    let prompt_tokens = estimate_message_tokens(&generation_messages);

    match response {
        Err(e) => {
            emit_trace(
                ctx.token_tx,
                ProgressStatus::Failed,
                "generation failed",
                false,
            );
            let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
        }
        Ok(response) => {
            let response_source = response.source;
            ctx.hooks.dispatch(HookEvent::AfterGeneration {
                backend: ctx.backend.name(),
                response_chars: response.text.chars().count(),
                from_cache: response.hit,
                elapsed_ms: response.elapsed_ms,
            });
            let full_response = response.text;
            info!(
                response_chars = full_response.chars().count(),
                reflection_enabled = state.reflection_enabled,
                "generation completed"
            );
            if !response.hit {
                record_generation_budget(
                    ctx.cfg,
                    &mut state.budget,
                    ctx.token_tx,
                    prompt_tokens,
                    &full_response,
                );
            }

            emit_trace(
                ctx.token_tx,
                ProgressStatus::Updated,
                "scanning tool calls...",
                false,
            );
            let tool_execution = ctx.tools.execute_tool_calls(&full_response);
            let tool_results = tool_execution.results;
            info!(
                tool_results = tool_results.len(),
                pending = tool_execution.pending.is_some(),
                "tool scan completed"
            );
            for result in &tool_results {
                ctx.hooks.dispatch(HookEvent::ToolExecuted {
                    tool_name: result.tool_name.clone(),
                    argument_chars: result.argument.chars().count(),
                    result_chars: result.output.chars().count(),
                });
                turn_memory.record_tool_result(
                    result.tool_name.clone(),
                    result.argument.clone(),
                    result.output.clone(),
                    false,
                );
            }

            if let Some(pending) = tool_execution.pending {
                emit_trace(
                    ctx.token_tx,
                    ProgressStatus::Updated,
                    "waiting for approval...",
                    false,
                );
                state
                    .session_messages
                    .push(Message::assistant(&full_response));
                if let Some(result_msg) = ToolRegistry::format_results_with_limit(
                    &tool_results,
                    eco_tool_result_limit(state.eco_enabled),
                ) {
                    state.session_messages.push(Message::user(&result_msg));
                }
                let action_id = state.next_action_id;
                let approval = approval_context(ctx, state, Some(&mut turn_memory));
                if let Err(e) = handle_pending_action(approval, action_id, pending, true) {
                    emit_trace(
                        ctx.token_tx,
                        ProgressStatus::Failed,
                        "approval flow failed",
                        false,
                    );
                    let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
                } else if let Some(store) = ctx.fact_store {
                    let update =
                        store.verify_and_store_turn(ctx.project_name, &turn_memory, ctx.backend);
                    apply_memory_update(ctx.token_tx, ctx.hooks, &mut state.memory_state, update);
                }
                state.next_action_id = state.next_action_id.saturating_add(1);
            } else if let Some(result_msg) = ToolRegistry::format_results_with_limit(
                &tool_results,
                eco_tool_result_limit(state.eco_enabled),
            ) {
                emit_trace(
                    ctx.token_tx,
                    ProgressStatus::Updated,
                    "running tool follow-up...",
                    false,
                );
                let _ = ctx.token_tx.send(InferenceEvent::ToolCall(
                    tool_results
                        .iter()
                        .map(|r| format!("{}({})", r.tool_name, r.argument))
                        .collect::<Vec<_>>()
                        .join(", "),
                ));

                state
                    .session_messages
                    .push(Message::assistant(&full_response));
                state.session_messages.push(Message::user(&result_msg));

                match generate_with_cache(
                    ctx.backend,
                    &state.session_messages,
                    ctx.cfg,
                    ctx.project_root,
                    ctx.token_tx.clone(),
                    !state.reflection_enabled,
                    ctx.exact_cache,
                    &mut state.cache_stats,
                    CacheMode::PreferPromptLevel,
                ) {
                    Ok(follow_up) => {
                        let follow_up_source = follow_up.source;
                        let follow_up_text = follow_up.text;
                        let prompt_tokens = estimate_message_tokens(&state.session_messages);
                        if !follow_up.hit {
                            record_generation_budget(
                                ctx.cfg,
                                &mut state.budget,
                                ctx.token_tx,
                                prompt_tokens,
                                &follow_up_text,
                            );
                        }
                        let final_response = if state.reflection_enabled {
                            emit_trace(
                                ctx.token_tx,
                                ProgressStatus::Updated,
                                "reflecting final answer...",
                                false,
                            );
                            reflect_response(
                                ctx.backend,
                                ctx.cfg,
                                ctx.project_root,
                                &mut state.budget,
                                ctx.token_tx,
                                ctx.exact_cache,
                                &mut state.cache_stats,
                                &state.session_messages,
                                &follow_up_text,
                            )
                        } else {
                            Ok(follow_up_text)
                        };

                        match final_response {
                            Ok(final_response) => {
                                turn_memory.set_final_response(final_response.clone());
                                if !final_response.trim().is_empty() {
                                    log_debug_response(
                                        state.debug_logging_enabled,
                                        &final_response,
                                        if state.reflection_enabled {
                                            debug_log::ResponseSource::Live
                                        } else {
                                            follow_up_source
                                        },
                                    );
                                    store_exact_cache(
                                        ctx.exact_cache,
                                        ctx.cfg,
                                        ctx.project_root,
                                        &ctx.backend.name(),
                                        &state.session_messages,
                                        &final_response,
                                    );
                                    state
                                        .session_messages
                                        .push(Message::assistant(&final_response));
                                }
                                emit_trace(
                                    ctx.token_tx,
                                    ProgressStatus::Finished,
                                    "answer ready",
                                    false,
                                );
                                if let Some(store) = ctx.fact_store {
                                    let update = store.verify_and_store_turn(
                                        ctx.project_name,
                                        &turn_memory,
                                        ctx.backend,
                                    );
                                    apply_memory_update(
                                        ctx.token_tx,
                                        ctx.hooks,
                                        &mut state.memory_state,
                                        update,
                                    );
                                }
                            }
                            Err(e) => {
                                warn!(error = %e, "reflection after tool follow-up failed");
                                emit_trace(
                                    ctx.token_tx,
                                    ProgressStatus::Failed,
                                    "follow-up failed",
                                    false,
                                );
                                let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
                            }
                        }
                    }
                    Err(e) => {
                        warn!(error = %e, "tool follow-up generation failed");
                        emit_trace(
                            ctx.token_tx,
                            ProgressStatus::Failed,
                            "tool follow-up failed",
                            false,
                        );
                        let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                }
            } else {
                let final_response = if state.reflection_enabled {
                    emit_trace(
                        ctx.token_tx,
                        ProgressStatus::Updated,
                        "reflecting final answer...",
                        false,
                    );
                    reflect_response(
                        ctx.backend,
                        ctx.cfg,
                        ctx.project_root,
                        &mut state.budget,
                        ctx.token_tx,
                        ctx.exact_cache,
                        &mut state.cache_stats,
                        &state.session_messages,
                        &full_response,
                    )
                } else {
                    Ok(full_response)
                };

                match final_response {
                    Ok(final_response) => {
                        turn_memory.set_final_response(final_response.clone());
                        if !final_response.trim().is_empty() {
                            log_debug_response(
                                state.debug_logging_enabled,
                                &final_response,
                                if state.reflection_enabled {
                                    debug_log::ResponseSource::Live
                                } else {
                                    response_source
                                },
                            );
                            store_exact_cache(
                                ctx.exact_cache,
                                ctx.cfg,
                                ctx.project_root,
                                &ctx.backend.name(),
                                &state.session_messages,
                                &final_response,
                            );
                            store_prompt_level_cache(
                                ctx.exact_cache,
                                ctx.cfg,
                                ctx.project_root,
                                &ctx.backend.name(),
                                &state.session_messages,
                                &final_response,
                            );
                            state
                                .session_messages
                                .push(Message::assistant(&final_response));
                        }
                        emit_trace(
                            ctx.token_tx,
                            ProgressStatus::Finished,
                            "answer ready",
                            false,
                        );
                        if let Some(store) = ctx.fact_store {
                            let update = store.verify_and_store_turn(
                                ctx.project_name,
                                &turn_memory,
                                ctx.backend,
                            );
                            apply_memory_update(
                                ctx.token_tx,
                                ctx.hooks,
                                &mut state.memory_state,
                                update,
                            );
                        }
                    }
                    Err(e) => {
                        warn!(error = %e, "final response post-processing failed");
                        emit_trace(
                            ctx.token_tx,
                            ProgressStatus::Failed,
                            "final answer failed",
                            false,
                        );
                        let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                }
            }

            save_session(
                ctx.session_store,
                &mut state.active_session,
                &state.session_messages,
                &ctx.backend.name(),
                ctx.token_tx,
            );
            let _ = ctx.token_tx.send(InferenceEvent::Done);
        }
    }
}
