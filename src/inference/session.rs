use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};

use tracing::{debug, info, warn};

use crate::cache::ExactCache;
use crate::config;
use crate::debug_log;
use crate::events::{InferenceEvent, ProgressStatus};
use crate::hooks::{HookEvent, Hooks};
use crate::memory::{compression, facts::FactStore, index::ProjectIndex};
use crate::session::SessionStore;
use crate::tools::{BashTool, Tool, ToolRegistry};

use super::approval::{handle_pending_action, ApprovalContext};
use super::budget::{
    emit_budget_update, emit_cache_update, estimate_message_tokens, record_generation_budget,
    SessionBudget, SessionCacheStats,
};
use super::cache::{generate_with_cache, store_exact_cache, store_prompt_level_cache, CacheMode};
use super::indexing::{run_idle_index_step, IncrementalIndexState, IDLE_INDEX_POLL_INTERVAL};
use super::reflection::reflect_response;
use super::runtime::{
    eco_tool_result_limit, effective_reflection, emit_generation_started, emit_trace,
    log_debug_response, summary_limit,
};
use super::{build_system_prompt, load_backend_with_fallback, Message, SessionCommand};

fn save_session(store: Option<&SessionStore>, messages: &[Message], backend_name: &str) {
    if let Some(s) = store {
        if let Err(e) = s.save(messages, backend_name) {
            warn!(error = %e, "session save failed");
        }
    }
}

/// Persistent model thread — loads the backend once, handles prompts in a loop.
/// After each response it checks for tool calls and runs a follow-up if needed.
pub fn model_thread(prompt_rx: Receiver<SessionCommand>, token_tx: Sender<InferenceEvent>) {
    let cfg = match config::load_with_profile() {
        Ok(c) => c,
        Err(e) => {
            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
            return;
        }
    };

    let backend = match load_backend_with_fallback(&cfg, &token_tx) {
        Ok(backend) => backend,
        Err(e) => {
            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
            return;
        }
    };
    info!(backend = backend.name(), "model thread initialized");
    let hooks = Hooks::default();

    let _ = token_tx.send(InferenceEvent::Ready);
    let _ = token_tx.send(InferenceEvent::BackendName(backend.name()));
    let mut eco_enabled = cfg.eco.enabled;
    let mut reflection_requested = cfg.reflection.enabled;
    let mut reflection_enabled = effective_reflection(reflection_requested, eco_enabled);
    let mut debug_logging_enabled = cfg.debug_logging.content;
    info!(enabled = eco_enabled, "eco initial state");
    info!(enabled = reflection_enabled, "reflection initial state");
    info!(
        enabled = debug_logging_enabled,
        "debug logging initial state"
    );
    let _ = token_tx.send(InferenceEvent::EcoEnabled(eco_enabled));
    let _ = token_tx.send(InferenceEvent::ReflectionEnabled(reflection_enabled));
    let _ = token_tx.send(InferenceEvent::DebugLoggingEnabled(debug_logging_enabled));

    if let Some(ref profile_path) = cfg.active_profile {
        let name = profile_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(config::PROJECT_PROFILE_FILE);
        info!(profile = name, "project profile active");
        emit_trace(
            &token_tx,
            ProgressStatus::Finished,
            &format!("profile: {name}"),
            true,
        );
    }

    let tools = ToolRegistry::default();
    let project_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let project_name = project_root.to_string_lossy().to_string();
    let exact_cache = ExactCache::open().ok();
    let fact_store = FactStore::open().ok();
    let project_index = ProjectIndex::open_for(&project_root).ok();
    let mut index_state = if cfg.backend == "llama_cpp" {
        info!("idle incremental indexing disabled for llama_cpp backend");
        None
    } else {
        project_index.as_ref().map(|_| IncrementalIndexState::new())
    };
    let session_facts = fact_store
        .as_ref()
        .and_then(|store| store.get_relevant_facts(&project_name, "", 5).ok())
        .unwrap_or_default();
    let mut session_messages = vec![Message::system(&build_system_prompt(
        &tools,
        &session_facts,
        &[],
        eco_enabled,
    ))];
    let mut budget = SessionBudget {
        has_cost_estimate: cfg.backend == "llama_cpp" || cfg.backend == "ollama",
        ..SessionBudget::default()
    };
    let mut cache_stats = SessionCacheStats::default();
    emit_cache_update(&cache_stats, false, &token_tx);
    let mut next_action_id = 1u64;

    let session_store = SessionStore::open().ok();
    if let Some(ref store) = session_store {
        match store.load() {
            Ok(Some(saved)) => {
                let restored_count = saved.messages.len();
                info!(
                    msg_count = restored_count,
                    saved_at = saved.saved_at,
                    "restoring previous session"
                );
                let display_messages: Vec<(String, String)> = saved
                    .messages
                    .iter()
                    .map(|m| (m.role.clone(), m.content.clone()))
                    .collect();
                session_messages.extend(saved.messages);
                let _ = token_tx.send(InferenceEvent::SessionRestored {
                    display_messages,
                    saved_at: saved.saved_at,
                });
                hooks.dispatch(HookEvent::SessionRestored {
                    message_count: restored_count,
                    saved_at: saved.saved_at,
                });
            }
            Ok(None) => {}
            Err(e) => {
                warn!(error = %e, "session load failed — starting fresh");
            }
        }
    }

    loop {
        if let (Some(index), Some(state)) = (project_index.as_ref(), index_state.as_mut()) {
            run_idle_index_step(state, index, &project_root, &*backend);
        }

        let command = match prompt_rx.recv_timeout(IDLE_INDEX_POLL_INTERVAL) {
            Ok(command) => command,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };

        match command {
            SessionCommand::ClearSession => {
                info!("session cleared");
                if let Some(ref store) = session_store {
                    if let Err(e) = store.clear() {
                        warn!(error = %e, "session clear failed");
                    }
                }
                session_messages.clear();
                session_messages.push(Message::system(&build_system_prompt(
                    &tools,
                    &session_facts,
                    &[],
                    eco_enabled,
                )));
                budget = SessionBudget {
                    has_cost_estimate: cfg.backend == "llama_cpp" || cfg.backend == "ollama",
                    ..SessionBudget::default()
                };
                cache_stats = SessionCacheStats::default();
                emit_budget_update(&budget, &token_tx);
                emit_cache_update(&cache_stats, false, &token_tx);
                continue;
            }
            SessionCommand::InjectUserContext(content) => {
                info!(chars = content.chars().count(), "user context injected");
                session_messages.push(Message::user(&content));
                continue;
            }
            SessionCommand::RequestShellCommand(command) => {
                info!("shell command approval requested");
                match BashTool.run(&command) {
                    Ok(crate::tools::ToolRunResult::RequiresApproval(pending)) => {
                        if let Err(e) = handle_pending_action(
                            ApprovalContext {
                                prompt_rx: &prompt_rx,
                                token_tx: &token_tx,
                                backend: &*backend,
                                tools: &tools,
                                exact_cache: exact_cache.as_ref(),
                                session_messages: &mut session_messages,
                                cfg: &cfg,
                                project_root: &project_root,
                                budget: &mut budget,
                                cache_stats: &mut cache_stats,
                                debug_logging_enabled,
                                reflection_enabled,
                                eco_enabled,
                                hooks: &hooks,
                                index_state: index_state.as_mut(),
                            },
                            next_action_id,
                            pending,
                            false,
                        ) {
                            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                        } else {
                            save_session(
                                session_store.as_ref(),
                                &session_messages,
                                &backend.name(),
                            );
                        }
                    }
                    Ok(crate::tools::ToolRunResult::Immediate(output)) => {
                        let _ = token_tx.send(InferenceEvent::ContextMessage(output));
                    }
                    Err(e) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                }
                let _ = token_tx.send(InferenceEvent::Done);
                next_action_id = next_action_id.saturating_add(1);
                continue;
            }
            SessionCommand::RequestFileWrite { path, content } => {
                info!(path = path.as_str(), "file write approval requested");
                match crate::tools::build_pending_write_request(&path, &content) {
                    Ok(pending) => {
                        if let Err(e) = handle_pending_action(
                            ApprovalContext {
                                prompt_rx: &prompt_rx,
                                token_tx: &token_tx,
                                backend: &*backend,
                                tools: &tools,
                                exact_cache: exact_cache.as_ref(),
                                session_messages: &mut session_messages,
                                cfg: &cfg,
                                project_root: &project_root,
                                budget: &mut budget,
                                cache_stats: &mut cache_stats,
                                debug_logging_enabled,
                                reflection_enabled,
                                eco_enabled,
                                hooks: &hooks,
                                index_state: index_state.as_mut(),
                            },
                            next_action_id,
                            pending,
                            false,
                        ) {
                            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                        } else {
                            save_session(
                                session_store.as_ref(),
                                &session_messages,
                                &backend.name(),
                            );
                        }
                    }
                    Err(e) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                }
                let _ = token_tx.send(InferenceEvent::Done);
                next_action_id = next_action_id.saturating_add(1);
                continue;
            }
            SessionCommand::SetReflection(enabled) => {
                reflection_requested = enabled;
                reflection_enabled = effective_reflection(reflection_requested, eco_enabled);
                info!(
                    requested = enabled,
                    effective = reflection_enabled,
                    eco_enabled,
                    "reflection state updated"
                );
                let _ = token_tx.send(InferenceEvent::ReflectionEnabled(reflection_enabled));
                continue;
            }
            SessionCommand::SetEco(enabled) => {
                eco_enabled = enabled;
                reflection_enabled = effective_reflection(reflection_requested, eco_enabled);
                info!(
                    enabled = eco_enabled,
                    reflection_enabled, "eco state updated"
                );
                if let Some(first) = session_messages.first_mut() {
                    if first.role == "system" {
                        first.content =
                            build_system_prompt(&tools, &session_facts, &[], eco_enabled);
                    }
                }
                let _ = token_tx.send(InferenceEvent::EcoEnabled(eco_enabled));
                let _ = token_tx.send(InferenceEvent::ReflectionEnabled(reflection_enabled));
                continue;
            }
            SessionCommand::SetDebugLogging(enabled) => {
                debug_logging_enabled = enabled;
                info!(enabled, "debug logging state updated");
                let _ = token_tx.send(InferenceEvent::DebugLoggingEnabled(enabled));
                continue;
            }
            SessionCommand::ClearDebugLog => {
                match debug_log::clear() {
                    Ok(()) => {
                        info!("debug content log cleared");
                    }
                    Err(e) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                }
                continue;
            }
            SessionCommand::ClearCache => {
                match exact_cache.as_ref().map(|cache| cache.clear()) {
                    Some(Ok(deleted)) => {
                        info!(deleted, "exact cache cleared");
                    }
                    Some(Err(e)) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                    None => {
                        let _ = token_tx
                            .send(InferenceEvent::Error("Cache is unavailable".to_string()));
                    }
                }
                continue;
            }
            SessionCommand::ApproveAction(_) | SessionCommand::RejectAction(_) => {
                warn!("approval command received with no pending action");
                let _ = token_tx.send(InferenceEvent::Error(
                    "No action is currently awaiting approval".to_string(),
                ));
                continue;
            }
            SessionCommand::SubmitUser(prompt) => {
                info!(
                    reflection_enabled,
                    existing_messages = session_messages.len(),
                    "user turn submitted"
                );
                if debug_logging_enabled {
                    if let Err(e) = debug_log::append_user_prompt(&prompt) {
                        warn!(error = %e, "debug user prompt logging failed");
                    }
                }
                session_messages.push(Message::user(&prompt));

                let relevant_summaries = project_index
                    .as_ref()
                    .and_then(|index| {
                        index
                            .find_relevant(&prompt, summary_limit(eco_enabled))
                            .ok()
                    })
                    .unwrap_or_default();

                if let Some(first) = session_messages.first_mut() {
                    if first.role == "system" {
                        first.content = build_system_prompt(
                            &tools,
                            &session_facts,
                            &relevant_summaries,
                            eco_enabled,
                        );
                    }
                }

                compression::compress_history(&mut session_messages, &*backend, eco_enabled);

                emit_generation_started(&token_tx, "generating...", false);
                emit_trace(
                    &token_tx,
                    ProgressStatus::Started,
                    "drafting answer...",
                    false,
                );
                hooks.dispatch(HookEvent::BeforeGeneration {
                    backend: backend.name(),
                    message_count: session_messages.len(),
                    eco: eco_enabled,
                    reflection: reflection_enabled,
                });
                let response = generate_with_cache(
                    &*backend,
                    &session_messages,
                    &cfg,
                    &project_root,
                    token_tx.clone(),
                    !reflection_enabled,
                    exact_cache.as_ref(),
                    &mut cache_stats,
                    CacheMode::PreferPromptLevel,
                );
                debug!(
                    reflection_enabled,
                    message_count = session_messages.len(),
                    "generation started"
                );
                let prompt_tokens = estimate_message_tokens(&session_messages);

                match response {
                    Err(e) => {
                        emit_trace(
                            &token_tx,
                            ProgressStatus::Failed,
                            "generation failed",
                            false,
                        );
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                    Ok(response) => {
                        let response_source = response.source;
                        hooks.dispatch(HookEvent::AfterGeneration {
                            backend: backend.name(),
                            response_chars: response.text.chars().count(),
                            from_cache: response.hit,
                            elapsed_ms: response.elapsed_ms,
                        });
                        let full_response = response.text;
                        info!(
                            response_chars = full_response.chars().count(),
                            reflection_enabled, "generation completed"
                        );
                        if !response.hit {
                            record_generation_budget(
                                &cfg,
                                &mut budget,
                                &token_tx,
                                prompt_tokens,
                                &full_response,
                            );
                        }

                        emit_trace(
                            &token_tx,
                            ProgressStatus::Updated,
                            "scanning tool calls...",
                            false,
                        );
                        let tool_execution = tools.execute_tool_calls(&full_response);
                        let tool_results = tool_execution.results;
                        info!(
                            tool_results = tool_results.len(),
                            pending = tool_execution.pending.is_some(),
                            "tool scan completed"
                        );
                        for result in &tool_results {
                            hooks.dispatch(HookEvent::ToolExecuted {
                                tool_name: result.tool_name.clone(),
                                argument_chars: result.argument.chars().count(),
                                result_chars: result.output.chars().count(),
                            });
                        }

                        if let Some(pending) = tool_execution.pending {
                            emit_trace(
                                &token_tx,
                                ProgressStatus::Updated,
                                "waiting for approval...",
                                false,
                            );
                            session_messages.push(Message::assistant(&full_response));
                            if let Some(result_msg) = ToolRegistry::format_results_with_limit(
                                &tool_results,
                                eco_tool_result_limit(eco_enabled),
                            ) {
                                session_messages.push(Message::user(&result_msg));
                            }
                            if let Err(e) = handle_pending_action(
                                ApprovalContext {
                                    prompt_rx: &prompt_rx,
                                    token_tx: &token_tx,
                                    backend: &*backend,
                                    tools: &tools,
                                    exact_cache: exact_cache.as_ref(),
                                    session_messages: &mut session_messages,
                                    cfg: &cfg,
                                    project_root: &project_root,
                                    budget: &mut budget,
                                    cache_stats: &mut cache_stats,
                                    debug_logging_enabled,
                                    reflection_enabled,
                                    eco_enabled,
                                    hooks: &hooks,
                                    index_state: index_state.as_mut(),
                                },
                                next_action_id,
                                pending,
                                true,
                            ) {
                                emit_trace(
                                    &token_tx,
                                    ProgressStatus::Failed,
                                    "approval flow failed",
                                    false,
                                );
                                let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                            }
                            next_action_id = next_action_id.saturating_add(1);
                        } else if let Some(result_msg) = ToolRegistry::format_results_with_limit(
                            &tool_results,
                            eco_tool_result_limit(eco_enabled),
                        ) {
                            emit_trace(
                                &token_tx,
                                ProgressStatus::Updated,
                                "running tool follow-up...",
                                false,
                            );
                            let _ = token_tx.send(InferenceEvent::ToolCall(
                                tool_results
                                    .iter()
                                    .map(|r| format!("{}({})", r.tool_name, r.argument))
                                    .collect::<Vec<_>>()
                                    .join(", "),
                            ));

                            session_messages.push(Message::assistant(&full_response));
                            session_messages.push(Message::user(&result_msg));

                            match generate_with_cache(
                                &*backend,
                                &session_messages,
                                &cfg,
                                &project_root,
                                token_tx.clone(),
                                !reflection_enabled,
                                exact_cache.as_ref(),
                                &mut cache_stats,
                                CacheMode::PreferPromptLevel,
                            ) {
                                Ok(follow_up) => {
                                    let follow_up_source = follow_up.source;
                                    let follow_up_text = follow_up.text;
                                    let prompt_tokens = estimate_message_tokens(&session_messages);
                                    if !follow_up.hit {
                                        record_generation_budget(
                                            &cfg,
                                            &mut budget,
                                            &token_tx,
                                            prompt_tokens,
                                            &follow_up_text,
                                        );
                                    }
                                    let final_response = if reflection_enabled {
                                        emit_trace(
                                            &token_tx,
                                            ProgressStatus::Updated,
                                            "reflecting final answer...",
                                            false,
                                        );
                                        reflect_response(
                                            &*backend,
                                            &cfg,
                                            &project_root,
                                            &mut budget,
                                            &token_tx,
                                            exact_cache.as_ref(),
                                            &mut cache_stats,
                                            &session_messages,
                                            &follow_up_text,
                                        )
                                    } else {
                                        Ok(follow_up_text)
                                    };

                                    match final_response {
                                        Ok(final_response) => {
                                            if !final_response.trim().is_empty() {
                                                log_debug_response(
                                                    debug_logging_enabled,
                                                    &final_response,
                                                    if reflection_enabled {
                                                        debug_log::ResponseSource::Live
                                                    } else {
                                                        follow_up_source
                                                    },
                                                );
                                                store_exact_cache(
                                                    exact_cache.as_ref(),
                                                    &cfg,
                                                    &project_root,
                                                    &backend.name(),
                                                    &session_messages,
                                                    &final_response,
                                                );
                                                session_messages
                                                    .push(Message::assistant(&final_response));
                                            }
                                            emit_trace(
                                                &token_tx,
                                                ProgressStatus::Finished,
                                                "answer ready",
                                                false,
                                            );
                                        }
                                        Err(e) => {
                                            warn!(error = %e, "reflection after tool follow-up failed");
                                            emit_trace(
                                                &token_tx,
                                                ProgressStatus::Failed,
                                                "follow-up failed",
                                                false,
                                            );
                                            let _ =
                                                token_tx.send(InferenceEvent::Error(e.to_string()));
                                        }
                                    }
                                }
                                Err(e) => {
                                    warn!(error = %e, "tool follow-up generation failed");
                                    emit_trace(
                                        &token_tx,
                                        ProgressStatus::Failed,
                                        "tool follow-up failed",
                                        false,
                                    );
                                    let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                                }
                            }
                        } else {
                            let final_response = if reflection_enabled {
                                emit_trace(
                                    &token_tx,
                                    ProgressStatus::Updated,
                                    "reflecting final answer...",
                                    false,
                                );
                                reflect_response(
                                    &*backend,
                                    &cfg,
                                    &project_root,
                                    &mut budget,
                                    &token_tx,
                                    exact_cache.as_ref(),
                                    &mut cache_stats,
                                    &session_messages,
                                    &full_response,
                                )
                            } else {
                                Ok(full_response)
                            };

                            match final_response {
                                Ok(final_response) => {
                                    if !final_response.trim().is_empty() {
                                        log_debug_response(
                                            debug_logging_enabled,
                                            &final_response,
                                            if reflection_enabled {
                                                debug_log::ResponseSource::Live
                                            } else {
                                                response_source
                                            },
                                        );
                                        store_exact_cache(
                                            exact_cache.as_ref(),
                                            &cfg,
                                            &project_root,
                                            &backend.name(),
                                            &session_messages,
                                            &final_response,
                                        );
                                        store_prompt_level_cache(
                                            exact_cache.as_ref(),
                                            &cfg,
                                            &project_root,
                                            &backend.name(),
                                            &session_messages,
                                            &final_response,
                                        );
                                        session_messages.push(Message::assistant(&final_response));
                                    }
                                    emit_trace(
                                        &token_tx,
                                        ProgressStatus::Finished,
                                        "answer ready",
                                        false,
                                    );
                                }
                                Err(e) => {
                                    warn!(error = %e, "final response post-processing failed");
                                    emit_trace(
                                        &token_tx,
                                        ProgressStatus::Failed,
                                        "final answer failed",
                                        false,
                                    );
                                    let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                                }
                            }
                        }

                        save_session(session_store.as_ref(), &session_messages, &backend.name());
                        let _ = token_tx.send(InferenceEvent::Done);
                    }
                }
            }
        }
    }

    hooks.dispatch(HookEvent::SessionEnding {
        message_count: session_messages.len(),
    });

    if let Some(store) = fact_store.as_ref() {
        info!("persisting session facts");
        store.extract_and_store(&session_messages, &project_name, &*backend);
        match store.consolidate(&project_name, &cfg.memory) {
            Ok(stats) => {
                if stats.ttl_pruned + stats.dedup_removed + stats.cap_removed > 0 {
                    info!(
                        project = project_name.as_str(),
                        ttl_pruned = stats.ttl_pruned,
                        dedup_removed = stats.dedup_removed,
                        cap_removed = stats.cap_removed,
                        "memory consolidated"
                    );
                }
                hooks.dispatch(HookEvent::MemoryConsolidated {
                    facts_pruned: stats.ttl_pruned,
                    facts_deduped: stats.dedup_removed,
                    facts_capped: stats.cap_removed,
                });
            }
            Err(e) => warn!(error = %e, "memory consolidation failed"),
        }
    }
}
