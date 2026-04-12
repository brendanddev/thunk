use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};

use tracing::{debug, info, warn};

use crate::cache::ExactCache;
use crate::config;
use crate::debug_log;
use crate::events::{InferenceEvent, MemoryConsolidationView, ProgressStatus};
use crate::hooks::{HookEvent, Hooks};
use crate::memory::{
    compression,
    facts::{FactStore, TurnMemoryEvidence},
    index::ProjectIndex,
};
use crate::session::{display_name, SessionStore};
use crate::tools::{BashTool, Tool, ToolRegistry};

use super::investigation::{resolve_agentic_repo_turn, InvestigationState};
use super::memory::{
    apply_memory_update, clear_memory_retrieval, collect_retrieval_bundle, emit_memory_state,
    format_memory_recall, memory_fact_lines, refresh_loaded_facts, retrieval_trace_label,
    set_memory_retrieval, RuntimeMemoryState,
};
use super::support::{
    format_sessions_list, parse_export_format, reset_session_runtime, save_session, session_info,
};

use super::super::approval::{handle_pending_action, ApprovalContext};
use super::super::budget::{
    emit_cache_update, estimate_message_tokens, record_generation_budget, SessionBudget,
    SessionCacheStats,
};
use super::super::cache::{
    generate_with_cache, store_exact_cache, store_prompt_level_cache, CacheMode,
};
use super::super::indexing::{
    run_idle_index_step, IncrementalIndexState, IDLE_INDEX_POLL_INTERVAL,
};
use super::super::reflection::reflect_response;
use super::super::runtime::{
    eco_tool_result_limit, effective_reflection, emit_buffered_tokens, emit_generation_started,
    emit_trace, log_debug_response,
};
use super::super::tool_loop::run_read_only_tool_loop_with_resolution;
use super::super::{build_system_prompt, load_backend_with_fallback, Message, SessionCommand};

#[derive(Clone, Copy, Default)]
pub struct SessionRuntimeOptions {
    pub no_resume: bool,
}

/// After each response it checks for tool calls and runs a follow-up if needed.
#[allow(dead_code)]
pub fn model_thread(prompt_rx: Receiver<SessionCommand>, token_tx: Sender<InferenceEvent>) {
    model_thread_with_options(prompt_rx, token_tx, SessionRuntimeOptions::default());
}

pub fn model_thread_with_options(
    prompt_rx: Receiver<SessionCommand>,
    token_tx: Sender<InferenceEvent>,
    options: SessionRuntimeOptions,
) {
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
    let mut memory_state = RuntimeMemoryState::default();
    let mut investigation_state = InvestigationState::default();
    refresh_loaded_facts(&mut memory_state, fact_store.as_ref(), &project_name);
    hooks.dispatch(HookEvent::MemoryFactsLoaded {
        fact_count: memory_state.loaded_facts.len(),
    });
    let mut session_messages = vec![Message::system(&build_system_prompt(
        &tools,
        &[],
        &[],
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
    emit_memory_state(&token_tx, &memory_state);
    if !memory_state.loaded_facts.is_empty() {
        emit_trace(
            &token_tx,
            ProgressStatus::Finished,
            &format!(
                "memory: loaded {} fact{}",
                memory_state.loaded_facts.len(),
                if memory_state.loaded_facts.len() == 1 {
                    ""
                } else {
                    "s"
                }
            ),
            true,
        );
    }

    let session_store = SessionStore::open().ok();
    let mut active_session = None;
    if let Some(ref store) = session_store {
        if !options.no_resume {
            match store.load_most_recent() {
                Ok(Some(saved)) => {
                    let restored_count = saved.messages.len();
                    info!(
                        session_id = saved.summary.id.as_str(),
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
                    let info = session_info(&saved.summary);
                    active_session = Some(saved.summary.clone());
                    let _ = token_tx.send(InferenceEvent::SessionLoaded {
                        session: info,
                        display_messages,
                        saved_at: Some(saved.saved_at),
                    });
                    hooks.dispatch(HookEvent::SessionRestored {
                        message_count: restored_count,
                        saved_at: saved.saved_at,
                    });
                    hooks.dispatch(HookEvent::SessionResumed {
                        session_id: saved.summary.id.clone(),
                        named: saved.summary.name.is_some(),
                        message_count: restored_count,
                    });
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(error = %e, "session load failed — starting fresh");
                }
            }
        }

        if active_session.is_none() {
            match store.create_session(None, &backend.name()) {
                Ok(summary) => {
                    let info = session_info(&summary);
                    hooks.dispatch(HookEvent::SessionCreated {
                        session_id: summary.id.clone(),
                        named: false,
                    });
                    active_session = Some(summary);
                    let _ = token_tx.send(InferenceEvent::SessionStatus(info));
                }
                Err(e) => warn!(error = %e, "initial session create failed"),
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
                if let (Some(store), Some(current)) =
                    (session_store.as_ref(), active_session.as_ref())
                {
                    if let Err(e) = store.delete_session(&current.id) {
                        warn!(error = %e, "session clear failed");
                    } else {
                        hooks.dispatch(HookEvent::SessionCleared {
                            session_id: current.id.clone(),
                        });
                    }
                }
                reset_session_runtime(
                    &mut session_messages,
                    &tools,
                    eco_enabled,
                    &mut budget,
                    &mut cache_stats,
                    &cfg.backend,
                    &token_tx,
                );
                clear_memory_retrieval(&mut memory_state);
                memory_state.last_update = None;
                investigation_state.clear();
                emit_memory_state(&token_tx, &memory_state);
                if let Some(ref store) = session_store {
                    match store.create_session(None, &backend.name()) {
                        Ok(summary) => {
                            hooks.dispatch(HookEvent::SessionCreated {
                                session_id: summary.id.clone(),
                                named: false,
                            });
                            active_session = Some(summary.clone());
                            let _ = token_tx.send(InferenceEvent::SessionLoaded {
                                session: session_info(&summary),
                                display_messages: Vec::new(),
                                saved_at: None,
                            });
                            let _ = token_tx.send(InferenceEvent::SystemMessage(
                                "conversation cleared".to_string(),
                            ));
                        }
                        Err(e) => {
                            warn!(error = %e, "replacement session create failed");
                            active_session = None;
                            let _ = token_tx.send(InferenceEvent::SystemMessage(
                                "conversation cleared".to_string(),
                            ));
                        }
                    }
                }
                continue;
            }
            SessionCommand::ListSessions => {
                match session_store.as_ref().map(|store| store.list_sessions()) {
                    Some(Ok(sessions)) => {
                        let active_id = active_session.as_ref().map(|session| session.id.as_str());
                        let _ = token_tx.send(InferenceEvent::SystemMessage(format_sessions_list(
                            &sessions, active_id,
                        )));
                    }
                    Some(Err(e)) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                    None => {
                        let _ = token_tx.send(InferenceEvent::Error(
                            "Session store is unavailable".to_string(),
                        ));
                    }
                }
                continue;
            }
            SessionCommand::NewSession(name) => {
                save_session(
                    session_store.as_ref(),
                    &mut active_session,
                    &session_messages,
                    &backend.name(),
                    &token_tx,
                );
                reset_session_runtime(
                    &mut session_messages,
                    &tools,
                    eco_enabled,
                    &mut budget,
                    &mut cache_stats,
                    &cfg.backend,
                    &token_tx,
                );
                clear_memory_retrieval(&mut memory_state);
                memory_state.last_update = None;
                investigation_state.clear();
                emit_memory_state(&token_tx, &memory_state);
                match session_store
                    .as_ref()
                    .map(|store| store.create_session(name.as_deref(), &backend.name()))
                {
                    Some(Ok(summary)) => {
                        let session_label = display_name(&summary);
                        hooks.dispatch(HookEvent::SessionCreated {
                            session_id: summary.id.clone(),
                            named: summary.name.is_some(),
                        });
                        active_session = Some(summary.clone());
                        let _ = token_tx.send(InferenceEvent::SessionLoaded {
                            session: session_info(&summary),
                            display_messages: Vec::new(),
                            saved_at: None,
                        });
                        let _ = token_tx.send(InferenceEvent::SystemMessage(format!(
                            "started new session: {session_label}"
                        )));
                    }
                    Some(Err(e)) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                    None => {
                        let _ = token_tx.send(InferenceEvent::Error(
                            "Session store is unavailable".to_string(),
                        ));
                    }
                }
                continue;
            }
            SessionCommand::RenameSession(name) => {
                match (session_store.as_ref(), active_session.as_ref()) {
                    (Some(store), Some(current)) => {
                        match store.rename_session(&current.id, &name) {
                            Ok(updated) => {
                                hooks.dispatch(HookEvent::SessionRenamed {
                                    session_id: updated.id.clone(),
                                    named: updated.name.is_some(),
                                });
                                active_session = Some(updated.clone());
                                let _ = token_tx
                                    .send(InferenceEvent::SessionStatus(session_info(&updated)));
                                let _ = token_tx.send(InferenceEvent::SystemMessage(format!(
                                    "renamed session to {}",
                                    display_name(&updated)
                                )));
                            }
                            Err(e) => {
                                let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                            }
                        }
                    }
                    _ => {
                        let _ = token_tx.send(InferenceEvent::Error(
                            "No active session is available to rename".to_string(),
                        ));
                    }
                }
                continue;
            }
            SessionCommand::ResumeSession(selector) => {
                save_session(
                    session_store.as_ref(),
                    &mut active_session,
                    &session_messages,
                    &backend.name(),
                    &token_tx,
                );
                match session_store
                    .as_ref()
                    .map(|store| store.load_session(&selector))
                {
                    Some(Ok(saved)) => {
                        reset_session_runtime(
                            &mut session_messages,
                            &tools,
                            eco_enabled,
                            &mut budget,
                            &mut cache_stats,
                            &cfg.backend,
                            &token_tx,
                        );
                        clear_memory_retrieval(&mut memory_state);
                        memory_state.last_update = None;
                        investigation_state.clear();
                        emit_memory_state(&token_tx, &memory_state);
                        let display_messages = saved
                            .messages
                            .iter()
                            .map(|m| (m.role.clone(), m.content.clone()))
                            .collect::<Vec<_>>();
                        session_messages.extend(saved.messages);
                        hooks.dispatch(HookEvent::SessionResumed {
                            session_id: saved.summary.id.clone(),
                            named: saved.summary.name.is_some(),
                            message_count: saved.summary.message_count,
                        });
                        active_session = Some(saved.summary.clone());
                        let _ = token_tx.send(InferenceEvent::SessionLoaded {
                            session: session_info(&saved.summary),
                            display_messages,
                            saved_at: Some(saved.saved_at),
                        });
                    }
                    Some(Err(e)) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                    None => {
                        let _ = token_tx.send(InferenceEvent::Error(
                            "Session store is unavailable".to_string(),
                        ));
                    }
                }
                continue;
            }
            SessionCommand::DeleteSession(selector) => {
                match (session_store.as_ref(), active_session.as_ref()) {
                    (Some(store), Some(current_active)) => match store.resolve_session(&selector) {
                        Ok(summary) => {
                            let was_active = summary.id == current_active.id;
                            let deleted_label = display_name(&summary);
                            if let Err(e) = store.delete_session(&summary.id) {
                                let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                                continue;
                            }
                            hooks.dispatch(HookEvent::SessionDeleted {
                                session_id: summary.id.clone(),
                                was_active,
                            });

                            if was_active {
                                reset_session_runtime(
                                    &mut session_messages,
                                    &tools,
                                    eco_enabled,
                                    &mut budget,
                                    &mut cache_stats,
                                    &cfg.backend,
                                    &token_tx,
                                );
                                clear_memory_retrieval(&mut memory_state);
                                memory_state.last_update = None;
                                emit_memory_state(&token_tx, &memory_state);
                                match store.create_session(None, &backend.name()) {
                                    Ok(replacement) => {
                                        hooks.dispatch(HookEvent::SessionCreated {
                                            session_id: replacement.id.clone(),
                                            named: false,
                                        });
                                        active_session = Some(replacement.clone());
                                        let _ = token_tx.send(InferenceEvent::SessionLoaded {
                                            session: session_info(&replacement),
                                            display_messages: Vec::new(),
                                            saved_at: None,
                                        });
                                        let _ = token_tx.send(InferenceEvent::SystemMessage(
                                            format!(
                                                "deleted session: {deleted_label}; started fresh unnamed session"
                                            ),
                                        ));
                                    }
                                    Err(e) => {
                                        active_session = None;
                                        let _ = token_tx.send(InferenceEvent::Error(format!(
                                            "deleted session {deleted_label}, but failed to create replacement session: {e}"
                                        )));
                                    }
                                }
                            } else {
                                let _ = token_tx.send(InferenceEvent::SystemMessage(format!(
                                    "deleted session: {deleted_label}"
                                )));
                            }
                        }
                        Err(e) => {
                            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                        }
                    },
                    (Some(store), None) => match store.resolve_session(&selector) {
                        Ok(summary) => {
                            let deleted_label = display_name(&summary);
                            match store.delete_session(&summary.id) {
                                Ok(()) => {
                                    hooks.dispatch(HookEvent::SessionDeleted {
                                        session_id: summary.id.clone(),
                                        was_active: false,
                                    });
                                    let _ = token_tx.send(InferenceEvent::SystemMessage(format!(
                                        "deleted session: {deleted_label}"
                                    )));
                                }
                                Err(e) => {
                                    let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                                }
                            }
                        }
                        Err(e) => {
                            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                        }
                    },
                    (None, _) => {
                        let _ = token_tx.send(InferenceEvent::Error(
                            "Session store is unavailable".to_string(),
                        ));
                    }
                }
                continue;
            }
            SessionCommand::ExportSession { selector, format } => {
                match session_store.as_ref().map(|store| {
                    parse_export_format(format).and_then(|fmt| store.export_session(&selector, fmt))
                }) {
                    Some(Ok((summary, path))) => {
                        hooks.dispatch(HookEvent::SessionExported {
                            session_id: summary.id.clone(),
                            format: path
                                .extension()
                                .and_then(|ext| ext.to_str())
                                .unwrap_or("unknown")
                                .to_string(),
                        });
                        let _ = token_tx.send(InferenceEvent::SystemMessage(format!(
                            "exported session {} to {}",
                            display_name(&summary),
                            path.display()
                        )));
                    }
                    Some(Err(e)) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                    None => {
                        let _ = token_tx.send(InferenceEvent::Error(
                            "Session store is unavailable".to_string(),
                        ));
                    }
                }
                continue;
            }
            SessionCommand::InjectUserContext { content, metadata } => {
                info!(chars = content.chars().count(), "user context injected");
                investigation_state.apply_injected_context(metadata.as_ref(), &content);
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
                                turn_memory: None,
                            },
                            next_action_id,
                            pending,
                            false,
                        ) {
                            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                        } else {
                            save_session(
                                session_store.as_ref(),
                                &mut active_session,
                                &session_messages,
                                &backend.name(),
                                &token_tx,
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
                                turn_memory: None,
                            },
                            next_action_id,
                            pending,
                            false,
                        ) {
                            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                        } else {
                            save_session(
                                session_store.as_ref(),
                                &mut active_session,
                                &session_messages,
                                &backend.name(),
                                &token_tx,
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
            SessionCommand::RequestFileEdit { path, edits } => {
                info!(path = path.as_str(), "file edit approval requested");
                match crate::tools::build_pending_edit_request(&path, &edits) {
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
                                turn_memory: None,
                            },
                            next_action_id,
                            pending,
                            false,
                        ) {
                            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                        } else {
                            save_session(
                                session_store.as_ref(),
                                &mut active_session,
                                &session_messages,
                                &backend.name(),
                                &token_tx,
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
                        first.content = build_system_prompt(
                            &tools,
                            &memory_fact_lines(&memory_state.last_selected_facts),
                            &[],
                            &memory_state.last_selected_session_excerpts,
                            eco_enabled,
                        );
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
            SessionCommand::RecallMemory(query) => {
                let bundle = collect_retrieval_bundle(
                    &query,
                    eco_enabled,
                    &project_name,
                    project_index.as_ref(),
                    fact_store.as_ref(),
                    session_store.as_ref(),
                    active_session.as_ref().map(|session| session.id.as_str()),
                    &memory_state.loaded_facts,
                );
                let _ = token_tx.send(InferenceEvent::SystemMessage(format_memory_recall(
                    &query, &bundle,
                )));
                continue;
            }
            SessionCommand::PruneMemory => {
                match fact_store
                    .as_ref()
                    .map(|store| store.prune_irrelevant_facts(&project_name))
                {
                    Some(Ok(removed)) => {
                        refresh_loaded_facts(&mut memory_state, fact_store.as_ref(), &project_name);
                        clear_memory_retrieval(&mut memory_state);
                        emit_memory_state(&token_tx, &memory_state);
                        emit_trace(
                            &token_tx,
                            ProgressStatus::Finished,
                            &format!(
                                "memory: pruned {} fact{}",
                                removed,
                                if removed == 1 { "" } else { "s" }
                            ),
                            true,
                        );
                        let _ = token_tx.send(InferenceEvent::SystemMessage(format!(
                            "memory prune removed {removed} irrelevant fact{}",
                            if removed == 1 { "" } else { "s" }
                        )));
                    }
                    Some(Err(e)) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                    None => {
                        let _ = token_tx.send(InferenceEvent::Error(
                            "Memory store is unavailable".to_string(),
                        ));
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

                let tool_loop_resolution = resolve_agentic_repo_turn(&prompt, &investigation_state);

                if let Some(resolution) = tool_loop_resolution {
                    let intent = resolution.intent;
                    clear_memory_retrieval(&mut memory_state);
                    memory_state.last_update = None;
                    emit_memory_state(&token_tx, &memory_state);
                    hooks.dispatch(HookEvent::MemorySummariesSelected { summary_count: 0 });
                    if let Some(first) = session_messages.first_mut() {
                        if first.role == "system" {
                            first.content = build_system_prompt(&tools, &[], &[], &[], eco_enabled);
                        }
                    }
                    compression::compress_history(
                        &mut session_messages,
                        eco_enabled,
                        investigation_state.compression_context().as_ref(),
                    );

                    let mut tool_loop_messages = session_messages.clone();
                    if let Some(summary) = investigation_state.summary_message() {
                        let insert_at = tool_loop_messages.len().saturating_sub(1);
                        tool_loop_messages.insert(insert_at, summary);
                    }

                    let mut turn_memory = TurnMemoryEvidence::new(prompt.clone(), Vec::new());
                    hooks.dispatch(HookEvent::BeforeGeneration {
                        backend: backend.name(),
                        message_count: tool_loop_messages.len(),
                        eco: eco_enabled,
                        reflection: reflection_enabled,
                    });

                    match run_read_only_tool_loop_with_resolution(
                        intent,
                        &prompt,
                        Some(&resolution),
                        &tool_loop_messages,
                        &*backend,
                        &tools,
                        &cfg,
                        &project_root,
                        &token_tx,
                        None,
                        &mut cache_stats,
                        &mut budget,
                        eco_enabled,
                        reflection_enabled,
                    ) {
                        Ok(outcome) => {
                            investigation_state.note_tool_loop_outcome(
                                intent,
                                &prompt,
                                &outcome.tool_results,
                            );
                            for result in &outcome.tool_results {
                                hooks.dispatch(HookEvent::ToolExecuted {
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
                                if !reflection_enabled && !outcome.streamed_final_response {
                                    emit_buffered_tokens(&token_tx, &outcome.final_response);
                                }
                                log_debug_response(
                                    debug_logging_enabled,
                                    &outcome.final_response,
                                    debug_log::ResponseSource::Live,
                                );
                                session_messages.push(Message::assistant(&outcome.final_response));
                            }
                            hooks.dispatch(HookEvent::AfterGeneration {
                                backend: backend.name(),
                                response_chars: outcome.final_response.chars().count(),
                                from_cache: false,
                                elapsed_ms: 0,
                            });
                            emit_trace(&token_tx, ProgressStatus::Finished, "answer ready", false);
                            if let Some(store) = fact_store.as_ref() {
                                let update = store.verify_and_store_turn(
                                    &project_name,
                                    &turn_memory,
                                    &*backend,
                                );
                                apply_memory_update(&token_tx, &hooks, &mut memory_state, update);
                            }
                            save_session(
                                session_store.as_ref(),
                                &mut active_session,
                                &session_messages,
                                &backend.name(),
                                &token_tx,
                            );
                            let _ = token_tx.send(InferenceEvent::Done);
                        }
                        Err(e) => {
                            emit_trace(
                                &token_tx,
                                ProgressStatus::Failed,
                                "tool loop failed",
                                false,
                            );
                            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                        }
                    }
                    continue;
                }

                let retrieval = collect_retrieval_bundle(
                    &prompt,
                    eco_enabled,
                    &project_name,
                    project_index.as_ref(),
                    fact_store.as_ref(),
                    session_store.as_ref(),
                    active_session.as_ref().map(|session| session.id.as_str()),
                    &memory_state.loaded_facts,
                );
                let mut turn_memory =
                    TurnMemoryEvidence::new(prompt.clone(), retrieval.summaries.clone());
                set_memory_retrieval(&mut memory_state, &prompt, &retrieval);
                memory_state.last_update = None;
                emit_memory_state(&token_tx, &memory_state);
                hooks.dispatch(HookEvent::MemorySummariesSelected {
                    summary_count: retrieval.summaries.len(),
                });
                if let Some(label) = retrieval_trace_label(&retrieval) {
                    emit_trace(&token_tx, ProgressStatus::Finished, &label, false);
                }

                if let Some(first) = session_messages.first_mut() {
                    if first.role == "system" {
                        first.content = build_system_prompt(
                            &tools,
                            &memory_fact_lines(&retrieval.facts),
                            &retrieval.summaries,
                            &retrieval.session_excerpts,
                            eco_enabled,
                        );
                    }
                }

                compression::compress_history(
                    &mut session_messages,
                    eco_enabled,
                    investigation_state.compression_context().as_ref(),
                );

                let generation_messages = session_messages.clone();
                let generation_exact_cache = exact_cache.as_ref();

                emit_generation_started(&token_tx, "generating...", false);
                emit_trace(
                    &token_tx,
                    ProgressStatus::Started,
                    "drafting answer...",
                    false,
                );
                hooks.dispatch(HookEvent::BeforeGeneration {
                    backend: backend.name(),
                    message_count: generation_messages.len(),
                    eco: eco_enabled,
                    reflection: reflection_enabled,
                });
                let response = generate_with_cache(
                    &*backend,
                    &generation_messages,
                    &cfg,
                    &project_root,
                    token_tx.clone(),
                    !reflection_enabled,
                    generation_exact_cache,
                    &mut cache_stats,
                    CacheMode::PreferPromptLevel,
                );
                debug!(
                    reflection_enabled,
                    message_count = generation_messages.len(),
                    "generation started"
                );
                let prompt_tokens = estimate_message_tokens(&generation_messages);

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
                            turn_memory.record_tool_result(
                                result.tool_name.clone(),
                                result.argument.clone(),
                                result.output.clone(),
                                false,
                            );
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
                                    turn_memory: Some(&mut turn_memory),
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
                            } else if let Some(store) = fact_store.as_ref() {
                                let update = store.verify_and_store_turn(
                                    &project_name,
                                    &turn_memory,
                                    &*backend,
                                );
                                apply_memory_update(&token_tx, &hooks, &mut memory_state, update);
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
                                            turn_memory.set_final_response(final_response.clone());
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
                                            if let Some(store) = fact_store.as_ref() {
                                                let update = store.verify_and_store_turn(
                                                    &project_name,
                                                    &turn_memory,
                                                    &*backend,
                                                );
                                                apply_memory_update(
                                                    &token_tx,
                                                    &hooks,
                                                    &mut memory_state,
                                                    update,
                                                );
                                            }
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
                                    turn_memory.set_final_response(final_response.clone());
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
                                    if let Some(store) = fact_store.as_ref() {
                                        let update = store.verify_and_store_turn(
                                            &project_name,
                                            &turn_memory,
                                            &*backend,
                                        );
                                        apply_memory_update(
                                            &token_tx,
                                            &hooks,
                                            &mut memory_state,
                                            update,
                                        );
                                    }
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

                        save_session(
                            session_store.as_ref(),
                            &mut active_session,
                            &session_messages,
                            &backend.name(),
                            &token_tx,
                        );
                        let _ = token_tx.send(InferenceEvent::Done);
                    }
                }
            }
        }
    }

    hooks.dispatch(HookEvent::SessionEnding {
        message_count: session_messages.len(),
    });

    if let (Some(store), Some(current)) = (session_store.as_ref(), active_session.as_ref()) {
        if let Err(e) = store.delete_if_empty_unnamed(&current.id) {
            warn!(error = %e, "empty session cleanup failed");
        }
    }

    if let Some(store) = fact_store.as_ref() {
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
                memory_state.last_consolidation = Some(MemoryConsolidationView {
                    ttl_pruned: stats.ttl_pruned,
                    dedup_removed: stats.dedup_removed,
                    cap_removed: stats.cap_removed,
                });
                emit_memory_state(&token_tx, &memory_state);
                if stats.ttl_pruned + stats.dedup_removed + stats.cap_removed > 0 {
                    emit_trace(
                        &token_tx,
                        ProgressStatus::Finished,
                        &format!(
                            "memory: consolidated -{} dup",
                            stats.dedup_removed + stats.cap_removed + stats.ttl_pruned
                        ),
                        true,
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
