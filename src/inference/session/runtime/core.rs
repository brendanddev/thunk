use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};

use tracing::{info, warn};

use crate::cache::ExactCache;
use crate::config;
use crate::events::{InferenceEvent, MemoryConsolidationView, ProgressStatus};
use crate::hooks::{HookEvent, Hooks};
use crate::memory::{facts::FactStore, index::ProjectIndex};
use crate::session::SessionStore;
use crate::tools::ToolRegistry;

use super::action_commands::{
    handle_request_file_edit, handle_request_file_write, handle_request_shell_command,
};
use super::maintenance::{
    handle_clear_cache, handle_clear_debug_log, handle_orphaned_approval, handle_prune_memory,
    handle_recall_memory, handle_set_debug_logging, handle_set_eco, handle_set_reflection,
};
use super::session_commands::{
    handle_clear_session, handle_delete_session, handle_export_session, handle_list_sessions,
    handle_new_session, handle_rename_session, handle_resume_session,
};
use super::state::{RuntimeContext, RuntimeState, SessionRuntimeOptions};
use super::turns::{handle_injected_user_context, handle_submit_user};

use super::super::super::budget::{emit_cache_update, SessionBudget, SessionCacheStats};
use super::super::super::indexing::{
    run_idle_index_step, IncrementalIndexState, IDLE_INDEX_POLL_INTERVAL,
};
use super::super::super::runtime::{effective_reflection, emit_trace};
use super::super::super::{
    build_system_prompt, load_backend_with_fallback, Message, SessionCommand,
};
use super::super::investigation::InvestigationState;
use super::super::memory::{emit_memory_state, refresh_loaded_facts, RuntimeMemoryState};
use super::super::support::session_info;

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
    let eco_enabled = cfg.eco.enabled;
    let reflection_requested = cfg.reflection.enabled;
    let reflection_enabled = effective_reflection(reflection_requested, eco_enabled);
    let debug_logging_enabled = cfg.debug_logging.content;
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
    let index_state = if cfg.backend == "llama_cpp" {
        info!("idle incremental indexing disabled for llama_cpp backend");
        None
    } else {
        project_index.as_ref().map(|_| IncrementalIndexState::new())
    };
    let mut memory_state = RuntimeMemoryState::default();
    let investigation_state = InvestigationState::default();
    refresh_loaded_facts(&mut memory_state, fact_store.as_ref(), &project_name);
    hooks.dispatch(HookEvent::MemoryFactsLoaded {
        fact_count: memory_state.loaded_facts.len(),
    });

    let session_messages = vec![Message::system(&build_system_prompt(
        &tools,
        &[],
        &[],
        &[],
        eco_enabled,
    ))];
    let budget = SessionBudget {
        has_cost_estimate: cfg.backend == "llama_cpp" || cfg.backend == "ollama",
        ..SessionBudget::default()
    };
    let cache_stats = SessionCacheStats::default();
    emit_cache_update(&cache_stats, false, &token_tx);
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
    let mut state = RuntimeState {
        eco_enabled,
        reflection_requested,
        reflection_enabled,
        debug_logging_enabled,
        index_state,
        memory_state,
        investigation_state,
        session_messages,
        budget,
        cache_stats,
        next_action_id: 1,
        active_session: None,
    };

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
                    state.session_messages.extend(saved.messages);
                    let info = session_info(&saved.summary);
                    state.active_session = Some(saved.summary.clone());
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

        if state.active_session.is_none() {
            match store.create_session(None, &backend.name()) {
                Ok(summary) => {
                    let info = session_info(&summary);
                    hooks.dispatch(HookEvent::SessionCreated {
                        session_id: summary.id.clone(),
                        named: false,
                    });
                    state.active_session = Some(summary);
                    let _ = token_tx.send(InferenceEvent::SessionStatus(info));
                }
                Err(e) => warn!(error = %e, "initial session create failed"),
            }
        }
    }

    loop {
        if let (Some(index), Some(index_state)) =
            (project_index.as_ref(), state.index_state.as_mut())
        {
            run_idle_index_step(index_state, index, &project_root, &*backend);
        }

        let command = match prompt_rx.recv_timeout(IDLE_INDEX_POLL_INTERVAL) {
            Ok(command) => command,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };

        let ctx = RuntimeContext {
            prompt_rx: &prompt_rx,
            token_tx: &token_tx,
            cfg: &cfg,
            backend: &*backend,
            hooks: &hooks,
            tools: &tools,
            project_root: &project_root,
            project_name: &project_name,
            exact_cache: exact_cache.as_ref(),
            fact_store: fact_store.as_ref(),
            project_index: project_index.as_ref(),
            session_store: session_store.as_ref(),
        };

        match command {
            SessionCommand::ClearSession => handle_clear_session(&ctx, &mut state),
            SessionCommand::ListSessions => handle_list_sessions(&ctx, &state),
            SessionCommand::NewSession(name) => handle_new_session(&ctx, &mut state, name),
            SessionCommand::RenameSession(name) => handle_rename_session(&ctx, &mut state, name),
            SessionCommand::ResumeSession(selector) => {
                handle_resume_session(&ctx, &mut state, selector)
            }
            SessionCommand::DeleteSession(selector) => {
                handle_delete_session(&ctx, &mut state, selector)
            }
            SessionCommand::ExportSession { selector, format } => {
                handle_export_session(&ctx, selector, format)
            }
            SessionCommand::InjectUserContext { content, metadata } => {
                handle_injected_user_context(&ctx, &mut state, content, metadata)
            }
            SessionCommand::RequestShellCommand(command) => {
                handle_request_shell_command(&ctx, &mut state, command)
            }
            SessionCommand::RequestFileWrite { path, content } => {
                handle_request_file_write(&ctx, &mut state, path, content)
            }
            SessionCommand::RequestFileEdit { path, edits } => {
                handle_request_file_edit(&ctx, &mut state, path, edits)
            }
            SessionCommand::SetReflection(enabled) => {
                handle_set_reflection(&ctx, &mut state, enabled)
            }
            SessionCommand::SetEco(enabled) => handle_set_eco(&ctx, &mut state, enabled),
            SessionCommand::SetDebugLogging(enabled) => {
                handle_set_debug_logging(&ctx, &mut state, enabled)
            }
            SessionCommand::ClearDebugLog => handle_clear_debug_log(&ctx),
            SessionCommand::ClearCache => handle_clear_cache(&ctx),
            SessionCommand::RecallMemory(query) => handle_recall_memory(&ctx, &state, query),
            SessionCommand::PruneMemory => handle_prune_memory(&ctx, &mut state),
            SessionCommand::ApproveAction(_) | SessionCommand::RejectAction(_) => {
                warn!("approval command received with no pending action");
                handle_orphaned_approval(&ctx);
            }
            SessionCommand::SubmitUser(prompt) => handle_submit_user(&ctx, &mut state, prompt),
        }
    }

    hooks.dispatch(HookEvent::SessionEnding {
        message_count: state.session_messages.len(),
    });

    if let (Some(store), Some(current)) = (session_store.as_ref(), state.active_session.as_ref()) {
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
                state.memory_state.last_consolidation = Some(MemoryConsolidationView {
                    ttl_pruned: stats.ttl_pruned,
                    dedup_removed: stats.dedup_removed,
                    cap_removed: stats.cap_removed,
                });
                emit_memory_state(&token_tx, &state.memory_state);
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
