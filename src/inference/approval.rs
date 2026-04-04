use std::path::Path;
use std::sync::mpsc::{Receiver, Sender};

use tracing::{info, warn};

use crate::cache::ExactCache;
use crate::config;
use crate::debug_log;
use crate::error::{ParamsError, Result};
use crate::events::{InferenceEvent, PendingAction, PendingActionKind, ProgressStatus};
use crate::hooks::{HookEvent, Hooks};
use crate::memory::facts::TurnMemoryEvidence;
use crate::safety::InspectionReport;
use crate::tools::{PendingToolAction, ToolRegistry};

use super::budget::{
    estimate_message_tokens, record_generation_budget, SessionBudget, SessionCacheStats,
};
use super::cache::{generate_with_cache, store_exact_cache, CacheMode};
use super::indexing::IncrementalIndexState;
use super::reflection::reflect_response;
use super::runtime::{
    eco_tool_result_limit, emit_generation_started, emit_trace, log_debug_response,
};
use super::{InferenceBackend, Message, SessionCommand};

pub(super) struct ApprovalContext<'a> {
    pub prompt_rx: &'a Receiver<SessionCommand>,
    pub token_tx: &'a Sender<InferenceEvent>,
    pub backend: &'a dyn InferenceBackend,
    pub tools: &'a ToolRegistry,
    pub exact_cache: Option<&'a ExactCache>,
    pub session_messages: &'a mut Vec<Message>,
    pub cfg: &'a config::Config,
    pub project_root: &'a Path,
    pub budget: &'a mut SessionBudget,
    pub cache_stats: &'a mut SessionCacheStats,
    pub debug_logging_enabled: bool,
    pub reflection_enabled: bool,
    pub eco_enabled: bool,
    pub hooks: &'a Hooks,
    pub index_state: Option<&'a mut IncrementalIndexState>,
    pub turn_memory: Option<&'a mut TurnMemoryEvidence>,
}

pub(super) fn handle_pending_action(
    ctx: ApprovalContext<'_>,
    action_id: u64,
    mut pending: PendingToolAction,
    run_follow_up: bool,
) -> Result<()> {
    if pending.preview.is_empty() {
        pending.preview = pending.argument.clone();
    }

    let event = PendingAction {
        id: action_id,
        kind: pending.kind.clone(),
        title: pending.title.clone(),
        preview: pending.preview.clone(),
        inspection: pending.inspection.clone(),
    };
    dispatch_inspection_hook(ctx.hooks, &pending.inspection);
    info!(
        action_id,
        tool = pending.tool_name.as_str(),
        kind = ?pending.kind,
        run_follow_up,
        "pending action proposed"
    );
    ctx.hooks.dispatch(HookEvent::ApprovalRequested {
        tool_name: pending.tool_name.clone(),
        kind: format!("{:?}", pending.kind),
    });
    emit_trace(
        ctx.token_tx,
        ProgressStatus::Started,
        "awaiting approval...",
        false,
    );
    let _ = ctx.token_tx.send(InferenceEvent::PendingAction(event));
    let mut index_state = ctx.index_state;
    let mut turn_memory = ctx.turn_memory;

    loop {
        match ctx.prompt_rx.recv() {
            Ok(SessionCommand::ApproveAction(id)) if id == action_id => {
                info!(
                    action_id,
                    tool = pending.tool_name.as_str(),
                    "pending action approved"
                );
                ctx.hooks.dispatch(HookEvent::ApprovalResolved {
                    tool_name: pending.tool_name.clone(),
                    approved: true,
                });
                emit_trace(
                    ctx.token_tx,
                    ProgressStatus::Finished,
                    "approval received",
                    false,
                );
                emit_trace(
                    ctx.token_tx,
                    ProgressStatus::Started,
                    &format!("running {}", pending.tool_name),
                    false,
                );
                let result = ctx.tools.execute_pending_action(&pending);
                if let Some(memory) = turn_memory.as_deref_mut() {
                    memory.record_tool_result(
                        result.tool_name.clone(),
                        result.argument.clone(),
                        result.output.clone(),
                        true,
                    );
                }
                if matches!(
                    pending.kind,
                    PendingActionKind::FileWrite | PendingActionKind::FileEdit
                ) {
                    if let Some(state) = index_state.as_mut() {
                        state.request_scan_soon();
                    }
                }
                ctx.hooks.dispatch(HookEvent::ToolExecuted {
                    tool_name: result.tool_name.clone(),
                    argument_chars: pending.argument.chars().count(),
                    result_chars: result.output.chars().count(),
                });
                emit_trace(
                    ctx.token_tx,
                    ProgressStatus::Finished,
                    "approved action complete",
                    false,
                );
                let _ = ctx.token_tx.send(InferenceEvent::ToolCall(format!(
                    "{}({})",
                    result.tool_name, result.argument
                )));
                let result_msg = ToolRegistry::format_results_with_limit(
                    &[result],
                    eco_tool_result_limit(ctx.eco_enabled),
                );
                if let Some(result_msg) = result_msg {
                    ctx.session_messages.push(Message::user(&result_msg));
                    if !run_follow_up {
                        let _ = ctx
                            .token_tx
                            .send(InferenceEvent::ContextMessage(result_msg));
                    }
                }
                if run_follow_up {
                    emit_generation_started(ctx.token_tx, "generating...", true);
                    emit_trace(
                        ctx.token_tx,
                        ProgressStatus::Started,
                        "drafting follow-up...",
                        false,
                    );
                    let prompt_tokens = estimate_message_tokens(ctx.session_messages);
                    let follow_up = generate_with_cache(
                        ctx.backend,
                        ctx.session_messages,
                        ctx.cfg,
                        ctx.project_root,
                        ctx.token_tx.clone(),
                        !ctx.reflection_enabled,
                        ctx.exact_cache,
                        ctx.cache_stats,
                        CacheMode::PreferPromptLevel,
                    )?;
                    let follow_up_source = follow_up.source;
                    let follow_up_text = follow_up.text;
                    if !follow_up.hit {
                        record_generation_budget(
                            ctx.cfg,
                            ctx.budget,
                            ctx.token_tx,
                            prompt_tokens,
                            &follow_up_text,
                        );
                    }
                    let final_response = if ctx.reflection_enabled {
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
                            ctx.budget,
                            ctx.token_tx,
                            ctx.exact_cache,
                            ctx.cache_stats,
                            ctx.session_messages,
                            &follow_up_text,
                        )?
                    } else {
                        follow_up_text
                    };
                    if !final_response.trim().is_empty() {
                        if let Some(memory) = turn_memory.as_deref_mut() {
                            memory.set_final_response(final_response.clone());
                        }
                        log_debug_response(
                            ctx.debug_logging_enabled,
                            &final_response,
                            if ctx.reflection_enabled {
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
                            ctx.session_messages,
                            &final_response,
                        );
                        ctx.session_messages
                            .push(Message::assistant(&final_response));
                    }
                    emit_trace(
                        ctx.token_tx,
                        ProgressStatus::Finished,
                        "follow-up ready",
                        false,
                    );
                }
                return Ok(());
            }
            Ok(SessionCommand::RejectAction(id)) if id == action_id => {
                info!(
                    action_id,
                    tool = pending.tool_name.as_str(),
                    "pending action rejected"
                );
                ctx.hooks.dispatch(HookEvent::ApprovalResolved {
                    tool_name: pending.tool_name.clone(),
                    approved: false,
                });
                emit_trace(
                    ctx.token_tx,
                    ProgressStatus::Failed,
                    "action rejected",
                    false,
                );
                let rejection = format!(
                    "User rejected proposed action: {}",
                    pending.display_argument
                );
                ctx.session_messages.push(Message::user(&rejection));
                if !run_follow_up {
                    let _ = ctx.token_tx.send(InferenceEvent::ContextMessage(rejection));
                }
                if run_follow_up {
                    emit_generation_started(ctx.token_tx, "generating...", true);
                    emit_trace(
                        ctx.token_tx,
                        ProgressStatus::Started,
                        "drafting follow-up...",
                        false,
                    );
                    let follow_up = generate_with_cache(
                        ctx.backend,
                        ctx.session_messages,
                        ctx.cfg,
                        ctx.project_root,
                        ctx.token_tx.clone(),
                        !ctx.reflection_enabled,
                        ctx.exact_cache,
                        ctx.cache_stats,
                        CacheMode::PreferPromptLevel,
                    )?;
                    let follow_up_source = follow_up.source;
                    let follow_up_text = follow_up.text;
                    let prompt_tokens = estimate_message_tokens(ctx.session_messages);
                    if !follow_up.hit {
                        record_generation_budget(
                            ctx.cfg,
                            ctx.budget,
                            ctx.token_tx,
                            prompt_tokens,
                            &follow_up_text,
                        );
                    }
                    let final_response = if ctx.reflection_enabled {
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
                            ctx.budget,
                            ctx.token_tx,
                            ctx.exact_cache,
                            ctx.cache_stats,
                            ctx.session_messages,
                            &follow_up_text,
                        )?
                    } else {
                        follow_up_text
                    };
                    if let Some(memory) = turn_memory.as_deref_mut() {
                        memory.set_final_response(final_response.clone());
                    }
                    if !final_response.trim().is_empty() {
                        log_debug_response(
                            ctx.debug_logging_enabled,
                            &final_response,
                            if ctx.reflection_enabled {
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
                            ctx.session_messages,
                            &final_response,
                        );
                        ctx.session_messages
                            .push(Message::assistant(&final_response));
                    }
                    emit_trace(
                        ctx.token_tx,
                        ProgressStatus::Finished,
                        "follow-up ready",
                        false,
                    );
                }
                return Ok(());
            }
            Ok(SessionCommand::ClearSession) => {
                warn!(action_id, "clear requested while pending action active");
                emit_trace(
                    ctx.token_tx,
                    ProgressStatus::Failed,
                    "approval interrupted",
                    false,
                );
                return Err(ParamsError::Config(
                    "Cannot clear session while an action is awaiting approval".to_string(),
                ));
            }
            Ok(_) => {}
            Err(_) => {
                emit_trace(
                    ctx.token_tx,
                    ProgressStatus::Failed,
                    "approval channel closed",
                    false,
                );
                return Err(ParamsError::Inference(
                    "Approval channel closed while waiting for user decision".to_string(),
                ));
            }
        }
    }
}

fn dispatch_inspection_hook(hooks: &Hooks, inspection: &InspectionReport) {
    hooks.dispatch(HookEvent::InspectionEvaluated {
        operation: inspection.operation.clone(),
        decision: inspection.decision.to_string(),
        risk: inspection.risk.to_string(),
        target_count: inspection.targets.len() + inspection.network_targets.len(),
        blocked_reason_count: inspection.reasons.len(),
    });
}
