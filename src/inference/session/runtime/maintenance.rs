use tracing::info;

use crate::debug_log;
use crate::events::{InferenceEvent, ProgressStatus};

use super::super::super::runtime::{effective_reflection, emit_trace};
use super::super::memory::{
    clear_memory_retrieval, collect_retrieval_bundle, emit_memory_state, format_memory_recall,
    refresh_loaded_facts,
};
use super::state::{RuntimeContext, RuntimeState};

pub(super) fn handle_set_reflection(
    ctx: &RuntimeContext<'_>,
    state: &mut RuntimeState,
    enabled: bool,
) {
    state.reflection_requested = enabled;
    state.reflection_enabled = effective_reflection(state.reflection_requested, state.eco_enabled);
    info!(
        requested = enabled,
        effective = state.reflection_enabled,
        eco_enabled = state.eco_enabled,
        "reflection state updated"
    );
    let _ = ctx
        .token_tx
        .send(InferenceEvent::ReflectionEnabled(state.reflection_enabled));
}

pub(super) fn handle_set_eco(ctx: &RuntimeContext<'_>, state: &mut RuntimeState, enabled: bool) {
    state.eco_enabled = enabled;
    state.reflection_enabled = effective_reflection(state.reflection_requested, state.eco_enabled);
    info!(
        enabled = state.eco_enabled,
        reflection_enabled = state.reflection_enabled,
        "eco state updated"
    );
    state.refresh_base_system_prompt(ctx);
    let _ = ctx
        .token_tx
        .send(InferenceEvent::EcoEnabled(state.eco_enabled));
    let _ = ctx
        .token_tx
        .send(InferenceEvent::ReflectionEnabled(state.reflection_enabled));
}

pub(super) fn handle_set_debug_logging(
    ctx: &RuntimeContext<'_>,
    state: &mut RuntimeState,
    enabled: bool,
) {
    state.debug_logging_enabled = enabled;
    info!(enabled, "debug logging state updated");
    let _ = ctx
        .token_tx
        .send(InferenceEvent::DebugLoggingEnabled(enabled));
}

pub(super) fn handle_clear_debug_log(ctx: &RuntimeContext<'_>) {
    match debug_log::clear() {
        Ok(()) => {
            info!("debug content log cleared");
        }
        Err(e) => {
            let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
        }
    }
}

pub(super) fn handle_clear_cache(ctx: &RuntimeContext<'_>) {
    match ctx.exact_cache.map(|cache| cache.clear()) {
        Some(Ok(deleted)) => {
            info!(deleted, "exact cache cleared");
        }
        Some(Err(e)) => {
            let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
        }
        None => {
            let _ = ctx
                .token_tx
                .send(InferenceEvent::Error("Cache is unavailable".to_string()));
        }
    }
}

pub(super) fn handle_recall_memory(ctx: &RuntimeContext<'_>, state: &RuntimeState, query: String) {
    let bundle = collect_retrieval_bundle(
        &query,
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
    let _ = ctx
        .token_tx
        .send(InferenceEvent::SystemMessage(format_memory_recall(
            &query, &bundle,
        )));
}

pub(super) fn handle_prune_memory(ctx: &RuntimeContext<'_>, state: &mut RuntimeState) {
    match ctx
        .fact_store
        .map(|store| store.prune_irrelevant_facts(ctx.project_name))
    {
        Some(Ok(removed)) => {
            refresh_loaded_facts(&mut state.memory_state, ctx.fact_store, ctx.project_name);
            clear_memory_retrieval(&mut state.memory_state);
            emit_memory_state(ctx.token_tx, &state.memory_state);
            emit_trace(
                ctx.token_tx,
                ProgressStatus::Finished,
                &format!(
                    "memory: pruned {} fact{}",
                    removed,
                    if removed == 1 { "" } else { "s" }
                ),
                true,
            );
            let _ = ctx.token_tx.send(InferenceEvent::SystemMessage(format!(
                "memory prune removed {removed} irrelevant fact{}",
                if removed == 1 { "" } else { "s" }
            )));
        }
        Some(Err(e)) => {
            let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
        }
        None => {
            let _ = ctx.token_tx.send(InferenceEvent::Error(
                "Memory store is unavailable".to_string(),
            ));
        }
    }
}

pub(super) fn handle_orphaned_approval(ctx: &RuntimeContext<'_>) {
    let _ = ctx.token_tx.send(InferenceEvent::Error(
        "No action is currently awaiting approval".to_string(),
    ));
}
