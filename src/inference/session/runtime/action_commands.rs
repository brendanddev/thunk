use tracing::info;

use crate::events::InferenceEvent;
use crate::tools::{BashTool, Tool};

use super::super::super::approval::handle_pending_action;
use super::super::support::save_session;
use super::state::{approval_context, RuntimeContext, RuntimeState};

pub(super) fn handle_request_shell_command(
    ctx: &RuntimeContext<'_>,
    state: &mut RuntimeState,
    command: String,
) {
    info!("shell command approval requested");
    match BashTool.run(&command) {
        Ok(crate::tools::ToolRunResult::RequiresApproval(pending)) => {
            let action_id = state.next_action_id;
            if let Err(e) = handle_pending_action(
                approval_context(ctx, state, None),
                action_id,
                pending,
                false,
            ) {
                let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
            } else {
                save_session(
                    ctx.session_store,
                    &mut state.active_session,
                    &state.session_messages,
                    &ctx.backend.name(),
                    ctx.token_tx,
                );
            }
        }
        Ok(crate::tools::ToolRunResult::Immediate(output)) => {
            let _ = ctx.token_tx.send(InferenceEvent::ContextMessage(output));
        }
        Err(e) => {
            let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
        }
    }
    let _ = ctx.token_tx.send(InferenceEvent::Done);
    state.next_action_id = state.next_action_id.saturating_add(1);
}

pub(super) fn handle_request_file_write(
    ctx: &RuntimeContext<'_>,
    state: &mut RuntimeState,
    path: String,
    content: String,
) {
    info!(path = path.as_str(), "file write approval requested");
    match crate::tools::build_pending_write_request(&path, &content) {
        Ok(pending) => {
            let action_id = state.next_action_id;
            if let Err(e) = handle_pending_action(
                approval_context(ctx, state, None),
                action_id,
                pending,
                false,
            ) {
                let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
            } else {
                save_session(
                    ctx.session_store,
                    &mut state.active_session,
                    &state.session_messages,
                    &ctx.backend.name(),
                    ctx.token_tx,
                );
            }
        }
        Err(e) => {
            let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
        }
    }
    let _ = ctx.token_tx.send(InferenceEvent::Done);
    state.next_action_id = state.next_action_id.saturating_add(1);
}

pub(super) fn handle_request_file_edit(
    ctx: &RuntimeContext<'_>,
    state: &mut RuntimeState,
    path: String,
    edits: String,
) {
    info!(path = path.as_str(), "file edit approval requested");
    match crate::tools::build_pending_edit_request(&path, &edits) {
        Ok(pending) => {
            let action_id = state.next_action_id;
            if let Err(e) = handle_pending_action(
                approval_context(ctx, state, None),
                action_id,
                pending,
                false,
            ) {
                let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
            } else {
                save_session(
                    ctx.session_store,
                    &mut state.active_session,
                    &state.session_messages,
                    &ctx.backend.name(),
                    ctx.token_tx,
                );
            }
        }
        Err(e) => {
            let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
        }
    }
    let _ = ctx.token_tx.send(InferenceEvent::Done);
    state.next_action_id = state.next_action_id.saturating_add(1);
}
