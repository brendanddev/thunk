use tracing::warn;

use crate::events::InferenceEvent;
use crate::hooks::HookEvent;
use crate::session::display_name;

use super::super::support::{
    format_sessions_list, parse_export_format, save_session, session_info,
};
use super::state::{RuntimeContext, RuntimeState};

pub(super) fn handle_clear_session(ctx: &RuntimeContext<'_>, state: &mut RuntimeState) {
    if let (Some(store), Some(current)) = (ctx.session_store, state.active_session.as_ref()) {
        if let Err(e) = store.delete_session(&current.id) {
            warn!(error = %e, "session clear failed");
        } else {
            ctx.hooks.dispatch(HookEvent::SessionCleared {
                session_id: current.id.clone(),
            });
        }
    }

    state.reset_session_scope(ctx);
    if let Some(ref store) = ctx.session_store {
        match store.create_session(None, &ctx.backend.name()) {
            Ok(summary) => {
                ctx.hooks.dispatch(HookEvent::SessionCreated {
                    session_id: summary.id.clone(),
                    named: false,
                });
                state.active_session = Some(summary.clone());
                let _ = ctx.token_tx.send(InferenceEvent::SessionLoaded {
                    session: session_info(&summary),
                    display_messages: Vec::new(),
                    saved_at: None,
                });
                let _ = ctx.token_tx.send(InferenceEvent::SystemMessage(
                    "conversation cleared".to_string(),
                ));
            }
            Err(e) => {
                warn!(error = %e, "replacement session create failed");
                state.active_session = None;
                let _ = ctx.token_tx.send(InferenceEvent::SystemMessage(
                    "conversation cleared".to_string(),
                ));
            }
        }
    }
}

pub(super) fn handle_list_sessions(ctx: &RuntimeContext<'_>, state: &RuntimeState) {
    match ctx.session_store.map(|store| store.list_sessions()) {
        Some(Ok(sessions)) => {
            let active_id = state
                .active_session
                .as_ref()
                .map(|session| session.id.as_str());
            let _ = ctx
                .token_tx
                .send(InferenceEvent::SystemMessage(format_sessions_list(
                    &sessions, active_id,
                )));
        }
        Some(Err(e)) => {
            let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
        }
        None => {
            let _ = ctx.token_tx.send(InferenceEvent::Error(
                "Session store is unavailable".to_string(),
            ));
        }
    }
}

pub(super) fn handle_new_session(
    ctx: &RuntimeContext<'_>,
    state: &mut RuntimeState,
    name: Option<String>,
) {
    save_session(
        ctx.session_store,
        &mut state.active_session,
        &state.session_messages,
        &ctx.backend.name(),
        ctx.token_tx,
    );
    state.reset_session_scope(ctx);
    match ctx
        .session_store
        .map(|store| store.create_session(name.as_deref(), &ctx.backend.name()))
    {
        Some(Ok(summary)) => {
            let session_label = display_name(&summary);
            ctx.hooks.dispatch(HookEvent::SessionCreated {
                session_id: summary.id.clone(),
                named: summary.name.is_some(),
            });
            state.active_session = Some(summary.clone());
            let _ = ctx.token_tx.send(InferenceEvent::SessionLoaded {
                session: session_info(&summary),
                display_messages: Vec::new(),
                saved_at: None,
            });
            let _ = ctx.token_tx.send(InferenceEvent::SystemMessage(format!(
                "started new session: {session_label}"
            )));
        }
        Some(Err(e)) => {
            let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
        }
        None => {
            let _ = ctx.token_tx.send(InferenceEvent::Error(
                "Session store is unavailable".to_string(),
            ));
        }
    }
}

pub(super) fn handle_rename_session(
    ctx: &RuntimeContext<'_>,
    state: &mut RuntimeState,
    name: String,
) {
    match (ctx.session_store, state.active_session.as_ref()) {
        (Some(store), Some(current)) => match store.rename_session(&current.id, &name) {
            Ok(updated) => {
                ctx.hooks.dispatch(HookEvent::SessionRenamed {
                    session_id: updated.id.clone(),
                    named: updated.name.is_some(),
                });
                state.active_session = Some(updated.clone());
                let _ = ctx
                    .token_tx
                    .send(InferenceEvent::SessionStatus(session_info(&updated)));
                let _ = ctx.token_tx.send(InferenceEvent::SystemMessage(format!(
                    "renamed session to {}",
                    display_name(&updated)
                )));
            }
            Err(e) => {
                let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
            }
        },
        _ => {
            let _ = ctx.token_tx.send(InferenceEvent::Error(
                "No active session is available to rename".to_string(),
            ));
        }
    }
}

pub(super) fn handle_resume_session(
    ctx: &RuntimeContext<'_>,
    state: &mut RuntimeState,
    selector: String,
) {
    save_session(
        ctx.session_store,
        &mut state.active_session,
        &state.session_messages,
        &ctx.backend.name(),
        ctx.token_tx,
    );
    match ctx.session_store.map(|store| store.load_session(&selector)) {
        Some(Ok(saved)) => {
            state.reset_session_scope(ctx);
            let display_messages = saved
                .messages
                .iter()
                .map(|m| (m.role.clone(), m.content.clone()))
                .collect::<Vec<_>>();
            state.session_messages.extend(saved.messages);
            ctx.hooks.dispatch(HookEvent::SessionResumed {
                session_id: saved.summary.id.clone(),
                named: saved.summary.name.is_some(),
                message_count: saved.summary.message_count,
            });
            state.active_session = Some(saved.summary.clone());
            let _ = ctx.token_tx.send(InferenceEvent::SessionLoaded {
                session: session_info(&saved.summary),
                display_messages,
                saved_at: Some(saved.saved_at),
            });
        }
        Some(Err(e)) => {
            let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
        }
        None => {
            let _ = ctx.token_tx.send(InferenceEvent::Error(
                "Session store is unavailable".to_string(),
            ));
        }
    }
}

pub(super) fn handle_delete_session(
    ctx: &RuntimeContext<'_>,
    state: &mut RuntimeState,
    selector: String,
) {
    match (ctx.session_store, state.active_session.as_ref()) {
        (Some(store), Some(current_active)) => match store.resolve_session(&selector) {
            Ok(summary) => {
                let was_active = summary.id == current_active.id;
                let deleted_label = display_name(&summary);
                if let Err(e) = store.delete_session(&summary.id) {
                    let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
                    return;
                }
                ctx.hooks.dispatch(HookEvent::SessionDeleted {
                    session_id: summary.id.clone(),
                    was_active,
                });

                if was_active {
                    state.reset_session_scope(ctx);
                    match store.create_session(None, &ctx.backend.name()) {
                        Ok(replacement) => {
                            ctx.hooks.dispatch(HookEvent::SessionCreated {
                                session_id: replacement.id.clone(),
                                named: false,
                            });
                            state.active_session = Some(replacement.clone());
                            let _ = ctx.token_tx.send(InferenceEvent::SessionLoaded {
                                session: session_info(&replacement),
                                display_messages: Vec::new(),
                                saved_at: None,
                            });
                            let _ = ctx.token_tx.send(InferenceEvent::SystemMessage(format!(
                                "deleted session: {deleted_label}; started fresh unnamed session"
                            )));
                        }
                        Err(e) => {
                            state.active_session = None;
                            let _ = ctx.token_tx.send(InferenceEvent::Error(format!(
                                "deleted session {deleted_label}, but failed to create replacement session: {e}"
                            )));
                        }
                    }
                } else {
                    let _ = ctx.token_tx.send(InferenceEvent::SystemMessage(format!(
                        "deleted session: {deleted_label}"
                    )));
                }
            }
            Err(e) => {
                let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
            }
        },
        (Some(store), None) => match store.resolve_session(&selector) {
            Ok(summary) => {
                let deleted_label = display_name(&summary);
                match store.delete_session(&summary.id) {
                    Ok(()) => {
                        ctx.hooks.dispatch(HookEvent::SessionDeleted {
                            session_id: summary.id.clone(),
                            was_active: false,
                        });
                        let _ = ctx.token_tx.send(InferenceEvent::SystemMessage(format!(
                            "deleted session: {deleted_label}"
                        )));
                    }
                    Err(e) => {
                        let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                }
            }
            Err(e) => {
                let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
            }
        },
        (None, _) => {
            let _ = ctx.token_tx.send(InferenceEvent::Error(
                "Session store is unavailable".to_string(),
            ));
        }
    }
}

pub(super) fn handle_export_session(
    ctx: &RuntimeContext<'_>,
    selector: String,
    format: Option<String>,
) {
    match ctx.session_store.map(|store| {
        parse_export_format(format).and_then(|fmt| store.export_session(&selector, fmt))
    }) {
        Some(Ok((summary, path))) => {
            ctx.hooks.dispatch(HookEvent::SessionExported {
                session_id: summary.id.clone(),
                format: path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
            });
            let _ = ctx.token_tx.send(InferenceEvent::SystemMessage(format!(
                "exported session {} to {}",
                display_name(&summary),
                path.display()
            )));
        }
        Some(Err(e)) => {
            let _ = ctx.token_tx.send(InferenceEvent::Error(e.to_string()));
        }
        None => {
            let _ = ctx.token_tx.send(InferenceEvent::Error(
                "Session store is unavailable".to_string(),
            ));
        }
    }
}
