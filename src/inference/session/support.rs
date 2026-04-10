use std::sync::mpsc::Sender;

use tracing::warn;

use crate::error::Result;
use crate::events::{InferenceEvent, SessionInfo};
use crate::session::{list_label, short_id, SessionExportFormat, SessionStore, SessionSummary};
use crate::tools::ToolRegistry;

use super::super::budget::{
    emit_budget_update, emit_cache_update, SessionBudget, SessionCacheStats,
};
use super::super::{build_system_prompt, Message};

pub(super) fn session_info(summary: &SessionSummary) -> SessionInfo {
    SessionInfo {
        id: summary.id.clone(),
        name: summary.name.clone(),
        message_count: summary.message_count,
    }
}

pub(super) fn save_session(
    store: Option<&SessionStore>,
    active_session: &mut Option<SessionSummary>,
    messages: &[Message],
    backend_name: &str,
    token_tx: &Sender<InferenceEvent>,
) {
    if let (Some(s), Some(current)) = (store, active_session.as_ref()) {
        match s.save_messages(&current.id, messages, backend_name) {
            Ok(updated) => {
                *active_session = Some(updated.clone());
                let _ = token_tx.send(InferenceEvent::SessionStatus(session_info(&updated)));
            }
            Err(e) => warn!(error = %e, "session save failed"),
        }
    }
}

pub(super) fn reset_session_runtime(
    session_messages: &mut Vec<Message>,
    tools: &ToolRegistry,
    eco_enabled: bool,
    budget: &mut SessionBudget,
    cache_stats: &mut SessionCacheStats,
    backend_name: &str,
    token_tx: &Sender<InferenceEvent>,
) {
    session_messages.clear();
    session_messages.push(Message::system(&build_system_prompt(
        tools,
        &[],
        &[],
        &[],
        eco_enabled,
    )));
    *budget = SessionBudget {
        has_cost_estimate: backend_name == "llama_cpp" || backend_name == "ollama",
        ..SessionBudget::default()
    };
    *cache_stats = SessionCacheStats::default();
    emit_budget_update(budget, token_tx);
    emit_cache_update(cache_stats, false, token_tx);
}

pub(super) fn format_sessions_list(
    sessions: &[SessionSummary],
    active_session_id: Option<&str>,
) -> String {
    let mut lines = vec![format!("sessions · {}", sessions.len())];
    if sessions.is_empty() {
        lines.push("  (none saved for this project)".to_string());
        return lines.join("\n");
    }

    for session in sessions {
        let marker = if Some(session.id.as_str()) == active_session_id {
            "●"
        } else {
            "·"
        };
        let message_label = if session.message_count == 0 {
            "empty".to_string()
        } else {
            format!(
                "{} msg{}",
                session.message_count,
                if session.message_count == 1 { "" } else { "s" }
            )
        };
        lines.push(format!(
            "  {marker} {} · {} · {} · #{}",
            list_label(session),
            message_label,
            crate::session::describe_session_age(session.updated_at),
            short_id(&session.id)
        ));
    }

    lines.push("  /sessions resume|delete|export <name-or-id>".to_string());

    lines.join("\n")
}

pub(super) fn parse_export_format(raw: Option<String>) -> Result<SessionExportFormat> {
    match raw {
        None => Ok(SessionExportFormat::Markdown),
        Some(value) => SessionExportFormat::from_str(value.trim()).ok_or_else(|| {
            crate::error::ParamsError::Config(
                "Export format must be `markdown` or `json`".to_string(),
            )
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::format_sessions_list;
    use crate::session::SessionSummary;

    fn summary(
        id: &str,
        name: Option<&str>,
        updated_at: u64,
        message_count: usize,
    ) -> SessionSummary {
        SessionSummary {
            id: id.to_string(),
            project_root: "/tmp/project".to_string(),
            name: name.map(str::to_string),
            backend: "llama.cpp".to_string(),
            created_at: updated_at,
            updated_at,
            last_opened_at: updated_at,
            message_count,
        }
    }

    #[test]
    fn sessions_list_marks_current_and_shows_selector_hint() {
        let output = format_sessions_list(
            &[
                summary("7353e9a31234", Some("b"), 0, 2),
                summary("cbc8da921234", None, 0, 0),
            ],
            Some("7353e9a31234"),
        );

        assert!(output.contains("sessions · 2"));
        assert!(output.contains("● b · 2 msgs"));
        assert!(output.contains("· unnamed · empty"));
        assert!(output.contains("#7353e9a3"));
        assert!(!output.contains("id 7353e9a3"));
        assert!(output.contains("/sessions resume|delete|export <name-or-id>"));
    }
}
