use super::AppState;
use crate::commands::{builtin_command_specs, CommandRegistry};
use crate::events::{FactProvenance, MemorySnapshot};

pub(super) fn custom_help_text() -> String {
    let mut lines = vec!["built-in slash commands:".to_string()];
    for spec in builtin_command_specs() {
        let mut line = format!("  {:<18} — {}", spec.usage, spec.description);
        if !spec.aliases.is_empty() {
            line.push_str(&format!(" (aliases: {})", spec.aliases.join(", ")));
        }
        lines.push(line);
    }
    lines.push("".to_string());
    lines.push("input: Enter sends • Shift+Enter or Ctrl+J insert newlines".to_string());
    lines.push("transcript: Ctrl+O toggle • [ / ] move focus when input is empty".to_string());
    lines.push("custom commands: /commands list • /commands reload".to_string());
    lines.join("\n")
}

pub(super) fn format_custom_commands_list(registry: &CommandRegistry) -> String {
    let mut lines = vec!["built-ins:".to_string()];
    for spec in builtin_command_specs() {
        lines.push(format!("  {:<12} — {}", spec.canonical, spec.description));
    }
    lines.push(String::new());
    lines.push("custom commands:".to_string());
    if registry.list().is_empty() {
        lines.push("  (none loaded)".to_string());
    } else {
        for command in registry.list() {
            let usage = command
                .usage
                .as_ref()
                .map(|value| format!(" — {value}"))
                .unwrap_or_default();
            lines.push(format!(
                "  {:<12} [{}] — {}{}",
                command.name, command.origin, command.description, usage
            ));
        }
    }
    lines.join("\n")
}

pub(super) fn format_display_status(state: &AppState) -> String {
    format!(
        "display:\n  tokens: {}\n  time: {}",
        if state.show_top_bar_tokens {
            "on"
        } else {
            "off"
        },
        if state.show_top_bar_time { "on" } else { "off" }
    )
}

pub(super) fn format_memory_status(snapshot: &MemorySnapshot) -> String {
    let accepted = snapshot
        .last_update
        .as_ref()
        .map(|update| update.accepted_facts.len())
        .unwrap_or(0);
    let skipped = snapshot
        .last_update
        .as_ref()
        .map(|update| {
            update
                .skipped_reasons
                .iter()
                .map(|reason| reason.count)
                .sum::<usize>()
        })
        .unwrap_or(0);
    let mut lines = vec![
        format!("loaded facts: {}", snapshot.loaded_facts.len()),
        format!("last summaries: {}", snapshot.last_summary_paths.len()),
        format!("last fact matches: {}", snapshot.last_selected_facts.len()),
        format!(
            "last session excerpts: {}",
            snapshot.last_selected_session_excerpts.len()
        ),
        format!("last update: +{accepted} / -{skipped}"),
    ];
    if let Some(query) = &snapshot.last_retrieval_query {
        lines.push(format!("last retrieval query: {query}"));
    } else {
        lines.push("last retrieval query: (none)".to_string());
    }
    if snapshot.last_summary_paths.is_empty() {
        lines.push("recent summary paths: (none)".to_string());
    } else {
        lines.push("recent summary paths:".to_string());
        for path in &snapshot.last_summary_paths {
            lines.push(format!("  - {path}"));
        }
    }
    lines.join("\n")
}

pub(super) fn format_memory_facts(snapshot: &MemorySnapshot) -> String {
    if snapshot.loaded_facts.is_empty() {
        return "loaded memory facts:\n  (none)".to_string();
    }

    let mut lines = vec!["loaded memory facts:".to_string()];
    for fact in &snapshot.loaded_facts {
        let label = match fact.provenance {
            FactProvenance::Legacy => "legacy",
            FactProvenance::Verified => "verified",
        };
        lines.push(format!("  [{label}] {}", fact.content));
    }
    lines.join("\n")
}

pub(super) fn format_memory_last(snapshot: &MemorySnapshot) -> String {
    let mut lines = vec!["last memory update:".to_string()];
    match &snapshot.last_update {
        Some(update) => {
            lines.push(format!("  accepted: {}", update.accepted_facts.len()));
            lines.push(format!("  duplicates: {}", update.duplicate_count));
            if update.accepted_facts.is_empty() {
                lines.push("  accepted facts: (none)".to_string());
            } else {
                lines.push("  accepted facts:".to_string());
                for fact in &update.accepted_facts {
                    lines.push(format!("    - {}", fact.content));
                }
            }
            if update.skipped_reasons.is_empty() {
                lines.push("  skipped: (none)".to_string());
            } else {
                lines.push("  skipped:".to_string());
                for reason in &update.skipped_reasons {
                    lines.push(format!("    - {}: {}", reason.reason, reason.count));
                }
            }
        }
        None => lines.push("  (no verified update yet)".to_string()),
    }

    lines.push(String::new());
    lines.push("last retrieval:".to_string());
    match &snapshot.last_retrieval_query {
        Some(query) => {
            lines.push(format!("  query: {query}"));
            lines.push(format!(
                "  summaries: {}",
                snapshot.last_summary_paths.len()
            ));
            lines.push(format!(
                "  fact matches: {}",
                snapshot.last_selected_facts.len()
            ));
            lines.push(format!(
                "  session excerpts: {}",
                snapshot.last_selected_session_excerpts.len()
            ));
        }
        None => lines.push("  (no retrieval yet)".to_string()),
    }

    lines.push(String::new());
    lines.push("last consolidation:".to_string());
    match &snapshot.last_consolidation {
        Some(consolidation) => {
            lines.push(format!("  ttl pruned: {}", consolidation.ttl_pruned));
            lines.push(format!("  dedup removed: {}", consolidation.dedup_removed));
            lines.push(format!("  cap removed: {}", consolidation.cap_removed));
        }
        None => lines.push("  (no consolidation recorded yet)".to_string()),
    }

    lines.join("\n")
}

pub(super) fn format_transcript_status(total: usize, collapsed: usize) -> String {
    let mode = if total == 0 {
        "no collapsible blocks".to_string()
    } else if collapsed == total {
        "fully collapsed".to_string()
    } else if collapsed == 0 {
        "fully expanded".to_string()
    } else if collapsed * 2 >= total {
        "mostly collapsed".to_string()
    } else {
        "mostly expanded".to_string()
    };

    format!("transcript blocks: {total} collapsible, {collapsed} collapsed ({mode})")
}
