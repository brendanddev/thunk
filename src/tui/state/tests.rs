use super::*;
use crate::events::{PendingActionKind, ProgressStatus, ProgressTrace};
use crate::safety::{InspectionDecision, InspectionReport, RiskLevel};
use std::time::Duration;

#[test]
fn work_timer_accumulates_and_finishes() {
    let mut state = AppState::new();
    state.start_generation("generating...", false);
    std::thread::sleep(Duration::from_millis(5));
    state.set_pending_action(crate::events::PendingAction {
        id: 1,
        kind: PendingActionKind::ShellCommand,
        title: "Approve".to_string(),
        preview: "echo hi".to_string(),
        inspection: InspectionReport {
            operation: "bash".to_string(),
            decision: InspectionDecision::NeedsApproval,
            risk: RiskLevel::Low,
            summary: "test".to_string(),
            reasons: Vec::new(),
            targets: Vec::new(),
            segments: vec!["echo hi".to_string()],
            network_targets: Vec::new(),
        },
    });
    let paused = state.current_turn_duration().unwrap();
    assert!(paused >= Duration::from_millis(1));

    state.start_generation("generating...", false);
    std::thread::sleep(Duration::from_millis(5));
    state.finish_response();

    assert!(state.current_turn_duration().is_none());
    assert!(state.last_work_duration().unwrap() >= paused);
}

#[test]
fn grouped_traces_collapse_at_turn_end() {
    let mut state = AppState::new();
    state.start_generation("generating...", false);
    state.apply_trace(ProgressTrace {
        status: ProgressStatus::Finished,
        label: "drafting answer...".to_string(),
        persist: false,
    });
    state.apply_trace(ProgressTrace {
        status: ProgressStatus::Finished,
        label: "answer ready".to_string(),
        persist: false,
    });

    assert!(state.recent_traces.is_empty());
    state.finish_response();

    assert_eq!(state.recent_traces.len(), 1);
    assert_eq!(
        state.recent_traces.front().unwrap().label,
        "drafting answer... -> answer ready"
    );
    assert!(state.recent_traces.front().unwrap().success);
}

#[test]
fn standalone_traces_still_show_individually() {
    let mut state = AppState::new();
    state.apply_trace(ProgressTrace {
        status: ProgressStatus::Finished,
        label: "profile: .params.toml".to_string(),
        persist: true,
    });

    assert_eq!(state.recent_traces.len(), 1);
    assert_eq!(
        state.recent_traces.front().unwrap().label,
        "profile: .params.toml"
    );
}

#[test]
fn non_persisted_traces_do_not_inject_chat_messages() {
    let mut state = AppState::new();
    state.apply_trace(ProgressTrace {
        status: ProgressStatus::Finished,
        label: "memory: stored 1 fact".to_string(),
        persist: false,
    });

    assert!(state.messages.is_empty());
    assert_eq!(state.recent_traces.len(), 1);
    assert_eq!(
        state.recent_traces.front().unwrap().label,
        "memory: stored 1 fact"
    );
}

#[test]
fn command_autocomplete_cycles_matches() {
    let mut state = AppState::new();
    state.input = "/d".to_string();
    state.cursor = 2;

    assert!(state.autocomplete_command(&["/def", "/diag", "/debug-log"], false));
    assert_eq!(state.input, "/def");

    assert!(state.autocomplete_command(&["/def", "/diag", "/debug-log"], false));
    assert_eq!(state.input, "/diag");
}

#[test]
fn command_autocomplete_adds_space_for_unique_match() {
    let mut state = AppState::new();
    state.input = "/reject".to_string();
    state.cursor = state.input.len();

    assert!(state.autocomplete_command(&["/reject"], false));
    assert_eq!(state.input, "/reject ");
}

#[test]
fn pending_action_sets_waiting_status_without_chat_message() {
    let mut state = AppState::new();
    assert_eq!(state.messages.len(), 0);

    state.set_pending_action(crate::events::PendingAction {
        id: 1,
        kind: PendingActionKind::ShellCommand,
        title: "Approve shell command".to_string(),
        preview: "cargo check".to_string(),
        inspection: InspectionReport {
            operation: "bash".to_string(),
            decision: InspectionDecision::NeedsApproval,
            risk: RiskLevel::Low,
            summary: "Shell command requires approval before execution".to_string(),
            reasons: Vec::new(),
            targets: Vec::new(),
            segments: vec!["cargo check".to_string()],
            network_targets: Vec::new(),
        },
    });

    assert!(state.has_pending_action());
    assert_eq!(state.status, "awaiting approval");
    assert!(state.messages.is_empty());
}

#[test]
fn clearing_pending_action_removes_card_state() {
    let mut state = AppState::new();
    state.set_pending_action(crate::events::PendingAction {
        id: 1,
        kind: PendingActionKind::ShellCommand,
        title: "Approve shell command".to_string(),
        preview: "cargo check".to_string(),
        inspection: InspectionReport {
            operation: "bash".to_string(),
            decision: InspectionDecision::NeedsApproval,
            risk: RiskLevel::Low,
            summary: "Shell command requires approval before execution".to_string(),
            reasons: Vec::new(),
            targets: Vec::new(),
            segments: vec!["cargo check".to_string()],
            network_targets: Vec::new(),
        },
    });

    state.clear_pending_action();

    assert!(!state.has_pending_action());
    assert_eq!(state.status, "ready");
}

#[test]
fn insert_newline_preserves_multiline_input() {
    let mut state = AppState::new();
    state.input = "hello".to_string();
    state.cursor = state.input.len();

    state.insert_newline();
    state.insert_str("world");

    assert_eq!(state.input, "hello\nworld");
}

#[test]
fn submit_input_returns_multiline_content_unchanged() {
    let mut state = AppState::new();
    state.input = "one\ntwo\nthree".to_string();
    state.cursor = state.input.len();

    let submitted = state.submit_input();

    assert_eq!(submitted, "one\ntwo\nthree");
    assert!(state.input.is_empty());
}

#[test]
fn normalized_paste_preserves_newlines() {
    assert_eq!(
        AppState::normalized_paste("one\r\ntwo\rthree"),
        "one\ntwo\nthree"
    );
}

#[test]
fn home_and_end_operate_on_current_line() {
    let mut state = AppState::new();
    state.input = "first\nsecond\nthird".to_string();
    state.cursor = "first\nsec".len();

    state.cursor_end();
    assert_eq!(state.cursor, "first\nsecond".len());

    state.cursor_home();
    assert_eq!(state.cursor, "first\n".len());
}

#[test]
fn input_display_lines_wrap_and_keep_cursor_visible() {
    let mut state = AppState::new();
    state.input = "123456789\nabc".to_string();
    state.cursor = state.input.len();

    let (lines, cursor_row, cursor_col) = state.input_display_lines(4, 3);

    assert!(!lines.is_empty());
    assert!(lines.iter().any(|line| line == "abc"));
    assert_eq!(cursor_row, 2);
    assert_eq!(cursor_col, 3);
}

#[test]
fn history_recall_restores_previous_submission_and_draft() {
    let mut state = AppState::new();
    state.input = "first".to_string();
    state.cursor = state.input.len();
    assert_eq!(state.submit_input(), "first");

    state.input = "second".to_string();
    state.cursor = state.input.len();
    assert_eq!(state.submit_input(), "second");

    state.input = "draft".to_string();
    state.cursor = state.input.len();

    assert!(state.recall_previous_input());
    assert_eq!(state.input, "second");
    assert!(state.recall_previous_input());
    assert_eq!(state.input, "first");
    assert!(state.recall_next_input());
    assert_eq!(state.input, "second");
    assert!(state.recall_next_input());
    assert_eq!(state.input, "draft");
}

#[test]
fn scroll_and_input_mutations_mark_dirty_sections() {
    let mut state = AppState::new();
    state.clear_dirty_sections();

    state.insert_char('a');
    assert!(state.dirty_sections().contains(DirtySections::INPUT));

    state.clear_dirty_sections();
    state.add_user_message("hello");
    assert!(state.dirty_sections().contains(DirtySections::CHAT));
    assert!(state.dirty_sections().contains(DirtySections::SIDEBAR));

    state.clear_dirty_sections();
    state.scroll_up(1);
    assert!(state.dirty_sections().contains(DirtySections::CHAT));
}

#[test]
fn injected_context_messages_default_to_collapsed() {
    let mut state = AppState::new();
    state.add_user_message(
        "Tool results:\n\n--- read_file(src/main.rs) ---\nFile: src/main.rs\n\n```",
    );

    let message = state.messages.last().expect("message");
    assert!(message.transcript.collapsible);
    assert!(message.transcript.collapsed);
    assert_eq!(
        message.transcript.summary.as_deref(),
        Some("tool results • 1 tool")
    );
}

#[test]
fn normal_messages_stay_non_collapsible() {
    let mut state = AppState::new();
    state.add_user_message("hello there");

    let message = state.messages.last().expect("message");
    assert!(!message.transcript.collapsible);
    assert!(!message.transcript.collapsed);
}

#[test]
fn restore_session_collapses_injected_context_rows() {
    let mut state = AppState::new();
    state.restore_session(
        SessionInfo {
            id: "session".to_string(),
            name: None,
            message_count: 1,
        },
        vec![(
            "user".to_string(),
            "Search results:\n\nSearch results for 'cache' (1 matches):".to_string(),
        )],
        None,
    );

    let message = state.messages.last().expect("restored message");
    assert_eq!(message.role, Role::System);
    assert!(message.transcript.collapsible);
    assert!(message.transcript.collapsed);
}

#[test]
fn transcript_focus_and_toggle_use_visible_collapsible_items() {
    let mut state = AppState::new();
    state.add_user_message("hello");
    state.add_user_message("Tool results:\n\n--- read_file(src/main.rs) ---");
    state.add_user_message("Directory listing:\n\nDirectory: src\n\nmain.rs");

    let first_id = state.messages[1].id;
    let second_id = state.messages[2].id;
    state.set_visible_collapsible_ids(vec![first_id, second_id]);

    assert!(state.is_focused_collapsible(second_id));
    assert!(state.focus_next_visible_collapsible());
    assert!(state.is_focused_collapsible(first_id));
    assert!(state.toggle_focused_collapsible());
    assert!(!state.messages[1].transcript.collapsed);
    assert!(state.focus_prev_visible_collapsible());
    assert!(state.is_focused_collapsible(second_id));
}

#[test]
fn transcript_global_collapse_expand_preserves_content() {
    let mut state = AppState::new();
    state.add_user_message("Tool results:\n\n--- read_file(src/main.rs) ---");
    let original = state.messages[0].content.clone();

    assert_eq!(state.expand_all_transcript_items(), 1);
    assert!(!state.messages[0].transcript.collapsed);
    assert_eq!(state.collapse_all_transcript_items(), 1);
    assert!(state.messages[0].transcript.collapsed);
    assert_eq!(state.messages[0].content, original);
}

#[test]
fn reverse_search_recalls_previous_submission_without_submitting() {
    let mut state = AppState::new();
    state.input = "first prompt".to_string();
    state.cursor = state.input.len();
    assert_eq!(state.submit_input(), "first prompt");

    state.input = "second prompt".to_string();
    state.cursor = state.input.len();
    assert_eq!(state.submit_input(), "second prompt");

    state.input = "draft".to_string();
    state.cursor = state.input.len();

    assert!(state.activate_reverse_search());
    state.reverse_search_push_char('f');
    state.reverse_search_push_char('i');

    assert!(state.is_reverse_search_active());
    assert_eq!(state.input, "first prompt");
    assert!(state.accept_reverse_search());
    assert_eq!(state.input, "first prompt");
    assert!(!state.is_reverse_search_active());
}

#[test]
fn reverse_search_cancel_restores_original_draft() {
    let mut state = AppState::new();
    state.input = "alpha".to_string();
    state.cursor = state.input.len();
    assert_eq!(state.submit_input(), "alpha");

    state.input = "draft text".to_string();
    state.cursor = state.input.len();

    assert!(state.activate_reverse_search());
    state.reverse_search_push_char('a');
    assert_eq!(state.input, "alpha");

    assert!(state.cancel_reverse_search());
    assert_eq!(state.input, "draft text");
    assert!(!state.is_reverse_search_active());
}

#[test]
fn reverse_search_cycle_walks_matching_history() {
    let mut state = AppState::new();
    for value in ["fix lint", "find bug", "finish docs"] {
        state.input = value.to_string();
        state.cursor = state.input.len();
        assert_eq!(state.submit_input(), value);
    }

    assert!(state.activate_reverse_search());
    state.reverse_search_push_char('f');
    state.reverse_search_push_char('i');
    assert_eq!(state.input, "finish docs");

    assert!(state.reverse_search_cycle());
    assert_eq!(state.input, "find bug");
}

#[test]
fn command_launcher_selects_command_without_submitting() {
    let mut state = AppState::new();
    state.input = "draft".to_string();
    state.cursor = state.input.len();

    assert!(state.activate_command_launcher(vec![
        crate::commands::CommandSuggestion {
            name: "/read".to_string(),
            usage: "/read <path>".to_string(),
            description: "load a file".to_string(),
            source: "builtin",
            group: "context",
            aliases: vec!["/r".to_string()],
        },
        crate::commands::CommandSuggestion {
            name: "/search".to_string(),
            usage: "/search <query>".to_string(),
            description: "search project files".to_string(),
            source: "builtin",
            group: "context",
            aliases: vec!["/s".to_string()],
        },
    ]));

    state.command_launcher_push_char('s');
    let accepted = state.accept_command_launcher().expect("command");
    assert_eq!(accepted, "/search ");
    assert_eq!(state.input, "/search ");
    assert!(!state.is_command_launcher_active());
}

#[test]
fn command_launcher_cancel_restores_draft() {
    let mut state = AppState::new();
    state.input = "draft".to_string();
    state.cursor = state.input.len();

    assert!(
        state.activate_command_launcher(vec![crate::commands::CommandSuggestion {
            name: "/read".to_string(),
            usage: "/read <path>".to_string(),
            description: "load a file".to_string(),
            source: "builtin",
            group: "context",
            aliases: vec!["/r".to_string()],
        }])
    );

    state.command_launcher_push_char('r');
    assert!(state.cancel_command_launcher());
    assert_eq!(state.input, "draft");
}

#[test]
fn command_launcher_matches_aliases_and_usage() {
    let mut state = AppState::new();
    assert!(state.activate_command_launcher(vec![
        crate::commands::CommandSuggestion {
            name: "/read".to_string(),
            usage: "/read <path>".to_string(),
            description: "load a file".to_string(),
            source: "builtin",
            group: "context",
            aliases: vec!["/r".to_string()],
        },
        crate::commands::CommandSuggestion {
            name: "/sessions".to_string(),
            usage: "/sessions <list|new|rename|resume|export>".to_string(),
            description: "manage saved sessions".to_string(),
            source: "builtin",
            group: "session",
            aliases: Vec::new(),
        },
    ]));

    state.command_launcher_push_char('r');
    let view = state.command_launcher_view(5).expect("view");
    assert_eq!(view.1[0].0.name, "/read");

    state.command_launcher_backspace();
    for ch in "resume".chars() {
        state.command_launcher_push_char(ch);
    }
    let view = state.command_launcher_view(5).expect("view");
    assert_eq!(view.1[0].0.name, "/sessions");
}
