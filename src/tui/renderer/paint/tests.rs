use super::*;
use crate::commands::CommandSuggestion;
use crate::events::PendingAction;
use crate::safety::{InspectionDecision, InspectionReport, RiskLevel};

#[test]
fn collapsed_context_hides_controls_until_focused() {
    let message = ChatMessage {
        id: 1,
        role: Role::User,
        content: "Tool results:\n\n--- read_file(src/main.rs) ---".to_string(),
        transcript: crate::tui::state::TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some("tool results • 1 tool".to_string()),
            preview_lines: vec!["read_file(src/main.rs)".to_string()],
        },
    };

    let unfocused = build_collapsed_context(&message, Theme::default(), 80, false);
    let focused = build_collapsed_context(&message, Theme::default(), 80, true);

    let unfocused_text = unfocused
        .iter()
        .flat_map(|line| line.spans.iter().map(|span| span.text.as_str()))
        .collect::<String>();
    let focused_text = focused
        .iter()
        .flat_map(|line| line.spans.iter().map(|span| span.text.as_str()))
        .collect::<String>();

    assert!(!unfocused_text.contains("Ctrl+O"));
    assert!(focused_text.contains("Ctrl+O"));
    // Navigation noise [ ] removed from hint line.
    assert!(!focused_text.contains("[ ]"));
}

#[test]
fn composer_uses_search_prompt_marker_in_reverse_search_mode() {
    let mut state = AppState::new();
    state.input = "first".to_string();
    state.cursor = state.input.len();
    assert_eq!(state.submit_input(), "first");
    state.input = "draft".to_string();
    state.cursor = state.input.len();
    assert!(state.activate_reverse_search());

    let (lines, _) = build_composer(&state, Theme::default(), 80, None, None);
    assert!(lines[0].spans.is_empty());
    assert_eq!(lines[1].spans[0].text, "? ");
}

#[test]
fn user_messages_render_with_conversation_gutter() {
    let message = ChatMessage {
        id: 1,
        role: Role::User,
        content: "hello".to_string(),
        transcript: crate::tui::state::TranscriptPresentation {
            collapsible: false,
            collapsed: false,
            summary: None,
            preview_lines: Vec::new(),
        },
    };

    let lines = build_standard_message(&message, Theme::default(), 80, false, 0);
    assert_eq!(lines[0].spans[0].text, CONVERSATION_GUTTER);
    assert_eq!(lines[0].spans[1].text, "you");
}

#[test]
fn system_messages_render_with_dim_gutter() {
    let message = ChatMessage {
        id: 1,
        role: Role::System,
        content: "memory: loaded 2 facts".to_string(),
        transcript: crate::tui::state::TranscriptPresentation {
            collapsible: false,
            collapsed: false,
            summary: None,
            preview_lines: Vec::new(),
        },
    };

    let lines = build_standard_message(&message, Theme::default(), 80, false, 0);
    assert_eq!(lines[0].spans[0].text, SYSTEM_GUTTER);
    assert!(lines[0].spans[1].text.contains("memory: loaded"));
}

#[test]
fn composer_uses_command_prompt_marker_in_launcher_mode() {
    let mut state = AppState::new();
    assert!(state.activate_command_launcher(vec![CommandSuggestion {
        name: "/read".to_string(),
        usage: "/read <path>".to_string(),
        description: "load a file".to_string(),
        source: "builtin",
        group: "context",
        aliases: vec!["/r".to_string()],
    }]));

    let (lines, _) = build_composer(&state, Theme::default(), 80, None, None);
    assert!(lines[0].spans.is_empty());
    assert_eq!(lines[1].spans[0].text, ": ");
}

#[test]
fn idle_empty_composer_renders_bare_prompt_without_placeholder() {
    let state = AppState::new();
    let (lines, _) = build_composer(&state, Theme::default(), 80, None, None);
    let text = lines
        .iter()
        .flat_map(|line| line.spans.iter().map(|span| span.text.as_str()))
        .collect::<String>();

    // No placeholder text, no tutorial hints — just the prompt marker.
    assert!(!text.contains("ask about the code"));
    assert!(!text.contains("Enter send"));
    assert!(text.contains("› "));
}

#[test]
fn composer_reserves_a_spacer_row_above_prompt() {
    let state = AppState::new();
    let (lines, cursor) = build_composer(&state, Theme::default(), 80, None, None);

    assert!(!lines.is_empty());
    assert!(lines[0].spans.is_empty());
    assert_eq!(lines[1].spans[0].text, "› ");
    assert_eq!(cursor, (2, 1));
}

#[test]
fn active_command_launcher_rows_omit_per_row_group_metadata() {
    let mut state = AppState::new();
    assert!(state.activate_command_launcher(vec![
        CommandSuggestion {
            name: "/read".to_string(),
            usage: "/read <path>".to_string(),
            description: "load a file".to_string(),
            source: "builtin",
            group: "context",
            aliases: vec!["/r".to_string()],
        },
        CommandSuggestion {
            name: "/sessions".to_string(),
            usage: "/sessions list".to_string(),
            description: "manage sessions".to_string(),
            source: "builtin",
            group: "session",
            aliases: Vec::new(),
        },
    ]));
    state.command_launcher_push_char('r');

    let (lines, _) = build_composer(&state, Theme::default(), 80, None, None);
    let text = lines
        .iter()
        .flat_map(|line| line.spans.iter().map(|span| span.text.as_str()))
        .collect::<String>();

    // Per-row group/source labels are not shown (not useful to users).
    assert!(!text.contains("[context]"));
    assert!(!text.contains("[context · builtin]"));
    // group/source metadata removed from detail block to reduce palette noise.
    assert!(!text.contains("group:"));
    assert!(!text.contains("source:"));
    // The entry name and description should still appear.
    assert!(text.contains("/read"));
    assert!(text.contains("load a file"));
}

#[test]
fn approval_preview_uses_inline_gutter_prefix() {
    let mut state = AppState::new();
    state.set_pending_action(PendingAction {
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

    let approval = build_approval(&state, Theme::default(), 80).expect("approval");
    let joined = approval
        .iter()
        .flat_map(|line| line.spans.iter().map(|span| span.text.as_str()))
        .collect::<String>();
    assert!(joined.contains("shell"));
    assert!(joined.contains("│ cargo check"));
    assert!(joined.contains("^Y approve  ^N reject"));
    assert!(!joined.contains("/approve"));
}

#[test]
fn approval_uses_first_reason_instead_of_duplicate_summary_block() {
    let mut state = AppState::new();
    state.set_pending_action(PendingAction {
        id: 1,
        kind: PendingActionKind::ShellCommand,
        title: "Approve shell command".to_string(),
        preview: "cargo check".to_string(),
        inspection: InspectionReport {
            operation: "bash".to_string(),
            decision: InspectionDecision::NeedsApproval,
            risk: RiskLevel::Low,
            summary: "Shell command requires approval before execution".to_string(),
            reasons: vec!["runs build in current dir".to_string()],
            targets: Vec::new(),
            segments: vec!["cargo check".to_string()],
            network_targets: Vec::new(),
        },
    });

    let approval = build_approval(&state, Theme::default(), 80).expect("approval");
    let text = approval
        .iter()
        .flat_map(|line| line.spans.iter().map(|span| span.text.as_str()))
        .collect::<String>();

    assert!(text.contains("· runs build in current dir"));
    assert!(!text.contains("Shell command requires approval before execution"));
}

#[test]
fn approval_preview_caps_match_kind_and_terminal_budget() {
    assert_eq!(
        approval_preview_line_cap(PendingActionKind::ShellCommand, 80, Some(20)),
        1
    );
    assert_eq!(
        approval_preview_line_cap(PendingActionKind::ShellCommand, 80, Some(36)),
        2
    );
    assert_eq!(
        approval_preview_line_cap(PendingActionKind::FileEdit, 68, Some(20)),
        2
    );
    assert_eq!(
        approval_preview_line_cap(PendingActionKind::FileWrite, 96, Some(36)),
        4
    );
}

#[test]
fn estimated_approval_rows_follow_compact_interrupt_shape() {
    assert_eq!(
        estimated_approval_rows(PendingActionKind::ShellCommand, 80, Some(20)),
        4
    );
    assert_eq!(
        estimated_approval_rows(PendingActionKind::ShellCommand, 80, Some(36)),
        5
    );
    assert_eq!(
        estimated_approval_rows(PendingActionKind::FileEdit, 68, Some(20)),
        5
    );
    assert_eq!(
        estimated_approval_rows(PendingActionKind::FileWrite, 96, Some(36)),
        7
    );
}

#[test]
fn transcript_only_adds_blank_line_when_message_kind_changes() {
    let mut state = AppState::new();
    state.messages.push(ChatMessage {
        id: 1,
        role: Role::User,
        content: "first".to_string(),
        transcript: crate::tui::state::TranscriptPresentation {
            collapsible: false,
            collapsed: false,
            summary: None,
            preview_lines: Vec::new(),
        },
    });
    state.messages.push(ChatMessage {
        id: 2,
        role: Role::User,
        content: "second".to_string(),
        transcript: crate::tui::state::TranscriptPresentation {
            collapsible: false,
            collapsed: false,
            summary: None,
            preview_lines: Vec::new(),
        },
    });
    state.messages.push(ChatMessage {
        id: 3,
        role: Role::Assistant,
        content: "reply".to_string(),
        transcript: crate::tui::state::TranscriptPresentation {
            collapsible: false,
            collapsed: false,
            summary: None,
            preview_lines: Vec::new(),
        },
    });

    let blocks = build_transcript(&mut state, Theme::default(), 80, |message, _, width| {
        build_transcript_block(message, Theme::default(), width, false)
    });

    assert!(!blocks[0].lines.last().unwrap().spans.is_empty());
    assert!(blocks[1].lines.last().unwrap().spans.is_empty());
    assert!(!blocks[2].lines.last().unwrap().spans.is_empty());
}

#[test]
fn composer_sheet_no_longer_paints_a_full_width_divider() {
    let theme = Theme::default();
    let mut symbols = SymbolPool::new();
    let blank = Cell {
        symbol_id: symbols.blank_id(),
        style: PackedStyle::new(theme.text, theme.background),
    };
    let mut buffer = CellBuffer::new(20, 4, blank);
    let rect = Rect::new(0, 0, 20, 4);

    let offset = paint_sheet(
        &mut buffer,
        &mut symbols,
        &rect,
        &[single_span("› hello", theme.base())],
        theme,
        "",
    );

    assert_eq!(offset, 0);
    let divider = symbols.intern("─");
    let row_has_divider = (0..buffer.width()).all(|x| buffer.get(x, 0).symbol_id == divider);
    assert!(!row_has_divider);
}

#[test]
fn top_bar_is_one_row_and_has_no_full_width_divider() {
    let theme = Theme::default();
    let mut symbols = SymbolPool::new();
    let blank = Cell {
        symbol_id: symbols.blank_id(),
        style: PackedStyle::new(theme.text, theme.background),
    };
    let mut buffer = CellBuffer::new(20, 6, blank);
    let mut state = AppState::new();
    let model = RenderModel {
        top_bar: vec![single_span("params · ready · fresh", theme.base())],
        transcript: vec![],
        composer: vec![single_span("› ", theme.base())],
        composer_cursor: (2, 0),
    };
    let layout = LayoutPlan {
        top_bar: Rect::new(0, 0, 20, 1),
        transcript: Rect::new(0, 1, 20, 3),
        composer: Rect::new(0, 4, 20, 2),
    };

    let _ = paint_model(&mut buffer, &mut symbols, &model, layout, theme, &mut state);

    // No row should be fully covered by the horizontal divider glyph.
    let divider = symbols.intern("─");
    for row in 0..buffer.height() {
        let row_all_divider = (0..buffer.width()).all(|x| buffer.get(x, row).symbol_id == divider);
        assert!(
            !row_all_divider,
            "unexpected full-width divider on row {row}"
        );
    }
}

#[test]
fn top_bar_is_always_one_line_and_reacts_to_states() {
    let mut state = AppState::new();

    // Loading state: 1 line, accent color.
    let loading = build_top_bar(&state, Theme::default(), 120);
    assert_eq!(loading.len(), 1, "top bar must always be 1 line");
    let loading_text = loading[0]
        .spans
        .iter()
        .map(|span| span.text.as_str())
        .collect::<String>();
    assert!(loading_text.contains("loading"));
    assert!(loading[0]
        .spans
        .iter()
        .any(|span| span.style == Theme::default().chip_accent()));

    // Streaming with active trace: still 1 line — activity trace moved to composer.
    state.model_ready = true;
    state.is_generating = true;
    state.tick = 5;
    state.current_trace = Some("indexing summaries".to_string());
    let streaming = build_top_bar(&state, Theme::default(), 120);
    assert_eq!(
        streaming.len(),
        1,
        "top bar stays 1 line even with active trace"
    );
    let streaming_text = streaming[0]
        .spans
        .iter()
        .map(|span| span.text.as_str())
        .collect::<String>();
    assert!(streaming_text.contains("generating"));
    // Activity trace must NOT appear in the top bar.
    assert!(!streaming_text.contains("indexing summaries"));
    assert!(streaming[0]
        .spans
        .iter()
        .any(|span| span.style == Theme::default().badge_assistant()));

    // Approval state: 1 line, warning color.
    state.is_generating = false;
    state.current_trace = None;
    state.set_pending_action(PendingAction {
        id: 1,
        kind: PendingActionKind::ShellCommand,
        title: "Approve shell command".to_string(),
        preview: "cargo check".to_string(),
        inspection: InspectionReport {
            operation: "bash".to_string(),
            decision: InspectionDecision::NeedsApproval,
            risk: RiskLevel::Medium,
            summary: "Shell command requires approval before execution".to_string(),
            reasons: Vec::new(),
            targets: Vec::new(),
            segments: vec!["cargo check".to_string()],
            network_targets: Vec::new(),
        },
    });
    let approval = build_top_bar(&state, Theme::default(), 120);
    assert_eq!(approval.len(), 1, "top bar stays 1 line during approval");
    let approval_text = approval[0]
        .spans
        .iter()
        .map(|span| span.text.as_str())
        .collect::<String>();
    assert!(approval_text.contains("awaiting approval"));
    assert!(approval[0]
        .spans
        .iter()
        .any(|span| span.style == Theme::default().chip_warning()));
}

#[test]
fn activity_trace_renders_in_composer_not_top_bar() {
    let mut state = AppState::new();
    state.model_ready = true;
    state.is_generating = true;
    state.current_trace = Some("indexing summaries".to_string());

    let activity = build_activity_line(&state, Theme::default(), 80);
    assert!(
        activity.is_some(),
        "active trace should produce an activity line"
    );
    let activity_text = activity
        .as_ref()
        .unwrap()
        .spans
        .iter()
        .map(|span| span.text.as_str())
        .collect::<String>();
    assert_eq!(activity.as_ref().unwrap().spans[0].text, SYSTEM_GUTTER);
    assert!(activity_text.contains("indexing summaries"));

    // Activity line is the first line in the composer content when present.
    let (lines, _cursor) = build_composer(&state, Theme::default(), 80, None, activity);
    assert!(lines[0].spans.is_empty());
    let first_line_text = lines[1]
        .spans
        .iter()
        .map(|span| span.text.as_str())
        .collect::<String>();
    assert!(first_line_text.contains("indexing summaries"));
}

#[test]
fn spinner_frame_is_tick_driven_and_stable() {
    assert_eq!(spinner_frame(0), "·");
    assert_eq!(spinner_frame(3), "•");
    assert_eq!(spinner_frame(6), "◦");
    assert_eq!(spinner_frame(9), "•");
}

#[test]
fn narrow_top_bar_keeps_runtime_segment() {
    let mut state = AppState::new();
    state.model_ready = true;
    state.is_generating = true;

    let line = build_top_bar(&state, Theme::default(), 28);
    let text = line[0]
        .spans
        .iter()
        .map(|span| span.text.as_str())
        .collect::<String>();

    assert!(text.contains("generating"));
    assert!(!text.contains("cache"));
    assert!(!text.contains("msgs"));
}

#[test]
fn top_bar_can_show_tokens_and_time() {
    let mut state = AppState::new();
    state.model_ready = true;
    state.update_budget(512, 722, 1234, Some(0.0));
    state.start_generation("generating...", false);

    let expected_time = format_duration_display(
        state
            .current_turn_duration()
            .expect("active generation should expose a duration"),
    );
    let line = build_top_bar(&state, Theme::default(), 120);
    let spans = &line[0].spans;

    assert!(spans.iter().any(|span| span.text == "1234 tok"));
    assert!(spans.iter().any(|span| span.text == expected_time));
}

#[test]
fn top_bar_hides_tokens_and_time_when_disabled() {
    let mut state = AppState::new();
    state.model_ready = true;
    state.update_budget(128, 128, 256, Some(0.0));
    state.start_generation("generating...", false);
    let hidden_time = format_duration_display(
        state
            .current_turn_duration()
            .expect("active generation should expose a duration"),
    );
    state.set_show_top_bar_tokens(false);
    state.set_show_top_bar_time(false);

    let line = build_top_bar(&state, Theme::default(), 120);
    let spans = &line[0].spans;

    assert!(!spans.iter().any(|span| span.text == "256 tok"));
    assert!(!spans.iter().any(|span| span.text == hidden_time));
}

#[test]
fn transcript_scroll_indicators_match_hidden_direction() {
    let theme = Theme::default();
    let mut state = AppState::new();
    state.scroll_offset = 2;
    let blocks = vec![RenderBlock {
        message_id: None,
        lines: (0..6)
            .map(|idx| single_span(&format!("line {idx}"), theme.base()))
            .collect(),
    }];

    let mut symbols = SymbolPool::new();
    let blank = Cell {
        symbol_id: symbols.blank_id(),
        style: PackedStyle::new(theme.text, theme.background),
    };
    let mut buffer = CellBuffer::new(40, 4, blank);

    paint_transcript(
        &mut buffer,
        &mut symbols,
        &Rect::new(0, 0, 40, 4),
        &blocks,
        theme,
        &mut state,
    );

    let top = (0..12)
        .map(|x| symbols.get(buffer.get(x, 0).symbol_id))
        .collect::<String>();
    let bottom = (0..20)
        .map(|x| symbols.get(buffer.get(x, 3).symbol_id))
        .collect::<String>();

    assert!(top.contains("↑ 2 above"));
    assert!(bottom.contains("↓ jump to latest"));
}

#[test]
fn active_assistant_message_gets_cursor_span() {
    let message = ChatMessage {
        id: 1,
        role: Role::Assistant,
        content: "working".to_string(),
        transcript: crate::tui::state::TranscriptPresentation {
            collapsible: false,
            collapsed: false,
            summary: None,
            preview_lines: Vec::new(),
        },
    };

    let lines = build_standard_message(&message, Theme::default(), 80, true, 0);
    let text = lines[0]
        .spans
        .iter()
        .map(|span| span.text.as_str())
        .collect::<String>();
    assert!(text.contains("▍") || text.contains("▌"));
}

#[test]
fn active_empty_assistant_message_only_shows_one_cursor_marker() {
    let message = ChatMessage {
        id: 1,
        role: Role::Assistant,
        content: String::new(),
        transcript: crate::tui::state::TranscriptPresentation {
            collapsible: false,
            collapsed: false,
            summary: None,
            preview_lines: Vec::new(),
        },
    };

    let lines = build_standard_message(&message, Theme::default(), 80, true, 0);
    let text = lines[0]
        .spans
        .iter()
        .map(|span| span.text.as_str())
        .collect::<String>();

    assert!(text.contains("params"));
    assert_eq!(text.matches('▍').count() + text.matches('▌').count(), 1);
}

#[test]
fn palette_command_names_are_padded_to_fixed_column() {
    let mut state = AppState::new();
    assert!(state.activate_command_launcher(vec![
        CommandSuggestion {
            name: "/read".to_string(),
            usage: "/read <path>".to_string(),
            description: "load a file".to_string(),
            source: "builtin",
            group: "context",
            aliases: vec![],
        },
        CommandSuggestion {
            name: "/sessions".to_string(),
            usage: "/sessions list".to_string(),
            description: "manage sessions".to_string(),
            source: "builtin",
            group: "session",
            aliases: vec![],
        },
    ]));

    let (lines, _) = build_composer(&state, Theme::default(), 80, None, None);

    // Entry rows have 4 spans: marker, name, separator, description.
    // Identify them by checking that the second span (index 1) contains a padded name.
    let entry_rows: Vec<&StyledLine> = lines
        .iter()
        .filter(|line| {
            // Entry rows: marker span + name span + sep span + description span = 4 spans
            if line.spans.len() != 4 {
                return false;
            }
            let marker = &line.spans[0].text;
            (marker == "→ " || marker == "  ")
                && (line.spans[1].text.contains("/read")
                    || line.spans[1].text.contains("/sessions"))
        })
        .collect();

    assert_eq!(
        entry_rows.len(),
        2,
        "both entries should appear as 4-span rows"
    );
    // The name spans must be the same display width (PALETTE_NAME_COL),
    // so descriptions (4th span) start at the same column.
    let name_widths: Vec<usize> = entry_rows
        .iter()
        .map(|line| line.spans[1].text.chars().count())
        .collect();
    assert_eq!(
        name_widths[0], name_widths[1],
        "name spans must have equal display width"
    );
    assert_eq!(
        name_widths[0], PALETTE_NAME_COL,
        "name spans must be padded to PALETTE_NAME_COL"
    );
}

#[test]
fn palette_usage_line_has_no_usage_prefix() {
    let mut state = AppState::new();
    assert!(state.activate_command_launcher(vec![CommandSuggestion {
        name: "/read".to_string(),
        usage: "/read <path>".to_string(),
        description: "load a file".to_string(),
        source: "builtin",
        group: "context",
        aliases: vec!["/r".to_string()],
    }]));

    let (lines, _) = build_composer(&state, Theme::default(), 80, None, None);
    let text: String = lines
        .iter()
        .flat_map(|line| line.spans.iter().map(|s| s.text.as_str()))
        .collect();

    // Usage shown without "usage:" label.
    assert!(!text.contains("usage:"));
    assert!(text.contains("/read <path>"));
    // Alias shown with "aka" prefix (shorter than "aliases:").
    assert!(!text.contains("aliases:"));
    assert!(text.contains("aka /r"));
}

#[test]
fn composer_dims_input_text_while_generating() {
    let mut state = AppState::new();
    state.input = "draft while busy".to_string();
    state.cursor = state.input.len();

    state.is_generating = false;
    let (idle_lines, _) = build_composer(&state, Theme::default(), 80, None, None);
    state.is_generating = true;
    let (busy_lines, _) = build_composer(&state, Theme::default(), 80, None, None);

    // Find the input text span in each case (second span of the prompt row).
    let idle_text_style = idle_lines[1].spans[1].style;
    let busy_text_style = busy_lines[1].spans[1].style;

    let theme = Theme::default();
    assert_eq!(idle_text_style, theme.base(), "idle input uses base style");
    assert_eq!(
        busy_text_style,
        theme.muted(),
        "input dims to muted while generating"
    );
}
