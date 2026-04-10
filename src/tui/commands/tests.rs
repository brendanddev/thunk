use super::{
    decode_slash_write_content, format_display_status, format_memory_facts, format_memory_last,
    format_memory_status, format_transcript_status, parse_sessions_export_args,
    parse_slash_edit_body,
};
use crate::events::{
    FactProvenance, MemoryConsolidationView, MemoryFactView, MemorySessionExcerptView,
    MemorySkippedReasonCount, MemorySnapshot, MemoryUpdateReport,
};
use crate::tui::state::AppState;

#[test]
fn decode_slash_write_content_expands_common_escapes() {
    let decoded = decode_slash_write_content("hello\\nfrom\\tparams\\\\");
    assert_eq!(decoded, "hello\nfrom\tparams\\");
}

#[test]
fn parse_slash_edit_body_extracts_path_and_multiline_body() {
    let parsed = parse_slash_edit_body(
        "src/main.rs\n```params-edit\n<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n```",
    )
    .expect("parse edit body");

    assert_eq!(parsed.0, "src/main.rs");
    assert!(parsed.1.starts_with("```params-edit"));
}

#[test]
fn parse_sessions_export_args_supports_optional_format() {
    assert_eq!(
        parse_sessions_export_args("alpha markdown"),
        Some(("alpha".to_string(), Some("markdown".to_string())))
    );
    assert_eq!(
        parse_sessions_export_args("named session"),
        Some(("named session".to_string(), None))
    );
}

#[test]
fn memory_formatters_include_counts_and_labels() {
    let snapshot = MemorySnapshot {
        loaded_facts: vec![
            MemoryFactView {
                content: "src/main.rs updates cache stats".to_string(),
                provenance: FactProvenance::Verified,
            },
            MemoryFactView {
                content: "legacy session note".to_string(),
                provenance: FactProvenance::Legacy,
            },
        ],
        last_summary_paths: vec!["src/main.rs".to_string()],
        last_retrieval_query: Some("cache stats".to_string()),
        last_selected_facts: vec![MemoryFactView {
            content: "src/main.rs updates cache stats".to_string(),
            provenance: FactProvenance::Verified,
        }],
        last_selected_session_excerpts: vec![MemorySessionExcerptView {
            session_label: "alpha".to_string(),
            role: "assistant".to_string(),
            excerpt: "cache stats are reported in the status line".to_string(),
        }],
        last_update: Some(MemoryUpdateReport {
            accepted_facts: vec![MemoryFactView {
                content: "src/main.rs updates cache stats".to_string(),
                provenance: FactProvenance::Verified,
            }],
            skipped_reasons: vec![MemorySkippedReasonCount {
                reason: "missing evidence anchor".to_string(),
                count: 2,
            }],
            duplicate_count: 1,
        }),
        last_consolidation: Some(MemoryConsolidationView {
            ttl_pruned: 1,
            dedup_removed: 2,
            cap_removed: 0,
        }),
    };

    let status = format_memory_status(&snapshot);
    let facts = format_memory_facts(&snapshot);
    let last = format_memory_last(&snapshot);

    assert!(status.contains("loaded facts: 2"));
    assert!(status.contains("last fact matches: 1"));
    assert!(status.contains("last session excerpts: 1"));
    assert!(status.contains("last retrieval query: cache stats"));
    assert!(status.contains("src/main.rs"));
    assert!(facts.contains("[verified]"));
    assert!(facts.contains("[legacy]"));
    assert!(last.contains("duplicates: 1"));
    assert!(last.contains("query: cache stats"));
    assert!(last.contains("dedup removed: 2"));
}

#[test]
fn transcript_status_reports_collapse_mode() {
    assert!(format_transcript_status(0, 0).contains("no collapsible blocks"));
    assert!(format_transcript_status(3, 3).contains("fully collapsed"));
    assert!(format_transcript_status(3, 0).contains("fully expanded"));
    assert!(format_transcript_status(4, 3).contains("mostly collapsed"));
    assert!(format_transcript_status(4, 1).contains("mostly expanded"));
}

#[test]
fn display_status_reports_toggle_state() {
    let mut state = AppState::new();
    state.set_show_top_bar_tokens(false);
    state.set_show_top_bar_time(true);

    let output = format_display_status(&state);

    assert!(output.contains("tokens: off"));
    assert!(output.contains("time: on"));
}
