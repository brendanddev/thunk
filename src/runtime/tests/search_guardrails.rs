use super::*;
use crate::runtime::types::RuntimeTerminalReason;
use super::super::tool_surface::{select_tool_surface, ToolSurface};

#[test]
fn weak_search_query_rejects_first_attempt_then_allows_specific_recovery() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::write(
        tmp.path().join("src/git_status.rs"),
        "fn render_git_status() {}\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: git]",
            "[search_code: git_status]",
            "[read_file: src/git_status.rs]",
            "git_status is implemented in src/git_status.rs.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "where is git_status implemented".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "turn must recover after one weak-query correction: {events:?}"
    );
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_error: search_code ===")
                && m.content.contains("too broad for an investigation turn (git)")),
        "first weak search must be rejected with a runtime correction"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: search_code ===")),
        "specific recovery search should execute"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: read_file ===")),
        "recovery should still read grounded evidence"
    );
}

#[test]
fn repeated_weak_search_query_terminates_distinct_policy_reason() {
    let mut rt = make_runtime(vec!["[search_code: git]", "[search_code: g]"]);

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "where is git_status implemented".into(),
        },
    );

    let answer_source = events.iter().find_map(|event| {
        if let RuntimeEvent::AnswerReady(source) = event {
            Some(source)
        } else {
            None
        }
    });
    assert!(
        matches!(
            answer_source,
            Some(AnswerSource::RuntimeTerminal {
                reason: RuntimeTerminalReason::RepeatedWeakSearchQuery,
                ..
            })
        ),
        "repeated weak search queries must terminate on a distinct runtime reason: {events:?}"
    );
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .all(|m| !m.content.contains("=== tool_result: search_code ===")),
        "rejected weak searches must not dispatch"
    );
}

#[test]
fn rendered_lookup_enables_weak_query_guard_after_surface_rejection() {
    let mut rt = make_runtime(vec![
        "[git_status]",
        "[search_code: git]",
        "[search_code: git]",
    ]);

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "where is git status rendered".into(),
        },
    );

    assert_eq!(
        select_tool_surface("where is git status rendered", true, false, false),
        ToolSurface::RetrievalFirst
    );
    let answer_source = events.iter().find_map(|event| {
        if let RuntimeEvent::AnswerReady(source) = event {
            Some(source)
        } else {
            None
        }
    });
    assert!(
        matches!(
            answer_source,
            Some(AnswerSource::RuntimeTerminal {
                reason: RuntimeTerminalReason::RepeatedWeakSearchQuery,
                ..
            })
        ),
        "rendered lookup should become investigation-required and hit weak-query guard: {events:?}"
    );
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_error: git_status ===")
                && m.content.contains("retrieval tools only")),
        "wrong-surface git_status should still be rejected first"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_error: search_code ===")
                && m.content.contains("too broad for an investigation turn (git)")),
        "first weak git search should receive the weak-query correction"
    );
    assert!(
        snapshot
            .iter()
            .all(|m| !m.content.contains("=== tool_result: git_status ===")
                && !m.content.contains("=== tool_result: search_code ===")
                && !m.content.contains("=== tool_result: read_file ===")),
        "wrong-surface and weak-query attempts must not dispatch or read evidence"
    );
}

#[test]
fn lockfile_read_rejected_when_matched_source_candidate_exists() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::write(tmp.path().join("Cargo.lock"), "git_status = \"1.0.0\"\n").unwrap();
    fs::write(
        tmp.path().join("src/git_status.rs"),
        "fn render_git_status() {}\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: git_status]",
            "[read_file: Cargo.lock]",
            "[read_file: src/git_status.rs]",
            "git_status is implemented in src/git_status.rs.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "where is git_status implemented".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "turn must recover from lockfile read: {events:?}"
    );
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .filter(|m| m.content.contains("=== tool_result: read_file ==="))
            .count()
            >= 2,
        "lockfile read should execute, then recovery should read source evidence"
    );
    assert!(
        snapshot.iter().any(|m| m
            .content
            .contains("[runtime:correction] The file just read is a lockfile")
            && m.content.contains("[read_file: ")
            && m.content.contains("src/git_status.rs")),
        "runtime should issue one lockfile-specific recovery to the source candidate"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("git_status is implemented in src/git_status.rs.")
    );
}

#[test]
fn lockfile_read_accepted_when_no_matched_source_candidate_exists() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("Cargo.lock"), "git_status = \"1.0.0\"\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: git_status]",
            "[read_file: Cargo.lock]",
            "git_status only appears in Cargo.lock.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "where is git_status implemented".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "lockfile-only result should fall back to normal grounded behavior: {events:?}"
    );
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot.iter().all(|m| !m
            .content
            .contains("[runtime:correction] The file just read is a lockfile")),
        "lockfile guard must not fire when no stronger source candidate exists"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("git_status only appears in Cargo.lock.")
    );
}

#[test]
fn lockfile_guard_preserves_config_lookup_recovery_priority() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
    fs::write(
        tmp.path().join("sandbox/Cargo.lock"),
        "database = \"postgres\"\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("sandbox/database.py"),
        "database = \"runtime default\"\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("sandbox/database.yaml"),
        "database: postgres\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: database]",
            "[read_file: sandbox/Cargo.lock]",
            "[read_file: sandbox/database.yaml]",
            "database is configured in sandbox/database.yaml.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Find where database is configured in sandbox/".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "config lookup should recover to config candidate before lockfile guard: {events:?}"
    );
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot.iter().any(|m| m
            .content
            .contains("[runtime:correction] This is a config lookup")
            && m.content.contains("sandbox/database.yaml")),
        "config recovery should remain the active mode-specific gate"
    );
    assert!(
        snapshot.iter().all(|m| !m
            .content
            .contains("[runtime:correction] The file just read is a lockfile")),
        "lockfile guard must not override ConfigLookup recovery"
    );
}
