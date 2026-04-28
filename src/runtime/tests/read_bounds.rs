use super::*;

#[test]
fn read_cap_blocks_reads_beyond_limit() {
    // On non-investigation turns, answer_phase fires after the first read.
    // The second read attempt is blocked by the answer_phase gate, not the cap.
    // This verifies that post-read tool drift is prevented for non-investigation turns.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("a.rs"), "fn a() {}\n").unwrap();
    fs::write(tmp.path().join("b.rs"), "fn b() {}\n").unwrap();

    let final_answer = "I have read the file.";
    let mut rt = make_runtime_in(
        vec!["[read_file: a.rs]", "[read_file: b.rs]", final_answer],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read a.rs".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "turn must complete without failure: {events:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::AnswerReady(_))),
        "turn must complete with AnswerReady: {events:?}"
    );

    let snapshot = rt.messages_snapshot();
    let all_user: String = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::User)
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    assert_eq!(
        all_user.matches("=== tool_result: read_file ===").count(),
        1,
        "answer_phase fires after first read; second read must not dispatch"
    );
    assert!(
        all_user.contains("[runtime:correction]") && all_user.contains("already read this turn"),
        "second read attempt must be blocked by answer_phase correction"
    );
}

#[test]
fn duplicate_read_is_blocked_within_same_turn() {
    // On non-investigation turns, answer_phase fires after the first read.
    // The duplicate read attempt is blocked by the answer_phase gate (not the dedup
    // guard) — both mechanisms prevent the read, but answer_phase fires first.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[read_file: engine.rs]",
            "[read_file: engine.rs]",
            "I already have the file contents in context.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read engine.rs".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "turn must complete without failure: {events:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::AnswerReady(_))),
        "turn must complete with AnswerReady: {events:?}"
    );

    let snapshot = rt.messages_snapshot();
    let all_user: String = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::User)
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    // First read must succeed.
    assert_eq!(
        all_user.matches("=== tool_result: read_file ===").count(),
        1,
        "first read must succeed; duplicate must not dispatch"
    );

    // On non-investigation turns, answer_phase fires after the first read.
    // The duplicate read attempt is intercepted by the answer_phase gate (pre-dispatch),
    // which injects a runtime correction rather than the dedup tool error.
    assert!(
        all_user.contains("[runtime:correction]") && all_user.contains("already read this turn"),
        "duplicate read attempt must be blocked by answer_phase correction"
    );
}
