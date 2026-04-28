use super::*;

#[test]
fn cycle_detection_blocks_second_identical_call() {
    // Model emits the same list_dir call twice in one response.
    // First call executes; second is blocked with a cycle error.
    // A synthesis response is provided so the loop can complete normally.
    let mut rt = make_runtime(vec!["[list_dir: .][list_dir: .]", "Done."]);
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "list files".into(),
        },
    );

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: list_dir ===")),
        "first call must produce a tool result"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_error: list_dir ===")
                && m.content.contains("identical arguments twice in a row")),
        "second identical call must produce a cycle error"
    );
}

#[test]
fn cycle_detection_allows_different_args() {
    // Two list_dir calls with different paths — neither should be blocked.
    // A synthesis response is provided so the loop can complete normally.
    let mut rt = make_runtime(vec!["[list_dir: .][list_dir: src/]", "Listed both."]);
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "list files".into(),
        },
    );

    let snapshot = rt.messages_snapshot();
    assert!(
        !snapshot
            .iter()
            .any(|m| m.content.contains("identical arguments twice in a row")),
        "different args must not trigger cycle detection"
    );
}

#[test]
fn tool_round_followed_by_synthesized_answer() {
    // After a tool call completes, the model is re-invoked in the same turn.
    // The synthesis response (no tool calls) produces the final AnswerReady event.
    let mut rt = make_runtime(vec!["[list_dir: .]", "The root contains several files."]);
    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "list files".into(),
        },
    );

    assert!(!has_failed(&events), "unexpected failure: {events:?}");

    // AnswerReady must be ToolAssisted from the synthesis response
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(
            answer_source,
            Some(AnswerSource::ToolAssisted { rounds: 1 })
        ),
        "expected ToolAssisted(1), got: {answer_source:?}"
    );

    // Conversation must contain: user prompt, assistant tool call, user tool result,
    // assistant synthesis — i.e. two assistant messages.
    let snapshot = rt.messages_snapshot();
    let assistant_msgs: Vec<_> = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::Assistant)
        .collect();
    assert_eq!(
        assistant_msgs.len(),
        2,
        "expected tool-call + synthesis, got: {assistant_msgs:?}"
    );
    assert!(
        assistant_msgs[1].content.contains("several files"),
        "synthesis must contain model's response text"
    );
}

#[test]
fn multi_tool_round_synthesizes_after_all_rounds() {
    // Model calls list_dir twice across two separate rounds, then synthesizes.
    // tool_rounds must reflect both rounds in the AnswerReady source.
    let mut rt = make_runtime(vec![
        "[list_dir: .]",
        "[list_dir: src/]",
        "Found everything I need.",
    ]);
    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "list files".into(),
        },
    );

    assert!(!has_failed(&events), "unexpected failure: {events:?}");

    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(
            answer_source,
            Some(AnswerSource::ToolAssisted { rounds: 2 })
        ),
        "expected ToolAssisted(2), got: {answer_source:?}"
    );

    let snapshot = rt.messages_snapshot();
    let assistant_msgs: Vec<_> = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::Assistant)
        .collect();
    assert_eq!(
        assistant_msgs.len(),
        3,
        "expected two tool calls + synthesis"
    );
}

#[test]
fn malformed_block_triggers_correction_and_retries() {
    // Model emits [test_file]...[/write_file] — wrong opening tag, correct closing tag.
    // The engine should detect the malformed block, discard the response, inject
    // a correction, and re-invoke the model. The synthesis response closes the loop.
    let malformed = "[test_file]\npath: f.txt\n---content---\nhello\n[/write_file]";
    let mut rt = make_runtime(vec![malformed, "Done."]);
    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "create f.txt".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "must not fail permanently: {events:?}"
    );

    // The final answer must be the synthesis response, not the malformed block.
    let snapshot = rt.messages_snapshot();
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("Done."),
        "last assistant message must be synthesis"
    );

    // The correction message must have been injected into the conversation.
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.starts_with("[runtime:correction]")
                && m.content.contains("unrecognized opening tag")),
        "malformed block correction must be in conversation"
    );
}

#[test]
fn cycle_detection_allows_retry_after_tool_error() {
    // A tool fails (bad path), then the model retries with the same args.
    // The retry must NOT be blocked as a cycle — only successful calls set the key.
    // list_dir returns an IO error for a non-existent path, then succeeds on ".".
    // The synthesis response closes the loop.
    let mut rt = make_runtime(vec!["[list_dir: /nonexistent/path][list_dir: .]", "Done."]);
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "list files".into(),
        },
    );

    let snapshot = rt.messages_snapshot();
    // The error from the first call must appear
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_error: list_dir ===")),
        "first call error must be in conversation"
    );
    // The successful retry must produce a result — no cycle error
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: list_dir ===")),
        "successful retry must not be blocked by cycle detection"
    );
    assert!(
        !snapshot
            .iter()
            .any(|m| m.content.contains("identical arguments")
                && m.content.contains("=== tool_error: list_dir ===")
                && m.content.contains(".")),
        "retry with same args after error must not trigger cycle detection"
    );
}

#[test]
fn missing_read_file_error_terminates_without_retry_loop() {
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    let mut rt = make_runtime_in(
        vec![
            "[read_file: missing_file_phase75.rs]",
            "[read_file: missing_file_phase75.rs]",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Read missing_file_phase75.rs".into(),
        },
    );
    assert!(
        !has_failed(&events),
        "missing read should terminate cleanly: {events:?}"
    );

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_error: read_file ===")),
        "read_file failure must be surfaced as a tool_error"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("No file contents were read.")),
        "runtime terminal answer must explain that no contents were read"
    );
    let assistant_read_calls = snapshot
        .iter()
        .filter(|m| {
            m.role == crate::llm::backend::Role::Assistant
                && m.content.contains("[read_file: missing_file_phase75.rs]")
        })
        .count();
    assert_eq!(
        assistant_read_calls, 1,
        "read_file must not be retried in a loop"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::AnswerReady(AnswerSource::ToolLimitReached))),
        "missing read must not hit the tool-round limit"
    );
}
