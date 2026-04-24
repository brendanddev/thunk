use super::*;

#[test]
fn approve_with_no_pending_fires_failed() {
    let mut rt = make_runtime(vec!["hello"]);
    let events = collect_events(&mut rt, RuntimeRequest::Approve);
    assert!(has_failed(&events), "expected Failed, got: {events:?}");
    assert_eq!(
        failed_message(&events).as_deref(),
        Some("No pending action to approve.")
    );
}

#[test]
fn reject_with_no_pending_fires_failed() {
    let mut rt = make_runtime(vec!["hello"]);
    let events = collect_events(&mut rt, RuntimeRequest::Reject);
    assert!(has_failed(&events), "expected Failed, got: {events:?}");
    assert_eq!(
        failed_message(&events).as_deref(),
        Some("No pending action to reject.")
    );
}

#[test]
fn reject_uses_runtime_cancellation_even_if_model_would_claim_success() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    let mut rt = make_runtime_in(
        vec![
            "[write_file]\npath: reject_test_phase75.txt\n---content---\nshould not exist\n[/write_file]",
            "I created reject_test_phase75.txt.",
        ],
        tmp.path(),
    );

    let submit_events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Create a file reject_test_phase75.txt with the content should not exist"
                .into(),
        },
    );
    assert!(
        !has_failed(&submit_events),
        "submit failed: {submit_events:?}"
    );
    assert!(
        submit_events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ApprovalRequired(_))),
        "write_file must request approval"
    );

    let reject_events = collect_events(&mut rt, RuntimeRequest::Reject);
    assert!(
        !has_failed(&reject_events),
        "reject failed: {reject_events:?}"
    );
    assert!(
        !tmp.path().join("reject_test_phase75.txt").exists(),
        "rejected write must not create the file"
    );

    let snapshot = rt.messages_snapshot();
    // Runtime owns the cancellation answer — the model's synthesis response must not be used.
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("Canceled. No file was created")),
        "runtime cancellation answer must be recorded"
    );
    assert!(
        !snapshot
            .iter()
            .any(|m| m.content.contains("I created reject_test_phase75.txt.")),
        "backend response after reject must not be used"
    );
    assert!(fs::read_dir(tmp.path()).unwrap().next().is_none());
}

#[test]
fn submit_while_pending_fires_failed() {
    let mut rt = make_runtime(vec!["hello"]);
    rt.set_pending_for_test(PendingAction {
        tool_name: "edit_file".into(),
        summary: "edit src/lib.rs".into(),
        risk: RiskLevel::Medium,
        payload: "{}".into(),
    });
    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "continue".into(),
        },
    );
    assert!(has_failed(&events), "expected Failed, got: {events:?}");
    assert!(failed_message(&events)
        .as_deref()
        .unwrap_or("")
        .contains("pending"),);
}

#[test]
fn reset_clears_pending_state() {
    let mut rt = make_runtime(vec!["hello"]);
    rt.set_pending_for_test(PendingAction {
        tool_name: "write_file".into(),
        summary: "write src/new.rs".into(),
        risk: RiskLevel::High,
        payload: "{}".into(),
    });
    collect_events(&mut rt, RuntimeRequest::Reset);
    // After reset, approve should fail with "no pending" — not "submit blocked"
    let events = collect_events(&mut rt, RuntimeRequest::Approve);
    assert!(
        has_failed(&events),
        "expected Failed after reset, got: {events:?}"
    );
    assert_eq!(
        failed_message(&events).as_deref(),
        Some("No pending action to approve.")
    );
}

#[test]
fn edit_repair_correction_injected_on_garbled_repair_after_failure() {
    // First response: edit_file with empty search text — produces an Immediate tool error.
    // Second response: [edit_file] tags present but unrecognized delimiters (zero parse).
    // Engine must inject EDIT_REPAIR_CORRECTION rather than accepting as Direct.
    // Third response: synthesis after correction.
    let bad_edit = "[edit_file]\npath: foo.rs\n---replace---\nnew text\n[/edit_file]";
    let garbled_repair =
        "[edit_file]\npath: foo.rs\nFind: old text\nReplace: new text\n[/edit_file]";
    let synthesis = "I was unable to apply the edit.";

    let mut rt = make_runtime(vec![bad_edit, garbled_repair, synthesis]);
    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "edit foo.rs".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "must not fail permanently: {events:?}"
    );

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.starts_with("[runtime:correction]")
                && m.content.contains("edit_file")),
        "edit repair correction must be injected: {snapshot:?}"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(last_assistant, Some(synthesis));
}

#[test]
fn edit_old_new_content_format_requests_approval_and_executes() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    let file = tmp.path().join("test_phase82.txt");
    fs::write(&file, "hello world").unwrap();

    let edit = "[edit_file]\npath: test_phase82.txt\nold content: hello world\nnew content: hello params\n[/edit_file]";
    let mut rt = make_runtime_in(vec![edit, "Updated."], tmp.path());

    let submit_events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Edit test_phase82.txt and change hello world to hello params".into(),
        },
    );
    assert!(
        !has_failed(&submit_events),
        "submit failed: {submit_events:?}"
    );
    assert!(
        submit_events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ApprovalRequired(p)
            if p.tool_name == "edit_file")),
        "edit must request approval instead of falling back to Direct: {submit_events:?}"
    );
    assert_eq!(fs::read_to_string(&file).unwrap(), "hello world");

    let approve_events = collect_events(&mut rt, RuntimeRequest::Approve);
    assert!(
        !has_failed(&approve_events),
        "approve failed: {approve_events:?}"
    );
    assert_eq!(fs::read_to_string(&file).unwrap(), "hello params");

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: edit_file ===")),
        "approved edit result must be injected: {snapshot:?}"
    );
}

#[test]
fn approve_produces_runtime_owned_answer_after_successful_mutation() {
    // After approving a mutation, the runtime must finalize directly without
    // re-entering model generation. The answer is built from the tool output summary.
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "hello").unwrap();
    let path = f.path().to_string_lossy().into_owned();
    let payload = format!("{}\x00hello\x00world", path);

    // No model responses needed — the runtime owns the answer.
    let mut rt = make_runtime(Vec::<&str>::new());
    let before_count = rt.messages_snapshot().len();

    rt.set_pending_for_test(PendingAction {
        tool_name: "edit_file".into(),
        summary: format!("edit {path}"),
        risk: RiskLevel::Medium,
        payload,
    });

    let events = collect_events(&mut rt, RuntimeRequest::Approve);
    assert!(!has_failed(&events), "approve must not fail: {events:?}");

    // finish_with_runtime_answer emits AssistantMessageChunk for the runtime-owned answer.
    assert!(
        events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::AssistantMessageChunk(_))),
        "runtime-owned answer must emit AssistantMessageChunk"
    );

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
        "mutation finalization must use ToolAssisted {{ rounds: 1 }}: {answer_source:?}"
    );

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot.len() > before_count,
        "snapshot must grow after approve + runtime finalization"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: edit_file ===")),
        "tool result must be in conversation after approve"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant);
    assert!(
        last_assistant
            .map(|m| m.content.starts_with("edit_file result:"))
            .unwrap_or(false),
        "last assistant message must be the runtime-owned mutation answer: {last_assistant:?}"
    );
}
