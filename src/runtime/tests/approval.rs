use super::*;
use crate::app::config::Config;
use crate::llm::backend::GenerateRequest;
use crate::runtime::types::RuntimeTerminalReason;
use crate::tools::default_registry;
use std::sync::{Arc, Mutex};

fn make_runtime_in_with_recorded_requests(
    responses: Vec<impl Into<String>>,
    root: &std::path::Path,
) -> (Runtime, Arc<Mutex<Vec<GenerateRequest>>>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let project_root = ProjectRoot::new(root.to_path_buf()).unwrap();
    let runtime = Runtime::new(
        &Config::default(),
        project_root.clone(),
        Box::new(RecordingBackend::new(responses, Arc::clone(&requests))),
        default_registry().with_project_root(project_root.as_path_buf()),
    );
    (runtime, requests)
}

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
            text: "Create a file reject_test_phase75.txt with the content should not exist".into(),
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
fn repeated_garbled_edit_repair_terminals_without_surfacing_malformed_block() {
    let bad_edit = "[edit_file]\npath: foo.rs\n---replace---\nnew text\n[/edit_file]";
    let garbled_repair =
        "[edit_file]\npath: foo.rs\nFind: old text\nReplace: new text\n[/edit_file]";

    let mut rt = make_runtime(vec![
        bad_edit,
        garbled_repair,
        garbled_repair,
        "This response should not be consumed.",
    ]);
    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "edit foo.rs".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "repeated garbled edit repair must terminate cleanly: {events:?}"
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
            Some(AnswerSource::RuntimeTerminal {
                reason: RuntimeTerminalReason::RepeatedGarbledEditRepair,
                ..
            })
        ),
        "second garbled edit repair must use deterministic runtime terminal: {answer_source:?}"
    );

    let snapshot = rt.messages_snapshot();
    let assistant_messages: Vec<&str> = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str())
        .collect();
    assert!(
        !assistant_messages
            .iter()
            .any(|m| m.contains("Find: old text") || m.contains("Replace: new text")),
        "garbled edit repair must not surface as a final assistant answer: {assistant_messages:?}"
    );
    let last_assistant = assistant_messages.last().copied();
    assert!(
        matches!(last_assistant, Some(s) if s.contains("invalid edit_file repair block")),
        "last assistant message must be the runtime garbled-repair terminal: {last_assistant:?}"
    );
}

#[test]
fn edit_old_new_content_format_requests_approval_and_executes() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    let file = tmp.path().join("test_phase82.txt");
    fs::write(&file, "hello world").unwrap();

    let edit = "[edit_file]\npath: test_phase82.txt\nold content: hello world\nnew content: hello thunk\n[/edit_file]";
    let mut rt = make_runtime_in(vec![edit, "Updated."], tmp.path());

    let submit_events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Edit test_phase82.txt and change hello world to hello thunk".into(),
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
    assert_eq!(fs::read_to_string(&file).unwrap(), "hello thunk");

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: edit_file ===")),
        "approved edit result must be injected: {snapshot:?}"
    );
}

#[test]
fn simple_edit_prompt_seeds_edit_file_and_requests_approval() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    let file = tmp.path().join("test.txt");
    fs::write(&file, "hello world").unwrap();

    let (mut rt, requests) =
        make_runtime_in_with_recorded_requests(vec!["should not be used"], tmp.path());
    let submit_events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Edit the file test.txt replace the content hello world with hello thunk".into(),
        },
    );

    assert!(
        !has_failed(&submit_events),
        "submit failed: {submit_events:?}"
    );
    assert!(
        submit_events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ApprovalRequired(p) if p.tool_name == "edit_file")),
        "simple edit prompt must request edit_file approval: {submit_events:?}"
    );
    assert!(
        requests.lock().unwrap().is_empty(),
        "seeded simple edit must reach approval before any model generation"
    );
    assert_eq!(
        fs::read_to_string(&file).unwrap(),
        "hello world",
        "file must not change before approval"
    );
}

#[test]
fn seeded_simple_edit_executes_only_after_approval() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    let file = tmp.path().join("hello.txt");
    fs::write(&file, "hello root").unwrap();

    let (mut rt, requests) =
        make_runtime_in_with_recorded_requests(vec!["still unused"], tmp.path());
    let submit_events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Edit hello.txt replace hello root with hello runtime".into(),
        },
    );

    assert!(
        !has_failed(&submit_events),
        "submit failed: {submit_events:?}"
    );
    assert!(
        submit_events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ApprovalRequired(p) if p.tool_name == "edit_file")),
        "seeded simple edit must enter the normal approval path: {submit_events:?}"
    );
    assert_eq!(
        fs::read_to_string(&file).unwrap(),
        "hello root",
        "file must not change before approval"
    );

    let approve_events = collect_events(&mut rt, RuntimeRequest::Approve);
    assert!(
        !has_failed(&approve_events),
        "approve failed: {approve_events:?}"
    );
    assert_eq!(
        fs::read_to_string(&file).unwrap(),
        "hello runtime",
        "seeded simple edit must execute only after approval"
    );
    assert!(
        requests.lock().unwrap().is_empty(),
        "seeded simple edit must stay on the runtime-owned resolver/approval path"
    );
}

#[test]
fn simple_edit_prompt_outside_root_is_rejected_before_approval() {
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    let outside = tmp.path().parent().unwrap().join("outside.txt");

    let (mut rt, requests) =
        make_runtime_in_with_recorded_requests(vec!["must not be used"], tmp.path());
    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: format!(
                "Edit {} replace hello world with hello thunk",
                outside.display()
            ),
        },
    );

    assert!(!has_failed(&events), "must terminate cleanly: {events:?}");
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ApprovalRequired(_))),
        "outside-root seeded simple edit must terminate before approval: {events:?}"
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
            Some(AnswerSource::RuntimeTerminal {
                reason: RuntimeTerminalReason::MutationFailed,
                ..
            })
        ),
        "outside-root seeded simple edit must end as MutationFailed: {answer_source:?}"
    );
    assert!(
        requests.lock().unwrap().is_empty(),
        "outside-root seeded simple edit must terminate before any model generation"
    );
}

#[test]
fn approve_produces_runtime_owned_answer_after_successful_mutation() {
    // After approving a mutation, the runtime must finalize directly without
    // re-entering model generation. The answer is built from the tool output summary.
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("hello.txt");
    std::fs::write(&path, "hello\n").unwrap();
    let path = path.to_string_lossy().into_owned();
    let payload = format!("{}\x00hello\x00world", path);

    // No model responses needed — the runtime owns the answer.
    let mut rt = make_runtime_in(Vec::<&str>::new(), tmp.path());
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

#[test]
fn mutation_turn_with_preparatory_read_still_reaches_edit_file_approval() {
    // Regression test for Fix 2: answer_phase must not fire on mutation-allowed turns
    // after a preparatory read, or the model can never proceed to call edit_file.
    //
    // Sequence: model reads target file first (confirming content), then calls edit_file.
    // Both calls must be allowed — the PostRead answer_phase gate must not intercept.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    let target = tmp.path().join("hello.txt");
    fs::write(&target, "hello root\n").unwrap();

    let read_then_edit = vec![
        "[read_file: hello.txt]",
        "[edit_file]\npath: hello.txt\n---search---\nhello root\n---replace---\nhello runtime\n[/edit_file]",
        "Done.",
    ];
    let mut rt = make_runtime_in(read_then_edit, tmp.path());

    let submit_events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Edit hello.txt and change hello root to hello runtime".into(),
        },
    );

    assert!(
        !has_failed(&submit_events),
        "mutation turn with prior read must not fail: {submit_events:?}"
    );
    assert!(
        submit_events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ApprovalRequired(p) if p.tool_name == "edit_file")),
        "edit_file must reach approval even after a preparatory read: {submit_events:?}"
    );
    assert_eq!(
        fs::read_to_string(&target).unwrap(),
        "hello root\n",
        "file must not be modified before approval"
    );

    let approve_events = collect_events(&mut rt, RuntimeRequest::Approve);
    assert!(
        !has_failed(&approve_events),
        "approve must succeed: {approve_events:?}"
    );
    assert_eq!(
        fs::read_to_string(&target).unwrap(),
        "hello runtime\n",
        "file must be updated after approval"
    );
}
