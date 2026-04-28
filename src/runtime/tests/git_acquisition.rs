use super::*;
use crate::runtime::types::RuntimeTerminalReason;

// Git tool / anchor isolation

#[test]
fn git_status_does_not_update_anchors() {
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    init_git_repo(tmp.path());
    let mut rt = make_runtime_in(vec!["[git_status]", "Working tree checked."], tmp.path());

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Show git status".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "git_status turn must not fail: {events:?}"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { name } if name == "read_file")),
        "git_status must not dispatch read_file"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { name } if name == "search_code")),
        "git_status must not dispatch search_code"
    );
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: git_status ===")),
        "git_status result must be injected as a normal tool result"
    );
}

#[test]
fn git_diff_does_not_update_anchors() {
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    init_git_repo(tmp.path());
    let mut rt = make_runtime_in(vec!["[git_diff]", "Working tree diff checked."], tmp.path());

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Show git diff".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "git_diff turn must not fail: {events:?}"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { name } if name == "read_file")),
        "git_diff must not dispatch read_file"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { name } if name == "search_code")),
        "git_diff must not dispatch search_code"
    );
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: git_diff ===")),
        "git_diff result must be injected as a normal tool result"
    );
}

#[test]
fn git_log_does_not_update_anchors() {
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    init_git_repo(tmp.path());
    let mut rt = make_runtime_in(vec!["[git_log]", "Recent commits checked."], tmp.path());

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Show git log".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "git_log turn must not fail: {events:?}"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { name } if name == "read_file")),
        "git_log must not dispatch read_file"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { name } if name == "search_code")),
        "git_log must not dispatch search_code"
    );
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: git_log ===")),
        "git_log result must be injected as a normal tool result"
    );
}

// Investigation evidence

#[test]
fn git_status_does_not_satisfy_investigation_evidence() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    init_git_repo(tmp.path());
    fs::write(
        tmp.path().join("a.rs"),
        "fn use_task_status() { TaskStatus; }\n",
    )
    .unwrap();
    let mut rt = make_runtime_in(
        vec![
            "[git_status]",
            "[search_code: TaskStatus]",
            "[read_file: a.rs]",
            "TaskStatus is used in a.rs.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus used?".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "turn must recover through search/read: {events:?}"
    );
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_error: git_status ===")
                && m.content.contains("retrieval tools only")),
        "git_status should be rejected before dispatch on RetrievalFirst turns"
    );
    assert!(
        snapshot
            .iter()
            .all(|m| !m.content.contains("=== tool_result: git_status ===")),
        "disallowed git_status must not execute"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: search_code ===")),
        "model must still search after git_status"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: read_file ===")),
        "model must still read matched code evidence"
    );
}

#[test]
fn git_diff_does_not_satisfy_investigation_evidence() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    init_git_repo(tmp.path());
    fs::write(
        tmp.path().join("a.rs"),
        "fn use_task_status() { TaskStatus; }\n",
    )
    .unwrap();
    let mut rt = make_runtime_in(
        vec![
            "[git_diff]",
            "[search_code: TaskStatus]",
            "[read_file: a.rs]",
            "TaskStatus is used in a.rs.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus used?".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "turn must recover through search/read: {events:?}"
    );
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_error: git_diff ===")
                && m.content.contains("retrieval tools only")),
        "git_diff should be rejected before dispatch on RetrievalFirst turns"
    );
    assert!(
        snapshot
            .iter()
            .all(|m| !m.content.contains("=== tool_result: git_diff ===")),
        "disallowed git_diff must not execute"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: search_code ===")),
        "model must still search after git_diff"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: read_file ===")),
        "model must still read matched code evidence"
    );
}

#[test]
fn git_log_does_not_satisfy_investigation_evidence() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    init_git_repo(tmp.path());
    fs::write(
        tmp.path().join("a.rs"),
        "fn use_task_status() { TaskStatus; }\n",
    )
    .unwrap();
    let mut rt = make_runtime_in(
        vec![
            "[git_log]",
            "[search_code: TaskStatus]",
            "[read_file: a.rs]",
            "TaskStatus is used in a.rs.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus used?".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "turn must recover through search/read: {events:?}"
    );
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_error: git_log ===")
                && m.content.contains("retrieval tools only")),
        "git_log should be rejected before dispatch on RetrievalFirst turns"
    );
    assert!(
        snapshot
            .iter()
            .all(|m| !m.content.contains("=== tool_result: git_log ===")),
        "disallowed git_log must not execute"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: search_code ===")),
        "model must still search after git_log"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: read_file ===")),
        "model must still read matched code evidence"
    );
}

// RetrievalFirst surface policy

#[test]
fn disallowed_git_tool_does_not_update_anchors() {
    let mut rt = make_runtime(vec!["[git_status]", "No git tool was needed."]);

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "display the structure".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "disallowed git tool should be surfaced as tool error without failing: {events:?}"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { name } if name == "read_file")),
        "rejected git tool must not dispatch read_file"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { name } if name == "search_code")),
        "rejected git tool must not dispatch search_code"
    );
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_error: git_status ===")
                && m.content.contains("retrieval tools only")),
        "git_status must be rejected before dispatch"
    );
    assert!(
        snapshot
            .iter()
            .all(|m| !m.content.contains("=== tool_result: git_status ===")),
        "rejected git_status must not execute"
    );
}

#[test]
fn first_disallowed_git_tool_on_retrieval_first_turn_gets_surface_correction() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("a.rs"), "fn render_git_status() {}\n").unwrap();
    let mut rt = make_runtime_in(
        vec![
            "[git_status]",
            "[search_code: git_status]",
            "[read_file: a.rs]",
            "git status is rendered in a.rs.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "where is git status rendered".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "turn should recover after first policy correction: {events:?}"
    );
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_error: git_status ===")
                && m.content.contains("retrieval tools only")),
        "first disallowed git tool must get retrieval surface correction"
    );
    assert!(
        snapshot
            .iter()
            .all(|m| !m.content.contains("=== tool_result: git_status ===")),
        "disallowed git_status must not execute"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: search_code ===")),
        "model should recover to retrieval tools"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: read_file ===")),
        "model should still read grounded file evidence"
    );
}

#[test]
fn second_disallowed_git_tool_on_retrieval_first_turn_terminates_policy_violation() {
    let mut rt = make_runtime(vec!["[git_status]", "[git_diff]"]);

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "where is git status rendered".into(),
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
                reason: RuntimeTerminalReason::RepeatedDisallowedTool,
                ..
            })
        ),
        "second disallowed git tool must terminate as policy violation: {events:?}"
    );
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot.iter().any(|m| m
            .content
            .contains("repeated unavailable tool use for this retrieval-first turn")),
        "terminal policy error must be surfaced"
    );
    assert!(
        snapshot
            .iter()
            .all(|m| !m.content.contains("=== tool_result: git_")),
        "disallowed git tools must not execute"
    );
}

// GitReadOnly surface policy

#[test]
fn git_read_only_surface_rejects_search_code_but_allows_git_tool() {
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    init_git_repo(tmp.path());
    let mut rt = make_runtime_in(
        vec![
            "[search_code: TaskStatus]",
            "[git_status]",
            "Working tree checked.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "show git status".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "GitReadOnly turn should recover to allowed git tool: {events:?}"
    );
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_error: search_code ===")
                && m.content.contains("Git read-only tools only")),
        "search_code must be rejected on GitReadOnly turns"
    );
    assert!(
        snapshot
            .iter()
            .all(|m| !m.content.contains("=== tool_result: search_code ===")),
        "rejected search_code must not execute"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: git_status ===")),
        "git_status should still dispatch on GitReadOnly turns"
    );
}

#[test]
fn first_disallowed_search_on_git_read_only_turn_gets_surface_correction() {
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    init_git_repo(tmp.path());
    let mut rt = make_runtime_in(
        vec![
            "[search_code: status]",
            "[git_status]",
            "Working tree checked.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "show git status".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "turn should recover after first GitReadOnly policy correction: {events:?}"
    );
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_error: search_code ===")
                && m.content.contains("Git read-only tools only")),
        "first disallowed retrieval tool must get Git surface correction"
    );
    assert!(
        snapshot
            .iter()
            .all(|m| !m.content.contains("=== tool_result: search_code ===")),
        "disallowed search_code must not execute"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: git_status ===")),
        "model should recover to allowed git tool"
    );
}

#[test]
fn git_read_only_first_generation_multi_tool_acquisition_is_allowed() {
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    init_git_repo(tmp.path());
    let mut rt = make_runtime_in(
        vec![
            "[git_status]\n[git_diff]",
            "This response should not be consumed.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "show git status".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "same-generation Git acquisition must not fail: {events:?}"
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
            Some(AnswerSource::ToolAssisted { rounds: 1 })
        ),
        "same-generation Git tools should consume one acquisition round: {answer_source:?}"
    );
    let snapshot = rt.messages_snapshot();
    let all_user = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::User)
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        all_user.contains("=== tool_result: git_status ==="),
        "git_status should execute in the acquisition round"
    );
    assert!(
        all_user.contains("=== tool_result: git_diff ==="),
        "git_diff should execute in the same acquisition round"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str())
        .unwrap_or("");
    assert!(
        last_assistant.contains("Git read-only result:"),
        "runtime should produce the final Git answer"
    );
    assert!(
        last_assistant.contains("git_status:"),
        "runtime answer should include git_status output"
    );
    assert!(
        last_assistant.contains("git_diff:"),
        "runtime answer should include git_diff output"
    );
    assert!(
        last_assistant.contains("working tree clean"),
        "runtime answer should reuse rendered git_status output"
    );
    assert!(
        last_assistant.contains("No unstaged changes."),
        "runtime answer should reuse rendered git_diff output"
    );
    assert!(
        snapshot
            .iter()
            .all(|m| !m.content.contains("This response should not be consumed.")),
        "runtime must not request model synthesis after Git acquisition"
    );
}

#[test]
fn git_read_only_runtime_answer_prevents_second_generation_git_tool() {
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    init_git_repo(tmp.path());
    let mut rt = make_runtime_in(vec!["[git_status]", "[git_diff]"], tmp.path());

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "show git status".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "GitReadOnly turn should finish immediately after acquisition: {events:?}"
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
            Some(AnswerSource::ToolAssisted { rounds: 1 })
        ),
        "runtime-produced Git answer should remain a successful tool-assisted answer: {answer_source:?}"
    );
    let snapshot = rt.messages_snapshot();
    let all_user = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::User)
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        all_user.matches("=== tool_result: git_status ===").count(),
        1,
        "first Git acquisition tool should dispatch exactly once"
    );
    assert_eq!(
        all_user.matches("=== tool_result: git_diff ===").count(),
        0,
        "second backend response must not be consumed after acquisition completes"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert!(
        last_assistant
            .is_some_and(|answer| answer.contains("git_status:") && !answer.contains("git_diff:")),
        "runtime answer should include only the completed acquisition output"
    );
}

#[test]
fn failed_git_acquisition_finishes_without_retrieval_drift() {
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    let mut rt = make_runtime_in(vec!["[git_diff]", "[search_code: git_diff]"], tmp.path());

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "show git diff".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "failed Git acquisition should finish through runtime synthesis: {events:?}"
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
            Some(AnswerSource::ToolAssisted { rounds: 1 })
        ),
        "failed Git acquisition still completes as a bounded tool-assisted answer: {answer_source:?}"
    );
    let snapshot = rt.messages_snapshot();
    let all_user = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::User)
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        all_user.contains("=== tool_error: git_diff ==="),
        "Git tool failure should be injected as the acquisition result"
    );
    assert!(
        !all_user.contains("=== tool_error: search_code ==="),
        "second backend response must not be consumed after failed Git acquisition"
    );
    assert!(
        !all_user.contains("=== tool_result: search_code ==="),
        "retrieval drift must not dispatch search_code"
    );
    assert!(
        !all_user.contains("Git read-only tools only"),
        "no post-acquisition surface correction should be needed"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert!(
        last_assistant.is_some_and(|answer| answer.contains("git_diff:")
            && answer.contains("git_diff failed: not a Git repository")),
        "runtime answer should include the Git tool error"
    );
}

#[test]
fn second_disallowed_retrieval_tool_on_git_read_only_turn_terminates_policy_violation() {
    let mut rt = make_runtime(vec!["[search_code: status]", "[read_file: src/main.rs]"]);

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "show git status".into(),
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
                reason: RuntimeTerminalReason::RepeatedDisallowedTool,
                ..
            })
        ),
        "second disallowed retrieval tool must terminate as policy violation: {events:?}"
    );
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot.iter().any(|m| m
            .content
            .contains("repeated unavailable tool use for this Git read-only turn")),
        "terminal policy error must be surfaced"
    );
    assert!(
        snapshot
            .iter()
            .all(|m| !m.content.contains("=== tool_result: search_code ===")
                && !m.content.contains("=== tool_result: read_file ===")),
        "disallowed retrieval tools must not execute"
    );
}

#[test]
fn allowed_tool_execution_failure_does_not_count_as_disallowed_tool_attempt() {
    let mut rt = make_runtime(vec!["[read_file: missing.rs]"]);

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read missing.rs".into(),
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
                reason: RuntimeTerminalReason::ReadFileFailed,
                ..
            })
        ),
        "allowed read failure must remain a read failure, not policy violation: {events:?}"
    );
    assert!(
        !matches!(
            answer_source,
            Some(AnswerSource::RuntimeTerminal {
                reason: RuntimeTerminalReason::RepeatedDisallowedTool,
                ..
            })
        ),
        "tool execution failures must not trigger surface-policy terminal reason"
    );
}
