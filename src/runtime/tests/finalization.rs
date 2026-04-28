use super::*;
use crate::runtime::types::RuntimeTerminalReason;

#[test]
fn definition_lookup_extra_tool_after_evidence_ready_enters_answer_only_mode() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox/models")).unwrap();
    fs::create_dir_all(tmp.path().join("sandbox/cli")).unwrap();
    fs::write(
        tmp.path().join("sandbox/models/enums.py"),
        "class TaskStatus(str, Enum):\n    TODO = \"todo\"\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("sandbox/cli/commands.py"),
        "def show_commands():\n    return []\n",
    )
    .unwrap();

    let final_answer = "TaskStatus is defined in sandbox/models/enums.py.";
    let mut rt = make_runtime_in(
        vec![
            "[search_code: TaskStatus]",
            "[read_file: sandbox/models/enums.py]",
            "[read_file: sandbox/cli/commands.py]",
            final_answer,
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus defined in sandbox/".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "extra post-evidence tool call must not fail the turn: {events:?}"
    );
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "model should synthesize after answer-only correction: {answer_source:?}"
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
        "extra read_file after sufficient evidence must not dispatch"
    );
    assert!(
        all_user.contains("Evidence is already ready"),
        "runtime must inject answer-only correction after evidence is ready"
    );
    assert!(
        !events.iter().any(|e| matches!(
            e,
            RuntimeEvent::Failed { message }
                if message == "Model kept searching after the search budget was closed."
        )),
        "post-evidence tool use must not reach the closed-search-budget failure path"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(last_assistant, Some(final_answer));
}

#[test]
fn initialization_recovery_extra_tool_after_evidence_ready_enters_answer_only_mode() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
    fs::write(
        tmp.path().join("sandbox/services/logging_usage.py"),
        "def emit_log(logger):\n    logger.info(\"logging event\")\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("sandbox/services/logging_init.py"),
        "def initialize_logging():\n    logging.basicConfig(level=\"INFO\")\n",
    )
    .unwrap();

    let final_answer = "Logging is initialized in sandbox/services/logging_init.py.";
    let mut rt = make_runtime_in(
        vec![
            "[search_code: logging]",
            "[read_file: sandbox/services/logging_usage.py]",
            "[read_file: sandbox/services/logging_init.py]",
            "[read_file: sandbox/services/logging_usage.py]",
            final_answer,
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Find where logging is initialized in sandbox/".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "post-recovery evidence-ready tool call must not fail the turn: {events:?}"
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
        2,
        "only the wrong first read and accepted recovery read should dispatch"
    );
    assert!(
        all_user.contains("This is an initialization lookup"),
        "initialization recovery must still be issued before evidence is ready"
    );
    assert!(
        all_user.contains("Evidence is already ready"),
        "runtime must switch to answer-only mode after accepted recovery evidence"
    );
    assert!(
        !all_user
            .contains("=== tool_result: read_file ===\npath: sandbox/services/logging_usage.py\n")
            || all_user
                .matches(
                    "=== tool_result: read_file ===\npath: sandbox/services/logging_usage.py\n"
                )
                .count()
                == 1,
        "extra post-evidence read of the usage file must not dispatch"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(last_assistant, Some(final_answer));
}

#[test]
fn repeated_post_evidence_tool_use_terminates_before_search_budget_failure() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox/models")).unwrap();
    fs::write(
        tmp.path().join("sandbox/models/enums.py"),
        "class TaskStatus(str, Enum):\n    TODO = \"todo\"\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: TaskStatus]",
            "[read_file: sandbox/models/enums.py]",
            "[search_code: TaskStatus]",
            "[search_code: TaskStatus]",
            "This response should not be consumed.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus defined in sandbox/".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "repeated post-evidence tools must terminate cleanly: {events:?}"
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
                reason: RuntimeTerminalReason::RepeatedToolAfterEvidenceReady,
                ..
            })
        ),
        "second post-evidence tool attempt must use dedicated terminal reason: {answer_source:?}"
    );

    let snapshot = rt.messages_snapshot();
    let all_user: String = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::User)
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        all_user.matches("=== tool_result: search_code ===").count(),
        1,
        "post-evidence search_code attempts must not dispatch"
    );
    assert!(
        all_user.contains("Evidence is already ready"),
        "first post-evidence tool attempt must receive answer-only correction"
    );
    assert!(
        all_user.matches("Search returned matches").count() == 1,
        "post-evidence tool attempts must not add another search-budget-closed correction"
    );
    assert!(
        !events.iter().any(|e| matches!(
            e,
            RuntimeEvent::Failed { message }
                if message == "Model kept searching after the search budget was closed."
        )),
        "post-evidence tool attempts must not fall into closed-search-budget failure"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert!(
        matches!(last_assistant, Some(s) if s.contains("sufficient file evidence was already read")),
        "last assistant must be the repeated-post-evidence-tool terminal: {last_assistant:?}"
    );
}

// Phase 11.2.1 — Runtime Turn Finalization (Stage 1)

#[test]
fn direct_read_blocks_post_read_tool_call_with_answer_phase_correction() {
    // Non-investigation direct read: after read_file succeeds, answer_phase = true.
    // A subsequent tool call must be blocked. The model then produces the final answer.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("foo.rs"), "fn foo() {}\n").unwrap();

    let final_answer = "foo.rs defines a single function.";
    let mut rt = make_runtime_in(
        vec!["[search_code: foo]", final_answer],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read foo.rs".into(),
        },
    );

    assert!(!has_failed(&events), "must not fail: {events:?}");

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
        "read_file must have executed exactly once"
    );
    assert_eq!(
        all_user.matches("=== tool_result: search_code ===").count(),
        0,
        "search_code after read must be blocked by answer_phase gate"
    );
    assert!(
        all_user.contains("[runtime:correction]") && all_user.contains("already read this turn"),
        "answer_phase correction must be injected after blocked search"
    );

    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(last_assistant, Some(final_answer));
}

#[test]
fn general_retrieval_blocks_post_read_search_with_answer_phase_correction() {
    // Non-investigation search + read: after read succeeds, answer_phase = true.
    // A further search attempt must be blocked. The model then answers.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::write(
        tmp.path().join("src/main.rs"),
        "fn main() { println!(\"hello\"); }\n",
    )
    .unwrap();

    let final_answer = "The project entry point is src/main.rs.";
    let mut rt = make_runtime_in(
        vec![
            "[search_code: main]",
            "[read_file: src/main.rs]",
            "[search_code: main]",
            final_answer,
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "list files in src/".into(),
        },
    );

    assert!(!has_failed(&events), "must not fail: {events:?}");

    let snapshot = rt.messages_snapshot();
    let all_user: String = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::User)
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    assert_eq!(
        all_user.matches("=== tool_result: search_code ===").count(),
        1,
        "only the first search_code (before any read) must dispatch"
    );
    assert_eq!(
        all_user.matches("=== tool_result: read_file ===").count(),
        1,
        "read_file must have executed once"
    );
    assert!(
        all_user.contains("[runtime:correction]") && all_user.contains("already read this turn"),
        "answer_phase correction must be injected after post-read search attempt"
    );

    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(last_assistant, Some(final_answer));
}

#[test]
fn repeated_tool_after_answer_phase_terminates_before_search_budget_failure() {
    // Non-investigation: after read, answer_phase = true.
    // First post-read tool call → answer_phase correction.
    // Second post-read tool call → RepeatedToolAfterAnswerPhase terminal.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("bar.rs"), "fn bar() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[read_file: bar.rs]",
            "[search_code: bar]",
            "[search_code: bar]",
            "This response must not be consumed.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read bar.rs".into(),
        },
    );

    assert!(!has_failed(&events), "must terminate cleanly: {events:?}");

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
                reason: RuntimeTerminalReason::RepeatedToolAfterAnswerPhase,
                ..
            })
        ),
        "second post-read tool attempt must use RepeatedToolAfterAnswerPhase: {answer_source:?}"
    );

    let snapshot = rt.messages_snapshot();
    let all_user: String = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::User)
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    assert_eq!(
        all_user.matches("=== tool_result: search_code ===").count(),
        0,
        "post-read search_code attempts must not dispatch"
    );
    assert!(
        all_user.contains("[runtime:correction]") && all_user.contains("already read this turn"),
        "first post-read tool attempt must receive answer_phase correction"
    );

    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert!(
        matches!(last_assistant, Some(s) if s.contains("model kept calling tools after the file was already read")),
        "last assistant must be the repeated-answer-phase terminal: {last_assistant:?}"
    );
}
