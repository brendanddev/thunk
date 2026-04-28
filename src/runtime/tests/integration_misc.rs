use super::*;

#[test]
fn list_dir_before_search_is_blocked_for_filtered_investigation() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(
        tmp.path().join("task_service.py"),
        "def completed_tasks(tasks):\n    filtered = [task for task in tasks if task.done]\n    return filtered\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[list_dir: .]",
            "[search_code: filtered]",
            "[read_file: task_service.py]",
            "Completed tasks are filtered in task_service.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where are completed tasks filtered?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot.iter().any(|m| {
            m.content.contains("=== tool_error: list_dir ===")
                && m.content.contains("require search_code")
        }),
        "list_dir before search must be blocked on investigation-required turns"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: search_code ===")),
        "model must recover by searching"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: read_file ===")),
        "model must read a matched file before answering"
    );
}

#[test]
fn mutating_tool_is_blocked_on_informational_turn() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("engine.rs"), "fn write_file() {}\n").unwrap();
    let blocked_path = tmp.path().join("should_not_exist.txt");

    let mut rt = make_runtime_in(
        vec![
            "[write_file]\npath: should_not_exist.txt\n---content---\nnope\n[/write_file]",
            "[search_code: write_file]",
            "[read_file: engine.rs]",
            "write_file is implemented in engine.rs.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is write_file implemented?".into(),
        },
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ApprovalRequired(_))),
        "read-only informational turn must not create a pending mutation"
    );
    assert!(
        !blocked_path.exists(),
        "blocked write_file must not create a file"
    );

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot.iter().any(|m| {
            m.content.contains("=== tool_error: write_file ===")
                && m.content.contains("mutating tools are not allowed")
        }),
        "blocked mutation must be surfaced as a tool error"
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
        "turn should continue with allowed read-only tools: {answer_source:?}"
    );
}

#[test]
fn initialization_lookup_non_initialization_read_triggers_recovery() {
    // Initialization lookup: two source candidates, but only one matched line
    // contains an exact initialization term. Reading the other candidate first
    // must trigger one bounded recovery to the initialization candidate.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("services").join("logging_factory.py"),
        "logger = logging.getLogger(__name__)\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("services").join("logging_setup.py"),
        "def initialize_logging():\n    logging.basicConfig(level=logging.INFO)\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: logging]",
            "[read_file: services/logging_factory.py]",
            "[read_file: services/logging_setup.py]",
            "Logging is initialized in services/logging_setup.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Find where logging is initialized".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "initialization recovery + initialization read must admit synthesis: {answer_source:?}"
    );

    let snapshot = rt.messages_snapshot();
    let expected_recovery_path = "services/logging_setup.py";
    assert!(
        snapshot.iter().any(|m| {
            m.content.contains("This is an initialization lookup")
                && m.content
                    .contains(&format!("[read_file: {expected_recovery_path}]"))
        }),
        "runtime must inject bounded initialization recovery"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("Logging is initialized in services/logging_setup.py.")
    );
}

#[test]
fn initialization_lookup_no_initialization_candidates_degrades_cleanly() {
    // Initialization lookup triggered, but no matched line contains an exact
    // initialization term. Gate 3 does not fire — existing candidate-read
    // behavior is preserved.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("services").join("logging_factory.py"),
        "logger = logging.getLogger(__name__)\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: logging]",
            "[read_file: services/logging_factory.py]",
            "Logging is handled in services/logging_factory.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Find where logging is initialized".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "initialization lookup with no initialization candidates must degrade: {answer_source:?}"
    );
    let snapshot = rt.messages_snapshot();
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("Logging is handled in services/logging_factory.py.")
    );
}

#[test]
fn create_lookup_does_not_affect_usage_lookup_regression() {
    // A usage-lookup prompt with create terms must remain UsageLookup (higher priority).
    // The create gate must not activate for UsageLookup turns.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::create_dir_all(tmp.path().join("models")).unwrap();
    fs::write(
        tmp.path().join("services").join("task_service.py"),
        "if task.status == TaskStatus.DONE:\n    pass\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("models").join("enums.py"),
        "class TaskStatus(Enum):\n    DONE = 'done'\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: TaskStatus]",
            "[read_file: services/task_service.py]",
            "TaskStatus is used in services/task_service.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            // "used" triggers UsageLookup; "created" present but must not win.
            text: "Where is TaskStatus used and created?".into(),
        },
    );

    assert!(!has_failed(&events), "regression: {events:?}");
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "UsageLookup must not be disrupted by create terms: {answer_source:?}"
    );
}
