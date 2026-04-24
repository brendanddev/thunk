use super::*;

#[test]
fn path_scope_narrows_search_to_specified_directory() {
    // Files exist both inside and outside sandbox/cli/.
    // The query scopes to sandbox/cli/.
    // search_code must only receive candidates from sandbox/cli/,
    // so the file outside that directory never becomes a candidate.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox/cli")).unwrap();
    fs::create_dir_all(tmp.path().join("sandbox/models")).unwrap();
    fs::write(
        tmp.path().join("sandbox/cli/handler.py"),
        "if task.status == TaskStatus.PENDING:\n    handle(task)\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("sandbox/models/enums.py"),
        "class TaskStatus(str, Enum):\n    PENDING = \"pending\"\n",
    )
    .unwrap();

    // Model searches (no path in tool call — runtime injects sandbox/cli/).
    // Only sandbox/cli/handler.py matches; sandbox/models/enums.py is outside scope.
    // Model reads handler.py → evidence ready → synthesis admitted.
    let mut rt = make_runtime_in(
        vec![
            "[search_code: TaskStatus]",
            "[read_file: sandbox/cli/handler.py]",
            "TaskStatus is handled in sandbox/cli/handler.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus handled in sandbox/cli/".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");

    let snapshot = rt.messages_snapshot();

    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: search_code ===")),
        "search must have executed"
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
        "scoped search + read must admit synthesis: {answer_source:?}"
    );

    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("TaskStatus is handled in sandbox/cli/handler.py.")
    );
}

#[test]
fn path_scope_after_list_dir_failure_keeps_search_candidates_inside_scope() {
    // Manual regression: "in the sandbox/ folder" must still produce sandbox/
    // as the prompt-derived upper bound after an initial list_dir failure.
    // The model later reads an out-of-scope matched-looking file; that read must
    // not satisfy evidence because it was never a scoped search candidate.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
    fs::create_dir_all(tmp.path().join("src/app")).unwrap();
    fs::write(
        tmp.path().join("sandbox").join("database.yaml"),
        "database:\n  url: sqlite:///sandbox.db\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("src/app").join("session.rs"),
        "/// Owns the active database handle and current session ID.\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[list_dir: .]",
            "[search_code: database]",
            "[read_file: src/app/session.rs]",
            "The database is configured in src/app/session.rs.",
            "[read_file: sandbox/database.yaml]",
            "The database is configured in sandbox/database.yaml.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Find where database is configured in the sandbox/ folder".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot.iter().any(|m| {
            m.content.contains("=== tool_error: list_dir ===")
                && m.content.contains("require search_code")
        }),
        "list_dir before scoped search must be blocked"
    );

    let search_result = snapshot
        .iter()
        .find(|m| m.content.contains("=== tool_result: search_code ==="))
        .map(|m| m.content.as_str())
        .unwrap_or("");
    assert!(
        search_result.contains("sandbox/database.yaml"),
        "scoped search must include the sandbox config candidate: {search_result}"
    );
    assert!(
        !search_result.contains("src/app/session.rs"),
        "scoped search must not include out-of-scope candidates: {search_result}"
    );

    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("The database is configured in sandbox/database.yaml.")
    );
}

#[test]
fn scope_upper_bound_clamps_broader_model_path() {
    // Verifies end-to-end that when a prompt scope is extracted and the search
    // produces results only within the scope, synthesis is admitted (ToolAssisted).
    // The injection (9.1.2 None arm) and the clamping guard (9.1.4 Some arm) both
    // live in the same match block; this test exercises the combined path.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::create_dir_all(tmp.path().join("models")).unwrap();
    fs::write(
        tmp.path().join("services").join("task_service.py"),
        "if task.status == TaskStatus.TODO:\n    pass\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("models").join("enums.py"),
        "class TaskStatus(str, Enum):\n    TODO = \"todo\"\n",
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
            text: "Where is TaskStatus used in services/".into(),
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
        "scoped search must admit synthesis: {answer_source:?}"
    );
    let snapshot = rt.messages_snapshot();
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("TaskStatus is used in services/task_service.py.")
    );
}

#[test]
fn no_scope_search_behavior_unchanged() {
    // Prompt has no path scope (no "in dir/" pattern).
    // Runtime must not inject or clamp — standard search behavior applies.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::create_dir_all(tmp.path().join("models")).unwrap();
    fs::write(
        tmp.path().join("services").join("task_service.py"),
        "if task.status == TaskStatus.TODO:\n    pass\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("models").join("enums.py"),
        "class TaskStatus(str, Enum):\n    TODO = \"todo\"\n",
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
            text: "Where is TaskStatus used?".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "no-scope turn must not fail: {events:?}"
    );

    let snapshot = rt.messages_snapshot();
    let search_result = snapshot
        .iter()
        .find(|m| m.content.contains("=== tool_result: search_code ==="))
        .map(|m| m.content.as_str())
        .unwrap_or("");
    assert!(
        search_result.contains("services/task_service.py"),
        "unscoped search must include services/task_service.py: {search_result}"
    );
    assert!(
        search_result.contains("models/enums.py"),
        "unscoped search must include models/enums.py (no clamping without scope): {search_result}"
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
        "unscoped search + read must admit synthesis: {answer_source:?}"
    );
}

#[test]
fn scope_upper_bound_forced_broader_path_clamped_end_to_end() {
    // Forced failure-path validation from the spec:
    // Prompt: "Find where logging is initialized in sandbox/services/"
    // The scope extracts to sandbox/services/.
    // Model issues search without path (codec limitation — path always None),
    // runtime injects sandbox/services/ → only in-scope files become candidates.
    // Out-of-scope src/ files are never candidates.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::write(
        tmp.path().join("sandbox/services").join("logger.py"),
        "def initialize_logging():\n    logging.basicConfig()\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("src").join("logger.py"),
        "def initialize_logging():\n    setup_logger()\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: logging]",
            "[read_file: sandbox/services/logger.py]",
            "Logging is initialized in sandbox/services/logger.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Find where logging is initialized in sandbox/services/".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");

    let snapshot = rt.messages_snapshot();
    let search_result = snapshot
        .iter()
        .find(|m| m.content.contains("=== tool_result: search_code ==="))
        .map(|m| m.content.as_str())
        .unwrap_or("");

    assert!(
        search_result.contains("sandbox/services/logger.py"),
        "scoped search must include in-scope candidate: {search_result}"
    );
    assert!(
        !search_result.contains("src/logger.py"),
        "scoped search must exclude out-of-scope src/ candidate: {search_result}"
    );

    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("Logging is initialized in sandbox/services/logger.py.")
    );
}

#[test]
fn scope_upper_bound_clamped_to_cli_not_sandbox() {
    // Forced failure-path validation case 2:
    // Prompt: "Where is TaskStatus used in sandbox/cli/"
    // Model would search sandbox/ (broader) — clamp must restrict to sandbox/cli/.
    // Since codec produces path: None, injection fires and restricts to sandbox/cli/.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox/cli")).unwrap();
    fs::create_dir_all(tmp.path().join("sandbox/models")).unwrap();
    fs::write(
        tmp.path().join("sandbox/cli").join("handler.py"),
        "if task.status == TaskStatus.PENDING:\n    handle(task)\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("sandbox/models").join("enums.py"),
        "class TaskStatus(str, Enum):\n    PENDING = \"pending\"\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: TaskStatus]",
            "[read_file: sandbox/cli/handler.py]",
            "TaskStatus is used in sandbox/cli/handler.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus used in sandbox/cli/".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");

    let snapshot = rt.messages_snapshot();
    let search_result = snapshot
        .iter()
        .find(|m| m.content.contains("=== tool_result: search_code ==="))
        .map(|m| m.content.as_str())
        .unwrap_or("");

    assert!(
        search_result.contains("sandbox/cli/handler.py"),
        "scoped search must include in-scope candidate: {search_result}"
    );
    assert!(
        !search_result.contains("sandbox/models/enums.py"),
        "scoped search must exclude out-of-scope sandbox/models/ candidate: {search_result}"
    );
}
