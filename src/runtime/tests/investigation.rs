use super::*;
use crate::runtime::types::RuntimeTerminalReason;

#[test]
fn premature_investigation_answer_is_not_admitted() {
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    let mut rt = make_runtime_in(
        vec!["run_turns drives the loop.", "It still drives the loop."],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "What does run_turns do?".into(),
        },
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
                reason: RuntimeTerminalReason::InsufficientEvidence,
                ..
            })
        ),
        "premature direct answers must not be admitted: {answer_source:?}"
    );

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content == "run_turns drives the loop."),
        "pre-evidence prose is kept in the trace"
    );
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("No final answer was accepted")),
        "runtime terminal must explain that no grounded answer was accepted"
    );
}

#[test]
fn search_results_require_matched_read_before_synthesis() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: run_turns]",
            "run_turns is in engine.rs.",
            "It is definitely in engine.rs.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "What does run_turns do?".into(),
        },
    );

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot.iter().any(|m| {
            m.content.starts_with("[runtime:correction]")
                && m.content.contains("no matched file has been read")
        }),
        "runtime must require read_file after non-empty search"
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
                reason: RuntimeTerminalReason::InsufficientEvidence,
                ..
            })
        ),
        "unread search results must not admit synthesis: {answer_source:?}"
    );
}

#[test]
fn read_before_answering_correction_discards_premature_synthesis() {
    // After search returns matches, the model synthesizes without reading (premature).
    // The READ_BEFORE_ANSWERING correction must fire AND discard the premature synthesis
    // from context before injecting the correction message.
    // Verified by checking: no premature synthesis message remains in the conversation.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: run_turns]",
            "run_turns is the main driver.",
            "[read_file: engine.rs]",
            "run_turns drives the main event loop.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "What does run_turns do?".into(),
        },
    );

    let snapshot = rt.messages_snapshot();

    assert!(
        snapshot.iter().any(|m| {
            m.content.starts_with("[runtime:correction]")
                && m.content.contains("no matched file has been read")
        }),
        "READ_BEFORE_ANSWERING correction must be injected: {snapshot:?}"
    );

    assert!(
        !snapshot
            .iter()
            .any(|m| m.content == "run_turns is the main driver."),
        "premature synthesis must be discarded from context before correction"
    );

    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("run_turns drives the main event loop."),
        "grounded synthesis must be the last assistant message"
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
        "turn must complete as ToolAssisted after evidence-ready synthesis: {answer_source:?}"
    );
}

#[test]
fn read_must_come_from_current_search_results() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();
    fs::write(tmp.path().join("notes.rs"), "fn unrelated() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: run_turns]",
            "[read_file: notes.rs]",
            "notes.rs explains it.",
            "Still enough.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "What does run_turns do?".into(),
        },
    );

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: read_file ===")),
        "unmatched read still executes as normal context"
    );
    assert!(
        snapshot.iter().any(|m| {
            m.content.starts_with("[runtime:correction]")
                && m.content.contains("no matched file has been read")
        }),
        "unmatched read must not satisfy evidence readiness"
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
                reason: RuntimeTerminalReason::InsufficientEvidence,
                ..
            })
        ),
        "read outside search candidates must not admit synthesis: {answer_source:?}"
    );
}

#[test]
fn usage_lookup_definition_only_read_does_not_satisfy_evidence_when_usage_candidates_exist() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("models")).unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("models").join("enums.py"),
        "from enum import Enum\n\nclass TaskStatus(str, Enum):\n    TODO = \"todo\"\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("services").join("task_service.py"),
        "from models.enums import TaskStatus\n\nif task.status == TaskStatus.TODO:\n    pass\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: TaskStatus]",
            "[read_file: models/enums.py]",
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

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot.iter().any(|m| {
            m.content
                .contains("[runtime:correction] This is a usage lookup")
                && m.content.contains("services/task_service.py]")
        }),
        "definition-only read must trigger a targeted usage-file recovery correction"
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
        "turn should complete after the usage file is read: {answer_source:?}"
    );
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
fn usage_lookup_all_definition_candidates_fallback_allows_definition_read() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("models")).unwrap();
    fs::write(
        tmp.path().join("models").join("enums.py"),
        "from enum import Enum\n\nclass TaskStatus(str, Enum):\n    TODO = \"todo\"\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: TaskStatus]",
            "[read_file: models/enums.py]",
            "Only the TaskStatus definition was found.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus used?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let snapshot = rt.messages_snapshot();
    assert!(
        !snapshot
            .iter()
            .any(|m| m.content.starts_with("[runtime:correction]")),
        "definition-only fallback should not inject a correction when no usage candidates exist"
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
        "definition-only fallback must admit synthesis: {answer_source:?}"
    );
}

#[test]
fn usage_lookup_mixed_definition_and_usage_file_is_useful_immediately() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("models")).unwrap();
    fs::write(
        tmp.path().join("models").join("task_status.py"),
        "class TaskStatus:\n    TODO = \"todo\"\n\nDEFAULT_STATUS = TaskStatus.TODO\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: TaskStatus]",
            "[read_file: models/task_status.py]",
            "TaskStatus is defined and used in models/task_status.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus used?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let snapshot = rt.messages_snapshot();
    assert!(
        !snapshot
            .iter()
            .any(|m| m.content.starts_with("[runtime:correction]")),
        "mixed definition+usage file should satisfy usage evidence immediately"
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
        "mixed candidate read must admit synthesis: {answer_source:?}"
    );
}

#[test]
fn definition_lookup_accepts_definition_read_when_usage_candidates_exist() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("models")).unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("models").join("enums.py"),
        "from enum import Enum\n\nclass TaskStatus(str, Enum):\n    TODO = \"todo\"\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("services").join("task_service.py"),
        "from models.enums import TaskStatus\n\nif task.status == TaskStatus.TODO:\n    pass\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: TaskStatus]",
            "[read_file: models/enums.py]",
            "TaskStatus is defined in models/enums.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus defined?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let snapshot = rt.messages_snapshot();
    assert!(
        !snapshot.iter().any(|m| {
            m.content.starts_with("[runtime:correction]")
                && m.content.contains("no matched file has been read")
        }),
        "definition lookup must accept the definition read as useful evidence"
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
        "definition lookup should complete after reading the definition: {answer_source:?}"
    );
}

#[test]
fn mixed_prose_and_tool_call_does_not_admit_prose() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();
    let mut rt = make_runtime_in(
        vec![
            "It is probably in engine.rs.\n[search_code: run_turns]",
            "[read_file: engine.rs]",
            "run_turns is in engine.rs.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is run_turns defined?".into(),
        },
    );

    let answer_ready_count = events
        .iter()
        .filter(|e| matches!(e, RuntimeEvent::AnswerReady(_)))
        .count();
    assert_eq!(
        answer_ready_count, 1,
        "only final synthesis may be admitted"
    );

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("It is probably in engine.rs.")),
        "mixed pre-evidence prose remains trace context but is not admitted"
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
        "final grounded answer should be tool-assisted: {answer_source:?}"
    );
}

#[test]
fn repeated_pre_evidence_synthesis_is_suppressed_until_read() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "run_turns drives the loop.",
            "[search_code: run_turns]",
            "run_turns is in engine.rs.",
            "[read_file: engine.rs]",
            "run_turns is grounded in engine.rs.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "What does run_turns do?".into(),
        },
    );

    let answer_sources: Vec<_> = events
        .iter()
        .filter_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(answer_sources.len(), 1, "only one answer may be admitted");
    assert!(
        matches!(answer_sources[0], AnswerSource::ToolAssisted { .. }),
        "the single admitted answer must be after evidence-ready"
    );

    let snapshot = rt.messages_snapshot();
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(last_assistant, Some("run_turns is grounded in engine.rs."));
}

#[test]
fn two_candidate_reads_second_satisfies_evidence_admits_synthesis() {
    // Usage lookup: two search candidates (definition + usage).
    // First read is definition-only -> recovery correction fires.
    // Second read is a usage candidate -> evidence ready -> synthesis admitted.
    // Validates that candidate_reads_count reaching 2 does not prematurely terminate
    // when the second read satisfies evidence.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("models")).unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("models").join("enums.py"),
        "class TaskStatus(str, Enum):\n    PENDING = \"pending\"\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("services").join("runner.py"),
        "from models.enums import TaskStatus\nif task.status == TaskStatus.PENDING:\n    run()\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: TaskStatus]",
            "[read_file: models/enums.py]",
            "[read_file: services/runner.py]",
            "TaskStatus is used in services/runner.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus used?".into(),
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
        "second candidate read satisfying evidence must admit synthesis: {answer_source:?}"
    );
    let snapshot = rt.messages_snapshot();
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("TaskStatus is used in services/runner.py.")
    );
}

#[test]
fn third_candidate_read_after_two_insufficient_reads_is_blocked_pre_dispatch() {
    // Usage lookup: two definition-only reads exhaust the candidate-read budget
    // without useful evidence. If the model then tries a third distinct matched
    // candidate read instead of synthesizing, runtime must stop before dispatch.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("models")).unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("models").join("enums.py"),
        "class TaskStatus(str, Enum):\n    TODO = \"todo\"\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("models").join("alt_enums.py"),
        "class TaskStatus:\n    DONE = \"done\"\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("services").join("task_service.py"),
        "from models.enums import TaskStatus\nif task.status == TaskStatus.TODO:\n    pass\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: TaskStatus]",
            "[read_file: models/enums.py]",
            "[read_file: models/alt_enums.py]",
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
        "turn must terminate cleanly: {events:?}"
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
                reason: RuntimeTerminalReason::InsufficientEvidence,
                ..
            })
        ),
        "third candidate read must terminate with InsufficientEvidence: {answer_source:?}"
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
        "third candidate read must not dispatch"
    );
    assert!(
        all_user.contains("candidate read limit for this investigation reached"),
        "blocked third read must be recorded as a runtime tool error"
    );
    assert!(
        !all_user.contains("=== tool_result: read_file ===\npath: services/task_service.py"),
        "usage candidate must not be read after the two-candidate cap"
    );
}

#[test]
fn import_only_candidate_rejected_when_non_import_candidate_exists() {
    // File A: only an import line -> classified import-only.
    // File B: a usage line -> classified as non-import candidate.
    // Model reads A first -> correction fires pointing to B.
    // Model reads B -> evidence ready -> ToolAssisted.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("init")).unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("init").join("header.py"),
        "from models.enums import TaskStatus\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("services").join("task_service.py"),
        "if task.status == TaskStatus.TODO:\n    pass\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: TaskStatus]",
            "[read_file: init/header.py]",
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
        "import-only rejection + non-import read must admit synthesis: {answer_source:?}"
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
fn import_only_fallback_accepts_when_all_candidates_are_import_only() {
    // Single candidate: only an import line.
    // has_non_import_candidates == false -> import-only gate does not fire.
    // File is accepted as evidence -> ToolAssisted.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("models")).unwrap();
    fs::write(
        tmp.path().join("models").join("enums.py"),
        "from models.enums import TaskStatus\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: TaskStatus]",
            "[read_file: models/enums.py]",
            "TaskStatus is imported from models.enums.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus used?".into(),
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
        "all-import-only candidates must fall back to accepting the read: {answer_source:?}"
    );
    let snapshot = rt.messages_snapshot();
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("TaskStatus is imported from models.enums.")
    );
}
