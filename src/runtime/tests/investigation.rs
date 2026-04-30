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
    // Phase 16.1: non-candidate reads are now blocked before dispatch.
    // The read produces tool_error (not tool_result) with a correction message.
    assert!(
        snapshot.iter().any(|m| {
            m.content.contains("=== tool_error: read_file ===")
                && m.content.contains("was not returned by the search")
        }),
        "non-candidate read must be blocked before dispatch with a correction: {snapshot:?}"
    );
    assert!(
        !snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: read_file ===")),
        "blocked non-candidate read must not produce a tool_result"
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
        "blocked non-candidate read must not admit synthesis: {answer_source:?}"
    );
}

#[test]
fn usage_lookup_runtime_dispatches_preferred_substantive_candidate_after_search() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("models")).unwrap();
    fs::create_dir_all(tmp.path().join("cli")).unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("models").join("enums.py"),
        "from enum import Enum\n\nclass TaskStatus(str, Enum):\n    TODO = \"todo\"\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("cli").join("header.py"),
        "from models.enums import TaskStatus\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("services").join("runner.py"),
        "from models.enums import TaskStatus\n\nif task.status == TaskStatus.TODO:\n    run()\nif previous_status == TaskStatus.TODO:\n    audit()\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: TaskStatus]",
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
        "one viable substantive candidate should stay single-read after search"
    );
    assert!(
        all_user.contains("audit()"),
        "preferred substantive candidate should be read first: {all_user}"
    );
    assert!(
        !all_user.contains("TODO = \"todo\"")
            && !all_user.contains(
                "=== tool_result: read_file ===\n[1 lines]\nfrom models.enums import TaskStatus"
            ),
        "definition-only and import-only files must not be selected first: {all_user}"
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
        Some("TaskStatus is used in services/runner.py.")
    );
}

#[test]
fn broad_usage_lookup_two_substantive_candidates_are_auto_read_before_synthesis() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("models")).unwrap();
    fs::create_dir_all(tmp.path().join("cli")).unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("models").join("enums.py"),
        "class TaskStatus(str, Enum):\n    UNUSED_ENUM_MEMBER = \"unused\"\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("cli").join("header.py"),
        "from models.enums import TaskStatus\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("services").join("runner_primary.py"),
        "if task.status == TaskStatus.PENDING:\n    primary()\nif previous_status == TaskStatus.PENDING:\n    audit_primary()\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("services").join("runner_secondary.py"),
        "status = TaskStatus.PENDING\nsecondary()\n",
    )
    .unwrap();

    let final_answer =
        "TaskStatus is used in services/runner_primary.py and services/runner_secondary.py.";
    let mut rt = make_runtime_in(vec!["[search_code: TaskStatus]", final_answer], tmp.path());

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus used?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
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
        "broad usage lookup should auto-read two substantive candidates"
    );
    assert!(
        all_user.contains("primary()") && all_user.contains("secondary()"),
        "both substantive usage files must be read before synthesis: {all_user}"
    );
    assert!(
        !all_user.contains("UNUSED_ENUM_MEMBER")
            && !all_user.contains(
                "=== tool_result: read_file ===\n[1 lines]\nfrom models.enums import TaskStatus"
            ),
        "definition-only and import-only fallbacks must not be auto-read when two substantive candidates exist: {all_user}"
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
        "two runtime-owned usage reads should still admit synthesis: {answer_source:?}"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(last_assistant, Some(final_answer));
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
    let all_user: String = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::User)
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        all_user.matches("=== tool_result: read_file ===").count(),
        1,
        "all-definition fallback should perform exactly one read"
    );
    assert!(
        all_user.contains("TODO = \"todo\""),
        "all-definition fallback should still read the definition candidate: {all_user}"
    );
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
    let all_user: String = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::User)
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        all_user.matches("=== tool_result: read_file ===").count(),
        1,
        "mixed candidate should still be read exactly once"
    );
    assert!(
        all_user.contains("TODO = \"todo\""),
        "mixed definition+usage candidate should still be read before synthesis: {all_user}"
    );
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
fn usage_lookup_prefers_normal_source_over_initialization_candidate() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("models")).unwrap();
    fs::create_dir_all(tmp.path().join("sandbox/init")).unwrap();
    fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
    fs::write(
        tmp.path().join("models").join("enums.py"),
        "class TaskStatus(str, Enum):\n    PENDING = \"pending\"\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("sandbox/init").join("bootstrap.py"),
        "initialize_task_status(TaskStatus.PENDING)\nINITIALIZED_STATUS = TaskStatus.PENDING\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("sandbox/services").join("runner.py"),
        "if task.status == TaskStatus.PENDING:\n    run()\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: TaskStatus]",
            "TaskStatus is used in sandbox/services/runner.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus used in sandbox/".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
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
        "broad scoped usage lookup should auto-read both substantive candidates"
    );
    assert!(
        all_user.contains("run()"),
        "normal source file should still win the first ranking slot: {all_user}"
    );
    assert!(
        all_user.contains("initialize_task_status("),
        "the second substantive candidate should also be auto-read for broad usage lookup: {all_user}"
    );
    let read_results: Vec<&str> = snapshot
        .iter()
        .filter(|m| {
            m.role == crate::llm::backend::Role::User
                && m.content.contains("=== tool_result: read_file ===")
        })
        .map(|m| m.content.as_str())
        .collect();
    assert!(
        read_results
            .first()
            .is_some_and(|body| body.contains("run()"))
            && read_results
                .get(1)
                .is_some_and(|body| body.contains("initialize_task_status(")),
        "normal source file must still be selected before the initialization candidate: {read_results:?}"
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
        "preferred usage candidate should admit synthesis immediately: {answer_source:?}"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("TaskStatus is used in sandbox/services/runner.py.")
    );
}

#[test]
fn broad_scoped_usage_lookup_reads_two_in_scope_substantive_files_only() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
    fs::create_dir_all(tmp.path().join("sandbox/controllers")).unwrap();
    fs::create_dir_all(tmp.path().join("sandbox/models")).unwrap();
    fs::write(
        tmp.path().join("sandbox/services").join("runner_primary.py"),
        "if task.status == TaskStatus.PENDING:\n    primary_service()\nif previous_status == TaskStatus.PENDING:\n    audit_service()\n",
    )
    .unwrap();
    fs::write(
        tmp.path()
            .join("sandbox/services")
            .join("runner_secondary.py"),
        "status = TaskStatus.PENDING\nsecondary_service()\n",
    )
    .unwrap();
    fs::write(
        tmp.path()
            .join("sandbox/controllers")
            .join("runner_outside.py"),
        "if task.status == TaskStatus.PENDING:\n    outside_controller()\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("sandbox/models").join("enums.py"),
        "class TaskStatus(str, Enum):\n    UNUSED_ENUM_MEMBER = \"unused\"\n",
    )
    .unwrap();

    let final_answer = "TaskStatus is used in sandbox/services/runner_primary.py and sandbox/services/runner_secondary.py.";
    let mut rt = make_runtime_in(vec!["[search_code: TaskStatus]", final_answer], tmp.path());

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus used in sandbox/services/".into(),
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
        search_result.contains("sandbox/services/runner_primary.py")
            && search_result.contains("sandbox/services/runner_secondary.py"),
        "scoped search must include both in-scope substantive candidates: {search_result}"
    );
    assert!(
        !search_result.contains("sandbox/controllers/runner_outside.py")
            && !search_result.contains("sandbox/models/enums.py"),
        "scoped search must exclude out-of-scope candidates: {search_result}"
    );

    let all_user: String = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::User)
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        all_user.matches("=== tool_result: read_file ===").count(),
        2,
        "broad scoped usage lookup should auto-read two in-scope substantive files"
    );
    assert!(
        all_user.contains("primary_service()") && all_user.contains("secondary_service()"),
        "both in-scope substantive usage files must be read: {all_user}"
    );
    assert!(
        !all_user.contains("outside_controller()") && !all_user.contains("UNUSED_ENUM_MEMBER"),
        "out-of-scope and fallback candidates must not be read: {all_user}"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(last_assistant, Some(final_answer));
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
        "from models.enums import TaskStatus\n",
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
        all_user.contains("task_service.py"),
        "runtime must auto-dispatch task_service.py as the second candidate read"
    );
    assert!(
        !all_user.contains("DONE = \"done\""),
        "alt candidate must not be dispatched after the two-candidate cap"
    );
}

#[test]
fn import_only_candidate_rejected_when_non_import_candidate_exists() {
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
    let all_user: String = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::User)
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        all_user.matches("=== tool_result: read_file ===").count(),
        1,
        "runtime should select one preferred candidate read"
    );
    assert!(
        all_user.contains("pass\n"),
        "substantive usage file must be selected: {all_user}"
    );
    assert!(
        !all_user.contains(
            "=== tool_result: read_file ===\n[1 lines]\nfrom models.enums import TaskStatus"
        ),
        "import-only candidate must lose to substantive usage file: {all_user}"
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
        "preferred substantive usage read must admit synthesis: {answer_source:?}"
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
        "all-import fallback should perform exactly one read"
    );
    assert!(
        all_user.contains("from models.enums import TaskStatus"),
        "all-import fallback should still read the single candidate: {all_user}"
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
        "all-import-only candidates must fall back to accepting the read: {answer_source:?}"
    );
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

// Phase 16.1: Retrieval Candidate Discipline

#[test]
fn non_candidate_read_after_search_produces_correction() {
    // After search returns a candidate, the model reads a file that was NOT in the
    // search results.  The guard must block the read before dispatch and inject a
    // [runtime:correction] message naming the path that was not a candidate.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
    fs::write(
        tmp.path().join("sandbox/init.rs"),
        "fn initialize_logging() {}\n",
    )
    .unwrap();
    fs::write(tmp.path().join("unrelated.rs"), "fn other() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: initialize_logging]",
            "[read_file: unrelated.rs]",
            "Logging is initialized in sandbox/init.rs.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Find where logging is initialized in sandbox/".into(),
        },
    );

    let snapshot = rt.messages_snapshot();

    assert!(
        snapshot.iter().any(|m| {
            m.content.contains("=== tool_error: read_file ===")
                && m.content.contains("was not returned by the search")
        }),
        "non-candidate read must produce a tool_error correction before dispatch: {snapshot:?}"
    );
    assert!(
        !snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: read_file ===")),
        "non-candidate read must not reach dispatch"
    );
    let _ = events; // turn may end at InsufficientEvidence — that is acceptable
}

#[test]
fn candidate_read_after_search_passes_guard() {
    // After search returns a candidate, the model reads that exact candidate.
    // The guard must NOT fire — the read should proceed and evidence should be ready.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
    fs::write(
        tmp.path().join("sandbox/init.rs"),
        "fn initialize_logging() {}\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: initialize_logging]",
            "[read_file: sandbox/init.rs]",
            "Logging is initialized in sandbox/init.rs.",
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
        "candidate read must not fail: {events:?}"
    );

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_result: read_file ===")),
        "candidate read must reach dispatch and produce a tool_result"
    );
    assert!(
        !snapshot.iter().any(|m| {
            m.content.contains("=== tool_error: read_file ===")
                && m.content.contains("was not returned by the search")
        }),
        "guard must not fire for a file that is in the search results"
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
        "candidate read must admit synthesis: {answer_source:?}"
    );
}

#[test]
fn non_candidate_read_before_search_is_not_blocked() {
    // The guard only activates after search_produced_results() is true.
    // A read_file call on an investigation turn with no prior search must reach
    // dispatch normally (tool_result present), even though it will not satisfy
    // evidence readiness.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[read_file: engine.rs]",
            "run_turns drives the loop.",
            "Still drives it.",
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
        "read before search must reach dispatch — guard must not fire without prior search results"
    );
    assert!(
        !snapshot.iter().any(|m| {
            m.content.contains("=== tool_error: read_file ===")
                && m.content.contains("was not returned by the search")
        }),
        "guard must not fire when no search has been performed"
    );
    let _ = events; // turn ends at InsufficientEvidence since no search was done — acceptable
}

#[test]
fn repeated_non_candidate_read_across_rounds_goes_terminal() {
    // First round: search succeeds, model reads a non-candidate → correction (attempts=1).
    // Second round: model reads another non-candidate → persistent counter reaches 2 → terminal.
    // Verifies that InvestigationState.non_candidate_read_attempts persists across
    // separate run_tool_round calls within the same user turn.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
    fs::write(
        tmp.path().join("sandbox/init.rs"),
        "fn initialize_logging() {}\n",
    )
    .unwrap();
    fs::write(tmp.path().join("unrelated.rs"), "fn other() {}\n").unwrap();
    fs::write(tmp.path().join("also_unrelated.rs"), "fn another() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: initialize_logging]",
            "[read_file: unrelated.rs]",
            "[read_file: also_unrelated.rs]",
            "Done.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Find where logging is initialized in sandbox/".into(),
        },
    );

    let snapshot = rt.messages_snapshot();

    // First offense: correction injected (attempts=1 from round 2).
    assert!(
        snapshot.iter().any(|m| {
            m.content.contains("=== tool_error: read_file ===")
                && m.content.contains("was not returned by the search")
        }),
        "first non-candidate read must produce a correction: {snapshot:?}"
    );

    // Second offense: terminal (attempts=2 from round 3, counter persisted from round 2).
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
                reason: RuntimeTerminalReason::ReadFileFailed,
                ..
            })
        ),
        "second non-candidate read must terminate with ReadFileFailed: {answer_source:?}"
    );
}

#[test]
fn repeated_non_candidate_read_does_not_become_search_budget_closed() {
    // Regression guard: when a non-candidate read causes a terminal, the reason must be
    // ReadFileFailed, not InsufficientEvidence or a search-budget-related terminal.
    // Before the fix the counter reset each round, causing the model to retry the bad read,
    // then attempt an extra search, and terminal with a misleading search-budget message.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
    fs::write(
        tmp.path().join("sandbox/init.rs"),
        "fn initialize_logging() {}\n",
    )
    .unwrap();
    fs::write(tmp.path().join("unrelated.rs"), "fn other() {}\n").unwrap();
    fs::write(tmp.path().join("also_unrelated.rs"), "fn another() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: initialize_logging]",
            "[read_file: unrelated.rs]",
            "[read_file: also_unrelated.rs]",
            "[search_code: initialize_logging]",
            "Done.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Find where logging is initialized in sandbox/".into(),
        },
    );

    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });

    // Must terminate as ReadFileFailed on the second non-candidate read (round 3),
    // before the model ever reaches the redundant search in round 4.
    assert!(
        matches!(
            answer_source,
            Some(AnswerSource::RuntimeTerminal {
                reason: RuntimeTerminalReason::ReadFileFailed,
                ..
            })
        ),
        "terminal must be ReadFileFailed, not a search-budget-closed terminal: {answer_source:?}"
    );

    // The snapshot must NOT contain any search-budget-exceeded messages.
    let snapshot = rt.messages_snapshot();
    assert!(
        !snapshot
            .iter()
            .any(|m| m.content.contains("search budget exceeded")),
        "search-budget message must not appear — turn must terminal before reaching the extra search"
    );
}

#[test]
fn initialization_lookup_non_candidate_correction_names_initialization_candidate() {
    // Phase 16.2: non-candidate correction on an InitializationLookup turn must name the
    // best initialization candidate so the model can act on it immediately.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
    fs::write(
        tmp.path().join("sandbox/init.rs"),
        "fn initialize_logging() {}\n",
    )
    .unwrap();
    fs::write(tmp.path().join("unrelated.rs"), "fn other() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: initialize_logging]",
            "[read_file: unrelated.rs]",
            "Logging is initialized in sandbox/init.rs.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Find where logging is initialized in sandbox/".into(),
        },
    );

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot.iter().any(|m| {
            m.content.contains("=== tool_error: read_file ===")
                && m.content.contains("was not returned by the search")
                && m.content.contains("[read_file: sandbox/init.rs]")
        }),
        "correction for InitializationLookup must name the initialization candidate: {snapshot:?}"
    );
    let _ = events;
}

#[test]
fn config_lookup_non_candidate_correction_names_config_candidate() {
    // Phase 16.2: non-candidate correction on a ConfigLookup turn must name the best
    // config-file candidate so the model reads the right file on the next attempt.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("config")).unwrap();
    fs::write(
        tmp.path().join("config/database.yaml"),
        "database: postgres\n",
    )
    .unwrap();
    fs::write(tmp.path().join("unrelated.rs"), "fn other() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: database]",
            "[read_file: unrelated.rs]",
            "The database is configured in config/database.yaml.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Find where the database is configured".into(),
        },
    );

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot.iter().any(|m| {
            m.content.contains("=== tool_error: read_file ===")
                && m.content.contains("was not returned by the search")
                && m.content.contains("[read_file: config/database.yaml]")
        }),
        "correction for ConfigLookup must name the config candidate: {snapshot:?}"
    );
    let _ = events;
}

#[test]
fn general_mode_non_candidate_correction_names_first_search_candidate() {
    // Phase 16.2: on a General-mode turn the mode-specific selector returns None, so the
    // correction must fall back to naming the first search result.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();
    fs::write(tmp.path().join("unrelated.rs"), "fn other() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: run_turns]",
            "[read_file: unrelated.rs]",
            "run_turns drives the loop.",
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
            m.content.contains("=== tool_error: read_file ===")
                && m.content.contains("was not returned by the search")
                && m.content.contains("[read_file: engine.rs]")
        }),
        "correction for General mode must name the first search candidate: {snapshot:?}"
    );
    let _ = events;
}

#[test]
fn non_candidate_correction_with_no_mode_specific_candidate_names_first_result() {
    // Phase 16.2: when the mode is InitializationLookup but no matched line contains an
    // initialization term, the mode-specific selector returns None and the correction must
    // fall back to naming the first search result.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
    // Content does NOT contain "initialize"/"initialization" → won't be an initialization candidate.
    fs::write(tmp.path().join("sandbox/other.rs"), "fn setup() {}\n").unwrap();
    fs::write(tmp.path().join("unrelated.rs"), "fn other() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: setup]",
            "[read_file: unrelated.rs]",
            "The setup function is in sandbox/other.rs.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            // "initialized" triggers InitializationLookup; "setup" is the identifier to find.
            text: "Find where the application is initialized using setup".into(),
        },
    );

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot.iter().any(|m| {
            m.content.contains("=== tool_error: read_file ===")
                && m.content.contains("was not returned by the search")
                && m.content.contains("[read_file: sandbox/other.rs]")
        }),
        "correction must fall back to first search result when mode-specific set is empty: {snapshot:?}"
    );
    let _ = events;
}
