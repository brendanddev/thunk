use super::*;
use crate::runtime::types::RuntimeTerminalReason;

// Phase 12.0.3 — Investigation + Recovery Validation

#[test]
fn recovery_flow_surface_hint_is_retrieval_first_before_evidence_and_answer_only_after() {
    // search → wrong read (definition-only, usage lookup) → recovery correction →
    // correct read (usage file, evidence ready) → AnswerOnly synthesis.
    //
    // Critical: AnswerOnly must NOT activate after the wrong read. It must activate
    // only after the correct read accepts evidence. Validates no premature answer-phase
    // activation and no wasted post_evidence_tool_call_rejected rounds.
    use std::fs;
    use std::sync::{Arc, Mutex};
    use tempfile::TempDir;
    use crate::app::config::Config;
    use crate::llm::backend::Role;
    use crate::tools::default_registry;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("models")).unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    // Definition-only candidate: every matched line for "TaskStatus" is a definition site.
    fs::write(
        tmp.path().join("models/enums.py"),
        "class TaskStatus(str, Enum):\n    PENDING = \"pending\"\n",
    )
    .unwrap();
    // Usage candidate: contains a non-definition, non-import match line.
    fs::write(
        tmp.path().join("services/runner.py"),
        "from models.enums import TaskStatus\nif task.status == TaskStatus.PENDING:\n    run()\n",
    )
    .unwrap();

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut rt = Runtime::new(
        &Config::default(),
        tmp.path(),
        Box::new(RecordingBackend::new(
            vec![
                "[search_code: TaskStatus]",              // round 1: search
                "[read_file: models/enums.py]",           // round 2: wrong read (definition-only)
                "[read_file: services/runner.py]",        // round 3: correct read (evidence ready)
                "TaskStatus is used in services/runner.py.", // round 4: synthesis
            ],
            Arc::clone(&requests),
        )),
        default_registry(tmp.path().to_path_buf()),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus used?".into(),
        },
    );

    assert!(!has_failed(&events), "recovery flow must not fail: {events:?}");
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "recovery flow must complete as ToolAssisted: {answer_source:?}"
    );

    let requests = requests.lock().unwrap();
    assert_eq!(
        requests.len(),
        4,
        "expected exactly 4 generation rounds (search + wrong-read + correct-read + synthesis): got {}",
        requests.len()
    );

    // Rounds 1–3: AnswerOnly must not appear before evidence is accepted.
    for (i, req) in requests[..3].iter().enumerate() {
        let hint = req
            .messages
            .iter()
            .find(|m| m.role == Role::System && m.content.starts_with("Active tool surface:"))
            .unwrap_or_else(|| {
                panic!("round {} must carry a surface hint", i + 1)
            });
        assert!(
            !hint.content.contains("AnswerOnly"),
            "round {} must NOT use AnswerOnly before evidence is accepted: {}",
            i + 1,
            hint.content
        );
    }

    // Round 4: synthesis must use AnswerOnly (evidence was accepted after the correct read).
    let synthesis_hint = requests[3]
        .messages
        .iter()
        .find(|m| m.role == Role::System && m.content.starts_with("Active tool surface:"))
        .expect("synthesis round must carry a surface hint");
    assert!(
        synthesis_hint.content.contains("AnswerOnly"),
        "synthesis round must use AnswerOnly after evidence is accepted: {}",
        synthesis_hint.content
    );
}

#[test]
fn recovery_flow_round_count_equals_four_no_wasted_correction_rounds() {
    // Confirm that the standard recovery path (wrong read → correction → correct read → synthesis)
    // consumes exactly 4 generation rounds. Before bounded synthesis (Phase 12.0.1), an extra
    // round was wasted when the model tried a tool call during answer-phase synthesis. This test
    // guards against that regression: if the model answers cleanly in round 4 and the surface
    // hint correctly sets AnswerOnly, no 5th round should occur.
    use std::fs;
    use std::sync::{Arc, Mutex};
    use tempfile::TempDir;
    use crate::app::config::Config;
    use crate::tools::default_registry;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("models")).unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("models/enums.py"),
        "class TaskStatus(str, Enum):\n    PENDING = \"pending\"\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("services/runner.py"),
        "from models.enums import TaskStatus\nif task.status == TaskStatus.PENDING:\n    run()\n",
    )
    .unwrap();

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut rt = Runtime::new(
        &Config::default(),
        tmp.path(),
        Box::new(RecordingBackend::new(
            vec![
                "[search_code: TaskStatus]",
                "[read_file: models/enums.py]",
                "[read_file: services/runner.py]",
                "TaskStatus is used in services/runner.py.",
            ],
            Arc::clone(&requests),
        )),
        default_registry(tmp.path().to_path_buf()),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus used?".into(),
        },
    );

    assert!(!has_failed(&events), "must not fail: {events:?}");

    let round_count = requests.lock().unwrap().len();
    assert_eq!(
        round_count, 4,
        "recovery flow must complete in exactly 4 rounds — extra rounds indicate a regression in bounded synthesis: got {round_count}"
    );

    // Confirm no ToolCallStarted for the blocked wrong-read recovery path.
    let started: Vec<&str> = events
        .iter()
        .filter_map(|e| {
            if let RuntimeEvent::ToolCallStarted { name } = e {
                Some(name.as_str())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(
        started,
        vec!["search_code", "read_file", "read_file"],
        "ToolCallStarted must fire for search and both reads (wrong + correct): {started:?}"
    );
}

#[test]
fn initialization_recovery_flow_answer_only_activates_only_after_correct_initialization_read() {
    // InitializationLookup recovery: search → wrong read (no init term) → recovery correction →
    // correct read (has init term, evidence ready) → AnswerOnly synthesis.
    // Validates the same surface-hint guarantee across a different recovery mode.
    use std::fs;
    use std::sync::{Arc, Mutex};
    use tempfile::TempDir;
    use crate::app::config::Config;
    use crate::llm::backend::Role;
    use crate::tools::default_registry;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    // Non-initialization candidate: matched line has no initialization term.
    fs::write(
        tmp.path().join("services/logging_usage.py"),
        "def emit_log(logger):\n    logger.info(\"logging event\")\n",
    )
    .unwrap();
    // Initialization candidate: matched line contains "initialize".
    fs::write(
        tmp.path().join("services/logging_init.py"),
        "def initialize_logging():\n    logging.basicConfig(level=\"INFO\")\n",
    )
    .unwrap();

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut rt = Runtime::new(
        &Config::default(),
        tmp.path(),
        Box::new(RecordingBackend::new(
            vec![
                "[search_code: logging]",                        // round 1: search
                "[read_file: services/logging_usage.py]",        // round 2: wrong read (no init)
                "[read_file: services/logging_init.py]",         // round 3: correct read (init)
                "Logging is initialized in services/logging_init.py.", // round 4: synthesis
            ],
            Arc::clone(&requests),
        )),
        default_registry(tmp.path().to_path_buf()),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Find where logging is initialized in services/".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "initialization recovery flow must not fail: {events:?}"
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
        "initialization recovery flow must complete as ToolAssisted: {answer_source:?}"
    );

    let requests = requests.lock().unwrap();
    assert_eq!(
        requests.len(),
        4,
        "initialization recovery must complete in exactly 4 rounds: got {}",
        requests.len()
    );

    // Rounds 1–3: must use RetrievalFirst (evidence not yet accepted).
    for (i, req) in requests[..3].iter().enumerate() {
        let hint = req
            .messages
            .iter()
            .find(|m| m.role == Role::System && m.content.starts_with("Active tool surface:"))
            .unwrap_or_else(|| panic!("round {} must carry a surface hint", i + 1));
        assert!(
            !hint.content.contains("AnswerOnly"),
            "round {} must NOT use AnswerOnly before initialization evidence is accepted: {}",
            i + 1,
            hint.content
        );
    }

    // Round 4: synthesis must use AnswerOnly (correct init read accepted evidence).
    let synthesis_hint = requests[3]
        .messages
        .iter()
        .find(|m| m.role == Role::System && m.content.starts_with("Active tool surface:"))
        .expect("synthesis round must carry a surface hint");
    assert!(
        synthesis_hint.content.contains("AnswerOnly"),
        "synthesis round must use AnswerOnly after initialization evidence is accepted: {}",
        synthesis_hint.content
    );

    // Verify the recovery correction was issued (not bypassed).
    let snapshot = rt.messages_snapshot();
    let all_user: String = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::User)
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        all_user.contains("This is an initialization lookup"),
        "initialization recovery correction must be injected after wrong read"
    );
    assert_eq!(
        all_user.matches("=== tool_result: read_file ===").count(),
        2,
        "both reads (wrong + correct) must be dispatched before evidence is accepted"
    );
}

// Phase 12.0.2 — Structural Fallback Validation

#[test]
fn post_read_answer_phase_tool_call_does_not_emit_tool_call_started() {
    // After read_file completes, answer_phase = PostRead. A subsequent tool call
    // is blocked before reaching run_tool_round, so ToolCallStarted must not fire for it.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("foo.rs"), "fn foo() {}\n").unwrap();

    let final_answer = "foo.rs defines a single function.";
    let mut rt = make_runtime_in(
        vec!["[read_file: foo.rs]", "[search_code: foo]", final_answer],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read foo.rs".into(),
        },
    );

    assert!(!has_failed(&events), "must not fail: {events:?}");

    let started: Vec<&str> = events
        .iter()
        .filter_map(|e| {
            if let RuntimeEvent::ToolCallStarted { name } = e {
                Some(name.as_str())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(
        started,
        vec!["read_file"],
        "only the initial read_file must emit ToolCallStarted; blocked post-read search_code must not: {started:?}"
    );
}

#[test]
fn correction_round_in_post_read_answer_phase_receives_answer_only_surface_hint() {
    // When a model emits a tool call during answer-phase synthesis (PostRead),
    // the correction round is a new generation. That generation must also receive
    // the AnswerOnly surface hint — not RetrievalFirst — so the model is never
    // offered tools while answer_phase remains active.
    use std::fs;
    use std::sync::{Arc, Mutex};
    use tempfile::TempDir;
    use crate::app::config::Config;
    use crate::llm::backend::Role;
    use crate::tools::default_registry;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("foo.rs"), "fn foo() {}\n").unwrap();

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut rt = Runtime::new(
        &Config::default(),
        tmp.path(),
        Box::new(RecordingBackend::new(
            vec![
                "[read_file: foo.rs]",              // round 1: legitimate read
                "[search_code: foo]",                // round 2: blocked in AnswerOnly synthesis
                "foo.rs defines a single function.", // round 3: correction synthesis
            ],
            Arc::clone(&requests),
        )),
        default_registry(tmp.path().to_path_buf()),
    );

    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read foo.rs".into(),
        },
    );

    let requests = requests.lock().unwrap();
    assert_eq!(
        requests.len(),
        3,
        "expected 3 backend calls (read + blocked synthesis + correction synthesis): got {}",
        requests.len()
    );

    for (i, req) in requests[1..].iter().enumerate() {
        let surface_hint = req
            .messages
            .iter()
            .find(|m| m.role == Role::System && m.content.starts_with("Active tool surface:"))
            .unwrap_or_else(|| {
                panic!(
                    "synthesis request {} must carry an Active tool surface hint: {:?}",
                    i + 1,
                    req.messages
                )
            });
        assert!(
            surface_hint.content.contains("AnswerOnly"),
            "synthesis request {} surface hint must name AnswerOnly: {}",
            i + 1,
            surface_hint.content
        );
    }
}

#[test]
fn investigation_answer_phase_tool_call_does_not_emit_tool_call_started() {
    // After evidence is accepted (InvestigationEvidenceReady), a subsequent tool call
    // is blocked before reaching run_tool_round. ToolCallStarted must not fire for it.
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
            "[read_file: sandbox/cli/commands.py]", // blocked — evidence already ready
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

    assert!(!has_failed(&events), "must not fail: {events:?}");

    let started: Vec<&str> = events
        .iter()
        .filter_map(|e| {
            if let RuntimeEvent::ToolCallStarted { name } = e {
                Some(name.as_str())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(
        started,
        vec!["search_code", "read_file"],
        "only search_code and the first read_file must emit ToolCallStarted; blocked post-evidence read must not: {started:?}"
    );
}

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
        !all_user.contains(
            "=== tool_result: read_file ===\npath: sandbox/services/logging_usage.py\n"
        ) || all_user
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
        vec!["[read_file: foo.rs]", "[search_code: foo]", final_answer],
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
            text: "describe what this project does".into(),
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
