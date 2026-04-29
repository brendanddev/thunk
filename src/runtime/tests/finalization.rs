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
            text: "display the structure".into(),
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

// ── Regression: Fix 1 ─────────────────────────────────────────────────────────
// When a seeded direct read succeeds, the runtime must finalize immediately with
// the file contents rather than entering post-read answer-phase synthesis.
#[test]
fn direct_read_finalizes_immediately_with_file_contents() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
    fs::write(
        tmp.path().join("sandbox/main.py"),
        "def main():\n    return 'ok'\n",
    )
    .unwrap();

    let (mut rt, requests) = make_runtime_in_with_recorded_requests(
        vec![
            "[read_file: sandbox/main.py]",
            "[search_code: main]",
            "This must not be consumed.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Read sandbox/main.py".into(),
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
            Some(AnswerSource::ToolAssisted { rounds: 1 })
        ),
        "direct read must finalize as a single tool-assisted turn: {answer_source:?}"
    );

    let snapshot = rt.messages_snapshot();
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());

    // The fallback must contain the actual file content, not a failure message.
    assert!(
        matches!(last_assistant, Some(s) if s.contains("def main()")),
        "fallback answer must contain file contents: {last_assistant:?}"
    );
    for forbidden in [
        "=== tool_result",
        "=== /tool_result",
        "=== end_tool_result",
        "[tool_result:",
        "[/tool_result]",
    ] {
        assert!(
            !matches!(last_assistant, Some(s) if s.contains(forbidden)),
            "fallback answer must not contain protocol wrapper `{forbidden}`: {last_assistant:?}"
        );
    }
    assert!(
        !matches!(
            answer_source,
            Some(AnswerSource::RuntimeTerminal {
                reason: RuntimeTerminalReason::RepeatedToolAfterAnswerPhase,
                ..
            })
        ),
        "direct read must not end as RepeatedToolAfterAnswerPhase: {answer_source:?}"
    );
    assert!(
        requests.lock().unwrap().is_empty(),
        "direct read must not perform any model generation"
    );
}

// ── Regression: Fix 2 ─────────────────────────────────────────────────────────
// When the model emits a block opening tag without the matching close tag
// (e.g. `[write_file] path: foo ---content--- bar`), the runtime must detect it
// as malformed and inject a correction rather than accepting it as a direct answer.
#[test]
fn malformed_write_open_without_close_triggers_correction() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("test.txt"), "hello world\n").unwrap();

    // First response: malformed block (open tag, inline content, no close tag).
    // Second response: proper tool call after correction.
    let malformed = "[write_file] path: test.txt\n---content---\nhello thunk";
    let proper_call = "[write_file]\npath: test.txt\n---content---\nhello thunk\n[/write_file]";
    let mut rt = make_runtime_in(vec![malformed, proper_call], tmp.path());

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Update test.txt by replacing hello world with hello thunk".into(),
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

    // The malformed block must trigger the specialized write_file correction, not the generic one.
    assert!(
        all_user.contains("[runtime:correction]")
            && all_user.contains("write_file block is malformed"),
        "runtime must inject specialized write_file correction for open-without-close: {all_user}"
    );

    // The malformed string must NOT appear verbatim as an assistant message.
    let assistant_messages: Vec<&str> = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str())
        .collect();
    assert!(
        !assistant_messages
            .iter()
            .any(|m| m.contains("[write_file] path: test.txt")),
        "malformed tool syntax must never surface as a final answer: {assistant_messages:?}"
    );
}

// ── Regression: Fix 3 ─────────────────────────────────────────────────────────
// When the resolver rejects a mutation tool call (path escapes project root),
// the runtime must terminate immediately with MutationFailed rather than
// continuing into more tool rounds (e.g. falling back to search_code).
#[test]
fn mutation_resolver_failure_terminates_immediately() {
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();

    // Model tries to write outside the project root, then would search if allowed to continue.
    let outside_write = format!(
        "[write_file]\npath: {}/outside.txt\n---content---\nhello\n[/write_file]",
        tmp.path().parent().unwrap().display()
    );
    let would_search = "[search_code: hello]".to_string();
    let mut rt = make_runtime_in(vec![outside_write, would_search], tmp.path());

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Write /tmp/outside.txt with content hello".into(),
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
                reason: RuntimeTerminalReason::MutationFailed,
                ..
            })
        ),
        "resolver-rejected mutation must terminate with MutationFailed: {answer_source:?}"
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
        "runtime must not fall back into retrieval after a mutation resolver failure"
    );
}
