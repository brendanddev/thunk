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
    let mut rt = make_runtime_in(vec!["[search_code: foo]", final_answer], tmp.path());

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
    // Fix 1: for a direct read, the runtime now falls back to the read content
    // rather than emitting the synthesis-failure message.
    assert!(
        matches!(last_assistant, Some(s) if s.contains("fn bar()")),
        "last assistant must contain the file content fallback, not a terminal error: {last_assistant:?}"
    );
}

#[test]
fn direct_read_discards_runtime_correction_echo_before_final_synthesis() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
    fs::write(
        tmp.path().join("sandbox/main.py"),
        "def main():\n    return 'ok'\n",
    )
    .unwrap();

    let correction = "[runtime:correction] The file was already read this turn. Do not call more tools. Provide your final answer now based on what was read.";
    let final_answer = "sandbox/main.py defines main(), which returns 'ok'.";
    let mut rt = make_runtime_in(
        vec!["[read_file: sandbox/main.py]", correction, final_answer],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Read sandbox/main.py".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "runtime must recover from a correction echo after a successful read: {events:?}"
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
        "the duplicate post-read tool attempt must still be blocked"
    );
    assert!(
        all_user.contains("[runtime:correction]") && all_user.contains("already read this turn"),
        "the answer-phase correction must still be injected for the blocked duplicate read"
    );

    let assistant_messages: Vec<&str> = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str())
        .collect();
    assert!(
        !assistant_messages
            .iter()
            .any(|m| m.trim_start().starts_with("[runtime:correction]")),
        "runtime corrections must remain internal and never become assistant-visible: {assistant_messages:?}"
    );
    assert_eq!(assistant_messages.last().copied(), Some(final_answer));
}

#[test]
fn correction_echo_without_sentinel_prefix_is_not_emitted_as_final_answer() {
    // Regression test for Fix 3: model echoes the correction text without the
    // "[runtime:correction]" prefix. The runtime must still detect this as an
    // echo and discard it, then accept the real final answer on the next round.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
    fs::write(
        tmp.path().join("sandbox/main.py"),
        "def main():\n    return 'ok'\n",
    )
    .unwrap();

    // Model's first synthesis response after the seeded read echoes correction text
    // without the "[runtime:correction]" sentinel prefix.
    let partial_echo =
        "The file was already read this turn. Based on the contents, main returns 'ok'.";
    let final_answer = "sandbox/main.py defines main(), which returns 'ok'.";
    let mut rt = make_runtime_in(vec![partial_echo, final_answer], tmp.path());

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Read sandbox/main.py".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "runtime must recover from prefix-less correction echo: {events:?}"
    );

    // The partial echo must not be emitted to the user.
    assert!(
        !events.iter().any(|e| matches!(
            e,
            RuntimeEvent::AssistantMessageChunk(text) if text.contains("The file was already read this turn")
        )),
        "correction echo must not be emitted as an AssistantMessageChunk"
    );

    let snapshot = rt.messages_snapshot();
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some(final_answer),
        "last assistant message must be the real final answer, not the echo"
    );
}

// ── Regression: Fix 1 ─────────────────────────────────────────────────────────
// When a seeded direct read succeeds but model synthesis repeatedly fails
// (keeps calling tools in answer phase), the runtime must serve the file content
// as a deterministic fallback rather than emitting a synthesis-failure message.
#[test]
fn direct_read_fallback_serves_file_content_when_model_loops() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
    fs::write(
        tmp.path().join("sandbox/main.py"),
        "def main():\n    return 'ok'\n",
    )
    .unwrap();

    // Model produces tool calls both times it is asked to synthesize — simulating
    // the local-model loop observed in QA.
    let mut rt = make_runtime_in(
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
            Some(AnswerSource::RuntimeTerminal {
                reason: RuntimeTerminalReason::RepeatedToolAfterAnswerPhase,
                ..
            })
        ),
        "terminal reason must be RepeatedToolAfterAnswerPhase: {answer_source:?}"
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
    assert!(
        !matches!(last_assistant, Some(s) if s.contains("model kept calling tools")),
        "failure message must not be emitted when direct_read_result is available: {last_assistant:?}"
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
            text: "Edit test.txt replace hello world with hello thunk".into(),
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
