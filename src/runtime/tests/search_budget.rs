use super::*;
use crate::runtime::types::RuntimeTerminalReason;

#[test]
fn search_budget_blocks_second_search_when_first_had_results() {
    // Both searches in one response. "ToolInput" is present in many source files,
    // so the first search will produce matches — the second must be budget-blocked.
    let mut rt = make_runtime(vec![
        "[search_code: ToolInput][search_code: EditFile]",
        "Done.",
    ]);
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "search".into(),
        },
    );

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("=== tool_error: search_code ===")
                && m.content.contains("search budget exceeded")),
        "second search must be blocked with budget error when first had results"
    );
}

#[test]
fn search_budget_closes_after_first_search_with_results_across_rounds() {
    use std::fs;
    use tempfile::TempDir;
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("logging.rs"), "fn logging() {}").unwrap();
    let synthesis = "Logging appears in logging.rs.";

    let mut rt = make_runtime_in(
        vec![
            "[search_code: logging initialization]",
            "Let me try another search.\n[search_code: logger initialization]",
            synthesis,
        ],
        tmp.path(),
    );
    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "list files".into(),
        },
    );
    assert!(
        !has_failed(&events),
        "must not fail permanently: {events:?}"
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
        "second search must be intercepted before another tool result"
    );
    assert!(
        all_user.contains("Search returned matches"),
        "closed-search guidance must be injected after the first successful search"
    );
    assert!(
        !snapshot
            .iter()
            .any(|m| m.content.contains("Let me try another search")),
        "narrated retry assistant message must be discarded from model context"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(last_assistant, Some(synthesis));
}

#[test]
fn search_budget_closes_after_empty_retry_across_rounds() {
    // Phase 8.3: after two empty searches and the third attempt discarded, the runtime
    // now emits the insufficient-evidence terminal answer rather than letting the model
    // synthesize without any grounded evidence.
    use std::fs;
    use tempfile::TempDir;
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("file.rs"), "fn unrelated() {}").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: logging initialization]",
            "[search_code: logger initialization]",
            "Trying one more.\n[search_code: tracing]",
            // This response is never consumed — R4 fires before invoking the backend.
            "No matching code was found for those searches.",
        ],
        tmp.path(),
    );
    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "list files".into(),
        },
    );
    assert!(
        !has_failed(&events),
        "must not fail permanently: {events:?}"
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
        2,
        "first empty search and one retry should execute"
    );
    assert!(
        all_user.contains("allowed search retry also returned no matches"),
        "empty-retry terminal guidance must be injected"
    );
    assert!(
        !snapshot
            .iter()
            .any(|m| m.content.contains("Trying one more")),
        "third narrated search must be discarded from model context"
    );

    // Phase 8.3: runtime-owned insufficient-evidence terminal fires instead of model synthesis.
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
        "empty-search no-read turn must produce InsufficientEvidence terminal: {answer_source:?}"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert!(
        matches!(last_assistant, Some(s) if s.contains("don't have enough information to answer")),
        "last assistant message must be the runtime terminal, not model synthesis: {last_assistant:?}"
    );
}

#[test]
fn search_budget_allows_second_search_when_first_empty() {
    // Controlled temp dir: no file matches "no_match_here" but one matches "find_me".
    // First search returns empty → second search must be allowed.
    use std::fs;
    use tempfile::TempDir;
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("file.rs"), "fn find_me() {}").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: no_match_here][search_code: find_me]",
            "Found it.",
        ],
        tmp.path(),
    );
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "search".into(),
        },
    );

    let snapshot = rt.messages_snapshot();
    assert!(
        !snapshot
            .iter()
            .any(|m| m.content.contains("search budget exceeded")),
        "second search must be allowed when first returned empty"
    );
    // Both results land in the same accumulated user message, so count occurrences.
    let all_user: String = snapshot
        .iter()
        .filter(|m| m.role == crate::llm::backend::Role::User)
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let result_count = all_user.matches("=== tool_result: search_code ===").count();
    assert_eq!(result_count, 2, "both searches must have tool results");
}

#[test]
fn empty_search_retry_exhausted_third_search_terminates_pre_dispatch() {
    // Controlled temp dir: first two searches return empty; a third search attempt
    // terminates without another correction loop or search tool_error.
    use std::fs;
    use tempfile::TempDir;
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("file.rs"), "fn find_me() {}").unwrap();

    // first=empty, second=empty (allowed by budget), third=terminal.
    let mut rt = make_runtime_in(
        vec![
            "[search_code: no_match_a][search_code: no_match_b][search_code: find_me]",
            "Done.",
        ],
        tmp.path(),
    );
    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "triple search".into(),
        },
    );
    assert!(
        !has_failed(&events),
        "empty search exhaustion must terminate cleanly: {events:?}"
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
        2,
        "only the first empty search and the allowed empty retry should execute"
    );
    assert!(
        !all_user.contains("search budget exceeded"),
        "empty retry exhaustion should terminal without another search-budget tool_error"
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
        "empty retry exhaustion must use InsufficientEvidence terminal: {answer_source:?}"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert!(
        matches!(last_assistant, Some(s) if s.contains("don't have enough information to answer")),
        "last assistant message must be runtime terminal answer: {last_assistant:?}"
    );
}

#[test]
fn duplicate_search_after_empty_result_terminates_without_cycle_loop() {
    // Manual regression: after an empty search, repeating the same search should
    // terminate as insufficient evidence instead of looping through cycle tool_errors.
    use std::fs;
    use tempfile::TempDir;
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("file.rs"), "fn unrelated() {}").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: clap]",
            "[search_code: clap]",
            "This response should not be consumed.",
        ],
        tmp.path(),
    );
    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is clap used".into(),
        },
    );
    assert!(
        !has_failed(&events),
        "duplicate empty-search retry must terminate cleanly: {events:?}"
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
        "only the first empty search should execute"
    );
    assert!(
        !all_user.contains("identical arguments twice in a row"),
        "duplicate empty-search retry should terminal without cycle tool_error"
    );
    assert!(
        !snapshot
            .iter()
            .any(|m| m.content.contains("This response should not be consumed")),
        "runtime terminal must stop before another model synthesis"
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
        "duplicate empty-search retry must use InsufficientEvidence terminal: {answer_source:?}"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert!(
        matches!(last_assistant, Some(s) if s.contains("don't have enough information to answer")),
        "last assistant message must be runtime terminal answer: {last_assistant:?}"
    );
}
