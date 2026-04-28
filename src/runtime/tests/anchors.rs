use super::*;
use crate::runtime::types::RuntimeTerminalReason;

// Read-file anchor tests

#[test]
fn successful_read_file_updates_last_read_file_anchor() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src/runtime")).unwrap();
    fs::write(
        tmp.path().join("src/runtime/engine.rs"),
        "fn run_turns() {}\n",
    )
    .unwrap();

    let expected_path = "src/runtime/engine.rs";
    let mut rt = make_runtime_in(
        vec![
            "[read_file: src/runtime/engine.rs]",
            "Read engine.rs.",
            "Re-read engine.rs.",
        ],
        tmp.path(),
    );
    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read src/runtime/engine.rs".into(),
        },
    );
    assert!(!has_failed(&events), "unexpected failure: {events:?}");

    let followup = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read that file again".into(),
        },
    );
    assert!(
        followup.iter().any(|e| {
            matches!(
                e,
                RuntimeEvent::ToolCallFinished {
                    name,
                    summary: Some(s)
                } if name == "read_file" && s.contains(&expected_path)
            )
        }),
        "successful read must set anchor to that path: {followup:?}"
    );
}

#[test]
fn read_that_file_again_dispatches_one_read_to_anchor() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::write(tmp.path().join("src/anchor.rs"), "fn anchor() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[read_file: src/anchor.rs]",
            "First read complete.",
            "Anchored read complete.",
        ],
        tmp.path(),
    );
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read src/anchor.rs".into(),
        },
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read that file again".into(),
        },
    );

    let read_starts = events
        .iter()
        .filter(|e| matches!(e, RuntimeEvent::ToolCallStarted { name } if name == "read_file"))
        .count();
    assert_eq!(read_starts, 1, "anchor prompt must dispatch one read");
    let expected_path = "src/anchor.rs";
    assert!(
        events.iter().any(|e| {
            matches!(
                e,
                RuntimeEvent::ToolCallFinished {
                    name,
                    summary: Some(summary)
                } if name == "read_file" && summary.contains(&expected_path)
            )
        }),
        "anchored read must target the last successful path: {events:?}"
    );

    let snapshot = rt.messages_snapshot();
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(last_assistant, Some("Anchored read complete."));
}

#[test]
fn open_the_last_file_resolves_to_last_read_file_anchor() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::write(tmp.path().join("src/last.rs"), "fn last() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[read_file: src/last.rs]",
            "First read complete.",
            "Opened last file.",
        ],
        tmp.path(),
    );
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read src/last.rs".into(),
        },
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "open the last file".into(),
        },
    );

    let expected_path = "src/last.rs";
    assert!(
        events.iter().any(|e| {
            matches!(
                e,
                RuntimeEvent::ToolCallFinished {
                    name,
                    summary: Some(summary)
                } if name == "read_file" && summary.contains(&expected_path)
            )
        }),
        "open the last file must read the anchored path: {events:?}"
    );
}

#[test]
fn reset_clears_last_read_file_anchor() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::write(tmp.path().join("src/reset.rs"), "fn reset_anchor() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec!["[read_file: src/reset.rs]", "First read complete."],
        tmp.path(),
    );
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read src/reset.rs".into(),
        },
    );
    collect_events(&mut rt, RuntimeRequest::Reset);

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read that file".into(),
        },
    );
    assert!(
        events.iter().any(|e| matches!(
            e,
            RuntimeEvent::AssistantMessageChunk(chunk) if chunk.contains("No previous file")
        )),
        "reset anchor prompt must produce deterministic no-anchor answer: {events:?}"
    );
}

#[test]
fn failed_read_file_does_not_update_last_read_file_anchor() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::write(tmp.path().join("src/good.rs"), "fn good() {}\n").unwrap();

    let good_path = "src/good.rs";
    let mut rt = make_runtime_in(
        vec![
            "[read_file: src/good.rs]",
            "First read complete.",
            "[read_file: src/missing.rs]",
            "",
            "Read good.rs again.",
        ],
        tmp.path(),
    );
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read src/good.rs".into(),
        },
    );
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read src/missing.rs".into(),
        },
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read that file again".into(),
        },
    );
    assert!(
        events.iter().any(|e| {
            matches!(
                e,
                RuntimeEvent::ToolCallFinished {
                    name,
                    summary: Some(s)
                } if name == "read_file" && s.contains(&good_path)
            )
        }),
        "failed reads must not replace the last successful read anchor: {events:?}"
    );
}

#[test]
fn no_anchor_followup_returns_deterministic_failure() {
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    let mut rt = make_runtime_in(Vec::<String>::new(), tmp.path());
    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read that file".into(),
        },
    );

    assert!(
        events.iter().any(|e| matches!(
            e,
            RuntimeEvent::AssistantMessageChunk(chunk) if chunk.contains("No previous file")
        )),
        "no-anchor prompt must produce deterministic runtime answer: {events:?}"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { .. })),
        "no-anchor prompt must not guess or dispatch tools: {events:?}"
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
                reason: RuntimeTerminalReason::ReadFileFailed,
                ..
            })
        ),
        "no-anchor prompt must terminate as runtime-owned read failure: {answer_source:?}"
    );
}

#[test]
fn unsupported_anchor_phrases_do_not_resolve_last_read_file() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::write(tmp.path().join("src/anchor.rs"), "fn anchor() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[read_file: src/anchor.rs]",
            "First read complete.",
            "Not an anchor.",
            "Still not an anchor.",
            "Also not an anchor.",
        ],
        tmp.path(),
    );
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read src/anchor.rs".into(),
        },
    );

    for phrase in ["open it", "read that", "open the second result"] {
        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: phrase.into(),
            },
        );
        assert!(
            !events.iter().any(
                |e| matches!(e, RuntimeEvent::ToolCallStarted { name } if name == "read_file")
            ),
            "unsupported phrase `{phrase}` must not resolve the last-read anchor: {events:?}"
        );
    }
}

#[test]
fn anchored_read_seeds_reads_this_turn_and_answer_phase_fires_after_model_initiated_read() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    for file in ["anchor.rs", "b.rs"] {
        fs::write(
            tmp.path().join("src").join(file),
            format!("fn {}() {{}}\n", file.replace(".rs", "")),
        )
        .unwrap();
    }

    let final_answer = "Read both files.";
    let mut rt = make_runtime_in(
        vec![
            "[read_file: src/anchor.rs]",
            "First read complete.",
            "[read_file: src/b.rs]",
            "[search_code: anchor]",
            final_answer,
        ],
        tmp.path(),
    );
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read src/anchor.rs".into(),
        },
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "read that file again".into(),
        },
    );

    assert!(
        !has_failed(&events),
        "turn must complete without failure: {events:?}"
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
        3,
        "turn 1 anchor + anchor re-read + one model-initiated read must succeed"
    );
    assert!(
        all_user.contains("The file was already read this turn"),
        "answer_phase correction must fire after model-initiated read in anchor turn"
    );
    assert_eq!(
        all_user.matches("=== tool_result: search_code ===").count(),
        0,
        "post-read search_code must be blocked by answer_phase gate"
    );

    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(last_assistant, Some(final_answer));
}

// Search anchor tests

#[test]
fn successful_search_code_updates_last_search_anchor() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("a.rs"), "fn needle() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec!["[search_code: needle]", "Search complete."],
        tmp.path(),
    );
    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "display the structure".into(),
        },
    );
    assert!(!has_failed(&events), "unexpected failure: {events:?}");

    let followup = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "repeat the last search".into(),
        },
    );
    assert!(
        followup
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { name } if name == "search_code")),
        "successful search must set the anchor so replay dispatches: {followup:?}"
    );
}

#[test]
fn repeat_last_search_dispatches_one_search_code() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("a.rs"), "fn needle() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec!["[search_code: needle]", "Search complete."],
        tmp.path(),
    );
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "display the structure".into(),
        },
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "repeat the last search".into(),
        },
    );

    let search_starts = events
        .iter()
        .filter(|e| matches!(e, RuntimeEvent::ToolCallStarted { name } if name == "search_code"))
        .count();
    assert_eq!(search_starts, 1, "replay must dispatch exactly one search");
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { name } if name == "read_file")),
        "search replay must not auto-read candidates: {events:?}"
    );
    assert!(
        events.iter().any(|e| matches!(
            e,
            RuntimeEvent::AssistantMessageChunk(chunk) if chunk.contains("Repeated the last search")
        )),
        "search replay must end with runtime-owned completion: {events:?}"
    );
}

#[test]
fn unscoped_search_replays_with_no_scope() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("one")).unwrap();
    fs::create_dir_all(tmp.path().join("two")).unwrap();
    fs::write(tmp.path().join("one/a.rs"), "fn needle_one() {}\n").unwrap();
    fs::write(tmp.path().join("two/b.rs"), "fn needle_two() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec!["[search_code: needle]", "Search complete."],
        tmp.path(),
    );
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "display the structure".into(),
        },
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "search the last query again".into(),
        },
    );

    assert!(
        events.iter().any(|e| {
            matches!(
                e,
                RuntimeEvent::ToolCallFinished {
                    name,
                    summary: Some(summary)
                } if name == "search_code"
                    && summary.contains("found 2 match(es)")
                    && summary.contains("needle")
            )
        }),
        "unscoped replay must search the whole project: {events:?}"
    );
}

#[test]
fn scoped_search_replay_uses_effective_prompt_scope() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::write(tmp.path().join("sandbox/in_scope.py"), "needle = True\n").unwrap();
    fs::write(tmp.path().join("src/outside.py"), "needle = False\n").unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: needle]",
            "[read_file: sandbox/in_scope.py]",
            "needle is in sandbox/in_scope.py.",
        ],
        tmp.path(),
    );
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is needle used in sandbox/".into(),
        },
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "run the last search again".into(),
        },
    );

    assert!(
        events.iter().any(|e| {
            matches!(
                e,
                RuntimeEvent::ToolCallFinished {
                    name,
                    summary: Some(summary)
                } if name == "search_code" && summary.contains("found 1 match(es)")
            )
        }),
        "scoped replay must preserve the effective prompt scope: {events:?}"
    );
}

#[test]
fn reset_clears_last_search_anchor() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("a.rs"), "fn needle() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec!["[search_code: needle]", "Search complete."],
        tmp.path(),
    );
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "display the structure".into(),
        },
    );
    collect_events(&mut rt, RuntimeRequest::Reset);

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "search that again".into(),
        },
    );
    assert!(
        events.iter().any(|e| matches!(
            e,
            RuntimeEvent::AssistantMessageChunk(chunk) if chunk.contains("No previous search")
        )),
        "reset must clear the search anchor: {events:?}"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { .. })),
        "reset search anchor must not dispatch tools: {events:?}"
    );
}

#[test]
fn no_search_anchor_replay_returns_deterministic_failure() {
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    let mut rt = make_runtime_in(Vec::<String>::new(), tmp.path());
    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "search that again".into(),
        },
    );

    assert!(
        events.iter().any(|e| matches!(
            e,
            RuntimeEvent::AssistantMessageChunk(chunk) if chunk.contains("No previous search")
        )),
        "no-search-anchor prompt must produce deterministic runtime answer: {events:?}"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { .. })),
        "no-search-anchor prompt must not dispatch tools: {events:?}"
    );
}

// Same-scope continuity tests

#[test]
fn same_scope_followup_reuses_last_successful_scoped_search_scope() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::write(
        tmp.path().join("sandbox/services/logging.py"),
        "def initialize_logging():\n    pass\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("sandbox/services/database.yaml"),
        "database: sqlite:///service.db\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("src/database.yaml"),
        "database: sqlite:///wrong.db\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: logging]",
            "[read_file: sandbox/services/logging.py]",
            "logging is initialized in sandbox/services/logging.py.",
            "[search_code: database]",
            "[read_file: sandbox/services/database.yaml]",
            "database is configured in sandbox/services/database.yaml.",
        ],
        tmp.path(),
    );
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Find where logging is initialized in sandbox/services/".into(),
        },
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Find where database is configured in the same folder".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let snapshot = rt.messages_snapshot();
    let search_result = snapshot
        .iter()
        .rev()
        .find(|m| m.content.contains("=== tool_result: search_code ==="))
        .map(|m| m.content.as_str())
        .unwrap_or("");
    assert!(
        search_result.contains("sandbox/services/database.yaml"),
        "same-scope search must include in-scope config: {search_result}"
    );
    assert!(
        !search_result.contains("src/database.yaml"),
        "same-scope search must exclude out-of-scope config: {search_result}"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("database is configured in sandbox/services/database.yaml.")
    );
}

#[test]
fn same_scope_followup_without_prior_search_fails_deterministically() {
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    let mut rt = make_runtime_in(Vec::<String>::new(), tmp.path());
    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Find where database is configured in the same folder".into(),
        },
    );

    assert!(
        events.iter().any(|e| matches!(
            e,
            RuntimeEvent::AssistantMessageChunk(chunk) if chunk.contains("No previous scoped search")
        )),
        "missing same-scope anchor must produce deterministic answer: {events:?}"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { .. })),
        "missing same-scope anchor must not dispatch tools: {events:?}"
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
        "missing same-scope anchor must terminate as insufficient evidence: {answer_source:?}"
    );
}

#[test]
fn same_scope_followup_after_unscoped_search_fails_deterministically() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("a.rs"), "fn needle() {}\n").unwrap();

    let mut rt = make_runtime_in(
        vec!["[search_code: needle]", "Search complete."],
        tmp.path(),
    );
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "tool check".into(),
        },
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Find where database is configured within the same scope".into(),
        },
    );

    assert!(
        events.iter().any(|e| matches!(
            e,
            RuntimeEvent::AssistantMessageChunk(chunk) if chunk.contains("No previous scoped search")
        )),
        "unscoped last search must not provide same-scope continuity: {events:?}"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { .. })),
        "unscoped last search must not fall back to global search: {events:?}"
    );
}

#[test]
fn same_scope_followup_explicit_concrete_path_takes_precedence() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox/config")).unwrap();
    fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
    fs::write(
        tmp.path().join("sandbox/config/database.yaml"),
        "database: sqlite:///config.db\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("sandbox/services/database.yaml"),
        "database: sqlite:///service.db\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: database]",
            "[read_file: sandbox/config/database.yaml]",
            "database is configured in sandbox/config/database.yaml.",
        ],
        tmp.path(),
    );
    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Find where database is configured in sandbox/config/ and in the same folder"
                .into(),
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
        search_result.contains("sandbox/config/database.yaml"),
        "explicit scope must be used even with same-scope phrase: {search_result}"
    );
    assert!(
        !search_result.contains("sandbox/services/database.yaml"),
        "same-scope phrase must not override explicit concrete scope: {search_result}"
    );
}
