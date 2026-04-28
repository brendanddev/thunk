use super::super::prompt;
use super::super::tool_surface::{
    select_tool_surface, tool_allowed_for_surface, SurfaceTool, ToolSurface,
};
use super::*;
use crate::llm::backend::Role;
use crate::tools::ToolInput;
use std::sync::{Arc, Mutex};

#[test]
fn tool_surface_defaults_to_retrieval_first_for_code_investigation_prompts() {
    assert_eq!(
        select_tool_surface("Where is TaskStatus used in sandbox/?", true, false, false),
        ToolSurface::RetrievalFirst
    );
    assert_eq!(
        select_tool_surface(
            "Find where database is configured in sandbox/",
            true,
            false,
            false
        ),
        ToolSurface::RetrievalFirst
    );
}

#[test]
fn tool_surface_selects_git_read_only_for_explicit_git_prompts() {
    for prompt_text in [
        "show git status",
        "show git diff",
        "show git log",
        "show working tree status",
        "show working-tree status",
        "show recent commits",
        "show latest commits",
        "git status",
        "git diff",
        "git log",
        "show recent git status",
        "show recent git diff",
        "show recent git log",
        "show latest git status",
        "show latest git diff",
        "show latest git log",
    ] {
        assert_eq!(
            select_tool_surface(prompt_text, false, false, false),
            ToolSurface::GitReadOnly,
            "prompt should select GitReadOnly: {prompt_text}"
        );
    }
}

#[test]
fn tool_surface_does_not_use_bare_overlapping_tokens() {
    // These prompts contain git-related words (status, diff, log, git, commit) but are
    // code investigation requests — the runtime sets investigation_required=true for them.
    // They must NOT be routed to GitReadOnly; RetrievalFirst is correct.
    for prompt_text in [
        "where is diff rendered",
        "find log initialization in sandbox/",
        "where is commit saved",
        "where is git integration implemented",
        "where is git status rendered",
    ] {
        assert_eq!(
            select_tool_surface(prompt_text, true, false, false),
            ToolSurface::RetrievalFirst,
            "prompt should remain RetrievalFirst: {prompt_text}"
        );
    }
}

// Phase 14.1.2 surface routing tests

#[test]
fn tool_surface_where_is_code_noun_selects_retrieval_first() {
    // "Where is <X> <code-noun>?" must reach RetrievalFirst without a secondary verb.
    for prompt_text in [
        "Where is the helper function?",
        "Where is the config module?",
        "Where is the parser file?",
        "Where is the command?",
        "Where is the tool?",
        "Where is the main class?",
    ] {
        assert_eq!(
            select_tool_surface(prompt_text, true, false, false),
            ToolSurface::RetrievalFirst,
            "code-noun where-is prompt should select RetrievalFirst: {prompt_text}"
        );
    }
}

#[test]
fn tool_surface_where_is_code_noun_does_not_match_non_code_nouns() {
    // "Where is <non-code-noun>?" must NOT promote to RetrievalFirst on the noun alone.
    // These prompts have no identifier signal, so investigation_required is false.
    for prompt_text in [
        "Where is the best place to start?",
        "Where is the issue?",
        "Where is the project summary?",
    ] {
        assert_eq!(
            select_tool_surface(prompt_text, false, false, false),
            ToolSurface::AnswerOnly,
            "non-code-noun where-is should remain AnswerOnly: {prompt_text}"
        );
    }
}

#[test]
fn tool_surface_bare_filename_selects_retrieval_first() {
    // A prompt containing a bare filename (known extension) must reach RetrievalFirst
    // via investigation_required even without a secondary condition verb.
    for prompt_text in [
        "What is in engine.rs?",
        "What does main.py do?",
        "Explain tool_surface.rs",
    ] {
        assert_eq!(
            select_tool_surface(prompt_text, true, false, false),
            ToolSurface::RetrievalFirst,
            "bare-filename prompt should select RetrievalFirst: {prompt_text}"
        );
    }
}

#[test]
fn tool_surface_explore_with_structural_cue_selects_retrieval_first() {
    for prompt_text in [
        "explore the files",
        "explore this directory",
        "explore the folder contents",
        "explore src/",
    ] {
        assert_eq!(
            select_tool_surface(prompt_text, false, false, false),
            ToolSurface::RetrievalFirst,
            "explore + structural cue should select RetrievalFirst: {prompt_text}"
        );
    }
}

#[test]
fn tool_surface_bare_explore_remains_answer_only() {
    assert_eq!(
        select_tool_surface("explore", false, false, false),
        ToolSurface::AnswerOnly,
        "bare explore without structural cue must remain AnswerOnly"
    );
    assert_eq!(
        select_tool_surface("explore ideas", false, false, false),
        ToolSurface::AnswerOnly,
        "explore + non-structural noun must remain AnswerOnly"
    );
}

#[test]
fn tool_surface_find_without_secondary_condition_is_answer_only() {
    // TODO(Phase 14.1.x): "find X in path/" and "find where X is computed" reach this test
    // as AnswerOnly because prompt_requires_investigation returns false for them — they lack
    // a recognised secondary condition ("implemented", "configured", "rendered", etc.) even
    // though they express clear code-search intent.  If benchmarks show these patterns are
    // common enough to warrant promotion, extend prompt_requests_directory_navigation to
    // catch path-scoped "find" queries, or add the missing secondary terms.  Do not widen
    // the policy until that data exists.
    for prompt_text in ["find where status is computed", "find git helper in src/"] {
        assert_eq!(
            select_tool_surface(prompt_text, false, false, false),
            ToolSurface::AnswerOnly,
            "prompt has no detected investigation intent and should be AnswerOnly: {prompt_text}"
        );
    }
}

#[test]
fn tool_surface_hint_renders_from_canonical_surface_membership() {
    assert_eq!(
        prompt::render_tool_surface_hint(
            ToolSurface::RetrievalFirst.as_str(),
            ToolSurface::RetrievalFirst.allowed_tool_names()
        ),
        "Active tool surface: RetrievalFirst. Available this turn: search_code, read_file, list_dir."
    );
    assert_eq!(
        prompt::render_tool_surface_hint(
            ToolSurface::GitReadOnly.as_str(),
            ToolSurface::GitReadOnly.allowed_tool_names()
        ),
        "Active tool surface: GitReadOnly. Available this turn: git_status, git_diff, git_log."
    );
}

#[test]
fn tool_surface_enforcement_uses_canonical_surface_membership() {
    let inputs = [
        ToolInput::SearchCode {
            query: "needle".into(),
            path: None,
        },
        ToolInput::ReadFile {
            path: "src/lib.rs".into(),
        },
        ToolInput::ListDir { path: ".".into() },
        ToolInput::GitStatus,
        ToolInput::GitDiff,
        ToolInput::GitLog,
    ];

    for surface in [ToolSurface::RetrievalFirst, ToolSurface::GitReadOnly] {
        for input in &inputs {
            let tool = SurfaceTool::from_input(input).expect("test inputs are surface-controlled");
            assert_eq!(
                tool_allowed_for_surface(input, surface),
                surface.tools().contains(&tool),
                "surface enforcement must match canonical membership for {} on {}",
                input.tool_name(),
                surface.as_str()
            );
        }
    }
}

// Phase 14.1.3 retrieval tool choice discipline tests

#[test]
fn path_qualified_file_prompt_reads_before_first_model_generation() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
    fs::write(
        tmp.path().join("sandbox/main.py"),
        "def main():\n    pass\n",
    )
    .unwrap();

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut rt = Runtime::new(
        &Config::default(),
        tmp.path(),
        Box::new(RecordingBackend::new(
            vec!["sandbox/main.py defines main()."],
            Arc::clone(&requests),
        )),
        default_registry(tmp.path().to_path_buf()),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "What is in sandbox/main.py?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let first_tool = events.iter().find_map(|e| match e {
        RuntimeEvent::ToolCallStarted { name } => Some(name.clone()),
        _ => None,
    });
    assert_eq!(first_tool.as_deref(), Some("read_file"));

    let requests = requests.lock().unwrap();
    assert_eq!(
        requests.len(),
        1,
        "model must not generate before read_file"
    );
    let first = requests.first().expect("backend request must be recorded");
    assert!(
        first
            .messages
            .iter()
            .any(|m| m.content.contains("=== tool_result: read_file ===")),
        "first backend request must occur after read_file"
    );
}

#[test]
fn explicit_directory_prompt_lists_before_first_model_generation() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
    fs::write(
        tmp.path().join("sandbox/main.py"),
        "def main():\n    pass\n",
    )
    .unwrap();

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut rt = Runtime::new(
        &Config::default(),
        tmp.path(),
        Box::new(RecordingBackend::new(
            vec!["sandbox contains main.py."],
            Arc::clone(&requests),
        )),
        default_registry(tmp.path().to_path_buf()),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "explore sandbox/".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let first_tool = events.iter().find_map(|e| match e {
        RuntimeEvent::ToolCallStarted { name } => Some(name.clone()),
        _ => None,
    });
    assert_eq!(first_tool.as_deref(), Some("list_dir"));

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 1, "model must not generate before list_dir");
    let first = requests.first().expect("backend request must be recorded");
    assert!(
        first
            .messages
            .iter()
            .any(|m| m.content.contains("=== tool_result: list_dir ===")),
        "first backend request must occur after list_dir"
    );
}

#[test]
fn structural_directory_prompt_lists_before_first_model_generation() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("main.py"), "def main():\n    pass\n").unwrap();

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut rt = Runtime::new(
        &Config::default(),
        tmp.path(),
        Box::new(RecordingBackend::new(
            vec!["The project root contains main.py."],
            Arc::clone(&requests),
        )),
        default_registry(tmp.path().to_path_buf()),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "explore the files".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let first_tool = events.iter().find_map(|e| match e {
        RuntimeEvent::ToolCallStarted { name } => Some(name.clone()),
        _ => None,
    });
    assert_eq!(first_tool.as_deref(), Some("list_dir"));

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 1, "model must not generate before list_dir");
    let first = requests.first().expect("backend request must be recorded");
    assert!(
        first
            .messages
            .iter()
            .any(|m| m.content.contains("=== tool_result: list_dir ===")),
        "first backend request must occur after list_dir"
    );
}

#[test]
fn investigation_prompt_still_generates_before_first_tool() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
    fs::write(
        tmp.path().join("sandbox/helper.py"),
        "def helper():\n    return 1\n",
    )
    .unwrap();

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut rt = Runtime::new(
        &Config::default(),
        tmp.path(),
        Box::new(RecordingBackend::new(
            vec![
                "[search_code: helper]",
                "[read_file: sandbox/helper.py]",
                "The helper function is in sandbox/helper.py.",
            ],
            Arc::clone(&requests),
        )),
        default_registry(tmp.path().to_path_buf()),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is helper function in sandbox/".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let first_tool = events.iter().find_map(|e| match e {
        RuntimeEvent::ToolCallStarted { name } => Some(name.clone()),
        _ => None,
    });
    assert_eq!(first_tool.as_deref(), Some("search_code"));

    let requests = requests.lock().unwrap();
    assert_eq!(
        requests.len(),
        3,
        "investigation turn must still generate first"
    );
    let first = requests.first().expect("backend request must be recorded");
    assert!(
        !first
            .messages
            .iter()
            .any(|m| m.content.contains("=== tool_result:")),
        "initial investigation request must reach the model before any tool result exists"
    );
}

#[test]
fn bare_explore_remains_answer_only_with_no_seeded_tool() {
    let (mut rt, requests) = make_runtime_with_recorded_requests(vec!["Done."]);
    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "explore".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { .. })),
        "bare explore must not seed a retrieval tool"
    );

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 1, "bare explore must generate directly");
    let first = requests.first().expect("backend request must be recorded");
    assert!(
        first.messages.iter().any(|m| {
            m.role == Role::System
                && m.content
                    == "Active tool surface: AnswerOnly. No tools are available. Provide your final answer now."
        }),
        "bare explore must remain AnswerOnly"
    );
}

#[test]
fn answer_only_surface_sent_for_plain_conversational_prompt() {
    // Phase 14.1: plain conversational prompts with no investigation intent, no mutation
    // request, and no direct read path receive the AnswerOnly surface — not RetrievalFirst.
    let (mut rt, requests) = make_runtime_with_recorded_requests(vec!["Done."]);
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "hello".into(),
        },
    );

    let requests = requests.lock().unwrap();
    let first = requests.first().expect("backend request must be recorded");
    assert!(
        first.messages.iter().any(|m| {
            m.role == Role::System
                && m.content
                    == "Active tool surface: AnswerOnly. No tools are available. Provide your final answer now."
        }),
        "AnswerOnly surface hint must be injected for plain conversational prompt: {:?}",
        first.messages
    );
}

#[test]
fn git_read_only_surface_hint_is_sent_to_model() {
    let (mut rt, requests) = make_runtime_with_recorded_requests(vec!["Done."]);
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "show git status".into(),
        },
    );

    let requests = requests.lock().unwrap();
    let first = requests.first().expect("backend request must be recorded");
    assert!(
        first.messages.iter().any(|m| {
            m.role == Role::System
                && m.content
                    == "Active tool surface: GitReadOnly. Available this turn: git_status, git_diff, git_log."
        }),
        "GitReadOnly surface hint must be injected into backend request: {:?}",
        first.messages
    );
}

#[test]
fn tool_surface_hint_is_ephemeral_not_persisted() {
    let (mut rt, _requests) = make_runtime_with_recorded_requests(vec!["Done."]);
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "hello".into(),
        },
    );

    assert!(
        !rt.messages_snapshot().iter().any(|m| {
            m.content
                .starts_with("Active tool surface: RetrievalFirst. Available this turn:")
                || m.content
                    .starts_with("Active tool surface: GitReadOnly. Available this turn:")
        }),
        "surface hint must not be persisted in conversation history"
    );
}

#[test]
fn tool_surface_hint_does_not_replace_original_user_prompt() {
    let (mut rt, requests) = make_runtime_with_recorded_requests(vec!["Done."]);
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "where is serde used".into(),
        },
    );

    let requests = requests.lock().unwrap();
    let first = requests.first().expect("backend request must be recorded");
    assert!(
        first
            .messages
            .iter()
            .any(|m| m.role == Role::User && m.content == "where is serde used"),
        "original user prompt must remain in backend request: {:?}",
        first.messages
    );
    assert!(
        first.messages.iter().any(|m| {
            m.role == Role::System
                && m.content
                    .starts_with("Active tool surface: RetrievalFirst. Available this turn:")
        }),
        "surface hint must be additional system context"
    );
}

#[test]
fn mutation_turn_still_receives_surface_hint() {
    let (mut rt, requests) = make_runtime_with_recorded_requests(vec!["Done."]);
    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "create a new file src/new.rs".into(),
        },
    );

    let requests = requests.lock().unwrap();
    let first = requests.first().expect("backend request must be recorded");
    assert!(
        first.messages.iter().any(|m| {
            m.role == Role::System
                && m.content
                    == "Active tool surface: RetrievalFirst. Available this turn: search_code, read_file, list_dir."
        }),
        "mutation-intent turns still expose active surface hint: {:?}",
        first.messages
    );
}

#[test]
fn answer_only_surface_hint_declares_no_tools() {
    // Phase 12.0.1: AnswerOnly surface hint must list zero tools and
    // explicitly tell the model to provide its final answer.
    let hint = prompt::render_tool_surface_hint(
        ToolSurface::AnswerOnly.as_str(),
        ToolSurface::AnswerOnly.allowed_tool_names(),
    );
    assert!(
        hint.contains("AnswerOnly"),
        "hint must name the AnswerOnly surface: {hint}"
    );
    assert!(
        !hint.contains("search_code"),
        "AnswerOnly surface must not offer search_code: {hint}"
    );
    assert!(
        !hint.contains("read_file"),
        "AnswerOnly surface must not offer read_file: {hint}"
    );
    assert!(
        !hint.contains("Available this turn:"),
        "AnswerOnly surface must not use the tool-list format: {hint}"
    );
}

#[test]
fn answer_only_surface_hint_sent_to_model_during_post_read_synthesis() {
    // Phase 12.0.1: after a successful read the runtime sets answer_phase = PostRead.
    // The synthesis generation must receive the AnswerOnly surface hint so the model
    // is not offered any tools — eliminating the post_evidence_tool_call_rejected round.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
    fs::write(tmp.path().join("sandbox/main.py"), "def main(): pass\n").unwrap();

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut rt = Runtime::new(
        &Config::default(),
        tmp.path(),
        Box::new(RecordingBackend::new(
            vec![
                "[read_file: sandbox/main.py]", // round 1: model reads the requested file
                "Here is what I found.",        // round 2: synthesis — must get AnswerOnly hint
            ],
            Arc::clone(&requests),
        )),
        default_registry(tmp.path().to_path_buf()),
    );

    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Read sandbox/main.py".into(),
        },
    );

    let requests = requests.lock().unwrap();
    assert_eq!(
        requests.len(),
        2,
        "expected exactly 2 backend calls (read + synthesis): {requests:?}"
    );

    let synthesis = &requests[1];
    // Find the ephemeral per-turn surface hint (starts with "Active tool surface:").
    // This is distinct from the main system prompt, which always describes all tools.
    let surface_hint = synthesis
        .messages
        .iter()
        .find(|m| m.role == Role::System && m.content.starts_with("Active tool surface:"))
        .expect("synthesis request must carry a surface hint");
    assert!(
        surface_hint.content.contains("AnswerOnly"),
        "synthesis surface hint must name AnswerOnly: {}",
        surface_hint.content
    );
    assert!(
        !surface_hint.content.contains("search_code"),
        "AnswerOnly surface hint must not offer search_code: {}",
        surface_hint.content
    );
    assert!(
        !surface_hint.content.contains("read_file"),
        "AnswerOnly surface hint must not offer read_file: {}",
        surface_hint.content
    );
}

#[test]
fn answer_only_surface_hint_sent_after_second_runtime_owned_usage_read() {
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
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
        tmp.path().join("sandbox/models").join("enums.py"),
        "class TaskStatus(str, Enum):\n    UNUSED_ENUM_MEMBER = \"unused\"\n",
    )
    .unwrap();

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut rt = Runtime::new(
        &Config::default(),
        tmp.path(),
        Box::new(RecordingBackend::new(
            vec![
                "[search_code: TaskStatus]",
                "TaskStatus is used in sandbox/services/runner_primary.py and sandbox/services/runner_secondary.py.",
            ],
            Arc::clone(&requests),
        )),
        default_registry(tmp.path().to_path_buf()),
    );

    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus used in sandbox/services/".into(),
        },
    );

    let requests = requests.lock().unwrap();
    assert_eq!(
        requests.len(),
        2,
        "expected exactly 2 backend calls (search + synthesis after two runtime-owned reads): {requests:?}"
    );

    let synthesis = &requests[1];
    let surface_hint = synthesis
        .messages
        .iter()
        .find(|m| m.role == Role::System && m.content.starts_with("Active tool surface:"))
        .expect("synthesis request must carry a surface hint");
    assert!(
        surface_hint.content.contains("AnswerOnly"),
        "synthesis surface hint must name AnswerOnly after the second runtime-owned usage read: {}",
        surface_hint.content
    );
    assert!(
        !surface_hint.content.contains("search_code"),
        "AnswerOnly surface hint must not offer search_code: {}",
        surface_hint.content
    );
    assert!(
        !surface_hint.content.contains("read_file"),
        "AnswerOnly surface hint must not offer read_file: {}",
        surface_hint.content
    );
}
