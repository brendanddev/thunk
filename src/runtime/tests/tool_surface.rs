use super::*;
use crate::llm::backend::Role;
use crate::tools::ToolInput;
use super::super::prompt;
use super::super::tool_surface::{
    select_tool_surface, tool_allowed_for_surface, SurfaceTool, ToolSurface,
};

#[test]
fn tool_surface_defaults_to_retrieval_first_for_code_investigation_prompts() {
    assert_eq!(
        select_tool_surface("Where is TaskStatus used in sandbox/?"),
        ToolSurface::RetrievalFirst
    );
    assert_eq!(
        select_tool_surface("Find where database is configured in sandbox/"),
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
            select_tool_surface(prompt_text),
            ToolSurface::GitReadOnly,
            "prompt should select GitReadOnly: {prompt_text}"
        );
    }
}

#[test]
fn tool_surface_does_not_use_bare_overlapping_tokens() {
    for prompt_text in [
        "find where status is computed",
        "where is diff rendered",
        "find log initialization in sandbox/",
        "where is commit saved",
        "where is git integration implemented",
        "find git helper in src/",
        "where is git status rendered",
    ] {
        assert_eq!(
            select_tool_surface(prompt_text),
            ToolSurface::RetrievalFirst,
            "prompt should remain RetrievalFirst: {prompt_text}"
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
            let tool =
                SurfaceTool::from_input(input).expect("test inputs are surface-controlled");
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

#[test]
fn retrieval_first_surface_hint_is_sent_to_model() {
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
                    == "Active tool surface: RetrievalFirst. Available this turn: search_code, read_file, list_dir."
        }),
        "RetrievalFirst surface hint must be injected into backend request: {:?}",
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
