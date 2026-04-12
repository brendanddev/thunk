use super::evidence::{
    has_relevant_file_evidence, investigation_readiness, targeted_investigation_followup,
    InvestigationReadiness,
};
use super::intent::suggested_search_query;
use super::prompting::with_progress_heartbeat_interval;
use super::*;
use std::fs;
use std::path::Path;
use std::sync::{mpsc, Mutex, OnceLock};
use std::time::Duration;

use crate::events::InferenceEvent;
use crate::inference::session::investigation::{
    InvestigationAnchor, InvestigationLatencyPolicy, InvestigationResolution,
};

struct ScriptedBackend {
    name: String,
    responses: std::sync::Mutex<Vec<String>>,
}

impl ScriptedBackend {
    fn new(responses: Vec<&str>) -> Self {
        Self::with_name("scripted", responses)
    }

    fn with_name(name: &str, responses: Vec<&str>) -> Self {
        Self {
            name: name.to_string(),
            responses: std::sync::Mutex::new(
                responses.into_iter().rev().map(str::to_string).collect(),
            ),
        }
    }
}

impl InferenceBackend for ScriptedBackend {
    fn generate(&self, _messages: &[Message], tx: Sender<InferenceEvent>) -> Result<()> {
        let next = self
            .responses
            .lock()
            .expect("responses lock")
            .pop()
            .unwrap_or_default();
        let _ = tx.send(InferenceEvent::Token(next));
        Ok(())
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}

struct InspectingBackend {
    steps: std::sync::Mutex<Vec<(Option<String>, String)>>,
}

impl InspectingBackend {
    fn new(steps: Vec<(Option<&str>, &str)>) -> Self {
        Self {
            steps: std::sync::Mutex::new(
                steps
                    .into_iter()
                    .rev()
                    .map(|(expected, response)| {
                        (expected.map(str::to_string), response.to_string())
                    })
                    .collect(),
            ),
        }
    }
}

impl InferenceBackend for InspectingBackend {
    fn generate(&self, messages: &[Message], tx: Sender<InferenceEvent>) -> Result<()> {
        let (expected, response) = self
            .steps
            .lock()
            .expect("steps lock")
            .pop()
            .unwrap_or((None, String::new()));
        if let Some(expected) = expected {
            assert!(
                messages
                    .iter()
                    .any(|message| message.content.contains(&expected)),
                "expected tool-loop messages to contain `{expected}`, got:\n{}",
                messages
                    .iter()
                    .map(|message| format!("{}: {}", message.role, message.content))
                    .collect::<Vec<_>>()
                    .join("\n\n")
            );
        }
        let _ = tx.send(InferenceEvent::Token(response));
        Ok(())
    }

    fn name(&self) -> String {
        "inspecting".to_string()
    }
}

fn with_test_cwd<T>(root: &Path, run: impl FnOnce() -> T) -> T {
    static CWD_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = CWD_GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("cwd guard");
    let original = std::env::current_dir().expect("cwd");
    std::env::set_current_dir(root).expect("set cwd");
    let result = run();
    std::env::set_current_dir(original).expect("restore cwd");
    result
}

#[test]
fn detect_tool_loop_intent_handles_typoed_where_prompt() {
    assert_eq!(
        detect_tool_loop_intent("WWhere is memory retrieval implemented?"),
        Some(ToolLoopIntent::CodeNavigation)
    );
    assert_eq!(
        detect_tool_loop_intent("Where is eco mode configged"),
        Some(ToolLoopIntent::ConfigLocate)
    );
    assert_eq!(
        suggested_search_query(
            "Where is session restore implemented?",
            ToolLoopIntent::CodeNavigation
        )
        .as_deref(),
        Some("load_most_recent")
    );
}

#[test]
fn tool_loop_system_prompt_uses_read_only_tools() {
    let dir = std::env::temp_dir().join("params-tool-loop-prompt");
    let _ = std::fs::create_dir_all(dir.join("docs/context"));
    let _ = std::fs::write(dir.join("README.md"), "# params-cli");
    let _ = std::fs::write(dir.join("docs/context/CLAUDE.md"), "# CLAUDE");

    let prompt = build_tool_loop_system_prompt(
        &ToolRegistry::default(),
        &dir,
        ToolLoopIntent::CodeNavigation,
        false,
    );
    assert!(prompt.contains("read-only repo inspection tools"));
    assert!(prompt.contains("read_file"));
    assert!(prompt.contains("search"));
    assert!(!prompt.contains("write_file"));
    assert!(prompt.contains("README.md"));
    assert!(prompt.contains("docs/context/CLAUDE.md"));

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn llama_tool_loop_always_uses_bounded_tool_result_context() {
    assert_eq!(
        tool_loop_result_limit("llama.cpp · qwen", false),
        Some(1800)
    );
    assert_eq!(tool_loop_result_limit("llama.cpp · qwen", true), Some(900));
    assert_eq!(tool_loop_result_limit("openai_compat · gpt-5", false), None);
}

#[test]
fn heartbeat_emits_progress_updates_during_long_internal_generation() {
    let (tx, rx) = mpsc::channel();
    let result = with_progress_heartbeat_interval(
        &tx,
        "planning investigation...",
        Duration::from_millis(25),
        || {
            std::thread::sleep(Duration::from_millis(70));
            Ok::<_, crate::error::ParamsError>("done")
        },
    )
    .expect("heartbeat wrapper should return inner result");

    assert_eq!(result, "done");
    let traces = rx
        .try_iter()
        .filter_map(|event| match event {
            InferenceEvent::Trace(trace) => Some(trace.label),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        traces
            .iter()
            .any(|label| label.contains("planning investigation...")),
        "expected heartbeat trace updates, got {traces:?}"
    );
}

#[test]
fn tool_loop_seed_messages_drop_old_history_for_standalone_query() {
    let seed = build_tool_loop_seed_messages(
        &[
            Message::system("old system"),
            Message::user(&"previous context ".repeat(800)),
            Message::assistant(&"previous answer ".repeat(800)),
            Message::user("Where is session restore implemented?"),
        ],
        "tool loop system",
        "Where is session restore implemented?",
    );

    assert_eq!(seed.len(), 2);
    assert_eq!(seed[0].role, "system");
    assert_eq!(seed[0].content, "tool loop system");
    assert_eq!(seed[1].role, "user");
    assert_eq!(seed[1].content, "Where is session restore implemented?");
}

#[test]
fn tool_loop_seed_messages_keep_short_context_for_referential_follow_up() {
    let seed = build_tool_loop_seed_messages(
        &[
            Message::system("old system"),
            Message::user("Where is session restore implemented?"),
            Message::assistant("It looks related to session persistence."),
            Message::user("What calls it?"),
        ],
        "tool loop system",
        "What calls it?",
    );

    assert_eq!(seed.len(), 4);
    assert_eq!(seed[1].content, "Where is session restore implemented?");
    assert_eq!(seed[2].content, "It looks related to session persistence.");
    assert_eq!(seed[3].content, "What calls it?");
}

#[test]
fn read_only_tool_loop_bootstraps_with_shaped_search_target() {
    let dir = std::env::temp_dir().join("params-tool-loop-bootstrap-hint");
    let _ = fs::create_dir_all(dir.join("src/session"));
    let _ = fs::write(
        dir.join("src/session/mod.rs"),
        "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
    );

    let backend = InspectingBackend::new(vec![
        (
            Some("Start with `[search: load_most_recent]`"),
            "[search: load_most_recent]",
        ),
        (
            Some("Next read: `[read_file: src/session/mod.rs]`"),
            "[read_file: src/session/mod.rs]",
        ),
        (
            None,
            "The implementation is in `src/session/mod.rs` at line 1.",
        ),
    ]);
    let (tx, _rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::CodeNavigation,
            "Where is session restore implemented?",
            &[
                Message::system("system"),
                Message::user("Where is session restore implemented?"),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
    })
    .expect("tool loop");

    assert_eq!(
        outcome.final_response,
        "The implementation is in `src/session/mod.rs` at line 1."
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn read_only_tool_loop_runs_search_then_answers() {
    let dir = std::env::temp_dir().join("params-tool-loop-run");
    let _ = std::fs::create_dir_all(dir.join("src/session"));
    let _ = std::fs::write(
        dir.join("src/session/mod.rs"),
        "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
    );

    let backend = ScriptedBackend::new(vec![
        "[search: load_most_recent]",
        "[read_file: src/session/mod.rs]",
        "The implementation is in `src/session/mod.rs` at line 1.",
    ]);
    let (tx, _rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::CodeNavigation,
            "Where is session restore implemented?",
            &[
                Message::system("system"),
                Message::user("Where is session restore implemented?"),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
    })
    .expect("tool loop");

    assert_eq!(
        outcome.final_response,
        "The implementation is in `src/session/mod.rs` at line 1."
    );
    assert_eq!(outcome.tool_results.len(), 2);
    assert_eq!(outcome.tool_results[0].tool_name, "search");
    assert_eq!(outcome.tool_results[1].tool_name, "read_file");

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn llama_tool_loop_bootstraps_search_and_read_before_first_generation() {
    let dir = std::env::temp_dir().join("params-tool-loop-llama-bootstrap");
    let _ = fs::create_dir_all(dir.join("src/session"));
    let _ = fs::write(
        dir.join("src/session/mod.rs"),
        "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
    );

    let backend = ScriptedBackend::with_name(
        "llama.cpp · qwen",
        vec!["The implementation is in `src/session/mod.rs` at line 1."],
    );
    let (tx, _rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::CodeNavigation,
            "Where is session restore implemented?",
            &[
                Message::system("system"),
                Message::user("Where is session restore implemented?"),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
    })
    .expect("tool loop");

    assert_eq!(
        outcome.final_response,
        "The implementation is in `src/session/mod.rs` at line 1."
    );
    assert_eq!(outcome.tool_results.len(), 2);
    assert_eq!(outcome.tool_results[0].tool_name, "search");
    assert_eq!(outcome.tool_results[1].tool_name, "read_file");

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn grounded_tool_loop_generation_streams_final_answer_tokens() {
    let dir = std::env::temp_dir().join("params-tool-loop-streaming");
    let _ = fs::create_dir_all(dir.join("src/session"));
    let _ = fs::write(
        dir.join("src/session/mod.rs"),
        "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
    );

    let final_answer = "The implementation is in `src/session/mod.rs` at line 1.";
    let backend = ScriptedBackend::new(vec![
        "[search: load_most_recent]",
        "[read_file: src/session/mod.rs]",
        final_answer,
    ]);
    let (tx, rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::CodeNavigation,
            "Where is session restore implemented?",
            &[
                Message::system("system"),
                Message::user("Where is session restore implemented?"),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
    })
    .expect("tool loop");

    assert_eq!(outcome.final_response, final_answer);
    assert!(outcome.streamed_final_response);
    let streamed = rx
        .try_iter()
        .filter_map(|event| match event {
            InferenceEvent::Token(text) => Some(text),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    assert!(
        streamed.contains(final_answer),
        "expected streamed final answer tokens, got {streamed:?}"
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn tool_loop_does_not_stream_intermediate_tool_tags() {
    let dir = std::env::temp_dir().join("params-tool-loop-no-tag-stream");
    let _ = fs::create_dir_all(dir.join("src/session"));
    let _ = fs::write(
        dir.join("src/session/mod.rs"),
        "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
    );

    let backend = ScriptedBackend::new(vec![
        "[search: load_most_recent]",
        "[read_file: src/session/mod.rs]",
        "The implementation is in `src/session/mod.rs:1`.",
    ]);
    let (tx, rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::CodeNavigation,
            "Where is session restore implemented?",
            &[
                Message::system("system"),
                Message::user("Where is session restore implemented?"),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
    })
    .expect("tool loop");

    let streamed = rx
        .try_iter()
        .filter_map(|event| match event {
            InferenceEvent::Token(text) => Some(text),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    assert_eq!(
        outcome.final_response,
        "The implementation is in `src/session/mod.rs` at line 1."
    );
    assert!(
        !streamed.contains("[search:"),
        "intermediate tool tags must never be streamed to the UI"
    );
    assert!(
        !streamed.contains("[read_file:"),
        "intermediate tool tags must never be streamed to the UI"
    );
    assert!(streamed.contains("The implementation is in"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn grounded_answer_guidance_prefers_definition_and_real_body_lines() {
    let guidance = grounded_answer_guidance(
            ToolLoopIntent::CodeNavigation,
            "Where is session restore implemented?",
            None,
            &[ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/session/mod.rs".to_string(),
                output: "File: src/session/mod.rs\nLines: 6\n\n```\npub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    let Some(summary) = self.list_sessions()?.into_iter().next() else {\n        return Ok(None);\n    };\n    self.load_session_by_id(&summary.id)\n}\n```\n".to_string(),
            }],
        )
        .expect("guidance");

    assert!(guidance.contains("Primary implementation: src/session/mod.rs:1"));
    // Body lines must be listed one-per-line with file:line prefix so the model cannot
    // confuse which number belongs to which content.
    assert!(
        guidance.contains("src/session/mod.rs:"),
        "body lines must use file:line prefix"
    );
    assert!(guidance.contains("self.load_session_by_id(&summary.id)"));
    assert!(guidance.contains("Do not include code fences"));
    assert!(guidance.contains("Do not quote full function bodies"));
    // Verbatim-quoting rule: must explicitly ban identifier drift.
    assert!(
        guidance.contains("do not rename or substitute"),
        "guidance must forbid renaming identifiers"
    );
    // Anti-hedging: must explicitly list banned words.
    assert!(
        guidance.contains("presumably"),
        "guidance must list `presumably` as a banned hedging word"
    );
    assert!(
        guidance.contains("Do not use hedging words"),
        "guidance must explicitly ban hedging words"
    );
}

#[test]
fn grounded_answer_guidance_captures_ok_none_and_load_by_id_for_load_most_recent() {
    // Regression: both the Ok(None) early-return line and the load_session_by_id call
    // must appear in the observed body lines so the model cannot add hedging around them.
    let fixture_output = concat!(
        "File: src/session/mod.rs\nLines: 6\n\n```\n",
        "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n",
        "    let Some(summary) = self.list_sessions()?.into_iter().next() else {\n",
        "        return Ok(None);\n",
        "    };\n",
        "    self.load_session_by_id(&summary.id)\n",
        "}\n",
        "```\n"
    );
    let guidance = grounded_answer_guidance(
        ToolLoopIntent::CodeNavigation,
        "Where is session restore implemented?",
        None,
        &[ToolResult {
            tool_name: "read_file".to_string(),
            argument: "src/session/mod.rs".to_string(),
            output: fixture_output.to_string(),
        }],
    )
    .expect("guidance");

    // Both critical lines must be present so the model has direct evidence.
    assert!(
        guidance.contains("Ok(None)"),
        "guidance must expose the Ok(None) early-return line"
    );
    assert!(
        guidance.contains("load_session_by_id(&summary.id)"),
        "guidance must expose the load_session_by_id call line"
    );
    // The anti-hedging rule must be present.
    assert!(guidance.contains("presumably"));
    assert!(guidance.contains("Do not use hedging words"));
}

#[test]
fn grounded_answer_guidance_anti_hedging_is_in_final_synthesis_context() {
    // Regression: when the final synthesis step runs, the messages must include the
    // anti-hedging instruction so the model cannot produce `presumably` or `likely` even
    // when the body lines are directly observed.
    let dir = std::env::temp_dir().join("params-tool-loop-anti-hedge");
    let _ = fs::create_dir_all(dir.join("src/session"));
    let _ = fs::write(
        dir.join("src/session/mod.rs"),
        concat!(
            "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n",
            "    let Some(summary) = self.list_sessions()?.into_iter().next() else {\n",
            "        return Ok(None);\n",
            "    };\n",
            "    self.load_session_by_id(&summary.id)\n",
            "}\n"
        ),
    );

    let backend = InspectingBackend::new(vec![
            (
                Some("Start with `[search: load_most_recent]`"),
                "[search: load_most_recent]",
            ),
            (
                Some("Next read: `[read_file: src/session/mod.rs]`"),
                "[read_file: src/session/mod.rs]",
            ),
            (
                // The final synthesis prompt must contain both the anti-hedging instruction
                // and the verbatim-quoting rule so the model cannot rename identifiers.
                Some("do not rename or substitute"),
                "Session restore is at src/session/mod.rs:1. Line 3 returns Ok(None) when no session exists. Line 5 calls load_session_by_id(&summary.id).",
            ),
        ]);
    let (tx, _rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::CodeNavigation,
            "Where is session restore implemented?",
            &[
                Message::system("system"),
                Message::user("Where is session restore implemented?"),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
    })
    .expect("tool loop");

    assert!(outcome.final_response.contains("src/session/mod.rs"));
    assert!(outcome.final_response.contains("Ok(None)"));
    assert!(outcome
        .final_response
        .contains("load_session_by_id(&summary.id)"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn grounded_answer_guidance_body_lines_use_per_line_file_line_format() {
    // Regression for line-number drift: body lines must be formatted one-per-line with
    // a `file:line` prefix so the model cannot confuse which number belongs to which content.
    // Previously they were comma-separated on a single line, causing the model to cite the
    // wrong line number (e.g. "line 275" instead of "line 276").
    let fixture_output = concat!(
        "File: src/session/mod.rs\nLines: 6\n\n```\n",
        "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n",
        "    let Some(summary) = self.list_sessions()?.into_iter().next() else {\n",
        "        return Ok(None);\n",
        "    };\n",
        "    self.load_session_by_id(&summary.id)\n",
        "}\n",
        "```\n"
    );
    let guidance = grounded_answer_guidance(
        ToolLoopIntent::CodeNavigation,
        "Where is session restore implemented?",
        None,
        &[ToolResult {
            tool_name: "read_file".to_string(),
            argument: "src/session/mod.rs".to_string(),
            output: fixture_output.to_string(),
        }],
    )
    .expect("guidance");

    // Each body line must appear on its own line with a `path:N` prefix, not comma-separated.
    // This ensures the model can unambiguously map line numbers to line content.
    assert!(
        guidance.contains("src/session/mod.rs:3"),
        "Ok(None) must appear as src/session/mod.rs:3"
    );
    assert!(
        guidance.contains("src/session/mod.rs:5"),
        "load_session_by_id must appear as src/session/mod.rs:5"
    );
    // Must not format as comma-separated on a single line.
    let body_section = guidance
        .lines()
        .skip_while(|l| !l.contains("Observed body lines"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !body_section.contains("Ok(None)`) ,"),
        "body lines must not be comma-separated"
    );
    // Verbatim rule must be present.
    assert!(guidance.contains("do not rename or substitute"));
}

#[test]
fn tool_loop_formats_read_file_results_as_compact_plaintext_evidence() {
    let message = format_tool_loop_results_with_limit(
            ToolLoopIntent::CodeNavigation,
            "Where is session restore implemented?",
            None,
            &[ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/session/mod.rs".to_string(),
                output: "File: src/session/mod.rs\nLines: 6\n\n```\npub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    let Some(summary) = self.list_sessions()?.into_iter().next() else {\n        return Ok(None);\n    };\n    self.load_session_by_id(&summary.id)\n}\n```\n".to_string(),
            }],
            None,
        )
        .expect("formatted tool results");

    assert!(message.contains("Primary implementation: src/session/mod.rs:1"));
    assert!(message.contains("Observed body lines:"));
    assert!(message.contains("self.load_session_by_id(&summary.id)"));
    assert!(!message.contains("```"));
}

#[test]
fn grounded_answer_guidance_summarizes_loaded_file_from_observed_declarations() {
    let guidance = grounded_answer_guidance(
        ToolLoopIntent::CodeNavigation,
        "What does this file do?",
        None,
        &[ToolResult {
            tool_name: "read_file".to_string(),
            argument: "src/tui/state/helpers.rs".to_string(),
            output: concat!(
                "File: src/tui/state/helpers.rs\n",
                "Lines: 5\n\n",
                "```\n",
                "use crate::events::ProgressStatus;\n",
                "use super::AppState;\n\n",
                "pub fn summarize_trace_steps() {}\n",
                "fn extract_preview_lines() {}\n",
                "```\n"
            )
            .to_string(),
        }],
    )
    .expect("guidance");

    assert!(guidance.contains("Loaded file: `src/tui/state/helpers.rs`"));
    assert!(guidance.contains("Observed declarations:"));
    assert!(guidance.contains("src/tui/state/helpers.rs:1 `use crate::events::ProgressStatus;`"));
    assert!(guidance.contains("src/tui/state/helpers.rs:4 `pub fn summarize_trace_steps() {}`"));
    assert!(guidance.contains("Do not mention search results"));
}

#[test]
fn file_context_question_bootstraps_from_loaded_file_instead_of_searching_wrapper_text() {
    let dir = std::env::temp_dir().join("params-tool-loop-file-context");
    let _ = fs::create_dir_all(dir.join("src/tui/state"));
    let _ = fs::write(
        dir.join("src/tui/state/helpers.rs"),
        "use crate::events::ProgressStatus;\n\npub fn summarize_trace_steps() {}\n",
    );

    let loaded_context = concat!(
        "I've loaded this file for context:\n\n",
        "File: src/tui/state/helpers.rs\n",
        "Lines: 3\n\n",
        "```\n",
        "use crate::events::ProgressStatus;\n\n",
        "pub fn summarize_trace_steps() {}\n",
        "```\n"
    );

    let backend = ScriptedBackend::new(vec![
        "This file defines `summarize_trace_steps` in `src/tui/state/helpers.rs` and imports `ProgressStatus`, so it appears to hold helper logic for transcript/runtime state formatting.",
    ]);
    let (tx, _rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = run_read_only_tool_loop(
        ToolLoopIntent::CodeNavigation,
        "What does this file do?",
        &[
            Message::system("system"),
            Message::user(loaded_context),
            Message::user("What does this file do?"),
        ],
        &backend,
        &ToolRegistry::default(),
        &config::Config::default(),
        &dir,
        &tx,
        None,
        &mut cache_stats,
        &mut budget,
        false,
        false,
    )
    .expect("tool loop");

    assert_eq!(outcome.tool_results.len(), 1);
    assert_eq!(outcome.tool_results[0].tool_name, "read_file");
    assert_eq!(outcome.tool_results[0].argument, "src/tui/state/helpers.rs");
    assert!(outcome.final_response.contains("summarize_trace_steps"));
    assert!(outcome.final_response.contains("src/tui/state/helpers.rs"));
    assert!(outcome.final_response.contains("ProgressStatus"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn anchored_follow_up_uses_loaded_file_without_searching_elsewhere() {
    let dir = std::env::temp_dir().join("params-tool-loop-anchored-followup");
    let _ = fs::create_dir_all(dir.join("src"));
    let _ = fs::write(
        dir.join("src/main.rs"),
        "mod commands;\nmod config;\n\nfn main() -> Result<()> {\n    Ok(())\n}\n",
    );

    let resolution = InvestigationResolution {
        intent: ToolLoopIntent::CodeNavigation,
        anchor: Some(InvestigationAnchor::File("src/main.rs".to_string())),
        latency_policy: InvestigationLatencyPolicy::FastConvergence,
        anchored_file: Some("src/main.rs".to_string()),
        anchored_directory: None,
        anchored_query: None,
        prefer_answer_from_anchor: true,
    };
    let backend = ScriptedBackend::new(vec![
        "`src/main.rs:1` and `src/main.rs:2` declare top-level modules, and `src/main.rs:4` defines `main`, so this file is the CLI entrypoint.",
    ]);
    let (tx, rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop_with_resolution(
            ToolLoopIntent::CodeNavigation,
            "Can you tell me now?",
            Some(&resolution),
            &[
                Message::system("system"),
                Message::user(
                    "I've loaded this file for context:\n\nFile: src/main.rs\n\n```rust\nfn main() {}\n```",
                ),
                Message::user("Can you tell me now?"),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
    })
    .expect("tool loop");

    assert_eq!(outcome.tool_results.len(), 1);
    assert_eq!(outcome.tool_results[0].tool_name, "read_file");
    assert_eq!(outcome.tool_results[0].argument, "src/main.rs");
    assert!(
        !outcome
            .tool_results
            .iter()
            .any(|result| result.tool_name == "search"),
        "anchored follow-ups should answer from the loaded file, not restart broad searching"
    );
    let streamed = rx
        .try_iter()
        .filter_map(|event| match event {
            InferenceEvent::Token(text) => Some(text),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    assert!(!streamed.contains("[read_file:"));
    assert!(!streamed.contains("[search:"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn repo_overview_guidance_demands_short_structural_summary() {
    let guidance = grounded_answer_guidance(
        ToolLoopIntent::RepoOverview,
        "Can you see my project?",
        None,
        &[
            ToolResult {
                tool_name: "list_dir".to_string(),
                argument: ".".to_string(),
                output: ".\n  Cargo.toml\n  README.md\n  src/\n".to_string(),
            },
            ToolResult {
                tool_name: "read_file".to_string(),
                argument: "Cargo.toml".to_string(),
                output: "File: Cargo.toml\nLines: 3\n\n```\n[package]\nname = \"params-cli\"\nedition = \"2021\"\n```\n".to_string(),
            },
            ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/main.rs".to_string(),
                output: "File: src/main.rs\nLines: 4\n\n```\nmod inference;\nmod tui;\n\nfn main() {}\n```\n".to_string(),
            },
        ],
    )
    .expect("guidance");

    assert!(guidance.contains("2-4 short sentences"));
    assert!(guidance.contains("startup/entrypoints"));
    assert!(guidance.contains("main subsystems"));
    assert!(guidance.contains("dependency versions only if"));
}

#[test]
fn read_only_tool_loop_rejects_initial_prose_and_requires_tool_use() {
    let dir = std::env::temp_dir().join("params-tool-loop-requires-tool");
    let _ = std::fs::create_dir_all(dir.join("src/session"));
    let _ = std::fs::write(
        dir.join("src/session/mod.rs"),
        "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
    );

    let backend = ScriptedBackend::new(vec![
        "I will use the search tool to inspect the repo.",
        "[search: load_most_recent]",
        "[read_file: src/session/mod.rs]",
        "The implementation is in `src/session/mod.rs` at line 1.",
    ]);
    let (tx, _rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::CodeNavigation,
            "Where is session restore implemented?",
            &[
                Message::system("system"),
                Message::user("Where is session restore implemented?"),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
    })
    .expect("tool loop should recover after initial prose");

    assert_eq!(
        outcome.final_response,
        "The implementation is in `src/session/mod.rs` at line 1."
    );
    assert_eq!(outcome.tool_results.len(), 2);
    assert_eq!(outcome.tool_results[0].tool_name, "search");
    assert_eq!(outcome.tool_results[1].tool_name, "read_file");

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn code_navigation_requires_file_evidence_before_answering() {
    let dir = std::env::temp_dir().join("params-tool-loop-requires-read");
    let _ = std::fs::create_dir_all(dir.join("src/session"));
    let _ = std::fs::write(
        dir.join("src/session/mod.rs"),
        "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
    );

    let backend = ScriptedBackend::new(vec![
        "[search: load_most_recent]",
        "The implementation is at `src/inference/session.rs:3765`.",
        "[read_file: src/session/mod.rs]",
        "The implementation is in `src/session/mod.rs` at line 1.",
    ]);
    let (tx, _rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::CodeNavigation,
            "Where is session restore implemented?",
            &[
                Message::system("system"),
                Message::user("Where is session restore implemented?"),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
    })
    .expect("tool loop should require a file read before final answer");

    assert_eq!(
        outcome.final_response,
        "The implementation is in `src/session/mod.rs` at line 1."
    );
    assert_eq!(outcome.tool_results.len(), 2);
    assert_eq!(outcome.tool_results[0].tool_name, "search");
    assert_eq!(outcome.tool_results[1].tool_name, "read_file");

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn code_navigation_rejects_docs_only_read_and_forces_best_source_file() {
    let dir = std::env::temp_dir().join("params-tool-loop-noisy-search");
    let _ = fs::create_dir_all(dir.join("docs/context"));
    let _ = fs::create_dir_all(dir.join("src/inference"));
    let _ = fs::create_dir_all(dir.join("src/session"));
    let _ = fs::write(
        dir.join("docs/context/PLANS.md"),
        "load_most_recent overview\n",
    );
    let _ = fs::write(
            dir.join("src/inference/session.rs"),
            "fn prompt() {\n    let q = \"Where is session restore implemented?\";\n}\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn keeps_query() {\n        let x = \"load_most_recent\";\n    }\n}\n",
        );
    let _ = fs::write(
        dir.join("src/session/mod.rs"),
        "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
    );

    let backend = InspectingBackend::new(vec![
        (
            Some("Start with `[search: load_most_recent]`"),
            "[search: load_most_recent]",
        ),
        (
            Some("Next read: `[read_file: src/session/mod.rs]`"),
            "[read_file: docs/context/PLANS.md]",
        ),
        (None, "The implementation could not be clearly found."),
        (
            Some("Next read: `[read_file: src/session/mod.rs]`"),
            "[read_file: src/session/mod.rs]",
        ),
        (
            None,
            "The implementation is in `src/session/mod.rs` at line 1.",
        ),
    ]);
    let (tx, _rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::CodeNavigation,
            "Where is session restore implemented?",
            &[
                Message::system("system"),
                Message::user("Where is session restore implemented?"),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
    })
    .expect("tool loop should recover from docs-only evidence");

    assert_eq!(
        outcome.final_response,
        "The implementation is in `src/session/mod.rs` at line 1."
    );
    assert_eq!(
        outcome
            .tool_results
            .iter()
            .filter(|result| result.tool_name == "read_file")
            .count(),
        2
    );

    let _ = fs::remove_dir_all(dir);
}

// --- New routing pattern tests ---

#[test]
fn detect_tool_loop_intent_routes_explain_how_to_code_navigation() {
    assert_eq!(
        detect_tool_loop_intent("explain how session restore works"),
        Some(ToolLoopIntent::FlowTrace)
    );
    assert_eq!(
        detect_tool_loop_intent("Explain how the approval flow works"),
        Some(ToolLoopIntent::FlowTrace)
    );
    assert_eq!(
        detect_tool_loop_intent("explain how memory retrieval works"),
        Some(ToolLoopIntent::FlowTrace)
    );
}

#[test]
fn detect_tool_loop_intent_routes_describe_how_to_code_navigation() {
    assert_eq!(
        detect_tool_loop_intent("describe how the tool loop works"),
        Some(ToolLoopIntent::FlowTrace)
    );
    assert_eq!(
        detect_tool_loop_intent("Describe how fact promotion flows"),
        Some(ToolLoopIntent::FlowTrace)
    );
}

#[test]
fn detect_tool_loop_intent_routes_show_how_to_code_navigation() {
    assert_eq!(
        detect_tool_loop_intent("show me how session persistence works"),
        Some(ToolLoopIntent::FlowTrace)
    );
    assert_eq!(
        detect_tool_loop_intent("show how the approval path works"),
        Some(ToolLoopIntent::FlowTrace)
    );
}

#[test]
fn detect_tool_loop_intent_routes_walk_through_to_code_navigation() {
    assert_eq!(
        detect_tool_loop_intent("walk me through the session save flow"),
        Some(ToolLoopIntent::FlowTrace)
    );
    assert_eq!(
        detect_tool_loop_intent("walk through the memory compression logic"),
        Some(ToolLoopIntent::FlowTrace)
    );
}

#[test]
fn detect_tool_loop_intent_routes_what_calls_to_call_site_lookup() {
    assert_eq!(
        detect_tool_loop_intent("what calls load_most_recent?"),
        Some(ToolLoopIntent::CallSiteLookup)
    );
    assert_eq!(
        detect_tool_loop_intent("who calls run_read_only_tool_loop"),
        Some(ToolLoopIntent::CallSiteLookup)
    );
}

#[test]
fn detect_tool_loop_intent_routes_what_uses_to_usage_lookup() {
    assert_eq!(
        detect_tool_loop_intent("what uses SessionStore"),
        Some(ToolLoopIntent::UsageLookup)
    );
}

#[test]
fn detect_tool_loop_intent_routes_what_does_do_to_code_navigation() {
    assert_eq!(
        detect_tool_loop_intent("what does load_most_recent do"),
        Some(ToolLoopIntent::CodeNavigation)
    );
    assert_eq!(
        detect_tool_loop_intent("What does the session restore function do?"),
        Some(ToolLoopIntent::CodeNavigation)
    );
    // Should NOT match when it doesn't end with "do"
    assert_eq!(detect_tool_loop_intent("what does this mean"), None);
    assert_eq!(detect_tool_loop_intent("what does it return"), None);
}

#[test]
fn detect_tool_loop_intent_explain_how_config_routes_to_config_locate() {
    assert_eq!(
        detect_tool_loop_intent("explain how eco mode is configured"),
        Some(ToolLoopIntent::ConfigLocate)
    );
}

#[test]
fn suggested_search_query_extracts_subject_from_explain_how() {
    // "session" + "restore" triggers the hardcoded session-restore mapping.
    assert_eq!(
        suggested_search_query(
            "explain how session restore works",
            ToolLoopIntent::FlowTrace
        )
        .as_deref(),
        Some("load_most_recent")
    );
    // Non-session phrase: subject is extracted from the stripped phrase.
    assert_eq!(
        suggested_search_query(
            "explain how memory retrieval works",
            ToolLoopIntent::FlowTrace
        )
        .as_deref(),
        Some("retrieval")
    );
}

#[test]
fn suggested_search_query_extracts_subject_from_describe_how() {
    assert_eq!(
        suggested_search_query(
            "describe how the tool loop works",
            ToolLoopIntent::FlowTrace
        )
        .as_deref(),
        Some("loop")
    );
}

#[test]
fn suggested_search_query_extracts_subject_from_show_me_how() {
    // "approval" is the only non-stopword token after stripping the verb phrase.
    assert_eq!(
        suggested_search_query("show me how approval works", ToolLoopIntent::FlowTrace,).as_deref(),
        Some("approval")
    );
    assert_eq!(
        suggested_search_query("show how the search tool works", ToolLoopIntent::FlowTrace,)
            .as_deref(),
        Some("search")
    );
}

#[test]
fn suggested_search_query_extracts_subject_from_walk_me_through() {
    // "session" + "save" triggers the hardcoded session-save mapping.
    assert_eq!(
        suggested_search_query(
            "walk me through the session save flow",
            ToolLoopIntent::FlowTrace
        )
        .as_deref(),
        Some("save_messages")
    );
    // Non-session phrase: subject is extracted directly.
    assert_eq!(
        suggested_search_query(
            "walk me through the approval flow",
            ToolLoopIntent::FlowTrace
        )
        .as_deref(),
        Some("approval")
    );
}

#[test]
fn suggested_search_query_extracts_subject_from_what_calls() {
    // Underscores must be preserved — they are stripped by normalize_intent_text but
    // the raw extraction path preserves the full symbol name.
    assert_eq!(
        suggested_search_query(
            "what calls load_most_recent",
            ToolLoopIntent::CallSiteLookup
        )
        .as_deref(),
        Some("load_most_recent")
    );
    assert_eq!(
        suggested_search_query(
            "what calls load_most_recent?",
            ToolLoopIntent::CallSiteLookup
        )
        .as_deref(),
        Some("load_most_recent")
    );
}

#[test]
fn suggested_search_query_extracts_subject_from_who_calls() {
    assert_eq!(
        suggested_search_query(
            "who calls run_read_only_tool_loop",
            ToolLoopIntent::CallSiteLookup
        )
        .as_deref(),
        Some("run_read_only_tool_loop")
    );
}

#[test]
fn suggested_search_query_extracts_subject_from_what_uses() {
    assert_eq!(
        suggested_search_query("what uses SessionStore", ToolLoopIntent::UsageLookup).as_deref(),
        Some("sessionstore")
    );
}

#[test]
fn suggested_search_query_extracts_subject_from_what_does_do() {
    // Underscores are preserved via the raw extraction path.
    assert_eq!(
        suggested_search_query(
            "what does load_most_recent do",
            ToolLoopIntent::CodeNavigation
        )
        .as_deref(),
        Some("load_most_recent")
    );
    assert_eq!(
        suggested_search_query(
            "what does load_most_recent do?",
            ToolLoopIntent::CodeNavigation
        )
        .as_deref(),
        Some("load_most_recent")
    );
    assert_eq!(
        suggested_search_query("What does this file do?", ToolLoopIntent::CodeNavigation),
        None
    );
}

#[test]
fn tool_loop_runs_for_explain_how_pattern() {
    let dir = std::env::temp_dir().join("params-tool-loop-explain-how");
    let _ = fs::create_dir_all(dir.join("src/session"));
    let _ = fs::create_dir_all(dir.join("src/inference"));
    let _ = fs::write(
        dir.join("src/session/mod.rs"),
        "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    crate::inference::restore_session()\n}\n",
    );
    let _ = fs::write(
        dir.join("src/inference/mod.rs"),
        "pub fn restore_session() -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
    );

    let backend = ScriptedBackend::new(vec![
        "[search: session]",
        "[read_file: src/session/mod.rs]",
        "[read_file: src/inference/mod.rs]",
        "Session restore starts in src/session/mod.rs:1, where `load_most_recent` delegates to src/inference/mod.rs:1 `restore_session()`.",
    ]);
    let (tx, _rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::FlowTrace,
            "explain how session restore works",
            &[
                Message::system("system"),
                Message::user("explain how session restore works"),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
    })
    .expect("tool loop");

    assert!(outcome.final_response.contains("src/session/mod.rs"));
    assert!(outcome.final_response.contains("src/inference/mod.rs"));
    assert!(
        !outcome.final_response.contains("I couldn't gather enough"),
        "flow trace should produce grounded steps once cross-file evidence exists"
    );
    assert!(
        outcome
            .tool_results
            .iter()
            .filter(|r| r.tool_name == "read_file")
            .count()
            >= 2,
        "expected multi-file read evidence for explain-how query"
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn tool_loop_runs_for_what_calls_pattern() {
    let dir = std::env::temp_dir().join("params-tool-loop-what-calls");
    let _ = fs::create_dir_all(dir.join("src/inference"));
    let _ = fs::create_dir_all(dir.join("src/session"));
    let _ = fs::write(
        dir.join("src/session/mod.rs"),
        "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
    );
    let _ = fs::write(
        dir.join("src/inference/session.rs"),
        "fn restore(store: &SessionStore) {\n    let _ = store.load_most_recent();\n}\n",
    );

    let backend = ScriptedBackend::new(vec![
        "[search: load_most_recent]",
        "[read_file: src/inference/session.rs]",
        "load_most_recent is called from model_thread in src/inference/session.rs.",
        "load_most_recent is called from model_thread in src/inference/session.rs.",
    ]);
    let (tx, _rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::CallSiteLookup,
            "what calls load_most_recent",
            &[
                Message::system("system"),
                Message::user("what calls load_most_recent"),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
    })
    .expect("tool loop");

    assert!(outcome.final_response.contains("load_most_recent"));
    assert!(outcome
        .final_response
        .contains("src/inference/session.rs:2"));
    assert!(
        outcome.tool_results.iter().any(|r| r.tool_name == "search"),
        "expected a search result for what-calls query"
    );
    assert_eq!(
        outcome
            .tool_results
            .iter()
            .filter(|r| r.tool_name == "read_file")
            .count(),
        1,
        "call-site lookup should stop once one caller file provides direct evidence"
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn flow_trace_stops_once_cross_file_evidence_is_ready() {
    let dir = std::env::temp_dir().join("params-tool-loop-flow-ready");
    let _ = fs::create_dir_all(dir.join("src"));
    let _ = fs::write(
        dir.join("src/main.rs"),
        "fn main() {\n    init_logging();\n}\n",
    );
    let _ = fs::write(
        dir.join("src/logging.rs"),
        "pub fn init_logging() {\n    tracing_subscriber::fmt::init();\n}\n",
    );

    let backend = ScriptedBackend::new(vec![
        "[search: logging]",
        "[read_file: src/main.rs]",
        "[read_file: src/logging.rs]",
        "1. `src/main.rs:1` defines `main`. 2. `src/main.rs:2` calls `init_logging()`. 3. `src/logging.rs:1` defines `init_logging`. 4. `src/logging.rs:2` initializes tracing.",
    ]);
    let (tx, _rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::FlowTrace,
            "Trace how logging works.",
            &[
                Message::system("system"),
                Message::user("Trace how logging works."),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
    })
    .expect("tool loop");

    assert_eq!(
        outcome
            .tool_results
            .iter()
            .filter(|r| r.tool_name == "read_file")
            .count(),
        2,
        "flow trace should stop after the cross-file threshold is satisfied"
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn detect_tool_loop_intent_routes_trace_how_to_flow_trace() {
    assert_eq!(
        detect_tool_loop_intent("trace how session restore works"),
        Some(ToolLoopIntent::FlowTrace)
    );
    assert_eq!(
        detect_tool_loop_intent("how does session persistence work"),
        Some(ToolLoopIntent::FlowTrace)
    );
}

#[test]
fn call_site_lookup_requires_search_and_source_read_for_evidence() {
    // Search alone is not sufficient — need to also read a source file.
    assert!(!has_relevant_file_evidence(
        ToolLoopIntent::CallSiteLookup,
        "what calls load_most_recent",
        &[ToolResult {
            tool_name: "search".to_string(),
            argument: "load_most_recent".to_string(),
            output: "src/inference/session.rs:\n  42: load_most_recent()\n".to_string(),
        }],
    ));

    // Search + source read is sufficient.
    assert!(has_relevant_file_evidence(
        ToolLoopIntent::CallSiteLookup,
        "what calls load_most_recent",
        &[
            ToolResult {
                tool_name: "search".to_string(),
                argument: "load_most_recent".to_string(),
                output: "src/session/mod.rs:\n  1: pub fn load_most_recent() {\n\nsrc/main.rs:\n  12: store.load_most_recent();\n".to_string(),
            },
            ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/main.rs".to_string(),
                output:
                    "File: src/main.rs\nLines: 3\n\n```\nfn start() {\n    store.load_most_recent();\n}\n```\n"
                        .to_string(),
            },
        ],
    ));

    // Quoted symbol names and helper string construction are not call-sites.
    assert!(!has_relevant_file_evidence(
        ToolLoopIntent::CallSiteLookup,
        "what calls load_most_recent",
        &[
            ToolResult {
                tool_name: "search".to_string(),
                argument: "load_most_recent".to_string(),
                output: "src/inference/session/auto_inspect.rs:\n  257: return Some(\"load_most_recent\".to_string());\n".to_string(),
            },
            ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/inference/session/auto_inspect.rs".to_string(),
                output: "File: src/inference/session/auto_inspect.rs\nLines: 3\n\n```\nfn symbol_name() -> String {\n    return Some(\"load_most_recent\".to_string());\n}\n```\n".to_string(),
            },
        ],
    ));
}

#[test]
fn call_site_loop_shows_call_site_guidance_not_definition_guidance() {
    let guidance = grounded_answer_guidance(
            ToolLoopIntent::CallSiteLookup,
            "what calls load_most_recent",
            None,
            &[ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/inference/session.rs".to_string(),
                output: "File: src/inference/session.rs\nLines: 4\n\n```\nfn restore_session(store: &SessionStore) {\n    let s = store.load_most_recent();\n}\n```\n".to_string(),
            }],
        )
        .expect("guidance for call-site lookup");

    // Must list the invocation line, not the definition line
    assert!(
        guidance.contains("2"),
        "must reference the invocation line number"
    );
    assert!(
        guidance.contains("Observed call-sites:"),
        "must label evidence as call-sites, not definitions"
    );
    // Must not describe the implementation
    assert!(
        guidance.contains("Do NOT describe the symbol"),
        "must not instruct model to describe implementation"
    );
    // Anti-hedging rule must be present
    assert!(guidance.contains("presumably"));
}

#[test]
fn usage_lookup_guidance_labels_evidence_as_usages() {
    let guidance = grounded_answer_guidance(
            ToolLoopIntent::UsageLookup,
            "what uses SessionStore",
            None,
            &[ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/inference/session.rs".to_string(),
                output: "File: src/inference/session.rs\nLines: 4\n\n```\nuse crate::session::SessionStore;\n\nfn restore_session(store: &SessionStore) {}\n```\n".to_string(),
            }],
        )
        .expect("guidance for usage lookup");

    assert!(
        guidance.contains("Observed usages:"),
        "must label evidence as usages"
    );
    // The `use crate::session::SessionStore` import line (non-definition) should appear
    assert!(
        guidance.contains("5"),
        "must include the non-definition reference line"
    );
}

#[test]
fn flow_trace_guidance_shows_multi_file_definitions() {
    let guidance = grounded_answer_guidance(
            ToolLoopIntent::FlowTrace,
            "trace how session restore works",
            None,
            &[
                ToolResult {
                    tool_name: "read_file".to_string(),
                    argument: "src/session/mod.rs".to_string(),
                    output: "File: src/session/mod.rs\n\n```\npub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    self.load_session_by_id(&id)\n}\n```\n".to_string(),
                },
                ToolResult {
                    tool_name: "read_file".to_string(),
                    argument: "src/inference/session.rs".to_string(),
                    output: "File: src/inference/session.rs\n\n```\nfn restore_session(store: &SessionStore) {\n    let s = store.load_most_recent();\n}\n```\n".to_string(),
                },
            ],
        )
        .expect("guidance for flow trace");

    assert!(
        guidance.contains("Observed definitions across files:"),
        "must aggregate evidence from multiple files"
    );
    assert!(
        guidance.contains("src/session/mod.rs"),
        "must include first file"
    );
    assert!(
        guidance.contains("src/inference/session.rs"),
        "must include second file"
    );
    assert!(guidance.contains("presumably"));
}

#[test]
fn flow_trace_guidance_ignores_test_files_by_default() {
    let guidance = grounded_answer_guidance(
        ToolLoopIntent::FlowTrace,
        "explain how session restore works",
        None,
        &[
            ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/session/mod.rs".to_string(),
                output: "File: src/session/mod.rs\n\n```\npub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    self.load_session_by_id(&id)\n}\n```\n".to_string(),
            },
            ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/inference/session/auto_inspect/tests.rs".to_string(),
                output: "File: src/inference/session/auto_inspect/tests.rs\n\n```\nfn load_most_recent() -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n```\n".to_string(),
            },
        ],
    )
    .expect("guidance for flow trace");

    assert!(guidance.contains("src/session/mod.rs"));
    assert!(
        !guidance.contains("auto_inspect/tests.rs"),
        "flow trace guidance should ignore tests unless the prompt explicitly asks about them"
    );
}

#[test]
fn targeted_followup_for_call_site_lookup_prefers_invocation_files() {
    // When search hits include both a definition file and a caller file,
    // targeted_investigation_followup for CallSiteLookup should steer toward the caller.
    let search_output = concat!(
        "src/session/mod.rs:\n",
        "  1: pub fn load_most_recent(&self) -> Result<Option<SavedSession>>\n",
        "src/inference/session.rs:\n",
        "  42: store.load_most_recent()\n",
    );
    let results = &[ToolResult {
        tool_name: "search".to_string(),
        argument: "load_most_recent".to_string(),
        output: search_output.to_string(),
    }];

    let followup = targeted_investigation_followup(
        ToolLoopIntent::CallSiteLookup,
        "what calls load_most_recent",
        results,
    )
    .expect("should produce a followup");

    // Must steer toward the caller file (session.rs has the invocation on line 42)
    assert!(
        followup.contains("src/inference/session.rs"),
        "must suggest reading the file with the call-site, got: {followup}"
    );
}

#[test]
fn call_site_final_answer_text_is_not_misread_as_tool_tags() {
    assert!(
        !final_answer_contains_tool_tags(
            "load_most_recent is called from model_thread in src/inference/session.rs.",
            &ToolRegistry::default(),
        ),
        "plain prose caller answers must not be rejected as tool tags"
    );
}

#[test]
fn tool_loop_does_not_emit_thinking_system_message_rows() {
    let dir = std::env::temp_dir().join("params-tool-loop-no-thinking-row");
    let _ = fs::create_dir_all(dir.join("src/session"));
    let _ = fs::write(
        dir.join("src/session/mod.rs"),
        "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
    );

    let backend = ScriptedBackend::new(vec![
        "[search: load_most_recent]",
        "[read_file: src/session/mod.rs]",
        "The implementation is in `src/session/mod.rs` at line 1.",
    ]);
    let (tx, rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let _outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::CodeNavigation,
            "Where is session restore implemented?",
            &[
                Message::system("system"),
                Message::user("Where is session restore implemented?"),
            ],
            &backend,
            &ToolRegistry::default(),
            &config::Config::default(),
            &dir,
            &tx,
            None,
            &mut cache_stats,
            &mut budget,
            false,
            false,
        )
    })
    .expect("tool loop");

    let thinking_rows = rx
        .try_iter()
        .filter_map(|event| match event {
            InferenceEvent::SystemMessage(message) => Some(message),
            _ => None,
        })
        .filter(|message| message.starts_with("Thinking:"))
        .collect::<Vec<_>>();
    assert!(
        thinking_rows.is_empty(),
        "tool loop should expose Thinking through activity traces, not chat rows"
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn call_site_readiness_turns_ready_after_caller_file_is_read() {
    let readiness = investigation_readiness(
        ToolLoopIntent::CallSiteLookup,
        "what calls load_most_recent",
        None,
        &[
            ToolResult {
                tool_name: "search".to_string(),
                argument: "load_most_recent".to_string(),
                output: "src/inference/session.rs:\n   2: let _ = store.load_most_recent();\nsrc/session/mod.rs:\n   1: pub fn load_most_recent() {}\n".to_string(),
            },
            ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/inference/session.rs".to_string(),
                output: "File: src/inference/session.rs\nLines: 3\n\n```\nfn restore(store: &SessionStore) {\n    let _ = store.load_most_recent();\n}\n```".to_string(),
            },
        ],
    );

    assert!(
        matches!(readiness, InvestigationReadiness::Ready { .. }),
        "caller lookup should become answer-ready after one caller file is inspected"
    );
}
