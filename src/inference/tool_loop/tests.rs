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
        detect_tool_loop_intent("xplain how session restore works"),
        Some(ToolLoopIntent::FlowTrace)
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
fn synthesis_messages_drop_prior_answer_for_non_referential_prompt() {
    let messages = build_synthesis_messages(
        "Explain how session restore works",
        "Observed guidance",
        &[
            Message::system("old system"),
            Message::user("Tell me more"),
            Message::assistant("This is about src/main.rs."),
            Message::user("Explain how session restore works"),
        ],
    );

    assert_eq!(messages[0].role, "system");
    assert!(
        messages[0]
            .content
            .contains("Ignore unrelated prior conversation context"),
        "fresh synthesis prompts should explicitly ignore prior conversation context"
    );
    assert!(
        messages[0]
            .content
            .contains("Treat the current prompt as a standalone question"),
        "fresh synthesis prompts should treat the current question as standalone"
    );
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1].role, "user");
    assert_eq!(messages[1].content, "Explain how session restore works");
    assert_eq!(messages[2].content, "Observed guidance");
    assert!(
        !messages
            .iter()
            .any(|message| message.content == "This is about src/main.rs."),
        "non-referential synthesis should not include the prior assistant answer"
    );
}

#[test]
fn synthesis_messages_keep_prior_answer_for_referential_follow_up() {
    let messages = build_synthesis_messages(
        "Tell me more",
        "Observed guidance",
        &[
            Message::system("old system"),
            Message::user("What does this file do?"),
            Message::assistant("It is the CLI entrypoint."),
            Message::user("Tell me more"),
        ],
    );

    assert_eq!(messages[0].role, "system");
    assert!(
        messages[0]
            .content
            .contains("expand it with new detail rather than repeating it"),
        "referential follow-ups should still preserve the expansion instruction"
    );
    assert!(
        messages[0]
            .content
            .contains("Reuse prior assistant context only when it matches"),
        "referential synthesis should still be anchored to the current evidence"
    );
    assert!(
        messages
            .iter()
            .any(|message| message.content == "It is the CLI entrypoint."),
        "referential synthesis should keep the prior assistant answer"
    );
    assert!(
        messages
            .iter()
            .any(|message| message.content == "Tell me more"),
        "referential synthesis should keep the current follow-up prompt"
    );
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
    // The synthesis pass returns the scripted backend's natural-language response verbatim;
    // check that it contains the file reference and is not a raw tool tag.
    assert!(
        outcome.final_response.contains("src/session/mod.rs"),
        "final response must reference the implementation file"
    );
    assert!(
        !outcome.final_response.starts_with('['),
        "final response must not be a raw tool tag"
    );
    assert!(
        !streamed.contains("[search:"),
        "intermediate tool tags must never be streamed to the UI"
    );
    assert!(
        !streamed.contains("[read_file:"),
        "intermediate tool tags must never be streamed to the UI"
    );
    assert!(
        streamed.contains("src/session/mod.rs"),
        "implementation file must appear in streamed output"
    );

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
    assert!(guidance.contains("Observed file structure:"));
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
    let _ = fs::create_dir_all(dir.join("src/inference/session/runtime"));
    let _ = fs::write(
        dir.join("src/session/mod.rs"),
        "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    let Some(summary) = self.list_sessions()?.into_iter().next() else {\n        return Ok(None);\n    };\n    self.load_session_by_id(&summary.id)\n}\n",
    );
    let _ = fs::write(
        dir.join("src/inference/session/runtime/core.rs"),
        "fn restore_session(store: &SessionStore) -> Result<()> {\n    match store.load_most_recent()? {\n        Some(saved) => {\n            let _ = saved;\n        }\n        None => {}\n    }\n    Ok(())\n}\n",
    );

    let backend = ScriptedBackend::new(vec![
        "[search: load_most_recent]",
        "[read_file: src/session/mod.rs]",
        "[read_file: src/inference/session/runtime/core.rs]",
        "Session restore starts in `src/inference/session/runtime/core.rs:2`, where startup calls `load_most_recent`. \
         In `src/session/mod.rs:3`, the function returns `Ok(None)` if no session exists, and in \
         `src/session/mod.rs:5` it loads the saved session by ID.",
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
    assert!(outcome
        .final_response
        .contains("src/inference/session/runtime/core.rs"));
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
    // The synthesis pass returns natural-language prose; check file reference, not exact line format.
    assert!(outcome.final_response.contains("src/inference/session.rs"));
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
        guidance.contains("Observed cross-file evidence:"),
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
    // The guidance explicitly PROHIBITS hedging words in its rules section;
    // no separate "presumably" check needed here.
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
                argument: "src/inference/session/runtime/core.rs".to_string(),
                output: "File: src/inference/session/runtime/core.rs\n\n```\nfn restore_session(store: &SessionStore) {\n    let session = store.load_most_recent()?;\n}\n```\n".to_string(),
            },
            ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/inference/session/auto_inspect/tests.rs".to_string(),
                output: "File: src/inference/session/auto_inspect/tests.rs\n\n```\nfn load_most_recent() -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n```\n".to_string(),
            },
        ],
    )
    .expect("guidance for flow trace");

    assert!(guidance.contains("src/inference/session/runtime/core.rs"));
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

#[test]
fn llama_caller_lookup_bootstrap_reads_runtime_core_caller() {
    let dir = std::env::temp_dir().join("params-tool-loop-caller-bootstrap-core");
    let _ = fs::create_dir_all(dir.join("src/inference/session/runtime"));
    let _ = fs::create_dir_all(dir.join("src/session"));
    let _ = fs::create_dir_all(dir.join("src/tools"));
    let _ = fs::write(
        dir.join("src/session/mod.rs"),
        concat!(
            "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n",
            "    Ok(None)\n",
            "}\n\n",
            "#[cfg(test)]\n",
            "mod tests {\n",
            "    #[test]\n",
            "    fn restores() {\n",
            "        let _ = store.load_most_recent();\n",
            "    }\n",
            "}\n"
        ),
    );
    let _ = fs::write(
        dir.join("src/inference/session/runtime/core.rs"),
        concat!(
            "use crate::session::SessionStore;\n\n",
            "fn restore_previous_session(store: &SessionStore) {\n",
            "    let _ = store.load_most_recent();\n",
            "}\n"
        ),
    );
    let _ = fs::write(
        dir.join("src/tools/search.rs"),
        concat!(
            "fn fixture() {\n",
            "    let example = \"fn restore_previous_session(store: &SessionStore) { let _ = store.load_most_recent(); }\";\n",
            "    assert!(example.contains(\"load_most_recent\"));\n",
            "}\n"
        ),
    );

    let (tx, _rx) = mpsc::channel();
    let results = with_test_cwd(&dir, || {
        bootstrap_tool_results(
            ToolLoopIntent::CallSiteLookup,
            "What calls load_most_recent",
            None,
            &[Message::user("What calls load_most_recent")],
            &dir,
            "llama.cpp · qwen",
            &ToolRegistry::default(),
            &tx,
        )
    });

    let read = results
        .iter()
        .find(|result| result.tool_name == "read_file")
        .expect("bootstrap read");
    assert_eq!(read.argument, "src/inference/session/runtime/core.rs");

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn llama_usage_lookup_bootstrap_reads_real_source_import() {
    let dir = std::env::temp_dir().join("params-tool-loop-usage-bootstrap-import");
    let _ = fs::create_dir_all(dir.join("src/inference/session/runtime"));
    let _ = fs::create_dir_all(dir.join("src/session"));
    let _ = fs::write(
        dir.join("src/session/mod.rs"),
        "pub struct SessionStore {\n    pub id: String,\n}\n",
    );
    let _ = fs::write(
        dir.join("src/inference/session/runtime/core.rs"),
        concat!(
            "use crate::session::SessionStore;\n\n",
            "fn restore_previous_session(store: &SessionStore) {\n",
            "    let _ = store;\n",
            "}\n"
        ),
    );

    let (tx, _rx) = mpsc::channel();
    let results = with_test_cwd(&dir, || {
        bootstrap_tool_results(
            ToolLoopIntent::UsageLookup,
            "What uses SessionStore",
            None,
            &[Message::user("What uses SessionStore")],
            &dir,
            "llama.cpp · qwen",
            &ToolRegistry::default(),
            &tx,
        )
    });

    let read = results
        .iter()
        .find(|result| result.tool_name == "read_file")
        .expect("bootstrap read");
    assert_eq!(read.argument, "src/inference/session/runtime/core.rs");

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn llama_flow_trace_bootstrap_reads_restore_caller_after_store_definition() {
    let dir = std::env::temp_dir().join("params-tool-loop-flow-bootstrap-restore");
    let _ = fs::create_dir_all(dir.join("src/inference/session/runtime"));
    let _ = fs::create_dir_all(dir.join("src/session"));
    let _ = fs::create_dir_all(dir.join("src/tools"));
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
    let _ = fs::write(
        dir.join("src/inference/session/runtime/core.rs"),
        concat!(
            "use crate::session::SessionStore;\n\n",
            "fn restore_previous_session(store: &SessionStore) {\n",
            "    match store.load_most_recent() {\n",
            "        Ok(Some(saved)) => {\n",
            "            let _ = saved;\n",
            "        }\n",
            "        Ok(None) => {}\n",
            "        Err(_) => {}\n",
            "    }\n",
            "}\n"
        ),
    );
    let _ = fs::write(
        dir.join("src/tools/search.rs"),
        concat!(
            "fn search_fixture() {\n",
            "    let query = \"load_most_recent\";\n",
            "    walk_and_search(&dir, query, &mut matches).unwrap();\n",
            "    let example = \"match store.load_most_recent() { Ok(None) => {} }\";\n",
            "    assert!(example.contains(\"load_most_recent\"));\n",
            "}\n"
        ),
    );

    let (tx, _rx) = mpsc::channel();
    let results = with_test_cwd(&dir, || {
        bootstrap_tool_results(
            ToolLoopIntent::FlowTrace,
            "Explain how session restore works",
            None,
            &[Message::user("Explain how session restore works")],
            &dir,
            "llama.cpp · qwen",
            &ToolRegistry::default(),
            &tx,
        )
    });

    let read_paths = results
        .iter()
        .filter(|result| result.tool_name == "read_file")
        .map(|result| result.argument.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        read_paths,
        vec![
            "src/session/mod.rs",
            "src/inference/session/runtime/core.rs"
        ]
    );
    assert!(
        matches!(
            investigation_readiness(
                ToolLoopIntent::FlowTrace,
                "Explain how session restore works",
                None,
                &results
            ),
            InvestigationReadiness::Ready { .. }
        ),
        "flow trace bootstrap should gather cross-file evidence without waiting for another model planning turn"
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn anchored_main_followup_guidance_includes_cli_shape_and_startup() {
    let resolution = InvestigationResolution {
        intent: ToolLoopIntent::CodeNavigation,
        anchor: Some(InvestigationAnchor::File("src/main.rs".to_string())),
        latency_policy: InvestigationLatencyPolicy::FastConvergence,
        anchored_file: Some("src/main.rs".to_string()),
        anchored_directory: None,
        anchored_query: None,
        prefer_answer_from_anchor: true,
    };
    let guidance = grounded_answer_guidance(
        ToolLoopIntent::CodeNavigation,
        "Tell me more",
        Some(&resolution),
        &[ToolResult {
            tool_name: "read_file".to_string(),
            argument: "src/main.rs".to_string(),
            output: concat!(
                "File: src/main.rs\nLines: 18\n\n```\n",
                "mod commands;\n",
                "mod config;\n\n",
                "struct Cli {\n",
                "    prompt: Option<String>,\n",
                "    command: Option<Command>,\n",
                "}\n\n",
                "enum Command {\n",
                "    Pull,\n",
                "    Index,\n",
                "}\n\n",
                "fn main() -> Result<()> {\n",
                "    let cli = Cli::parse();\n",
                "    match cli.command {\n",
                "        Some(Command::Pull) => {}\n",
                "        Some(Command::Index) => {}\n",
                "        None => {}\n",
                "    }\n",
                "}\n",
                "```\n"
            )
            .to_string(),
        }],
    )
    .expect("guidance");

    assert!(guidance.contains("Observed file structure:"));
    assert!(guidance.contains("struct Cli"));
    assert!(guidance.contains("enum Command"));
    assert!(guidance.contains("Cli::parse()"));
    assert!(guidance.contains("match cli.command"));
    assert!(guidance.contains("focus on them instead of unrelated helper functions or module declarations"));
    assert!(guidance.contains("Do not infer hidden subcommands or describe logging setup, indexing behavior, benchmarking behavior"));
}

#[test]
fn anchored_main_whats_this_do_synthesis_stays_on_loaded_cli_lines() {
    let dir = std::env::temp_dir().join("params-tool-loop-main-anchored-shaping");
    let _ = fs::create_dir_all(dir.join("src"));
    let _ = fs::write(
        dir.join("src/main.rs"),
        concat!(
            "use clap::{Parser, Subcommand};\n",
            "use tracing::info;\n\n",
            "struct Cli {\n",
            "    prompt: Option<String>,\n",
            "    command: Option<Command>,\n",
            "}\n\n",
            "enum Command {\n",
            "    Pull,\n",
            "    Index,\n",
            "}\n\n",
            "fn init_logging() {}\n\n",
            "fn main() -> Result<()> {\n",
            "    let cli = Cli::parse();\n",
            "    match cli.command {\n",
            "        Some(Command::Pull) => {}\n",
            "        Some(Command::Index) => {}\n",
            "        None => { info!(\"tui\"); }\n",
            "    }\n",
            "}\n"
        ),
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
    let loaded_context = concat!(
        "I've loaded this file for context:\n\n",
        "File: src/main.rs\n",
        "Lines: 20\n\n",
        "```\n",
        "use clap::{Parser, Subcommand};\n",
        "use tracing::info;\n\n",
        "struct Cli {\n",
        "    prompt: Option<String>,\n",
        "    command: Option<Command>,\n",
        "}\n\n",
        "enum Command {\n",
        "    Pull,\n",
        "    Index,\n",
        "}\n\n",
        "fn init_logging() {}\n\n",
        "fn main() -> Result<()> {\n",
        "    let cli = Cli::parse();\n",
        "    match cli.command {\n",
        "        Some(Command::Pull) => {}\n",
        "        Some(Command::Index) => {}\n",
        "        None => { info!(\"tui\"); }\n",
        "    }\n",
        "}\n",
        "```\n"
    );
    let backend = InspectingBackend::new(vec![(
        Some("Do not infer hidden subcommands or describe logging setup, indexing behavior, benchmarking behavior"),
        "src/main.rs is the binary entrypoint. `src/main.rs:3` defines `Cli`, `src/main.rs:8` defines `Command`, and `src/main.rs:15` / `src/main.rs:16` show `Cli::parse()` followed by `match cli.command`.",
    )]);
    let (tx, _rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop_with_resolution(
            ToolLoopIntent::CodeNavigation,
            "Whats this do",
            Some(&resolution),
            &[
                Message::system("system"),
                Message::user(loaded_context),
                Message::user("Whats this do"),
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
    assert!(outcome.final_response.contains("binary entrypoint"));
    assert!(outcome.final_response.contains("Cli"));
    assert!(outcome.final_response.contains("Command"));
    assert!(outcome.final_response.contains("Cli::parse()"));
    assert!(outcome.final_response.contains("match cli.command"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn flow_trace_guidance_includes_runtime_caller_and_restore_branches() {
    let core_content = format!(
        "{}match store.load_most_recent() {{\n    Ok(Some(saved)) => {{\n        let _ = saved;\n    }}\n    Ok(None) => {{}}\n}}\n",
        "\n".repeat(166)
    );
    let restore_content = format!(
        "{}pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {{\n    let Some(summary) = self.list_sessions()?.into_iter().next() else {{\n        return Ok(None);\n    }};\n    self.load_session_by_id(&summary.id)\n}}\n",
        "\n".repeat(261)
    );
    let guidance = grounded_answer_guidance(
        ToolLoopIntent::FlowTrace,
        "Explain how session restore works",
        None,
        &[
            ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/inference/session/runtime/core.rs".to_string(),
                output: format!(
                    "File: src/inference/session/runtime/core.rs\nLines: 171\n\n```\n{core_content}```\n"
                ),
            },
            ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/session/mod.rs".to_string(),
                output: format!("File: src/session/mod.rs\nLines: 267\n\n```\n{restore_content}```\n"),
            },
        ],
    )
    .expect("guidance");

    assert!(guidance.contains("src/inference/session/runtime/core.rs:167"));
    assert!(guidance.contains("src/inference/session/runtime/core.rs:171"));
    assert!(guidance.contains("src/session/mod.rs:262"));
    assert!(guidance.contains("runtime branches around `match store.load_most_recent()`"));
    assert!(guidance.contains("list_sessions()?.into_iter().next()"));
    assert!(guidance.contains("not that it iterates all sessions"));
    assert!(guidance.contains("return Ok(None)"));
    assert!(guidance.contains("load_session_by_id(&summary.id)"));
    assert!(guidance.contains("Do not mention logging, message counts, restored message totals, `Ok(Some(saved))`, or broader success-path side effects"));
}

#[test]
fn flow_trace_render_fallback_prefers_runtime_caller_and_restore_handoff() {
    let outcome = investigation_readiness(
        ToolLoopIntent::FlowTrace,
        "Explain how session restore works",
        None,
        &[
            ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/inference/session/runtime/core.rs".to_string(),
                output: "File: src/inference/session/runtime/core.rs\nLines: 8\n\n```\nmatch store.load_most_recent() {\n    Ok(Some(saved)) => {\n        let restored_count = saved.messages.len();\n        info!(msg_count = restored_count, \"restoring previous session\");\n    }\n    Ok(None) => {}\n}\n```\n".to_string(),
            },
            ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/session/mod.rs".to_string(),
                output: "File: src/session/mod.rs\nLines: 5\n\n```\npub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    let Some(summary) = self.list_sessions()?.into_iter().next() else {\n        return Ok(None);\n    };\n    self.load_session_by_id(&summary.id)\n}\n```\n".to_string(),
            },
        ],
    );
    let answer = match outcome {
        InvestigationReadiness::Ready { evidence, .. } => {
            render_structured_answer("Explain how session restore works", &evidence)
        }
        _ => panic!("expected flow trace evidence"),
    };

    assert!(answer.contains("src/inference/session/runtime/core.rs:1"));
    assert!(answer.contains("match store.load_most_recent()"));
    assert!(answer.contains("load_most_recent"));
    assert!(answer.contains("list_sessions()?.into_iter().next()"));
    assert!(answer.contains("Ok(None)"));
    assert!(answer.contains("load_session_by_id(&summary.id)"));
    assert!(!answer.contains("restoring previous session"));
    assert!(!answer.contains("msg_count"));
    assert!(
        !answer.contains("iterates all sessions"),
        "fallback should stay close to the observed selection step"
    );
}

// ─── Benchmark Regression Tests ───────────────────────────────────────────────
//
// These tests correspond 1:1 to the benchmark cases defined in docs/BENCHMARKS.md.
// Each test verifies the minimum passing bar for the benchmark case it is named after.
// They are integration-level tests: they exercise run_read_only_tool_loop end-to-end
// with a ScriptedBackend, verifying intent routing, tool-loop convergence, and the
// final answer properties that distinguish a pass from a fail.

#[test]
fn benchmark_repo_overview_enters_tool_loop_and_returns_grounded_answer() {
    // Benchmark case: "Can you see my project?"
    // Minimum bar: enters tool loop (calls list_dir/read_file), final answer references
    // real repo files (Cargo.toml, src/main.rs), no raw tool tags, no "insufficient evidence".
    let dir = std::env::temp_dir().join("params-bench-repo-overview");
    let _ = fs::create_dir_all(dir.join("src"));
    let _ = fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"params-cli\"\nversion = \"0.1.0\"\n",
    );
    let _ = fs::write(
        dir.join("src/main.rs"),
        "mod commands;\nmod config;\nfn main() {\n    println!(\"hello\");\n}\n",
    );
    let _ = fs::write(dir.join("README.md"), "# params-cli\nA Rust CLI tool.\n");

    let backend = ScriptedBackend::new(vec![
        "[list_dir: .]",
        "[read_file: Cargo.toml]",
        "This is a Rust CLI project defined in `Cargo.toml`. \
         The entry point is `src/main.rs` which declares modules and calls `fn main`.",
    ]);
    let (tx, rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::RepoOverview,
            "Can you see my project?",
            &[
                Message::system("system"),
                Message::user("Can you see my project?"),
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

    // Must have used at least one tool (search, list_dir, or read_file)
    assert!(
        !outcome.tool_results.is_empty(),
        "repo overview must inspect the repo before answering"
    );
    // Final answer must reference real repo files
    assert!(
        outcome.final_response.contains("Cargo.toml")
            || outcome.final_response.contains("src/main.rs"),
        "repo overview answer must reference real repo files, got: {}",
        outcome.final_response
    );
    // Must not be an error fallback
    assert!(
        !outcome.final_response.contains("insufficient evidence"),
        "repo overview must not fall back to insufficient-evidence message"
    );
    // Must not leak raw tool tags
    assert!(
        !outcome.final_response.contains("[list_dir:")
            && !outcome.final_response.contains("[read_file:"),
        "repo overview answer must not contain raw tool tags"
    );
    // Final answer must have streamed (not just been buffer-emitted with no tokens)
    let streamed = rx
        .try_iter()
        .filter_map(|event| match event {
            InferenceEvent::Token(text) => Some(text),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    assert!(
        !streamed.is_empty(),
        "repo overview answer must stream tokens"
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn benchmark_file_summary_mentions_entrypoint_structure() {
    // Benchmark case: `/read src/main.rs` then `What does this file do?`
    // Minimum bar: final answer mentions the Rust entrypoint or main function,
    // not just the raw module-declaration line.
    let dir = std::env::temp_dir().join("params-bench-file-summary");
    let _ = fs::create_dir_all(dir.join("src"));
    let main_content = concat!(
        "mod cache;\nmod commands;\nmod config;\nmod debug_log;\n",
        "mod error;\nmod events;\nmod inference;\nmod session;\n",
        "use clap::Parser;\n",
        "#[derive(Parser)]\nstruct Cli { #[command(subcommand)] cmd: Commands }\n",
        "fn main() -> anyhow::Result<()> {\n    let cli = Cli::parse();\n    Ok(())\n}\n",
    );
    let _ = fs::write(dir.join("src/main.rs"), main_content);

    let backend = ScriptedBackend::new(vec![
        "[read_file: src/main.rs]",
        "`src/main.rs` is the Rust binary entrypoint. It declares modules (cache, commands, config, \
         debug_log, error, events, inference, session) and defines `fn main` which parses CLI \
         arguments via the `Cli` struct and dispatches to subcommands.",
    ]);
    let (tx, _rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::CodeNavigation,
            "What does this file do?",
            &[
                Message::system("system"),
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
    })
    .expect("tool loop");

    // Must reference src/main.rs — the file being asked about
    assert!(
        outcome.final_response.contains("src/main.rs"),
        "file summary must anchor to src/main.rs, got: {}",
        outcome.final_response
    );
    // Must mention more than just a raw module-declaration line
    assert!(
        outcome.final_response.contains("main")
            || outcome.final_response.contains("entrypoint")
            || outcome.final_response.contains("CLI")
            || outcome.final_response.contains("command"),
        "file summary must describe the entrypoint structure, got: {}",
        outcome.final_response
    );
    // Must not leak raw tool tags into the final answer
    assert!(
        !outcome.final_response.contains("[read_file:"),
        "file summary must not contain raw tool tags"
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn benchmark_followup_tell_me_more_expands_not_repeats() {
    // Benchmark case: `Tell me more` after a prior file-summary answer.
    // Minimum bar: the synthesis pass receives the prior answer as context
    // and the final response is non-empty and does not exactly repeat the prior answer.
    let dir = std::env::temp_dir().join("params-bench-followup-expand");
    let _ = fs::create_dir_all(dir.join("src"));
    let _ = fs::write(
        dir.join("src/main.rs"),
        concat!(
            "mod commands;\n",
            "mod config;\n\n",
            "struct Cli {\n",
            "    prompt: Option<String>,\n",
            "    command: Option<Command>,\n",
            "}\n\n",
            "enum Command {\n",
            "    Pull,\n",
            "    Index,\n",
            "}\n\n",
            "fn main() -> Result<()> {\n",
            "    let cli = Cli::parse();\n",
            "    match cli.command {\n",
            "        Some(Command::Pull) => {}\n",
            "        Some(Command::Index) => {}\n",
            "        None => {}\n",
            "    }\n",
            "}\n"
        ),
    );

    let prior_answer = "`src/main.rs:1` declares modules commands and config.";
    let expanded_answer = "`src/main.rs` is the CLI entrypoint. \
        It defines the `Cli` struct and the `Command` enum, then `fn main` parses \
        arguments with `Cli::parse()` and dispatches subcommands via `match cli.command`.";
    let resolution = InvestigationResolution {
        intent: ToolLoopIntent::CodeNavigation,
        anchor: Some(InvestigationAnchor::File("src/main.rs".to_string())),
        latency_policy: InvestigationLatencyPolicy::FastConvergence,
        anchored_file: Some("src/main.rs".to_string()),
        anchored_directory: None,
        anchored_query: None,
        prefer_answer_from_anchor: true,
    };

    // The synthesis pass will receive the prior answer as context (via base_messages tail)
    // and should return the expanded version rather than repeating the prior answer.
    let backend = ScriptedBackend::new(vec!["[read_file: src/main.rs]", expanded_answer]);
    let (tx, _rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop_with_resolution(
            ToolLoopIntent::CodeNavigation,
            "Tell me more",
            Some(&resolution),
            &[
                Message::system("system"),
                Message::user("What does this file do?"),
                Message::assistant(prior_answer),
                Message::user("Tell me more"),
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

    // Final response must be non-empty
    assert!(
        !outcome.final_response.trim().is_empty(),
        "Tell me more must produce a non-empty response"
    );
    // Must not be an exact repeat of the prior answer
    assert_ne!(
        outcome.final_response.trim(),
        prior_answer,
        "Tell me more must expand the answer, not repeat the prior response verbatim"
    );
    // Must contain additional detail beyond the shallow module list
    assert!(
        outcome.final_response.contains("Cli")
            || outcome.final_response.contains("Command")
            || outcome.final_response.contains("parse")
            || outcome.final_response.contains("dispatch"),
        "Tell me more must deepen the file-summary answer, got: {}",
        outcome.final_response
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn benchmark_caller_lookup_returns_real_caller_not_insufficient_evidence() {
    // Benchmark case: "What calls load_most_recent"
    // Minimum bar: returns real non-definition caller file reference, not "insufficient evidence".
    // Must converge within acceptable latency (enforced by ScriptedBackend exhaustion).
    let dir = std::env::temp_dir().join("params-bench-caller-lookup");
    let _ = fs::create_dir_all(dir.join("src/inference/session/runtime"));
    let _ = fs::create_dir_all(dir.join("src/session"));
    let _ = fs::write(
        dir.join("src/session/mod.rs"),
        "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
    );
    let _ = fs::write(
        dir.join("src/inference/session/runtime/core.rs"),
        "fn restore(store: &SessionStore) {\n    let _ = store.load_most_recent();\n}\n",
    );

    let backend = ScriptedBackend::new(vec![
        "[search: load_most_recent]",
        "[read_file: src/inference/session/runtime/core.rs]",
        "load_most_recent is called from `restore` in `src/inference/session/runtime/core.rs:2`.",
        "load_most_recent is called from `restore` in `src/inference/session/runtime/core.rs:2`.",
    ]);
    let (tx, _rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::CallSiteLookup,
            "What calls load_most_recent",
            &[
                Message::system("system"),
                Message::user("What calls load_most_recent"),
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

    // Must not return the investigation-budget fallback message
    assert!(
        !outcome.final_response.contains("insufficient evidence"),
        "caller lookup must not fall back to insufficient-evidence message, got: {}",
        outcome.final_response
    );
    // Must reference the symbol being looked up
    assert!(
        outcome.final_response.contains("load_most_recent"),
        "caller lookup answer must mention the symbol, got: {}",
        outcome.final_response
    );
    // Must reference the caller file (not the definition file)
    assert!(
        outcome
            .final_response
            .contains("src/inference/session/runtime/core.rs"),
        "caller lookup must reference the caller file, got: {}",
        outcome.final_response
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn benchmark_usage_lookup_returns_real_usage_not_insufficient_evidence() {
    // Benchmark case: "What uses SessionStore"
    // Minimum bar: returns real non-definition usage file reference, not "insufficient evidence".
    let dir = std::env::temp_dir().join("params-bench-usage-lookup");
    let _ = fs::create_dir_all(dir.join("src/inference/session/runtime"));
    let _ = fs::create_dir_all(dir.join("src/session"));
    let _ = fs::write(
        dir.join("src/session/mod.rs"),
        "pub struct SessionStore {\n    pub path: PathBuf,\n}\n",
    );
    let _ = fs::write(
        dir.join("src/inference/session/runtime/core.rs"),
        "use crate::session::SessionStore;\nfn restore(store: &SessionStore) {}\n",
    );

    let backend = ScriptedBackend::new(vec![
        "[search: SessionStore]",
        "[read_file: src/inference/session/runtime/core.rs]",
        "SessionStore is used in `src/inference/session/runtime/core.rs:1` as an import and in \
         the `restore` function signature on line 2.",
        "SessionStore is used in `src/inference/session/runtime/core.rs:1` as an import and in \
         the `restore` function signature on line 2.",
    ]);
    let (tx, _rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::UsageLookup,
            "What uses SessionStore",
            &[
                Message::system("system"),
                Message::user("What uses SessionStore"),
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

    assert!(
        !outcome.final_response.contains("insufficient evidence"),
        "usage lookup must not fall back to insufficient-evidence message, got: {}",
        outcome.final_response
    );
    assert!(
        outcome.final_response.contains("SessionStore"),
        "usage lookup answer must mention the symbol, got: {}",
        outcome.final_response
    );
    assert!(
        outcome
            .final_response
            .contains("src/inference/session/runtime/core.rs"),
        "usage lookup must reference the usage file, got: {}",
        outcome.final_response
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn benchmark_flow_trace_produces_prose_explanation_not_raw_code_dump() {
    // Benchmark case: "Explain how session restore works"
    // Minimum bar: the final answer is a plain-language prose explanation (not a
    // numbered list of raw code lines), references multiple source files, and does
    // not leak tool tags.
    let dir = std::env::temp_dir().join("params-bench-flow-trace");
    let _ = fs::create_dir_all(dir.join("src/inference/session/runtime"));
    let _ = fs::create_dir_all(dir.join("src/session"));
    let _ = fs::write(
        dir.join("src/session/mod.rs"),
        "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n\
         let Some(summary) = self.list_sessions()?.into_iter().next() else {\n\
         return Ok(None);\n};\nself.load_session_by_id(&summary.id)\n}\n",
    );
    let _ = fs::write(
        dir.join("src/inference/session/runtime/core.rs"),
        "fn restore_session(store: &SessionStore) -> Result<()> {\n\
         match store.load_most_recent()? {\n\
             Some(saved) => {\n\
                 let _ = saved;\n\
             }\n\
             None => {}\n\
         }\n\
         Ok(())\n}\n",
    );

    let backend = ScriptedBackend::new(vec![
        "[search: load_most_recent]",
        "[read_file: src/session/mod.rs]",
        "[read_file: src/inference/session/runtime/core.rs]",
        "Session restore works as follows: `restore_session` in `src/inference/session/runtime/core.rs:1` \
         calls `load_most_recent` from `src/session/mod.rs:1`. That function lists all sessions \
         and picks the most recent one (returning `None` early if none exist), then loads it \
         by ID via `load_session_by_id`.",
    ]);
    let (tx, _rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::FlowTrace,
            "Explain how session restore works",
            &[
                Message::system("system"),
                Message::user("Explain how session restore works"),
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

    // Must not be an investigation-budget fallback
    assert!(
        !outcome.final_response.contains("insufficient evidence"),
        "flow trace must not fall back to insufficient-evidence message"
    );
    // Must reference the key symbol
    assert!(
        outcome.final_response.contains("load_most_recent"),
        "flow trace must reference the traced symbol, got: {}",
        outcome.final_response
    );
    // Must reference at least one source file
    assert!(
        outcome.final_response.contains("src/session/mod.rs")
            || outcome
                .final_response
                .contains("src/inference/session/runtime/core.rs"),
        "flow trace must cite a source file, got: {}",
        outcome.final_response
    );
    assert!(
        outcome.final_response.contains("Ok(None)")
            || outcome.final_response.contains("None")
            || outcome.final_response.contains("load_session_by_id"),
        "flow trace must describe the restore branch behavior, got: {}",
        outcome.final_response
    );
    // Must not be a raw numbered code dump (every line starts with a digit colon)
    let lines: Vec<&str> = outcome
        .final_response
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();
    let all_lines_are_numbered_code = !lines.is_empty()
        && lines.iter().all(|l| {
            l.trim()
                .chars()
                .next()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false)
        });
    assert!(
        !all_lines_are_numbered_code,
        "flow trace must produce prose explanation, not a raw numbered code dump"
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn benchmark_config_lookup_finds_eco_mode_field_not_unrelated_struct() {
    // Benchmark case: "Where is eco mode configured?"
    // Minimum bar: the final answer references the eco-mode config field or struct,
    // not an unrelated nearby struct like `ProjectProfile`.
    let dir = std::env::temp_dir().join("params-bench-config-lookup");
    let _ = fs::create_dir_all(dir.join("src/config"));
    // The real project uses `eco.enabled` (dot-notation) for the TOML key, so the search
    // query is "eco.enabled". The file content must contain that substring (e.g. as a
    // nested field access like `profile.eco.enabled`) for grounded_answer_guidance to find it.
    let _ = fs::write(
        dir.join("src/config/profile.rs"),
        concat!(
            "pub struct ProjectProfile {\n",
            "    pub backend: Option<String>,\n",
            "    pub eco: ProjectEcoProfile,\n",
            "}\n",
            "pub struct ProjectEcoProfile {\n",
            "    pub enabled: bool,\n",
            "}\n",
            "fn apply(profile: &ProjectProfile, base: &mut Config) {\n",
            "    if let Some(e) = profile.eco.enabled {\n",
            "        base.eco.enabled = e;\n",
            "    }\n",
            "}\n",
        ),
    );

    let backend = ScriptedBackend::new(vec![
        "[search: eco.enabled]",
        "[read_file: src/config/profile.rs]",
        "Eco mode is configured in `src/config/profile.rs:9` via `profile.eco.enabled` \
         inside the `apply` function, which reads the `ProjectEcoProfile.enabled` field.",
    ]);
    let (tx, _rx) = mpsc::channel();
    let mut cache_stats = SessionCacheStats::default();
    let mut budget = SessionBudget::default();
    let outcome = with_test_cwd(&dir, || {
        run_read_only_tool_loop(
            ToolLoopIntent::ConfigLocate,
            "Where is eco mode configured?",
            &[
                Message::system("system"),
                Message::user("Where is eco mode configured?"),
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

    // Must not be the investigation-budget fallback
    assert!(
        !outcome.final_response.contains("insufficient evidence"),
        "config lookup must not fall back to insufficient-evidence message"
    );
    // Must reference the eco-mode config specifically (field, struct, or TOML key)
    assert!(
        outcome.final_response.contains("eco"),
        "config lookup must reference the eco-mode config field, got: {}",
        outcome.final_response
    );
    // Must reference the config file
    assert!(
        outcome.final_response.contains("src/config/profile.rs"),
        "config lookup must cite the config source file, got: {}",
        outcome.final_response
    );

    let _ = fs::remove_dir_all(dir);
}
