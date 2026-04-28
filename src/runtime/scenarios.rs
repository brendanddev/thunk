/// Scenario-based regression tests covering the exact failure classes fixed in Phase 7.5.
/// Each test encodes one concrete failure mode end-to-end: prompt -> backend response -> runtime handling -> conversation state.
///
/// These are complementary to the unit tests in engine.rs (which test engine internals)
/// and tool_codec.rs (which test parsing). Scenarios test full round-trips.
#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use crate::app::config::Config;
    use crate::llm::backend::{BackendCapabilities, BackendEvent, GenerateRequest, ModelBackend};
    use crate::runtime::types::{RuntimeEvent, RuntimeRequest};
    use crate::runtime::{ProjectRoot, Runtime};
    use crate::tools::default_registry;

    // Test backend

    struct TestBackend {
        responses: Vec<String>,
        call_count: usize,
    }

    impl TestBackend {
        fn new(responses: Vec<impl Into<String>>) -> Self {
            Self {
                responses: responses.into_iter().map(Into::into).collect(),
                call_count: 0,
            }
        }
    }

    impl ModelBackend for TestBackend {
        fn name(&self) -> &str {
            "test"
        }

        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                context_window_tokens: None,
                max_output_tokens: None,
            }
        }

        fn generate(
            &mut self,
            _request: GenerateRequest,
            on_event: &mut dyn FnMut(BackendEvent),
        ) -> crate::app::Result<()> {
            let reply = self
                .responses
                .get(self.call_count)
                .cloned()
                .unwrap_or_default();
            self.call_count += 1;
            if !reply.is_empty() {
                on_event(BackendEvent::TextDelta(reply));
            }
            on_event(BackendEvent::Finished);
            Ok(())
        }
    }

    // Helpers

    fn make_runtime(dir: &TempDir, responses: Vec<impl Into<String>>) -> Runtime {
        let project_root = ProjectRoot::new(dir.path().to_path_buf()).unwrap();
        Runtime::new(
            &Config::default(),
            project_root.clone(),
            Box::new(TestBackend::new(responses)),
            default_registry(project_root.as_path_buf()),
        )
    }

    fn collect_events(runtime: &mut Runtime, request: RuntimeRequest) -> Vec<RuntimeEvent> {
        let mut events = Vec::new();
        runtime.handle(request, &mut |e| events.push(e));
        events
    }

    fn has_failed(events: &[RuntimeEvent]) -> bool {
        events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::Failed { .. }))
    }

    fn has_approval(events: &[RuntimeEvent]) -> bool {
        events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ApprovalRequired(_)))
    }

    fn has_chunk(events: &[RuntimeEvent]) -> bool {
        events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::AssistantMessageChunk(_)))
    }

    fn assistant_chunks(events: &[RuntimeEvent]) -> Vec<String> {
        events
            .iter()
            .filter_map(|e| {
                if let RuntimeEvent::AssistantMessageChunk(chunk) = e {
                    Some(chunk.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    fn last_assistant_content(messages: &[crate::llm::backend::Message]) -> Option<&str> {
        messages
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str())
    }

    fn all_user_content(messages: &[crate::llm::backend::Message]) -> String {
        messages
            .iter()
            .filter(|m| m.role == crate::llm::backend::Role::User)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    // Scenario 1: Full reject flow
    //
    // Failure class: reject was framed as a retryable error, causing the model to
    // re-propose or misstate the same tool action. The rejection response must be
    // terminal and runtime-owned.
    // This scenario verifies: proposal fires ApprovalRequired, reject produces a
    // runtime cancellation answer, the file is not created, and no second
    // ApprovalRequired is fired.

    #[test]
    fn reject_flow_is_terminal_without_reproposal_or_success_claim() {
        let dir = TempDir::new().unwrap();
        let mut rt = make_runtime(
            &dir,
            vec![
                "[write_file: temp.rs]", // model proposes write
                "I created temp.rs.",    // must not be used after rejection
            ],
        );

        let submit_events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "create temp.rs".into(),
            },
        );
        assert!(
            !has_failed(&submit_events),
            "submit must not fail: {submit_events:?}"
        );
        assert!(
            has_approval(&submit_events),
            "submit must fire ApprovalRequired"
        );

        let reject_events = collect_events(&mut rt, RuntimeRequest::Reject);
        assert!(
            !has_failed(&reject_events),
            "reject must not fail: {reject_events:?}"
        );
        assert!(
            has_chunk(&reject_events),
            "reject must emit a runtime-authored cancellation answer"
        );
        // No second approval — reject must not re-enter a re-proposal path.
        assert!(
            !has_approval(&reject_events),
            "reject must not fire a second ApprovalRequired"
        );
        assert!(
            !dir.path().join("temp.rs").exists(),
            "rejected write must not create a file"
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| m.content.contains("do not retry")),
            "rejection error must contain terminal guidance"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("Canceled. No file was created")),
            "runtime cancellation response must be in conversation"
        );
        assert!(
            !snapshot
                .iter()
                .any(|m| m.content.contains("I created temp.rs.")),
            "model must not get a chance to contradict the rejected mutation"
        );
        assert_eq!(
            last_assistant_content(&snapshot),
            Some("Canceled. No file was created or changed."),
            "last assistant message must be the runtime cancellation"
        );
    }

    // Scenario 2: search_code colon-space prefix
    //
    // Failure class: model emits `pattern: X` (colon-space) instead of `pattern=X`.
    // The parser was not tolerant of this form; tool calls were silently dropped.
    // This scenario verifies: colon-space prefix is accepted, search executes, and
    // the model reads the matched file before synthesizing (Phase 8.3 golden path).

    #[test]
    fn search_code_colon_prefix_executes_successfully() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("hello.rs"), "fn main() {}\n").unwrap();

        let colon_form = "[search_code]\npattern: fn main\n[/search_code]";
        // Phase 8.3: search returns a match → model must read before answering.
        let mut rt = make_runtime(
            &dir,
            vec![colon_form, "[read_file: hello.rs]", "Found the function."],
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "search main".into(),
            },
        );
        assert!(
            !has_failed(&events),
            "colon-prefix search must not fail: {events:?}"
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: search_code ===")),
            "search_code must produce a tool_result"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: read_file ===")),
            "read_file must produce a tool_result after the search"
        );
    }

    // Scenario 2b: bounded search stops after one empty retry
    //
    // Failure class: live search used phrase queries, retried in prose, and kept
    // searching after the allowed retry. The runtime budget must execute only the
    // first empty search and one empty retry, discard later narrated search attempts.
    // Phase 8.3: after both empty searches, the runtime emits the insufficient-evidence
    // terminal answer rather than delegating synthesis to the model.

    #[test]
    fn search_empty_retry_is_bounded_and_stops_cleanly() {
        use crate::runtime::types::{AnswerSource, RuntimeTerminalReason};

        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("unrelated.rs"), "fn unrelated() {}\n").unwrap();

        let mut rt = make_runtime(
            &dir,
            vec![
                "[search_code: logging initialization]",
                "[search_code: logger initialization]",
                "Trying one more.\n[search_code: tracing]",
                // Not consumed — R4 fires before this backend response is requested.
                "No matching code was found for those searches.",
            ],
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where logging is initialized".into(),
            },
        );
        assert!(
            !has_failed(&events),
            "bounded search scenario must not fail permanently: {events:?}"
        );

        // Search budget behavior is unchanged.
        let snapshot = rt.messages_snapshot();
        let user_content = all_user_content(&snapshot);
        assert_eq!(
            user_content
                .matches("=== tool_result: search_code ===")
                .count(),
            2,
            "only the first empty search and one retry should execute"
        );
        assert!(
            user_content.contains("allowed search retry also returned no matches"),
            "runtime must inject closed-search guidance after the empty retry"
        );
        assert!(
            !snapshot
                .iter()
                .any(|m| m.content.contains("Trying one more")),
            "narrated third search attempt must be discarded"
        );

        // Phase 8.3: runtime-owned terminal fires instead of model synthesis.
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
        assert!(
            last_assistant_content(&snapshot)
                .unwrap_or("")
                .contains("found no matches"),
            "runtime terminal must state that no matches were found"
        );
    }

    // Scenario 8.3-A: non-empty search → synthesis without read → correction fires once
    //
    // Phase 8.3 behavior: after search returns matches, the model attempting synthesis
    // without reading any file triggers a one-time runtime correction. The model then
    // gets another attempt. The correction fires at most once per turn.

    #[test]
    fn non_empty_search_synthesis_without_read_fires_correction_once() {
        use crate::runtime::types::{AnswerSource, RuntimeTerminalReason};

        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("target.rs"), "fn target_fn() {}\n").unwrap();

        let mut rt = make_runtime(
            &dir,
            vec![
                "[search_code: target_fn]",      // produces matches
                "The function is in target.rs.", // synthesis without read → correction fires
                "The function is in target.rs.", // second synthesis: still no read → terminal
            ],
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is target_fn defined?".into(),
            },
        );
        assert!(
            !has_failed(&events),
            "must not fail permanently: {events:?}"
        );

        let snapshot = rt.messages_snapshot();

        // Correction must appear exactly once. Match the specific sentinel+text that only
        // READ_BEFORE_ANSWERING produces — not SEARCH_CLOSED_AFTER_RESULTS which also
        // mentions "Search returned matches" and "read_file" inside the results block.
        let correction_count = snapshot
            .iter()
            .filter(|m| {
                m.content.starts_with("[runtime:correction]")
                    && m.content.contains("no matched file has been read")
            })
            .count();
        assert_eq!(
            correction_count, 1,
            "read-before-answering correction must fire exactly once"
        );

        // Correction uses the [runtime:correction] sentinel.
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.starts_with("[runtime:correction]")
                    && m.content.contains("read_file")),
            "correction must use runtime:correction sentinel"
        );

        // Turn ends with a runtime terminal answer, not an admitted synthesis.
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
            "turn must terminate without admitting unread synthesis: {answer_source:?}"
        );
    }

    // Scenario 8.3-B: empty search → no read → runtime terminal insufficient-evidence answer
    //
    // Phase 8.3 behavior: when search was attempted but all results were empty and no file
    // was read, the runtime emits the insufficient-evidence answer without model synthesis.

    #[test]
    fn empty_search_no_read_emits_runtime_terminal_insufficient_evidence() {
        use crate::runtime::types::{AnswerSource, RuntimeTerminalReason};

        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("unrelated.rs"), "fn something_else() {}\n").unwrap();

        let mut rt = make_runtime(
            &dir,
            vec![
                "[search_code: nonexistent_symbol]", // returns empty
                // Pre-evidence prose may remain in the trace but is not admitted.
                "I couldn't find anything about that.",
            ],
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find nonexistent_symbol".into(),
            },
        );
        assert!(!has_failed(&events), "must terminate cleanly: {events:?}");

        // Runtime-owned terminal answer, not model synthesis.
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
            "empty search must produce InsufficientEvidence terminal: {answer_source:?}"
        );

        let snapshot = rt.messages_snapshot();

        // The model's "I couldn't find anything" response is trace context only.
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("I couldn't find anything")),
            "pre-evidence prose should remain in the trace"
        );

        // The runtime terminal message must appear in the conversation.
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("found no matches")),
            "runtime insufficient-evidence answer must be in conversation"
        );
    }

    // Scenario 8.3-C: non-empty search → read_file → synthesis allowed (golden path)
    //
    // Phase 8.3 behavior: search returns matches, model reads a matched file, then
    // synthesizes. No correction fires; turn completes as ToolAssisted normally.

    #[test]
    fn non_empty_search_then_read_then_synthesis_is_allowed() {
        use crate::runtime::types::AnswerSource;

        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("main.rs"),
            "fn main() { println!(\"hello\"); }\n",
        )
        .unwrap();

        let mut rt = make_runtime(
            &dir,
            vec![
                "[search_code: main]",
                "[read_file: main.rs]",
                "The main function prints hello.",
            ],
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "search main".into(),
            },
        );
        assert!(
            !has_failed(&events),
            "golden path must not fail: {events:?}"
        );

        let snapshot = rt.messages_snapshot();

        // No correction should fire.
        assert!(
            !snapshot
                .iter()
                .any(|m| m.content.starts_with("[runtime:correction]")),
            "no correction must fire when search → read → synthesis path is followed"
        );

        // Both tool results must be present.
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: search_code ===")),
            "search_code must have a tool_result"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: read_file ===")),
            "read_file must have a tool_result"
        );

        // Turn completes as ToolAssisted (2 rounds).
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
                Some(AnswerSource::ToolAssisted { rounds: 2 })
            ),
            "golden path must complete as ToolAssisted(2): {answer_source:?}"
        );

        assert_eq!(
            last_assistant_content(&snapshot),
            Some("The main function prints hello."),
        );
    }

    // Scenario 8.3-D: investigation state resets between turns
    //
    // Phase 8.3 behavior: per-turn investigation state must not bleed across turns.
    // A second turn submitted after a search-with-results turn must start fresh —
    // no correction fires if the second turn has no search tools at all.

    #[test]
    fn investigation_state_resets_between_turns() {
        use crate::runtime::types::AnswerSource;

        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("util.rs"), "fn helper() {}\n").unwrap();

        let mut rt = make_runtime(
            &dir,
            vec![
                // Turn 1: search (results) → read → synthesis. Sets investigation state.
                "[search_code: helper]",
                "[read_file: util.rs]",
                "The helper function is in util.rs.",
                // Turn 2: direct answer with no tools — must be Direct, no interference.
                "Sure, I can help with that.",
            ],
        );

        // Turn 1.
        let events1 = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "search helper".into(),
            },
        );
        assert!(!has_failed(&events1), "turn 1 must not fail: {events1:?}");
        let src1 = events1.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(s) = e {
                Some(s.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(src1, Some(AnswerSource::ToolAssisted { .. })),
            "turn 1 must be ToolAssisted: {src1:?}"
        );

        // Turn 2: no tools — must be Direct without investigation interference.
        let events2 = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Can you help me with something else?".into(),
            },
        );
        assert!(!has_failed(&events2), "turn 2 must not fail: {events2:?}");

        // No correction must fire in turn 2.
        let snapshot = rt.messages_snapshot();
        let turn2_corrections = snapshot
            .iter()
            .rev()
            .take_while(|m| !m.content.contains("The helper function is in util.rs."))
            .filter(|m| m.content.starts_with("[runtime:correction]"))
            .count();
        assert_eq!(
            turn2_corrections, 0,
            "no correction must fire in the second turn"
        );

        let src2 = events2.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(s) = e {
                Some(s.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(src2, Some(AnswerSource::Direct)),
            "turn 2 with no tools must be Direct: {src2:?}"
        );
    }

    // Scenario 3: partial edit_file (missing ---search---) surfaces error
    //
    // Failure class: model emitted [edit_file] block with ---replace--- but no
    // ---search--- section. The parser silently dropped the block. Now it produces
    // an empty search string, causing the tool to surface a clear, actionable error.

    #[test]
    fn partial_edit_file_missing_search_surfaces_clear_error() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("f.rs"), "fn old() {}\n").unwrap();

        let partial = "[edit_file]\npath: f.rs\n---replace---\nfn new() {}\n[/edit_file]";
        let mut rt = make_runtime(
            &dir,
            vec![partial, "I need to include the ---search--- section."],
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "edit f.rs".into(),
            },
        );
        assert!(
            !has_failed(&events),
            "partial edit must not permanently fail: {events:?}"
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| {
                m.content.contains("=== tool_error: edit_file ===")
                    && m.content.contains("---search---")
            }),
            "tool_error must contain ---search--- guidance"
        );
    }

    // Scenario 3b: malformed edit repair can recover to valid edit
    //
    // Failure class: an initial malformed edit failed, then a malformed repair
    // attempt fell through as a direct answer. The runtime now injects a targeted
    // correction, accepts the subsequent valid edit block, and routes it through
    // normal approval.

    #[test]
    fn malformed_edit_repair_can_recover_to_valid_approved_edit() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("f.rs");
        fs::write(&file, "hello world").unwrap();

        let first_bad_edit = "[edit_file]\npath: f.rs\n---replace---\nhello thunk\n[/edit_file]";
        let malformed_repair =
            "[edit_file]\npath: f.rs\nFind: hello world\nReplace: hello thunk\n[/edit_file]";
        let valid_edit = "[edit_file]\npath: f.rs\n---search---\nhello world\n---replace---\nhello thunk\n[/edit_file]";

        let mut rt = make_runtime(
            &dir,
            vec![
                first_bad_edit,
                malformed_repair,
                valid_edit,
                "Edit applied.",
            ],
        );

        let submit_events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Edit f.rs and change hello world to hello thunk".into(),
            },
        );
        assert!(
            !has_failed(&submit_events),
            "submit must recover to approval: {submit_events:?}"
        );
        assert!(
            has_approval(&submit_events),
            "valid repaired edit must request approval"
        );
        assert_eq!(
            fs::read_to_string(&file).unwrap(),
            "hello world",
            "file must not change before approval"
        );

        let approve_events = collect_events(&mut rt, RuntimeRequest::Approve);
        assert!(
            !has_failed(&approve_events),
            "approve must execute repaired edit: {approve_events:?}"
        );
        assert_eq!(
            fs::read_to_string(&file).unwrap(),
            "hello thunk",
            "approved repaired edit must update the file"
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| {
                m.content.starts_with("[runtime:correction]") && m.content.contains("edit_file")
            }),
            "runtime must inject edit repair correction"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: edit_file ===")),
            "approved repaired edit must inject a tool result"
        );
        // Phase 11.2.1: runtime-owned answer after mutation — no model synthesis.
        assert!(
            last_assistant_content(&snapshot)
                .map(|s| s.starts_with("edit_file result:"))
                .unwrap_or(false),
            "approval should produce runtime-owned answer after executing the repaired edit"
        );
    }

    // Scenario 3c: missing read_file terminates without retry loop
    //
    // Failure class: missing-file reads repeatedly executed failing read_file
    // calls. The runtime should surface the first tool error and end with a
    // clean no-contents-read answer instead of consuming later retry responses.

    #[test]
    fn missing_read_file_terminates_cleanly_without_looping() {
        let dir = TempDir::new().unwrap();
        let mut rt = make_runtime(
            &dir,
            vec![
                "[read_file: missing_file_phase75.rs]",
                "[read_file: missing_file_phase75.rs]",
            ],
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Read missing_file_phase75.rs".into(),
            },
        );
        assert!(
            !has_failed(&events),
            "missing read should terminate cleanly: {events:?}"
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_error: read_file ===")),
            "read_file failure must be surfaced as a tool_error"
        );
        let assistant_read_calls = snapshot
            .iter()
            .filter(|m| {
                m.role == crate::llm::backend::Role::Assistant
                    && m.content.contains("[read_file: missing_file_phase75.rs]")
            })
            .count();
        assert_eq!(
            assistant_read_calls, 0,
            "seeded direct-read failure must terminate before consuming any backend retry"
        );
        assert!(
            last_assistant_content(&snapshot)
                .unwrap_or("")
                .contains("No file contents were read."),
            "last assistant answer must clearly terminate the missing read"
        );
    }

    // Scenario 4: fabrication triggers one correction then succeeds
    //
    // Failure class: model emits [tool_result:] / [tool_error:] in its own response,
    // bypassing the protocol. The engine must detect this, discard the response,
    // inject a correction message, and re-invoke the model. One retry is allowed.

    #[test]
    fn fabrication_triggers_correction_then_normal_answer() {
        let dir = TempDir::new().unwrap();
        let fabricated = "=== tool_result: read_file ===\nsome fake content\n=== /tool_result ===";
        let mut rt = make_runtime(&dir, vec![fabricated, "Let me answer directly."]);

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "tell me something".into(),
            },
        );
        assert!(
            !has_failed(&events),
            "one fabrication must not permanently fail: {events:?}"
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| {
                m.content.starts_with("[runtime:correction]") && m.content.contains("result block")
            }),
            "correction message must be in conversation"
        );
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant);
        assert_eq!(
            last_assistant.map(|m| m.content.as_str()),
            Some("Let me answer directly."),
            "last assistant message must be the corrected response"
        );
    }

    // Scenario 5: malformed open tag correction
    //
    // Failure class: model used [test_file]...[/write_file] — mismatched tag name.
    // The engine must detect the orphan close tag, inject a correction, and retry.

    #[test]
    fn malformed_open_tag_triggers_correction_and_retries() {
        let dir = TempDir::new().unwrap();
        let malformed = "[test_file]\npath: f.txt\n---content---\nhello\n[/write_file]";
        let mut rt = make_runtime(&dir, vec![malformed, "Done."]);

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "create f.txt".into(),
            },
        );
        assert!(
            !has_failed(&events),
            "malformed tag must not permanently fail: {events:?}"
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| {
                m.content.starts_with("[runtime:correction]")
                    && m.content.contains("unrecognized opening tag")
            }),
            "malformed-block correction must be in conversation"
        );
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant);
        assert_eq!(
            last_assistant.map(|m| m.content.as_str()),
            Some("Done."),
            "last assistant message must be the retry response"
        );
    }

    // Scenario 6: cycle detection not triggered by errors
    //
    // Failure class: a successful call sets the cycle fingerprint. If the tool
    // then errors on retry, cycle detection must NOT block it — only successful
    // calls should set the cycle key.

    #[test]
    fn cycle_detection_allows_retry_after_error() {
        let dir = TempDir::new().unwrap();
        // First list_dir fails (bad path), second succeeds on ".".
        let mut rt = make_runtime(
            &dir,
            vec!["[list_dir: /no/such/path][list_dir: .]", "Done."],
        );
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "display the structure".into(),
            },
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_error: list_dir ===")),
            "failed call must produce a tool_error"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: list_dir ===")),
            "retry after error must not be blocked"
        );
        assert!(
            !snapshot.iter().any(|m| {
                m.content.contains("identical arguments twice in a row")
                    && m.content.contains("[list_dir: .]")
            }),
            "retry with same args after error must not trigger cycle detection"
        );
    }

    // Scenario 8.4-A: code identifier question forces investigation before Direct
    //
    // Phase 8.4 golden path: question contains a snake_case identifier → model attempts
    // Direct → R1 fires → model searches → reads matched file → synthesizes ToolAssisted.

    #[test]
    fn code_identifier_question_forces_investigation_before_direct() {
        use crate::runtime::types::AnswerSource;

        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();

        let mut rt = make_runtime(
            &dir,
            vec![
                "It runs the tool loop.",              // Direct attempt → R1 fires
                "[search_code: run_turns]",            // after R1 correction, model searches
                "[read_file: engine.rs]",              // after results, model reads
                "run_turns drives the generate-loop.", // synthesis
            ],
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "What does run_turns do?".into(),
            },
        );
        assert!(!has_failed(&events), "must not fail: {events:?}");

        let snapshot = rt.messages_snapshot();

        // R1 correction must appear in conversation.
        assert!(
            snapshot.iter().any(|m| {
                m.content.starts_with("[runtime:correction]")
                    && m.content.contains("specific code element")
            }),
            "R1 correction must be injected"
        );

        // Both tool results must appear.
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: search_code ===")),
            "search_code must produce a tool_result"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: read_file ===")),
            "read_file must produce a tool_result"
        );

        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(s) = e {
                Some(s.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "must complete as ToolAssisted: {answer_source:?}"
        );
        assert_eq!(
            last_assistant_content(&snapshot),
            Some("run_turns drives the generate-loop."),
        );
    }

    // Scenario 8.4-B: R1 fires at most once per turn
    //
    // Phase 8.4.x behavior: after R1 fires once, subsequent Direct attempts without tools
    // are not admitted; the runtime terminates with insufficient evidence.

    #[test]
    fn investigation_required_correction_fires_once_only() {
        use crate::runtime::types::{AnswerSource, RuntimeTerminalReason};

        let dir = TempDir::new().unwrap();

        let mut rt = make_runtime(
            &dir,
            vec![
                "It runs the loop.",       // Direct → R1 fires
                "It still runs the loop.", // Direct again → R1 already fired → allowed
            ],
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "What does run_turns do?".into(),
            },
        );
        assert!(!has_failed(&events), "must not fail: {events:?}");

        let snapshot = rt.messages_snapshot();

        let r1_count = snapshot
            .iter()
            .filter(|m| {
                m.content.starts_with("[runtime:correction]")
                    && m.content.contains("specific code element")
            })
            .count();
        assert_eq!(r1_count, 1, "R1 correction must fire exactly once");

        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(s) = e {
                Some(s.clone())
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
            "second Direct attempt after R1 must not be admitted: {answer_source:?}"
        );
    }

    // Scenario 8.4-C: non-identifier question allows Direct unchanged
    //
    // Phase 8.4 behavior: when the question contains no snake_case or PascalCase identifiers,
    // investigation_required is false and R1 never fires.

    #[test]
    fn non_identifier_question_allows_direct_unchanged() {
        use crate::runtime::types::AnswerSource;

        let dir = TempDir::new().unwrap();
        let mut rt = make_runtime(&dir, vec!["I'm doing well, thanks!"]);

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "How are you?".into(),
            },
        );
        assert!(!has_failed(&events), "must not fail: {events:?}");

        let snapshot = rt.messages_snapshot();
        assert!(
            !snapshot
                .iter()
                .any(|m| m.content.starts_with("[runtime:correction]")),
            "no correction must fire for a non-identifier question"
        );

        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(s) = e {
                Some(s.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::Direct)),
            "plain question must produce Direct: {answer_source:?}"
        );
    }

    // Scenario 8.4-D: model that searches first skips R1
    //
    // Phase 8.4 behavior: if the model uses tools before attempting synthesis, tool_rounds > 0
    // when calls.is_empty() is reached and R1 does not fire.

    #[test]
    fn already_investigated_skips_r1() {
        use crate::runtime::types::AnswerSource;

        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();

        let mut rt = make_runtime(
            &dir,
            vec![
                "[search_code: run_turns]", // model investigates first
                "[read_file: engine.rs]",
                "run_turns orchestrates the loop.", // synthesis after tools
            ],
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "What does run_turns do?".into(),
            },
        );
        assert!(!has_failed(&events), "must not fail: {events:?}");

        let snapshot = rt.messages_snapshot();
        assert!(
            !snapshot.iter().any(|m| {
                m.content.starts_with("[runtime:correction]")
                    && m.content.contains("specific code element")
            }),
            "R1 must not fire when model already investigated"
        );

        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(s) = e {
                Some(s.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "must complete as ToolAssisted: {answer_source:?}"
        );
    }

    // Scenario 8.4-E: approve-failure path does not trigger R1
    //
    // Phase 8.4 behavior: handle_approve failure injects a tool_error as the last user
    // message, then calls run_turns(0,...). The investigation_required filter must exclude
    // tool_error messages so R1 does not misfire on the model's error explanation.

    #[test]
    fn approve_path_does_not_trigger_r1() {
        use crate::runtime::types::AnswerSource;

        let dir = TempDir::new().unwrap();

        // Inject a bad edit_file payload — execute_approved will fail.
        let bad_payload = format!(
            "{}\x00nonexistent_search_text\x00replacement",
            dir.path().join("no_such_file.rs").display()
        );

        let mut rt = make_runtime(&dir, vec!["I couldn't apply the edit_file change."]);
        rt.set_pending_for_test(crate::tools::PendingAction {
            tool_name: "edit_file".into(),
            summary: "edit no_such_file.rs".into(),
            risk: crate::tools::RiskLevel::Medium,
            payload: bad_payload,
        });

        let events = collect_events(&mut rt, RuntimeRequest::Approve);
        assert!(
            !has_failed(&events),
            "approve failure must not permanently fail: {events:?}"
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            !snapshot.iter().any(|m| {
                m.content.starts_with("[runtime:correction]")
                    && m.content.contains("specific code element")
            }),
            "R1 must not fire on approve-failure path"
        );

        // handle_approve calls run_turns(0,...) on failure, so tool_rounds starts at 0.
        // The model explains the failure directly — Direct answer, no tool rounds.
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(s) = e {
                Some(s.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::Direct)),
            "approve-failure path must complete as Direct (run_turns(0,...)): {answer_source:?}"
        );
    }

    // Scenario 8.4-F: R1 composes with R4 — identifier question, R1 fires, search empty → terminal
    //
    // Phase 8.4 behavior: R1 fires (Direct → SEARCH_BEFORE_ANSWERING), model searches but
    // gets no results, R4 fires → InsufficientEvidence terminal. Demonstrates composition.

    #[test]
    fn r1_composes_with_r4_on_empty_search() {
        use crate::runtime::types::{AnswerSource, RuntimeTerminalReason};

        let dir = TempDir::new().unwrap();
        // No file with "run_turns" — search will return empty.
        fs::write(dir.path().join("unrelated.rs"), "fn something_else() {}\n").unwrap();

        let mut rt = make_runtime(
            &dir,
            vec![
                "It runs the loop.",        // Direct → R1 fires
                "[search_code: run_turns]", // after correction, model searches → empty
                // Pre-evidence prose may remain in the trace but is not admitted.
                "I couldn't find run_turns.",
            ],
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "What does run_turns do?".into(),
            },
        );
        assert!(!has_failed(&events), "must not fail: {events:?}");

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| {
                m.content.starts_with("[runtime:correction]")
                    && m.content.contains("specific code element")
            }),
            "R1 correction must have fired"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("I couldn't find run_turns")),
            "pre-evidence prose should remain in the trace"
        );

        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(s) = e {
                Some(s.clone())
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
            "R1+empty-search must produce InsufficientEvidence terminal: {answer_source:?}"
        );
    }

    // Scenario 8.4-G: R1 composes with R2 — identifier question, R1 fires, search returns
    // results, model skips read → R2 fires, model reads, synthesis allowed.

    #[test]
    fn r1_composes_with_r2_on_search_results_no_read() {
        use crate::runtime::types::AnswerSource;

        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();

        let mut rt = make_runtime(
            &dir,
            vec![
                "It runs the loop.",                         // Direct → R1 fires
                "[search_code: run_turns]", // after R1, model searches → has results
                "The function is in engine.rs.", // synthesis without read → R2 fires
                "[read_file: engine.rs]",   // after R2, model reads
                "run_turns is the main orchestration loop.", // synthesis allowed
            ],
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "What does run_turns do?".into(),
            },
        );
        assert!(!has_failed(&events), "must not fail: {events:?}");

        let snapshot = rt.messages_snapshot();

        // Both R1 and R2 corrections must appear.
        assert!(
            snapshot.iter().any(|m| {
                m.content.starts_with("[runtime:correction]")
                    && m.content.contains("specific code element")
            }),
            "R1 correction must be in conversation"
        );
        assert!(
            snapshot.iter().any(|m| {
                m.content.starts_with("[runtime:correction]")
                    && m.content.contains("no matched file has been read")
            }),
            "R2 correction must be in conversation"
        );

        // Both tool results must appear.
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: search_code ===")),
            "search must have a tool_result"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: read_file ===")),
            "read must have a tool_result"
        );

        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(s) = e {
                Some(s.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "must complete as ToolAssisted after R1+R2 composition: {answer_source:?}"
        );
        assert_eq!(
            last_assistant_content(&snapshot),
            Some("run_turns is the main orchestration loop."),
        );
    }

    // Scenario 8.4-H: natural-language lookup prompts require the same search → read discipline
    //
    // Failure class: prompts like "Find where logging is initialized" do not contain a
    // snake_case/PascalCase identifier, so they were admitted as Direct after search-only evidence.

    #[test]
    fn natural_language_lookup_requires_read_before_answer() {
        use crate::runtime::types::AnswerSource;

        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("logging.rs"), "pub fn init_logging() {}\n").unwrap();

        let mut rt = make_runtime(
            &dir,
            vec![
                "[search_code: logging]",
                "Logging is initialized in logging.rs.",
                "[read_file: logging.rs]",
                "Logging initialization is grounded in logging.rs.",
            ],
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where logging is initialized".into(),
            },
        );
        assert!(!has_failed(&events), "must not fail: {events:?}");

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| {
                m.content.starts_with("[runtime:correction]")
                    && m.content.contains("no matched file has been read")
            }),
            "natural-language lookup must still require a matched read"
        );

        let chunks = assistant_chunks(&events);
        assert_eq!(
            chunks,
            vec!["Logging initialization is grounded in logging.rs.".to_string()],
            "only the final post-read synthesis should be visible"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(s) = e {
                Some(s.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "final answer must be tool-assisted: {answer_source:?}"
        );
    }

    // Scenario 8.4-I: explicit missing-file read cannot drift into unrelated large reads
    //
    // Failure class: after a missing-file prompt, the model searched and then read a large
    // unrelated file, overflowing context. Runtime now blocks reads that do not match the
    // file path explicitly requested by the user.

    #[test]
    fn missing_file_prompt_blocks_unrelated_large_read() {
        use crate::runtime::types::{AnswerSource, RuntimeTerminalReason};

        let dir = TempDir::new().unwrap();
        let large = (0..500)
            .map(|i| format!("fn unrelated_{i}() {{}}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(dir.path().join("engine.rs"), large).unwrap();

        let mut rt = make_runtime(
            &dir,
            vec![
                "[search_code: unrelated]",
                "[read_file: engine.rs]",
                "The missing file is probably represented by engine.rs.",
            ],
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Read missing_file_phase84x.rs".into(),
            },
        );
        assert!(!has_failed(&events), "must terminate cleanly: {events:?}");

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_error: read_file ===")),
            "seeded missing-file read must surface a read_file tool_error"
        );
        assert!(
            !snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: read_file ===")),
            "no file contents should be read into context"
        );
        assert!(
            !snapshot.iter().any(|m| m.content.contains("unrelated_499")),
            "large unrelated file content must not enter the conversation"
        );
        assert!(
            !snapshot
                .iter()
                .any(|m| m.content.contains("[search_code: unrelated]")
                    || m.content.contains("[read_file: engine.rs]")
                    || m.content.contains("probably represented by engine.rs")),
            "seeded direct-read failure must not consume later backend guesses"
        );

        let chunks = assistant_chunks(&events);
        assert_eq!(
            chunks.len(),
            1,
            "visible output should be one runtime terminal"
        );
        assert!(
            chunks[0].contains("I couldn't read `missing_file_phase84x.rs`")
                && chunks[0].contains("No file contents were read."),
            "visible output should explain the missing direct read: {chunks:?}"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(s) = e {
                Some(s.clone())
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
            "seeded missing-file read must end as ReadFileFailed terminal: {answer_source:?}"
        );
    }

    // Scenario 8.4-J: pre-evidence prose is trace-only, not visible assistant output

    #[test]
    fn speculative_prose_before_evidence_is_not_visible() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();

        let mut rt = make_runtime(
            &dir,
            vec![
                "Speculative answer before evidence.",
                "[search_code: run_turns]",
                "[read_file: engine.rs]",
                "Grounded final answer.",
            ],
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "What does run_turns do?".into(),
            },
        );
        assert!(!has_failed(&events), "must not fail: {events:?}");

        let chunks = assistant_chunks(&events);
        assert_eq!(
            chunks,
            vec!["Grounded final answer.".to_string()],
            "speculative pre-evidence prose must not be emitted as visible assistant output"
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content == "Speculative answer before evidence."),
            "pre-evidence prose remains trace context"
        );
    }

    // Scenario 7: approve -> synthesis
    //
    // Phase 11.2.1: approved mutations now produce a runtime-owned answer directly.
    // The runtime calls finish_with_runtime_answer rather than re-entering generation.

    #[test]
    fn approve_produces_runtime_owned_answer_after_mutation() {
        use std::io::Write;

        use crate::tools::RiskLevel;
        use tempfile::NamedTempFile;

        let dir = TempDir::new().unwrap();

        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "hello").unwrap();
        let path = f.path().to_string_lossy().into_owned();

        let payload = format!("{}\x00hello\x00world", path);

        // No model responses needed — the runtime owns the answer.
        let mut rt = make_runtime(&dir, Vec::<&str>::new());
        rt.set_pending_for_test(crate::tools::PendingAction {
            tool_name: "edit_file".into(),
            summary: format!("edit {path}"),
            risk: RiskLevel::Medium,
            payload,
        });

        let events = collect_events(&mut rt, RuntimeRequest::Approve);
        assert!(!has_failed(&events), "approve must not fail: {events:?}");
        // finish_with_runtime_answer emits AssistantMessageChunk for the runtime-owned answer.
        assert!(
            has_chunk(&events),
            "approve must emit runtime-owned answer chunk"
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: edit_file ===")),
            "tool result must be in conversation after approve"
        );
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant);
        assert!(
            last_assistant
                .map(|m| m.content.starts_with("edit_file result:"))
                .unwrap_or(false),
            "last assistant message must be runtime-owned answer: {last_assistant:?}"
        );
    }
}
