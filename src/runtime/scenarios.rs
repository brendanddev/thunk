/// Scenario-based regression tests covering the exact failure classes fixed in Phase 7.5.
/// Each test encodes one concrete failure mode end-to-end: prompt → backend response →
/// runtime handling → conversation state.
///
/// These are complementary to the unit tests in engine.rs (which test engine internals)
/// and tool_codec.rs (which test parsing). Scenarios test full round-trips.
#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use crate::app::config::Config;
    use crate::llm::backend::{BackendEvent, GenerateRequest, ModelBackend};
    use crate::runtime::types::{RuntimeEvent, RuntimeRequest};
    use crate::runtime::Runtime;
    use crate::tools::default_registry;

    // ── Test backend ─────────────────────────────────────────────────────────

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

        fn generate(
            &mut self,
            _request: GenerateRequest,
            on_event: &mut dyn FnMut(BackendEvent),
        ) -> crate::app::Result<()> {
            let reply = self.responses.get(self.call_count).cloned().unwrap_or_default();
            self.call_count += 1;
            if !reply.is_empty() {
                on_event(BackendEvent::TextDelta(reply));
            }
            on_event(BackendEvent::Finished);
            Ok(())
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_runtime(dir: &TempDir, responses: Vec<impl Into<String>>) -> Runtime {
        Runtime::new(
            &Config::default(),
            dir.path(),
            Box::new(TestBackend::new(responses)),
            default_registry(dir.path().to_path_buf()),
        )
    }

    fn collect_events(runtime: &mut Runtime, request: RuntimeRequest) -> Vec<RuntimeEvent> {
        let mut events = Vec::new();
        runtime.handle(request, &mut |e| events.push(e));
        events
    }

    fn has_failed(events: &[RuntimeEvent]) -> bool {
        events.iter().any(|e| matches!(e, RuntimeEvent::Failed { .. }))
    }

    fn has_approval(events: &[RuntimeEvent]) -> bool {
        events.iter().any(|e| matches!(e, RuntimeEvent::ApprovalRequired(_)))
    }

    fn has_chunk(events: &[RuntimeEvent]) -> bool {
        events.iter().any(|e| matches!(e, RuntimeEvent::AssistantMessageChunk(_)))
    }

    // ── Scenario 1: Full reject flow ──────────────────────────────────────────
    //
    // Failure class: reject was framed as a retryable error, causing the model to
    // re-propose the same tool call. The rejection message must be terminal.
    // This scenario verifies: proposal fires ApprovalRequired, reject drives
    // synthesis, and no second ApprovalRequired is fired.

    #[test]
    fn reject_flow_drives_synthesis_without_reproposal() {
        let dir = TempDir::new().unwrap();
        let mut rt = make_runtime(
            &dir,
            vec![
                "[write_file: temp.rs]",        // model proposes write
                "Understood, I won't create it.", // synthesis after rejection
            ],
        );

        let submit_events =
            collect_events(&mut rt, RuntimeRequest::Submit { text: "create temp.rs".into() });
        assert!(!has_failed(&submit_events), "submit must not fail: {submit_events:?}");
        assert!(has_approval(&submit_events), "submit must fire ApprovalRequired");

        let reject_events = collect_events(&mut rt, RuntimeRequest::Reject);
        assert!(!has_failed(&reject_events), "reject must not fail: {reject_events:?}");
        assert!(has_chunk(&reject_events), "reject must drive synthesis");
        // No second approval — model synthesized without re-proposing
        assert!(
            !has_approval(&reject_events),
            "reject must not fire a second ApprovalRequired"
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| m.content.contains("do not retry")),
            "rejection error must contain terminal guidance"
        );
        assert!(
            snapshot.iter().any(|m| m.content.contains("Understood")),
            "synthesis response must be in conversation"
        );
    }

    // ── Scenario 2: search_code colon-space prefix ────────────────────────────
    //
    // Failure class: model emits `pattern: X` (colon-space) instead of `pattern=X`.
    // The parser was not tolerant of this form; tool calls were silently dropped.
    // This scenario verifies: colon-space prefix is accepted and search executes.

    #[test]
    fn search_code_colon_prefix_executes_successfully() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("hello.rs"), "fn main() {}\n").unwrap();

        let colon_form = "[search_code]\npattern: fn main\n[/search_code]";
        let mut rt = make_runtime(&dir, vec![colon_form, "Found the function."]);

        let events =
            collect_events(&mut rt, RuntimeRequest::Submit { text: "find main".into() });
        assert!(!has_failed(&events), "colon-prefix search must not fail: {events:?}");

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| m.content.contains("=== tool_result: search_code ===")),
            "search_code must produce a tool_result"
        );
    }

    // ── Scenario 3: partial edit_file (missing ---search---) surfaces error ────
    //
    // Failure class: model emitted [edit_file] block with ---replace--- but no
    // ---search--- section. The parser silently dropped the block. Now it produces
    // an empty search string, causing the tool to surface a clear, actionable error.

    #[test]
    fn partial_edit_file_missing_search_surfaces_clear_error() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("f.rs"), "fn old() {}\n").unwrap();

        let partial = "[edit_file]\npath: f.rs\n---replace---\nfn new() {}\n[/edit_file]";
        let mut rt =
            make_runtime(&dir, vec![partial, "I need to include the ---search--- section."]);

        let events = collect_events(&mut rt, RuntimeRequest::Submit { text: "edit f.rs".into() });
        assert!(!has_failed(&events), "partial edit must not permanently fail: {events:?}");

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| {
                m.content.contains("=== tool_error: edit_file ===")
                    && m.content.contains("---search---")
            }),
            "tool_error must contain ---search--- guidance"
        );
    }

    // ── Scenario 4: fabrication triggers one correction then succeeds ─────────
    //
    // Failure class: model emits [tool_result:] / [tool_error:] in its own response,
    // bypassing the protocol. The engine must detect this, discard the response,
    // inject a correction message, and re-invoke the model. One retry is allowed.

    #[test]
    fn fabrication_triggers_correction_then_normal_answer() {
        let dir = TempDir::new().unwrap();
        let fabricated = "=== tool_result: read_file ===\nsome fake content\n=== /tool_result ===";
        let mut rt = make_runtime(&dir, vec![fabricated, "Let me answer directly."]);

        let events =
            collect_events(&mut rt, RuntimeRequest::Submit { text: "tell me something".into() });
        assert!(!has_failed(&events), "one fabrication must not permanently fail: {events:?}");

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

    // ── Scenario 5: malformed open tag correction ─────────────────────────────
    //
    // Failure class: model used [test_file]...[/write_file] — mismatched tag name.
    // The engine must detect the orphan close tag, inject a correction, and retry.

    #[test]
    fn malformed_open_tag_triggers_correction_and_retries() {
        let dir = TempDir::new().unwrap();
        let malformed = "[test_file]\npath: f.txt\n---content---\nhello\n[/write_file]";
        let mut rt = make_runtime(&dir, vec![malformed, "Done."]);

        let events =
            collect_events(&mut rt, RuntimeRequest::Submit { text: "create f.txt".into() });
        assert!(!has_failed(&events), "malformed tag must not permanently fail: {events:?}");

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

    // ── Scenario 6: cycle detection not triggered by errors ───────────────────
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
        collect_events(&mut rt, RuntimeRequest::Submit { text: "list both".into() });

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| m.content.contains("=== tool_error: list_dir ===")),
            "failed call must produce a tool_error"
        );
        assert!(
            snapshot.iter().any(|m| m.content.contains("=== tool_result: list_dir ===")),
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

    // ── Scenario 7: approve → synthesis ──────────────────────────────────────
    //
    // Failure class: approved mutations did not trigger synthesis — the TUI froze
    // with no response after /approve. Now handle_approve calls run_turns(1, …)
    // so the model confirms what was done.

    #[test]
    fn approve_triggers_synthesis_after_mutation() {
        use std::io::Write;

        use tempfile::NamedTempFile;
        use crate::tools::RiskLevel;

        let dir = TempDir::new().unwrap();

        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "hello").unwrap();
        let path = f.path().to_string_lossy().into_owned();

        let payload = format!("{}\x00hello\x00world", path);

        let mut rt = make_runtime(&dir, vec!["Edit applied successfully."]);
        rt.set_pending_for_test(crate::tools::PendingAction {
            tool_name: "edit_file".into(),
            summary: format!("edit {path}"),
            risk: RiskLevel::Medium,
            payload,
        });

        let events = collect_events(&mut rt, RuntimeRequest::Approve);
        assert!(!has_failed(&events), "approve must not fail: {events:?}");
        assert!(has_chunk(&events), "approve must drive synthesis");

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| m.content.contains("=== tool_result: edit_file ===")),
            "tool result must be in conversation after approve"
        );
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant);
        assert!(
            last_assistant.map(|m| m.content.contains("Edit applied")).unwrap_or(false),
            "last assistant message must be synthesis"
        );
    }
}
