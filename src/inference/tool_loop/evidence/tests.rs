use super::*;
use crate::inference::tool_loop::ToolLoopIntent;
use crate::inference::Message;
use crate::tools::ToolRegistry;
use crate::tools::ToolResult;

use super::observe::{all_candidates_fully_read, all_search_candidate_paths, observed_read_paths};

#[test]
fn path_normalization_handles_dotslash_prefix() {
    let result = ToolResult {
        tool_name: "read_file".to_string(),
        argument: "./src/main.rs".to_string(),
        output: "File: ./src/main.rs\nLines: 1\n\n```\nfn main() {}\n```".to_string(),
    };
    let paths = observed_read_paths(&[result]);
    assert!(
        paths.contains("src/main.rs"),
        "should normalize ./src/main.rs to src/main.rs"
    );
}

#[test]
fn path_normalization_handles_backslash() {
    let result = ToolResult {
        tool_name: "read_file".to_string(),
        argument: "src\\main.rs".to_string(),
        output: "File: src\\main.rs\nLines: 1\n\n```\nfn main() {}\n```".to_string(),
    };
    let paths = observed_read_paths(&[result]);
    assert!(
        paths.contains("src/main.rs"),
        "should normalize src\\main.rs to src/main.rs"
    );
}

#[test]
fn declaration_extraction_stops_at_test_boundary() {
    let content = "mod foo {\n    pub fn test_fn() {}\n}\n\n#[cfg(test)]\nmod tests {\n    fn test_example() {}\n}";

    let declarations = super::parse::declaration_lines_with_numbers(content, 10);
    let has_test = declarations
        .iter()
        .any(|(_, line)| line.contains("test_example"));
    assert!(
        !has_test,
        "declarations should not include lines after test boundary"
    );
}

#[test]
fn implementation_evidence_from_test_file_is_rejected() {
    let test_read = ToolResult {
        tool_name: "read_file".to_string(),
        argument: "tests/integration_test.rs".to_string(),
        output: "File: tests/integration_test.rs\nLines: 3\n\n```\n#[test]\nfn it_works() {}\n```"
            .to_string(),
    };
    let search = ToolResult {
        tool_name: "search".to_string(),
        argument: "it_works".to_string(),
        output: "tests/integration_test.rs:\n  2: fn it_works()\n".to_string(),
    };

    assert!(
        !has_relevant_file_evidence(
            ToolLoopIntent::CodeNavigation,
            "Where is it_works defined?",
            &[search, test_read]
        ),
        "test file evidence should be rejected"
    );
}

#[test]
fn lookup_completeness_no_artificial_limit() {
    let results = vec![
        ToolResult {
            tool_name: "search".to_string(),
            argument: "my_func".to_string(),
            output: "src/a.rs:\n  1: fn my_func()\nsrc/b.rs:\n  2: fn my_func()\n".to_string(),
        },
        ToolResult {
            tool_name: "read_file".to_string(),
            argument: "src/a.rs".to_string(),
            output: "File: src/a.rs\nLines: 1\n\n```\nfn my_func() {}\n```".to_string(),
        },
        ToolResult {
            tool_name: "read_file".to_string(),
            argument: "src/b.rs".to_string(),
            output: "File: src/b.rs\nLines: 1\n\n```\nfn my_func() {}\n```".to_string(),
        },
    ];

    let outcome = investigation_outcome(
        ToolLoopIntent::CallSiteLookup,
        "what calls my_func",
        None,
        &results,
    );

    match outcome {
        InvestigationOutcome::Ready { evidence, .. } => {
            if let StructuredEvidence::CallSites(cse) = evidence {
                assert!(
                    cse.sites.len() >= 2,
                    "should include all observed call-sites without .take(2) limit"
                );
            }
        }
        _ => {}
    }
}

#[test]
fn repo_overview_bootstrap_reads_manifest_and_entrypoint() {
    let dir = std::env::temp_dir().join(format!("params-repo-bootstrap-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(dir.join("Cargo.toml"), "[package]\nname = \"demo\"\n").unwrap();
    std::fs::write(dir.join("README.md"), "# demo\n").unwrap();
    std::fs::write(dir.join("src/main.rs"), "fn main() {}\n").unwrap();

    let (tx, _rx) = std::sync::mpsc::channel();
    let results = bootstrap_tool_results(
        ToolLoopIntent::RepoOverview,
        "Can you see my project?",
        None,
        &[Message::user("Can you see my project?")],
        &dir,
        "llama.cpp",
        &ToolRegistry::default(),
        &tx,
    );

    assert!(results.iter().any(|r| r.tool_name == "list_dir"));
    assert!(results.iter().any(|r| r.argument == "Cargo.toml"));
    assert!(results.iter().any(|r| r.argument == "src/main.rs"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn callsite_lookup_requires_non_definition_read() {
    let definition_result = ToolResult {
        tool_name: "read_file".to_string(),
        argument: "src/session/mod.rs".to_string(),
        output: "File: src/session/mod.rs\nLines: 3\n\n```\npub fn load_most_recent() {\n}\n```"
            .to_string(),
    };
    let search_result = ToolResult {
        tool_name: "search".to_string(),
        argument: "load_most_recent".to_string(),
        output: "src/session/mod.rs:\n  1: pub fn load_most_recent() {\n\nsrc/main.rs:\n  12: store.load_most_recent();\n"
            .to_string(),
    };

    assert!(
        !has_relevant_file_evidence(
            ToolLoopIntent::CallSiteLookup,
            "what calls load_most_recent",
            &[search_result.clone(), definition_result]
        ),
        "definition-only reads should not satisfy call-site lookup"
    );

    let caller_result = ToolResult {
        tool_name: "read_file".to_string(),
        argument: "src/main.rs".to_string(),
        output: "File: src/main.rs\nLines: 5\n\n```\nfn start() {\n    store.load_most_recent();\n}\n```"
            .to_string(),
    };
    assert!(has_relevant_file_evidence(
        ToolLoopIntent::CallSiteLookup,
        "what calls load_most_recent",
        &[search_result, caller_result]
    ));
}

#[test]
fn flow_trace_requires_cross_file_evidence() {
    let one_read = ToolResult {
        tool_name: "read_file".to_string(),
        argument: "src/main.rs".to_string(),
        output: "File: src/main.rs\nLines: 4\n\n```\nfn main() {\n    init_logging();\n}\n```"
            .to_string(),
    };
    let search = ToolResult {
        tool_name: "search".to_string(),
        argument: "logging".to_string(),
        output:
            "src/main.rs:\n  2: init_logging();\n\nsrc/logging.rs:\n  1: pub fn init_logging() {}\n"
                .to_string(),
    };

    assert!(
        !has_relevant_file_evidence(
            ToolLoopIntent::FlowTrace,
            "Trace how logging works.",
            &[search.clone(), one_read.clone()]
        ),
        "single-file evidence should not satisfy flow tracing"
    );

    let second_read = ToolResult {
        tool_name: "read_file".to_string(),
        argument: "src/logging.rs".to_string(),
        output: "File: src/logging.rs\nLines: 4\n\n```\npub fn init_logging() {\n    configure_sink();\n}\n```"
            .to_string(),
    };
    assert!(has_relevant_file_evidence(
        ToolLoopIntent::FlowTrace,
        "Trace how logging works.",
        &[search, one_read, second_read]
    ));
}

#[test]
fn declaration_extraction_stops_at_brace_boundary() {
    let content = "pub struct Config {\n    pub name: String,\n}\n\npub fn other() {}\n";

    let declarations = super::parse::declaration_lines_with_numbers(content, 10);
    let has_config = declarations
        .iter()
        .any(|(_, line)| line.contains("pub struct Config"));
    let has_other = declarations
        .iter()
        .any(|(_, line)| line.contains("pub fn other"));
    assert!(
        has_config && has_other,
        "declarations should extract both top-level items"
    );
}

#[test]
fn path_normalization_handles_line_numbers() {
    let result = ToolResult {
        tool_name: "read_file".to_string(),
        argument: "src/main.rs:42".to_string(),
        output: "File: src/main.rs\nLines: 1\n\n```\nfn main() {}\n```".to_string(),
    };
    let paths = observed_read_paths(&[result]);
    assert!(
        paths.contains("src/main.rs"),
        "should normalize path:line to path only"
    );
}

#[test]
fn all_candidates_fully_read_detects_incomplete() {
    let search = ToolResult {
        tool_name: "search".to_string(),
        argument: "my_func".to_string(),
        output: "src/a.rs:\n  1: fn my_func()\n\nsrc/b.rs:\n  2: fn my_func()\n".to_string(),
    };
    let read_a = ToolResult {
        tool_name: "read_file".to_string(),
        argument: "src/a.rs".to_string(),
        output: "File: src/a.rs\nLines: 1\n\n```\nfn my_func() {}\n```".to_string(),
    };

    let candidate_paths = all_search_candidate_paths(
        ToolLoopIntent::CodeNavigation,
        "where is my_func defined",
        &[search.clone(), read_a.clone()],
    );
    assert!(candidate_paths.len() == 2, "should find 2 candidate files");

    let fully_read = all_candidates_fully_read(
        ToolLoopIntent::CodeNavigation,
        "where is my_func defined",
        &[search, read_a],
    );
    assert!(
        !fully_read,
        "should not be fully read when only 1 of 2 candidates read"
    );
}

#[test]
fn session_restore_trace_rejects_message_count_detail() {
    let search = ToolResult {
        tool_name: "search".to_string(),
        argument: "session restore".to_string(),
        output: "src/session/mod.rs:\n  1: pub fn load_most_recent()\nsrc/runtime/core.rs:\n  10: pub fn restore()\n".to_string(),
    };
    let read_session = ToolResult {
        tool_name: "read_file".to_string(),
        argument: "src/session/mod.rs".to_string(),
        output: "File: src/session/mod.rs\nLines: 5\n\n```\npub fn load_most_recent() {\n    store.load_most_recent()\n}\n```".to_string(),
    };
    let read_runtime = ToolResult {
        tool_name: "read_file".to_string(),
        argument: "src/runtime/core.rs".to_string(),
        output: "File: src/runtime/core.rs\nLines: 10\n\n```\npub fn restore() {\n    match store.load_most_recent() {\n        Some(summary) => load_session_by_id(summary.id),\n        None => {\n            Ok(None)\n        }\n    }\n}\n```".to_string(),
    };

    let outcome = investigation_outcome(
        ToolLoopIntent::FlowTrace,
        "Explain how session restore works",
        None,
        &[search, read_session, read_runtime],
    );

    let mut has_forbidden = false;
    if let InvestigationOutcome::Ready { evidence, .. } = &outcome {
        if let StructuredEvidence::FlowTrace(fte) = evidence {
            let text = fte.steps().iter().map(|s| s.path()).collect::<String>();
            has_forbidden = text.contains("message_count") || text.contains("log_messages");
        }
    }

    assert!(
        !has_forbidden,
        "flow trace should reject answers with forbidden message-count detail"
    );
}
