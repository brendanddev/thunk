use super::*;
use crate::inference::tool_loop::ToolLoopIntent;
use crate::inference::Message;
use crate::tools::ToolRegistry;
use crate::tools::ToolResult;

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
