use super::*;
use crate::inference::session::memory::suppress_retrieval_for_auto_inspection;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_project_root(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    std::env::temp_dir().join(format!("params-auto-inspect-{label}-{nonce}"))
}

#[test]
fn detect_auto_inspect_intent_matches_repo_prompts() {
    assert_eq!(
        detect_auto_inspect_intent("What's in this repo?"),
        Some(AutoInspectIntent::RepoOverview)
    );
    assert_eq!(
        detect_auto_inspect_intent("Summarize this codebase"),
        Some(AutoInspectIntent::RepoOverview)
    );
}

#[test]
fn detect_auto_inspect_intent_matches_directory_prompts() {
    assert_eq!(
        detect_auto_inspect_intent("What's in this directory?"),
        Some(AutoInspectIntent::DirectoryOverview)
    );
    assert_eq!(
        detect_auto_inspect_intent("What's here"),
        Some(AutoInspectIntent::DirectoryOverview)
    );
}

#[test]
fn detect_auto_inspect_intent_matches_workflow_prompts() {
    assert_eq!(
        detect_auto_inspect_intent("Where is cache implemented?"),
        Some(AutoInspectIntent::WhereIsImplementation)
    );
    assert_eq!(
        detect_auto_inspect_intent("Trace how sessions are saved"),
        Some(AutoInspectIntent::FeatureTrace)
    );
    assert_eq!(
        detect_auto_inspect_intent("Where is eco mode configured?"),
        Some(AutoInspectIntent::ConfigLocate)
    );
    assert_eq!(
        detect_auto_inspect_intent("Where is eco mode configged"),
        Some(AutoInspectIntent::ConfigLocate)
    );
}

#[test]
fn detect_auto_inspect_intent_skips_unrelated_prompts() {
    assert_eq!(detect_auto_inspect_intent("Explain the cache"), None);
    assert_eq!(detect_auto_inspect_intent("/read README.md"), None);
}

#[test]
fn repo_auto_inspection_prefers_main_over_lib() {
    let root = temp_project_root("repo-main");
    fs::create_dir_all(root.join("src")).expect("create src");
    fs::write(root.join("README.md"), "# params").expect("write readme");
    fs::write(root.join("Cargo.toml"), "[package]").expect("write cargo");
    fs::write(root.join("src/main.rs"), "fn main() {}").expect("write main");
    fs::write(root.join("src/lib.rs"), "pub fn lib() {}").expect("write lib");

    let plan = plan_auto_inspection(
        AutoInspectIntent::RepoOverview,
        "What's in this repo?",
        &root,
    );
    let labels = plan
        .steps
        .iter()
        .map(|step| step.label.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        labels,
        vec![
            "List .",
            "List src/",
            "Read README.md",
            "Read Cargo.toml",
            "Read src/main.rs"
        ]
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn directory_auto_inspection_is_bounded_and_chooses_single_manifest() {
    let root = temp_project_root("directory");
    fs::create_dir_all(&root).expect("create dir");
    fs::write(root.join("README.md"), "# params").expect("write readme");
    fs::write(root.join("Cargo.toml"), "[package]").expect("write cargo");
    fs::write(root.join("package.json"), "{}").expect("write package");

    let plan = plan_auto_inspection(AutoInspectIntent::DirectoryOverview, "What's here", &root);
    let labels = plan
        .steps
        .iter()
        .map(|step| step.label.as_str())
        .collect::<Vec<_>>();

    assert_eq!(labels, vec!["List .", "Read README.md", "Read Cargo.toml"]);

    let _ = fs::remove_dir_all(root);
}

#[test]
fn workflow_auto_inspection_starts_with_search() {
    let root = temp_project_root("workflow-plan");
    fs::create_dir_all(root.join("src")).expect("create src");

    let where_plan = plan_auto_inspection(
        AutoInspectIntent::WhereIsImplementation,
        "Where is cache implemented?",
        &root,
    );
    let trace_plan = plan_auto_inspection(
        AutoInspectIntent::FeatureTrace,
        "Trace how sessions are saved",
        &root,
    );
    let config_plan = plan_auto_inspection(
        AutoInspectIntent::ConfigLocate,
        "Where is eco mode configured?",
        &root,
    );

    assert_eq!(where_plan.steps[0].tool_name, "search");
    assert_eq!(where_plan.steps[0].argument, "cache");
    assert_eq!(trace_plan.steps[0].tool_name, "search");
    // save_session is a thin wrapper in the ~150 KB unreadable file; the
    // real persistence function save_messages is in the readable session
    // store, so we search for that instead.
    assert_eq!(trace_plan.steps[0].argument, "save_messages");
    assert_eq!(config_plan.steps[0].tool_name, "search");
    assert_eq!(config_plan.steps[0].argument, "eco.enabled");

    let _ = fs::remove_dir_all(root);
}

#[test]
fn workflow_query_extraction_prefers_salient_terms() {
    assert_eq!(
        extract_auto_inspect_query(
            "Where is session restore implemented?",
            AutoInspectIntent::WhereIsImplementation
        )
        .as_deref(),
        Some("load_most_recent")
    );
    assert_eq!(
        extract_auto_inspect_query(
            "Trace how sessions are saved",
            AutoInspectIntent::FeatureTrace
        )
        .as_deref(),
        Some("save_messages")
    );
    assert_eq!(
        extract_auto_inspect_query(
            "Where is eco mode configured?",
            AutoInspectIntent::ConfigLocate
        )
        .as_deref(),
        Some("eco.enabled")
    );
    assert_eq!(
        extract_auto_inspect_query(
            "Where is eco mode configged",
            AutoInspectIntent::ConfigLocate
        )
        .as_deref(),
        Some("eco.enabled")
    );
    assert_eq!(
        extract_auto_inspect_query(
            "Where is memory retrieval implemented?",
            AutoInspectIntent::WhereIsImplementation
        )
        .as_deref(),
        Some("retrieval")
    );
    assert_eq!(
        extract_auto_inspect_query(
            "Where is project indexing implemented?",
            AutoInspectIntent::WhereIsImplementation
        )
        .as_deref(),
        Some("indexing")
    );
}

#[test]
fn parse_search_output_groups_hits_by_file() {
    let hits = parse_search_output(
            "Search results for 'cache' (3 matches):\n\nsrc/main.rs:\n     4: mod cache;\n    18: cache::warm();\n\nsrc/cache/mod.rs:\n     2: pub fn warm() {}\n",
        );

    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].path, "src/main.rs");
    assert_eq!(hits[0].hits.len(), 2);
    assert_eq!(hits[1].path, "src/cache/mod.rs");
}

#[test]
fn config_candidate_ranking_prefers_config_paths() {
    let hits = vec![
        SearchFileHit {
            path: "docs/context/CLAUDE.md".to_string(),
            hits: vec![SearchLineHit {
                line_number: 1,
                line_content: "eco mode documentation".to_string(),
            }],
        },
        SearchFileHit {
            path: "src/config.rs".to_string(),
            hits: vec![SearchLineHit {
                line_number: 22,
                line_content: "pub struct EcoConfig".to_string(),
            }],
        },
    ];

    let ranked = rank_search_files(AutoInspectIntent::ConfigLocate, "eco mode", &hits);
    assert_eq!(ranked[0].path, "src/config.rs");
}

#[test]
fn choose_followup_read_steps_prefers_code_paths_when_available() {
    let root = temp_project_root("workflow-select");
    fs::create_dir_all(root.join("src/session")).expect("create src/session");
    fs::write(
        root.join("src/session/mod.rs"),
        "pub fn load_session() {}\n",
    )
    .expect("write session mod");
    fs::write(root.join("docs.md"), "load_most_recent docs\n").expect("write docs");

    let plan = AutoInspectPlan {
        intent: AutoInspectIntent::WhereIsImplementation,
        thinking: "Thinking: locating the most likely implementation files.",
        status_label: "locating implementation...",
        context_label: "this implementation lookup request",
        query: Some("load_most_recent".to_string()),
        steps: vec![],
    };

    let hits = vec![
        SearchFileHit {
            path: "docs/context/CLAUDE.md".to_string(),
            hits: vec![SearchLineHit {
                line_number: 10,
                line_content: "load_most_recent docs".to_string(),
            }],
        },
        SearchFileHit {
            path: "src/session/mod.rs".to_string(),
            hits: vec![SearchLineHit {
                line_number: 2,
                line_content: "pub fn load_most_recent()".to_string(),
            }],
        },
    ];

    let steps = choose_followup_read_steps(
        &plan,
        &root,
        &hits,
        auto_inspection_budget(
            AutoInspectIntent::WhereIsImplementation,
            "llama.cpp · qwen",
            false,
        ),
    );

    assert_eq!(steps[0].argument, "src/session/mod.rs");
    let _ = fs::remove_dir_all(root);
}

#[test]
fn choose_followup_read_steps_prefers_profile_config_for_eco_mode() {
    let root = temp_project_root("workflow-config-select");
    fs::create_dir_all(root.join("src/config")).expect("create src/config");
    fs::create_dir_all(root.join("src/tui")).expect("create src/tui");
    fs::write(
        root.join("src/config/profile.rs"),
        "if let Some(e) = profile.eco.enabled {\n    base.eco.enabled = e;\n}\n",
    )
    .expect("write profile");
    fs::write(
        root.join("src/config.rs"),
        "pub struct EcoConfig { pub enabled: bool }\n",
    )
    .expect("write config");
    fs::write(
        root.join("src/tui/commands.rs"),
        "state.set_eco_enabled(true);\n",
    )
    .expect("write commands");
    fs::write(root.join("Cargo.toml"), "[package]\nname = \"test\"\n").expect("write cargo");

    let plan = AutoInspectPlan {
        intent: AutoInspectIntent::ConfigLocate,
        thinking: "Thinking: checking the files that configure this behavior.",
        status_label: "locating configuration...",
        context_label: "this configuration lookup request",
        query: Some("eco.enabled".to_string()),
        steps: vec![],
    };

    let hits = vec![
        SearchFileHit {
            path: "src/tui/commands.rs".to_string(),
            hits: vec![SearchLineHit {
                line_number: 1,
                line_content: "state.set_eco_enabled(true);".to_string(),
            }],
        },
        SearchFileHit {
            path: "src/config.rs".to_string(),
            hits: vec![SearchLineHit {
                line_number: 1,
                line_content: "pub struct EcoConfig { pub enabled: bool }".to_string(),
            }],
        },
    ];

    let steps = choose_followup_read_steps(
        &plan,
        &root,
        &hits,
        auto_inspection_budget(AutoInspectIntent::ConfigLocate, "llama.cpp · qwen", false),
    );

    assert_eq!(steps[0].argument, "src/config/profile.rs");
    assert_eq!(steps[1].argument, "src/config.rs");
    let _ = fs::remove_dir_all(root);
}

#[test]
fn auto_inspection_hidden_context_is_compact_and_structural() {
    let plan = AutoInspectPlan {
        intent: AutoInspectIntent::RepoOverview,
        thinking: "Thinking: exploring the repo structure and key project docs.",
        status_label: "inspecting repo...",
        context_label: "this repo summary request",
        query: None,
        steps: vec![],
    };

    let hidden = synthesize_auto_inspection_context(
            &plan,
            &[
                ToolResult {
                    tool_name: "list_dir".to_string(),
                    argument: ".".to_string(),
                    output: "Directory: .\n\nsrc/\ndocs/\nREADME.md\nCargo.toml".to_string(),
                },
                ToolResult {
                    tool_name: "list_dir".to_string(),
                    argument: "src".to_string(),
                    output: "Directory: src\n\ncache/\ninference/\ntui/\nmain.rs".to_string(),
                },
                ToolResult {
                    tool_name: "read_file".to_string(),
                    argument: "Cargo.toml".to_string(),
                    output: "File: Cargo.toml\nLines: 8\n\n```\n[package]\nname = \"params-cli\"\ndescription = \"Personal AI coding assistant CLI\"\n\n[dependencies]\ncrossterm = \"0.28\"\nllama-cpp-2 = \"0.1\"\nserde = \"1\"\n```\n".to_string(),
                },
                ToolResult {
                    tool_name: "read_file".to_string(),
                    argument: "src/main.rs".to_string(),
                    output: "File: src/main.rs\nLines: 5\n\n```\nmod cache;\nmod inference;\nmod tui;\n\nfn main() {}\n```\n".to_string(),
                },
            ],
            auto_inspection_budget(AutoInspectIntent::RepoOverview, "llama.cpp · test", false),
        )
        .expect("hidden context");

    assert!(hidden.starts_with("Automatic inspection context for this repo summary request:"));
    assert!(hidden.contains("Repo type: Rust project"));
    assert!(hidden.contains("Code areas:"));
    assert!(hidden.contains("Manifest:"));
    assert!(hidden.contains("Entrypoint `src/main.rs`; modules: cache, inference, tui"));
    assert!(!hidden.contains("--- list_dir(.) ---"));
    assert!(!hidden.contains("Tool results:"));
    assert!(hidden.chars().count() <= 1000);
}

#[test]
fn auto_inspection_budget_is_tighter_for_llama_cpp() {
    let local = auto_inspection_budget(AutoInspectIntent::RepoOverview, "llama.cpp · qwen", false);
    let cloud = auto_inspection_budget(
        AutoInspectIntent::RepoOverview,
        "openai_compat · gpt",
        false,
    );

    assert!(local.total_chars < cloud.total_chars);
    assert!(local.readme_chars < cloud.readme_chars);
    assert!(local.entrypoint_chars < cloud.entrypoint_chars);
}

#[test]
fn auto_inspection_prefers_code_structure_over_large_readme_excerpt() {
    let plan = AutoInspectPlan {
        intent: AutoInspectIntent::RepoOverview,
        thinking: "Thinking: exploring the repo structure and key project docs.",
        status_label: "inspecting repo...",
        context_label: "this repo summary request",
        query: None,
        steps: vec![],
    };
    let long_readme = "README intro ".repeat(120);

    let hidden = synthesize_auto_inspection_context(
        &plan,
        &[
            ToolResult {
                tool_name: "list_dir".to_string(),
                argument: ".".to_string(),
                output: "Directory: .\n\nsrc/\nREADME.md\nCargo.toml".to_string(),
            },
            ToolResult {
                tool_name: "list_dir".to_string(),
                argument: "src".to_string(),
                output:
                    "Directory: src\n\ncache/\nconfig/\ninference/\nsession/\ntools/\ntui/\nmain.rs"
                        .to_string(),
            },
            ToolResult {
                tool_name: "read_file".to_string(),
                argument: "README.md".to_string(),
                output: format!("File: README.md\nLines: 40\n\n```\n{}\n```\n", long_readme),
            },
            ToolResult {
                tool_name: "list_dir".to_string(),
                argument: ".".to_string(),
                output: "Directory: .\n\nsrc/\nREADME.md\nCargo.toml".to_string(),
            },
        ],
        auto_inspection_budget(AutoInspectIntent::RepoOverview, "llama.cpp · qwen", false),
    )
    .expect("hidden context");

    assert!(hidden.contains("Code areas:"));
    assert!(hidden.contains("`cache/`"));
    assert!(hidden.contains("`inference/`"));
    assert!(hidden.chars().count() <= 1000);
}

#[test]
fn workflow_hidden_context_is_compact_and_query_driven() {
    let plan = AutoInspectPlan {
        intent: AutoInspectIntent::WhereIsImplementation,
        thinking: "Thinking: locating the most likely implementation files.",
        status_label: "locating implementation...",
        context_label: "this implementation lookup request",
        query: Some("cache".to_string()),
        steps: vec![],
    };

    let hidden = synthesize_auto_inspection_context(
            &plan,
            &[
                ToolResult {
                    tool_name: "search".to_string(),
                    argument: "cache".to_string(),
                    output: "Search results for 'cache' (3 matches):\n\nsrc/main.rs:\n     4: mod cache;\n    18: cache::warm();\n\nsrc/cache/mod.rs:\n     2: pub fn warm() {}\n".to_string(),
                },
                ToolResult {
                    tool_name: "read_file".to_string(),
                    argument: "src/cache/mod.rs".to_string(),
                    output: "File: src/cache/mod.rs\nLines: 4\n\n```\npub fn warm() {}\npub fn clear() {}\n```\n".to_string(),
                },
            ],
            auto_inspection_budget(
                AutoInspectIntent::WhereIsImplementation,
                "llama.cpp · qwen",
                false,
            ),
        )
        .expect("workflow context");

    assert!(
        hidden.starts_with("Automatic inspection context for this implementation lookup request:")
    );
    assert!(hidden.contains("Instruction: answer directly from this evidence."));
    assert!(hidden.contains("Prefer exact inspected-file evidence over supporting search hits."));
    assert!(hidden.contains("Do not emit tool calls or fenced code blocks."));
    assert!(hidden.contains("Query: cache"));
    assert!(hidden.contains("Likely files:"));
    assert!(hidden.contains("Implementation hints:"));
    assert!(hidden.contains("declarations: 1 `pub fn warm() {}`"));
    assert!(!hidden.contains("Supporting search hits:"));
    assert!(!hidden.contains("Tool results:"));
    assert!(hidden.chars().count() <= 900);
}

#[test]
fn workflow_hidden_context_prefers_inspected_file_hits_over_doc_search_hits() {
    let plan = AutoInspectPlan {
        intent: AutoInspectIntent::WhereIsImplementation,
        thinking: "Thinking: locating the most likely implementation files.",
        status_label: "locating implementation...",
        context_label: "this implementation lookup request",
        query: Some("load_most_recent".to_string()),
        steps: vec![],
    };

    let hidden = synthesize_auto_inspection_context(
            &plan,
            &[
                ToolResult {
                    tool_name: "search".to_string(),
                    argument: "load_most_recent".to_string(),
                    output: "Search results for 'load_most_recent' (4 matches):\n\nsrc/session/mod.rs:\n   272: pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n   844: let loaded = store.load_most_recent().unwrap().unwrap();\n\ndocs/context/PLANS.md:\n  3189: load_most_recent overview\n".to_string(),
                },
                ToolResult {
                    tool_name: "read_file".to_string(),
                    argument: "src/session/mod.rs".to_string(),
                    output: "File: src/session/mod.rs\nLines: 8\n\n```\npub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    let Some(summary) = self.list_sessions()?.into_iter().next() else {\n        return Ok(None);\n    };\n    self.load_session_by_id(&summary.id)\n}\n\npub fn load_session(&self, selector: &str) -> Result<SavedSession> {\n```\n".to_string(),
                },
            ],
            auto_inspection_budget(
                AutoInspectIntent::WhereIsImplementation,
                "llama.cpp · qwen",
                false,
            ),
        )
        .expect("workflow context");

    assert!(hidden.contains("Likely files: `src/session/mod.rs`"));
    assert!(hidden.contains("Primary definition: src/session/mod.rs:1 `pub fn load_most_recent"));
    assert!(!hidden.contains("Supporting search hits:"));
    assert!(!hidden.contains("src/session/mod.rs:844"));
    assert!(!hidden.contains("docs/context/PLANS.md:3189"));
}

#[test]
fn implementation_summary_prefers_definition_matches_over_use_sites() {
    let summary = summarize_workflow_read(
            "src/session/mod.rs",
            "fn unrelated() {}\n\npub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n\nfn later() {\n    let loaded = store.load_most_recent().unwrap().unwrap();\n}\n",
            "load_most_recent",
            AutoInspectIntent::WhereIsImplementation,
            260,
        )
        .expect("summary");

    assert!(summary.contains("exact lines: 3 `pub fn load_most_recent"));
    assert!(!summary.contains("7 `let loaded = store.load_most_recent().unwrap().unwrap();`"));
}

#[test]
fn primary_definition_location_uses_definition_line() {
    let location = primary_definition_location(
            "src/session/mod.rs",
            "pub use self::session::load_most_recent;\n\npub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
            "load_most_recent",
            72,
        )
        .expect("location");

    assert_eq!(
        location,
        "src/session/mod.rs:3 `pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {`"
    );
}

#[test]
fn implementation_workflows_suppress_general_retrieval_and_instruct_definition_only() {
    assert!(suppress_retrieval_for_auto_inspection(
        AutoInspectIntent::WhereIsImplementation
    ));
    assert!(suppress_retrieval_for_auto_inspection(
        AutoInspectIntent::FeatureTrace
    ));
    assert!(suppress_retrieval_for_auto_inspection(
        AutoInspectIntent::ConfigLocate
    ));
    assert!(!suppress_retrieval_for_auto_inspection(
        AutoInspectIntent::RepoOverview
    ));

    let plan = AutoInspectPlan {
        intent: AutoInspectIntent::WhereIsImplementation,
        thinking: "Thinking: locating the most likely implementation files.",
        status_label: "locating implementation...",
        context_label: "this implementation lookup request",
        query: Some("load_most_recent".to_string()),
        steps: vec![],
    };

    let hidden = synthesize_auto_inspection_context(
            &plan,
            &[ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/session/mod.rs".to_string(),
                output: "File: src/session/mod.rs\nLines: 5\n\n```\npub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n```\n".to_string(),
            }],
            auto_inspection_budget(
                AutoInspectIntent::WhereIsImplementation,
                "llama.cpp · qwen",
                false,
            ),
        )
        .expect("workflow context");

    assert!(hidden.contains("Report definition or implementation locations only"));
    assert!(hidden.contains("omit usage lines"));
}

// --- Tests that reproduce the actual runtime failure modes ---

#[test]
fn summarize_workflow_read_omits_unrelated_declarations_when_exact_match_found() {
    // Reproduces the "line 28 / line 34 / struct noise" failure: the old
    // code always appended the first-N declaration lines from the file even
    // when an exact definition match was already found. Those small line
    // numbers (top-of-file structs) were cited by the model instead of the
    // correct line.
    let content = "pub struct Other {}\n\npub struct AnotherThing {}\n\n\
                       pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n";
    let summary = summarize_workflow_read(
        "src/session/mod.rs",
        content,
        "load_most_recent",
        AutoInspectIntent::WhereIsImplementation,
        400,
    )
    .expect("summary");

    assert!(summary.contains("exact lines: 5 `pub fn load_most_recent"));
    // The unrelated structs at lines 1 and 3 must NOT appear.
    assert!(
        !summary.contains("declarations:"),
        "declarations section should be omitted when exact match exists"
    );
    assert!(!summary.contains("1 `pub struct Other"));
    assert!(!summary.contains("3 `pub struct AnotherThing"));
}

#[test]
fn primary_definition_location_returns_none_for_file_that_only_calls_the_function() {
    // Reproduces the "line 12 / wrong anchor" failure: the old fallback in
    // primary_definition_location would pick the first fn/struct in the file
    // even when the file never *defines* the queried symbol, producing a
    // completely wrong "Primary definition" anchor.
    let content = "fn call_something() {\n    store.load_most_recent().unwrap();\n}\n\
                       fn other() {\n    let x = load_most_recent();\n}\n";
    let location =
        primary_definition_location("src/inference/session.rs", content, "load_most_recent", 72);
    // Should return None — this file calls but does not define the function.
    assert!(
        location.is_none(),
        "expected None for a file that only calls the function, got: {location:?}"
    );
}

#[test]
fn synthesize_context_deduplicates_search_and_read_paths_when_formats_match() {
    // Reproduces the absolute-vs-relative path mismatch: SearchCode previously
    // produced absolute paths while ReadFile produced relative paths, causing
    // every file to appear twice in Likely files and supporting_hits to be
    // always empty. After the fix, both use relative paths and this test
    // verifies they are deduplicated correctly.
    let plan = AutoInspectPlan {
        intent: AutoInspectIntent::WhereIsImplementation,
        thinking: "t",
        status_label: "s",
        context_label: "this implementation lookup request",
        query: Some("load_most_recent".to_string()),
        steps: vec![],
    };

    // Both the search output and the read output use the same relative path.
    let results = vec![
            ToolResult {
                tool_name: "search".to_string(),
                argument: "load_most_recent".to_string(),
                output: "Search results for 'load_most_recent' (1 match):\n\nsrc/session/mod.rs:\n   272: pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n".to_string(),
            },
            ToolResult {
                tool_name: "read_file".to_string(),
                argument: "src/session/mod.rs".to_string(),
                output: "File: src/session/mod.rs\nLines: 3\n\n```\npub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n```\n".to_string(),
            },
        ];

    let hidden = synthesize_auto_inspection_context(
        &plan,
        &results,
        auto_inspection_budget(AutoInspectIntent::WhereIsImplementation, "openai", false),
    )
    .expect("context");

    // The file should appear exactly once in Likely files — no duplicate
    // caused by one absolute and one relative representation.
    let likely_start = hidden.find("Likely files:").expect("Likely files section");
    let likely_end = hidden[likely_start..]
        .find('\n')
        .map(|i| likely_start + i)
        .unwrap_or(hidden.len());
    let likely_line = &hidden[likely_start..likely_end];
    let occurrences = likely_line.matches("src/session/mod.rs").count();
    assert_eq!(
        occurrences, 1,
        "src/session/mod.rs should appear exactly once in Likely files, got: {likely_line}"
    );
}

#[test]
fn primary_definition_location_correct_for_deep_line_number() {
    // Verifies that the correct line number is reported when the target
    // function is deep in the file (e.g. line 272 in the real session store),
    // not confused by unrelated top-of-file structs.
    let mut content = String::new();
    // Pad with 271 blank lines so the function starts at line 272.
    for _ in 0..271 {
        content.push('\n');
    }
    content.push_str("pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n");
    content.push_str("    Ok(None)\n");
    content.push_str("}\n");

    let location =
        primary_definition_location("src/session/mod.rs", &content, "load_most_recent", 72)
            .expect("location");

    assert!(
        location.starts_with("src/session/mod.rs:272 "),
        "expected line 272, got: {location}"
    );
}

#[test]
fn feature_trace_summary_skips_unrelated_declarations_without_matches() {
    let summary = summarize_workflow_read(
        "src/session/mod.rs",
        "pub struct SessionSummary {}\n\npub struct SavedSession {}\n",
        "save_session",
        AutoInspectIntent::FeatureTrace,
        240,
    );

    assert!(summary.is_none());
}

#[test]
fn feature_trace_context_prefers_search_flow_anchors_when_large_file_cannot_be_read() {
    // When the target file is too large to read (e.g., src/inference/session.rs),
    // the workflow falls back to search-only evidence.  The synthesizer must
    // emit the "search anchors only" evidence-quality warning and the
    // anti-fabrication FeatureTrace instruction so the model does not invent
    // code bodies it never read.
    let plan = AutoInspectPlan {
        intent: AutoInspectIntent::FeatureTrace,
        thinking: "Thinking: tracing the main code path for this feature.",
        status_label: "tracing feature...",
        context_label: "this feature trace request",
        query: Some("save_messages".to_string()),
        steps: vec![],
    };

    // Use a non-constrained backend so the evidence warning is not truncated
    // by the tighter llama.cpp budget (900 chars).
    let hidden = synthesize_auto_inspection_context(
            &plan,
            &[ToolResult {
                tool_name: "search".to_string(),
                argument: "save_messages".to_string(),
                output: "Search results for 'save_messages' (4 matches):\n\nsrc/session/mod.rs:\n  224: pub fn save_messages(\n  241: save_messages(\n  260: save_messages(&conn, session_id, messages);\n  3635: assert_eq!(trace_plan.steps[0].argument, \"save_messages\");\n".to_string(),
            }],
            auto_inspection_budget(AutoInspectIntent::FeatureTrace, "openai_compat · gpt-4", false),
        )
        .expect("context");

    assert!(hidden.contains("Primary flow anchors:"));
    assert!(hidden.contains("src/session/mod.rs:224 `pub fn save_messages(`"));
    // assert_eq! call sites must be filtered out by is_feature_trace_anchor_line
    assert!(!hidden.contains("assert_eq!(trace_plan.steps[0].argument"));
    // Anti-fabrication instruction must be present
    assert!(hidden.contains("Do not invent function bodies, placeholder snippets"));
    // Evidence-quality warning must appear when no file content was read
    assert!(hidden.contains("Evidence: search anchors only"));
}

#[test]
fn feature_trace_context_grounded_when_file_is_readable() {
    // Happy path: the readable src/session/mod.rs contains save_messages,
    // so the synthesizer should produce "Flow hints" from actual file
    // content and NOT emit the "search anchors only" evidence warning.
    let plan = AutoInspectPlan {
        intent: AutoInspectIntent::FeatureTrace,
        thinking: "Thinking: tracing session save.",
        status_label: "tracing feature...",
        context_label: "this feature trace request",
        query: Some("save_messages".to_string()),
        steps: vec![],
    };

    let file_content = "pub fn save_messages(\n    conn: &Connection,\n    session_id: i64,\n    messages: &[Message],\n) -> Result<()> {\n    for msg in messages {\n        conn.execute(INSERT, params![session_id, msg.role, msg.content])?;\n    }\n    Ok(())\n}";

    let hidden = synthesize_auto_inspection_context(
            &plan,
            &[
                ToolResult {
                    tool_name: "search".to_string(),
                    argument: "save_messages".to_string(),
                    output: "Search results for 'save_messages' (2 matches):\n\nsrc/session/mod.rs:\n  224: pub fn save_messages(\n  241: save_messages(&conn, session_id, messages);\n".to_string(),
                },
                ToolResult {
                    tool_name: "read_file".to_string(),
                    argument: "src/session/mod.rs".to_string(),
                    output: format!("File: src/session/mod.rs\nLines: 10\n\n```\n{file_content}\n```"),
                },
            ],
            auto_inspection_budget(AutoInspectIntent::FeatureTrace, "llama.cpp · qwen", false),
        )
        .expect("context");

    // File content was read, so flow hints must appear
    assert!(hidden.contains("Flow hints"));
    assert!(hidden.contains("src/session/mod.rs"));
    // The "search anchors only" warning must NOT appear — we have real evidence
    assert!(!hidden.contains("Evidence: search anchors only"));
    // Anti-fabrication instruction must still be present
    assert!(hidden.contains("Do not invent function bodies, placeholder snippets"));
}

#[test]
fn feature_trace_summary_ignores_test_module_hits_in_read_file() {
    let content = "\
pub fn save_messages(\n\
    &self,\n\
    session_id: &str,\n\
) -> Result<SessionSummary> {\n\
    Ok(todo!())\n\
}\n\
\n\
#[cfg(test)]\n\
mod tests {\n\
    #[test]\n\
    fn saves_sessions() {\n\
        store.save_messages(\"id\", &[], \"llama.cpp\").unwrap();\n\
    }\n\
}\n";

    let summary = summarize_workflow_read(
        "src/session/mod.rs",
        content,
        "save_messages",
        AutoInspectIntent::FeatureTrace,
        260,
    )
    .expect("summary");

    assert!(summary.contains("flow lines: 1 `pub fn save_messages(`"));
    assert!(!summary.contains("store.save_messages"));
}

#[test]
fn feature_trace_context_ignores_test_region_search_hits_for_read_files() {
    let plan = AutoInspectPlan {
        intent: AutoInspectIntent::FeatureTrace,
        thinking: "Thinking: tracing the main code path for this feature.",
        status_label: "tracing feature...",
        context_label: "this feature trace request",
        query: Some("save_messages".to_string()),
        steps: vec![],
    };

    let file_content = "\
pub fn save_messages(\n\
    &self,\n\
    session_id: &str,\n\
) -> Result<SessionSummary> {\n\
    Ok(todo!())\n\
}\n\
\n\
#[cfg(test)]\n\
mod tests {\n\
    #[test]\n\
    fn saves_sessions() {\n\
        store.save_messages(\"id\", &[], \"llama.cpp\").unwrap();\n\
    }\n\
}\n";

    let hidden = synthesize_auto_inspection_context(
            &plan,
            &[
                ToolResult {
                    tool_name: "search".to_string(),
                    argument: "save_messages".to_string(),
                    output: "Search results for 'save_messages' (3 matches):\n\nsrc/session/mod.rs:\n  1: pub fn save_messages(\n  11: store.save_messages(\"id\", &[], \"llama.cpp\").unwrap();\n\nsrc/inference/session.rs:\n  1975: match s.save_messages(&current.id, messages, backend_name) {\n".to_string(),
                },
                ToolResult {
                    tool_name: "read_file".to_string(),
                    argument: "src/session/mod.rs".to_string(),
                    output: format!("File: src/session/mod.rs\nLines: 12\n\n```\n{file_content}\n```"),
                },
            ],
            auto_inspection_budget(AutoInspectIntent::FeatureTrace, "openai_compat · gpt-4", false),
        )
        .expect("context");

    assert!(hidden.contains("Primary flow anchors: src/session/mod.rs:1 `pub fn save_messages(`"));
    assert!(hidden.contains("src/inference/session.rs:1975"));
    assert!(!hidden.contains("src/session/mod.rs:11"));
}

#[test]
fn config_locate_context_surfaces_exact_merge_lines() {
    let plan = AutoInspectPlan {
        intent: AutoInspectIntent::ConfigLocate,
        thinking: "Thinking: checking the files that configure this behavior.",
        status_label: "locating configuration...",
        context_label: "this configuration lookup request",
        query: Some("eco.enabled".to_string()),
        steps: vec![],
    };

    let content = "\
pub struct ProjectEcoProfile {\n\
    pub enabled: Option<bool>,\n\
}\n\
\n\
pub fn apply_profile(mut base: Config, profile: ProjectProfile) -> Config {\n\
    if let Some(e) = profile.eco.enabled {\n\
        base.eco.enabled = e;\n\
    }\n\
    base\n\
}\n";

    let hidden = synthesize_auto_inspection_context(
        &plan,
        &[ToolResult {
            tool_name: "read_file".to_string(),
            argument: "src/config/profile.rs".to_string(),
            output: format!("File: src/config/profile.rs\nLines: 9\n\n```\n{content}\n```"),
        }],
        auto_inspection_budget(
            AutoInspectIntent::ConfigLocate,
            "openai_compat · gpt-4",
            false,
        ),
    )
    .expect("context");

    assert!(hidden.contains(
        "Primary config lines: src/config/profile.rs:6 `if let Some(e) = profile.eco.enabled {`"
    ));
    assert!(hidden.contains("src/config/profile.rs:7 `base.eco.enabled = e;`"));
}

#[test]
fn choose_followup_read_steps_skips_oversized_auto_inspection_files() {
    let root = temp_project_root("workflow-large-skip");
    fs::create_dir_all(root.join("src")).expect("create src");
    fs::write(root.join("src/small.rs"), "fn helper() {}\n").expect("write small file");
    let oversized = "a".repeat(100_500);
    fs::write(root.join("src/large.rs"), oversized).expect("write large file");

    let plan = AutoInspectPlan {
        intent: AutoInspectIntent::FeatureTrace,
        thinking: "Thinking: tracing the main code path for this feature.",
        status_label: "tracing feature...",
        context_label: "this feature trace request",
        query: Some("save_session".to_string()),
        steps: vec![],
    };

    let hits = vec![
        SearchFileHit {
            path: "src/large.rs".to_string(),
            hits: vec![SearchLineHit {
                line_number: 1,
                line_content: "fn save_session(".to_string(),
            }],
        },
        SearchFileHit {
            path: "src/small.rs".to_string(),
            hits: vec![SearchLineHit {
                line_number: 1,
                line_content: "fn save_session(".to_string(),
            }],
        },
    ];

    let steps = choose_followup_read_steps(
        &plan,
        &root,
        &hits,
        auto_inspection_budget(AutoInspectIntent::FeatureTrace, "llama.cpp · qwen", false),
    );

    assert!(steps.iter().all(|step| step.argument != "src/large.rs"));
    assert!(steps.iter().any(|step| step.argument == "src/small.rs"));

    let _ = fs::remove_dir_all(root);
}
