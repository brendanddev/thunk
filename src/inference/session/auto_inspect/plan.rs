use std::path::Path;

use super::intent::extract_auto_inspect_query;
use super::types::{AutoInspectBudget, AutoInspectIntent, AutoInspectPlan, AutoInspectStep};

fn push_read_step(steps: &mut Vec<AutoInspectStep>, project_root: &Path, rel: &str) {
    if project_root.join(rel).is_file() {
        steps.push(AutoInspectStep {
            label: format!("Read {rel}"),
            tool_name: "read_file",
            argument: rel.to_string(),
        });
    }
}

pub(crate) fn plan_auto_inspection(
    intent: AutoInspectIntent,
    prompt: &str,
    project_root: &Path,
) -> AutoInspectPlan {
    let mut steps = vec![AutoInspectStep {
        label: "List .".to_string(),
        tool_name: "list_dir",
        argument: ".".to_string(),
    }];

    match intent {
        AutoInspectIntent::RepoOverview => {
            if project_root.join("src").is_dir() {
                steps.push(AutoInspectStep {
                    label: "List src/".to_string(),
                    tool_name: "list_dir",
                    argument: "src".to_string(),
                });
            }
            push_read_step(&mut steps, project_root, "README.md");
            push_read_step(&mut steps, project_root, "Cargo.toml");
            if project_root.join("src/main.rs").is_file() {
                push_read_step(&mut steps, project_root, "src/main.rs");
            } else {
                push_read_step(&mut steps, project_root, "src/lib.rs");
            }

            AutoInspectPlan {
                intent,
                thinking: "Thinking: exploring the repo structure and key project docs.",
                status_label: "inspecting repo...",
                context_label: "this repo summary request",
                query: None,
                steps,
            }
        }
        AutoInspectIntent::DirectoryOverview => {
            push_read_step(&mut steps, project_root, "README.md");
            for manifest in ["Cargo.toml", "package.json", "pyproject.toml", "go.mod"] {
                if project_root.join(manifest).is_file() {
                    push_read_step(&mut steps, project_root, manifest);
                    break;
                }
            }

            AutoInspectPlan {
                intent,
                thinking: "Thinking: checking the current directory and its key files.",
                status_label: "inspecting directory...",
                context_label: "this directory summary request",
                query: None,
                steps,
            }
        }
        AutoInspectIntent::WhereIsImplementation => {
            let query = extract_auto_inspect_query(prompt, intent);
            let mut steps = Vec::new();
            if let Some(ref query) = query {
                steps.push(AutoInspectStep {
                    label: format!("Search {query}"),
                    tool_name: "search",
                    argument: query.clone(),
                });
            }
            AutoInspectPlan {
                intent,
                thinking: "Thinking: locating the most likely implementation files.",
                status_label: "locating implementation...",
                context_label: "this implementation lookup request",
                query,
                steps,
            }
        }
        AutoInspectIntent::FeatureTrace => {
            let query = extract_auto_inspect_query(prompt, intent);
            let mut steps = Vec::new();
            if let Some(ref query) = query {
                steps.push(AutoInspectStep {
                    label: format!("Search {query}"),
                    tool_name: "search",
                    argument: query.clone(),
                });
            }
            AutoInspectPlan {
                intent,
                thinking: "Thinking: tracing the main code path for this feature.",
                status_label: "tracing feature...",
                context_label: "this feature trace request",
                query,
                steps,
            }
        }
        AutoInspectIntent::ConfigLocate => {
            let query = extract_auto_inspect_query(prompt, intent);
            let mut steps = Vec::new();
            if let Some(ref query) = query {
                steps.push(AutoInspectStep {
                    label: format!("Search {query}"),
                    tool_name: "search",
                    argument: query.clone(),
                });
            }
            AutoInspectPlan {
                intent,
                thinking: "Thinking: checking the files that configure this behavior.",
                status_label: "locating configuration...",
                context_label: "this configuration lookup request",
                query,
                steps,
            }
        }
    }
}

pub(crate) fn auto_inspection_budget(
    intent: AutoInspectIntent,
    backend_name: &str,
    eco_enabled: bool,
) -> AutoInspectBudget {
    let constrained = backend_name.contains("llama.cpp");
    match (intent, constrained, eco_enabled) {
        (AutoInspectIntent::RepoOverview, true, true) => AutoInspectBudget {
            total_chars: 700,
            top_level_entries: 6,
            code_entries: 6,
            readme_chars: 120,
            manifest_chars: 160,
            entrypoint_chars: 160,
            search_files: 0,
            read_files: 0,
            key_hits_per_file: 0,
            workflow_summary_chars: 0,
        },
        (AutoInspectIntent::RepoOverview, true, false) => AutoInspectBudget {
            total_chars: 1000,
            top_level_entries: 8,
            code_entries: 8,
            readme_chars: 170,
            manifest_chars: 220,
            entrypoint_chars: 220,
            search_files: 0,
            read_files: 0,
            key_hits_per_file: 0,
            workflow_summary_chars: 0,
        },
        (AutoInspectIntent::RepoOverview, false, true) => AutoInspectBudget {
            total_chars: 1200,
            top_level_entries: 8,
            code_entries: 8,
            readme_chars: 180,
            manifest_chars: 240,
            entrypoint_chars: 240,
            search_files: 0,
            read_files: 0,
            key_hits_per_file: 0,
            workflow_summary_chars: 0,
        },
        (AutoInspectIntent::RepoOverview, false, false) => AutoInspectBudget {
            total_chars: 1700,
            top_level_entries: 10,
            code_entries: 10,
            readme_chars: 260,
            manifest_chars: 320,
            entrypoint_chars: 320,
            search_files: 0,
            read_files: 0,
            key_hits_per_file: 0,
            workflow_summary_chars: 0,
        },
        (AutoInspectIntent::DirectoryOverview, true, true) => AutoInspectBudget {
            total_chars: 550,
            top_level_entries: 6,
            code_entries: 0,
            readme_chars: 120,
            manifest_chars: 160,
            entrypoint_chars: 0,
            search_files: 0,
            read_files: 0,
            key_hits_per_file: 0,
            workflow_summary_chars: 0,
        },
        (AutoInspectIntent::DirectoryOverview, true, false) => AutoInspectBudget {
            total_chars: 800,
            top_level_entries: 8,
            code_entries: 0,
            readme_chars: 160,
            manifest_chars: 220,
            entrypoint_chars: 0,
            search_files: 0,
            read_files: 0,
            key_hits_per_file: 0,
            workflow_summary_chars: 0,
        },
        (AutoInspectIntent::DirectoryOverview, false, true) => AutoInspectBudget {
            total_chars: 950,
            top_level_entries: 8,
            code_entries: 0,
            readme_chars: 180,
            manifest_chars: 240,
            entrypoint_chars: 0,
            search_files: 0,
            read_files: 0,
            key_hits_per_file: 0,
            workflow_summary_chars: 0,
        },
        (AutoInspectIntent::DirectoryOverview, false, false) => AutoInspectBudget {
            total_chars: 1300,
            top_level_entries: 10,
            code_entries: 0,
            readme_chars: 240,
            manifest_chars: 320,
            entrypoint_chars: 0,
            search_files: 0,
            read_files: 0,
            key_hits_per_file: 0,
            workflow_summary_chars: 0,
        },
        (AutoInspectIntent::WhereIsImplementation, true, true)
        | (AutoInspectIntent::FeatureTrace, true, true)
        | (AutoInspectIntent::ConfigLocate, true, true) => AutoInspectBudget {
            total_chars: 650,
            top_level_entries: 0,
            code_entries: 0,
            readme_chars: 0,
            manifest_chars: 120,
            entrypoint_chars: 0,
            search_files: 3,
            read_files: 2,
            key_hits_per_file: 2,
            workflow_summary_chars: 140,
        },
        (AutoInspectIntent::WhereIsImplementation, true, false)
        | (AutoInspectIntent::FeatureTrace, true, false)
        | (AutoInspectIntent::ConfigLocate, true, false) => AutoInspectBudget {
            total_chars: 900,
            top_level_entries: 0,
            code_entries: 0,
            readme_chars: 0,
            manifest_chars: 160,
            entrypoint_chars: 0,
            search_files: 3,
            read_files: 2,
            key_hits_per_file: 2,
            workflow_summary_chars: 180,
        },
        (AutoInspectIntent::WhereIsImplementation, false, true)
        | (AutoInspectIntent::FeatureTrace, false, true)
        | (AutoInspectIntent::ConfigLocate, false, true) => AutoInspectBudget {
            total_chars: 1100,
            top_level_entries: 0,
            code_entries: 0,
            readme_chars: 0,
            manifest_chars: 180,
            entrypoint_chars: 0,
            search_files: 4,
            read_files: 2,
            key_hits_per_file: 2,
            workflow_summary_chars: 220,
        },
        (AutoInspectIntent::WhereIsImplementation, false, false)
        | (AutoInspectIntent::FeatureTrace, false, false)
        | (AutoInspectIntent::ConfigLocate, false, false) => AutoInspectBudget {
            total_chars: 1400,
            top_level_entries: 0,
            code_entries: 0,
            readme_chars: 0,
            manifest_chars: 220,
            entrypoint_chars: 0,
            search_files: 5,
            read_files: 3,
            key_hits_per_file: 2,
            workflow_summary_chars: 260,
        },
    }
}
