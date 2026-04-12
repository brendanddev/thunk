mod display;
mod parse;

use std::sync::mpsc;
use std::thread;

use tracing::info;

use super::format::sanitize_for_display;
use super::state::AppState;
use crate::commands::{
    resolve_builtin_command, BuiltinKind, CommandRegistry, CustomCommand, CustomCommandBody,
    CustomCommandStep,
};
use crate::events::{ProgressStatus, ProgressTrace};
use crate::inference::{InjectedContextMetadata, SessionCommand};
use display::{
    custom_help_text, format_custom_commands_list, format_display_status, format_memory_facts,
    format_memory_last, format_memory_status, format_transcript_status,
};
use parse::{
    decode_slash_write_content, parse_command_parts, parse_sessions_export_args,
    parse_slash_edit_body,
};

pub(crate) enum SlashJobOutcome {
    Trace(ProgressTrace),
    Context {
        finished_trace: ProgressTrace,
        payload: InjectedContextPayload,
    },
    ContextBatch {
        finished_trace: ProgressTrace,
        payloads: Vec<InjectedContextPayload>,
    },
    WorkflowPrompt {
        finished_trace: ProgressTrace,
        payloads: Vec<InjectedContextPayload>,
        prompt: String,
    },
    WorkflowShell {
        finished_trace: ProgressTrace,
        payloads: Vec<InjectedContextPayload>,
        command: String,
    },
    WorkflowWrite {
        finished_trace: ProgressTrace,
        payloads: Vec<InjectedContextPayload>,
        path: String,
        content: String,
    },
    WorkflowEdit {
        finished_trace: ProgressTrace,
        payloads: Vec<InjectedContextPayload>,
        path: String,
        edits: String,
    },
    Error {
        failed_trace: ProgressTrace,
        message: String,
    },
}

type SlashWork = Box<dyn FnOnce() -> std::result::Result<String, String> + Send>;

#[derive(Clone)]
pub(crate) struct InjectedContextPayload {
    pub content: String,
    pub metadata: Option<InjectedContextMetadata>,
}

struct SlashContextSpec {
    running_status: String,
    started_trace: String,
    finished_trace: String,
    failed_trace: String,
    context_prefix: String,
    metadata: Option<InjectedContextMetadata>,
    work: SlashWork,
}

fn make_trace(status: ProgressStatus, label: impl Into<String>, persist: bool) -> ProgressTrace {
    ProgressTrace {
        status,
        label: label.into(),
        persist,
    }
}

fn run_tool_immediate<T: crate::tools::Tool>(
    tool: T,
    arg: &str,
) -> std::result::Result<String, String> {
    match crate::tools::Tool::run(&tool, arg) {
        Ok(crate::tools::ToolRunResult::Immediate(output)) => Ok(output),
        Ok(crate::tools::ToolRunResult::RequiresApproval(_)) => {
            Err("requested approval unexpectedly".to_string())
        }
        Err(e) => Err(e.to_string()),
    }
}

fn spawn_slash_context_job(
    state: &mut AppState,
    slash_tx: &mpsc::Sender<SlashJobOutcome>,
    spec: SlashContextSpec,
    persist: bool,
) {
    state.start_generation(&spec.running_status, false);
    info!(label = spec.started_trace.as_str(), "trace.started");
    state.apply_trace(make_trace(
        ProgressStatus::Started,
        spec.started_trace.clone(),
        false,
    ));
    let tx = slash_tx.clone();
    thread::spawn(move || {
        let outcome = match (spec.work)() {
            Ok(output) => {
                let safe = sanitize_for_display(&output);
                let context = format!("{}\n\n{safe}", spec.context_prefix);
                SlashJobOutcome::Context {
                    finished_trace: make_trace(
                        ProgressStatus::Finished,
                        spec.finished_trace,
                        persist,
                    ),
                    payload: InjectedContextPayload {
                        content: context,
                        metadata: spec.metadata,
                    },
                }
            }
            Err(error) => SlashJobOutcome::Error {
                failed_trace: make_trace(ProgressStatus::Failed, spec.failed_trace, persist),
                message: error,
            },
        };
        let _ = tx.send(outcome);
    });
}

fn build_context_spec(cmd: &str, arg: &str) -> Option<SlashContextSpec> {
    let canonical = resolve_builtin_command(cmd)?.canonical;
    match canonical {
        "/read" => {
            if arg.is_empty() {
                return None;
            }
            let arg_owned = arg.to_string();
            Some(SlashContextSpec {
                running_status: "reading file...".to_string(),
                started_trace: format!("reading {arg}"),
                finished_trace: format!("loaded {arg}"),
                failed_trace: format!("read failed for {arg}"),
                context_prefix: "I've loaded this file for context:".to_string(),
                metadata: Some(InjectedContextMetadata {
                    file_path: Some(arg.to_string()),
                    directory_path: None,
                    search_query: None,
                    tool_name: Some("read_file".to_string()),
                }),
                work: Box::new(move || {
                    run_tool_immediate(crate::tools::ReadFile, &arg_owned)
                        .map_err(|e| format!("error reading {arg_owned}: {e}"))
                }),
            })
        }
        "/ls" => {
            let path = if arg.is_empty() { "." } else { arg };
            let path_owned = path.to_string();
            Some(SlashContextSpec {
                running_status: "listing directory...".to_string(),
                started_trace: format!("listing {path}"),
                finished_trace: format!("listed {path}"),
                failed_trace: format!("list failed for {path}"),
                context_prefix: "Directory listing:".to_string(),
                metadata: Some(InjectedContextMetadata {
                    file_path: None,
                    directory_path: Some(path.to_string()),
                    search_query: None,
                    tool_name: Some("list_dir".to_string()),
                }),
                work: Box::new(move || {
                    run_tool_immediate(crate::tools::ListDir, &path_owned)
                        .map_err(|e| format!("error listing {path_owned}: {e}"))
                }),
            })
        }
        "/search" => {
            if arg.is_empty() {
                return None;
            }
            let arg_owned = arg.to_string();
            Some(SlashContextSpec {
                running_status: "searching code...".to_string(),
                started_trace: format!("searching for {arg}"),
                finished_trace: format!("search complete for {arg}"),
                failed_trace: format!("search failed for {arg}"),
                context_prefix: "Search results:".to_string(),
                metadata: Some(InjectedContextMetadata {
                    file_path: None,
                    directory_path: None,
                    search_query: Some(arg.to_string()),
                    tool_name: Some("search".to_string()),
                }),
                work: Box::new(move || {
                    run_tool_immediate(crate::tools::SearchCode, &arg_owned)
                        .map_err(|e| format!("error searching: {e}"))
                }),
            })
        }
        "/git" => {
            let git_arg = if arg.is_empty() { "status" } else { arg };
            let git_arg_owned = git_arg.to_string();
            Some(SlashContextSpec {
                running_status: "running git...".to_string(),
                started_trace: format!("running git {git_arg}"),
                finished_trace: format!("git: {git_arg}"),
                failed_trace: format!("git failed for {git_arg}"),
                context_prefix: format!("Git context ({git_arg}):"),
                metadata: Some(InjectedContextMetadata {
                    file_path: None,
                    directory_path: None,
                    search_query: None,
                    tool_name: Some("git".to_string()),
                }),
                work: Box::new(move || {
                    run_tool_immediate(crate::tools::GitTool, &git_arg_owned)
                        .map_err(|e| format!("git error: {e}"))
                }),
            })
        }
        "/diag" => {
            if arg.is_empty() {
                return None;
            }
            let arg_owned = arg.to_string();
            Some(SlashContextSpec {
                running_status: "running diagnostics...".to_string(),
                started_trace: format!("running diagnostics for {arg}"),
                finished_trace: format!("diagnostics ready for {arg}"),
                failed_trace: format!("diagnostics failed for {arg}"),
                context_prefix: "LSP diagnostics:".to_string(),
                metadata: Some(InjectedContextMetadata {
                    file_path: Some(arg.to_string()),
                    directory_path: None,
                    search_query: None,
                    tool_name: Some("lsp_diagnostics".to_string()),
                }),
                work: Box::new(move || {
                    run_tool_immediate(crate::tools::LspDiagnosticsTool, &arg_owned)
                        .map_err(|e| format!("diagnostics error: {e}"))
                }),
            })
        }
        "/hover" => {
            if arg.is_empty() {
                return None;
            }
            let arg_owned = arg.to_string();
            Some(SlashContextSpec {
                running_status: "loading hover info...".to_string(),
                started_trace: format!("loading hover for {arg}"),
                finished_trace: format!("hover ready for {arg}"),
                failed_trace: format!("hover failed for {arg}"),
                context_prefix: "LSP hover:".to_string(),
                metadata: Some(InjectedContextMetadata {
                    file_path: Some(arg.to_string()),
                    directory_path: None,
                    search_query: None,
                    tool_name: Some("lsp_hover".to_string()),
                }),
                work: Box::new(move || {
                    run_tool_immediate(crate::tools::LspHoverTool, &arg_owned)
                        .map_err(|e| format!("hover error: {e}"))
                }),
            })
        }
        "/def" => {
            if arg.is_empty() {
                return None;
            }
            let arg_owned = arg.to_string();
            Some(SlashContextSpec {
                running_status: "resolving definition...".to_string(),
                started_trace: format!("resolving definition for {arg}"),
                finished_trace: format!("definition ready for {arg}"),
                failed_trace: format!("definition failed for {arg}"),
                context_prefix: "LSP definition:".to_string(),
                metadata: Some(InjectedContextMetadata {
                    file_path: Some(arg.to_string()),
                    directory_path: None,
                    search_query: None,
                    tool_name: Some("lsp_definition".to_string()),
                }),
                work: Box::new(move || {
                    run_tool_immediate(crate::tools::LspDefinitionTool, &arg_owned)
                        .map_err(|e| format!("definition error: {e}"))
                }),
            })
        }
        "/lcheck" => Some(SlashContextSpec {
            running_status: "checking rust lsp...".to_string(),
            started_trace: "checking rust lsp".to_string(),
            finished_trace: "rust lsp check complete".to_string(),
            failed_trace: "rust lsp check failed".to_string(),
            context_prefix: "LSP check:".to_string(),
            metadata: Some(InjectedContextMetadata {
                file_path: None,
                directory_path: None,
                search_query: None,
                tool_name: Some("lsp_health".to_string()),
            }),
            work: Box::new(move || Ok(crate::tools::rust_lsp_health_report())),
        }),
        "/fetch" => {
            if arg.is_empty() {
                return None;
            }
            let arg_owned = arg.to_string();
            Some(SlashContextSpec {
                running_status: "fetching webpage...".to_string(),
                started_trace: format!("fetching {arg}"),
                finished_trace: format!("fetched {arg}"),
                failed_trace: format!("fetch failed for {arg}"),
                context_prefix: "Fetched web context:".to_string(),
                metadata: Some(InjectedContextMetadata {
                    file_path: None,
                    directory_path: None,
                    search_query: Some(arg.to_string()),
                    tool_name: Some("fetch_url".to_string()),
                }),
                work: Box::new(move || {
                    run_tool_immediate(crate::tools::FetchUrlTool, &arg_owned)
                        .map_err(|e| format!("fetch error: {e}"))
                }),
            })
        }
        _ => None,
    }
}

fn execute_custom_template(
    command: &CustomCommand,
    args: &[&str],
    state: &mut AppState,
    prompt_tx: &mpsc::Sender<SessionCommand>,
) {
    let CustomCommandBody::Prompt(template) = &command.body else {
        return;
    };
    let expanded = crate::commands::expand_positional_args(template, args);
    info!(
        command = command.name.as_str(),
        origin = command.origin.as_str(),
        "custom command started"
    );
    state.start_generation(&format!("running {}...", command.name), false);
    state.apply_trace(make_trace(
        ProgressStatus::Started,
        format!("running {}", command.name),
        false,
    ));
    state.apply_trace(make_trace(
        ProgressStatus::Finished,
        format!("prepared {}", command.name),
        false,
    ));
    state.add_user_message(&expanded);
    let _ = prompt_tx.send(SessionCommand::SubmitUser(expanded));
    state.start_generation("generating...", true);
}

fn execute_custom_workflow(
    command: CustomCommand,
    args: Vec<&str>,
    state: &mut AppState,
    slash_tx: &mpsc::Sender<SlashJobOutcome>,
) {
    let CustomCommandBody::Workflow(steps) = command.body.clone() else {
        return;
    };

    let expanded_steps = steps
        .into_iter()
        .map(|step| match step {
            CustomCommandStep::Slash(text) => {
                CustomCommandStep::Slash(crate::commands::expand_positional_args(&text, &args))
            }
            CustomCommandStep::Prompt(text) => {
                CustomCommandStep::Prompt(crate::commands::expand_positional_args(&text, &args))
            }
        })
        .collect::<Vec<_>>();

    let workflow_name = command.name.clone();
    let origin = command.origin;

    state.start_generation(&format!("running {}...", workflow_name), false);
    state.apply_trace(make_trace(
        ProgressStatus::Started,
        format!("running {workflow_name}"),
        false,
    ));
    info!(
        command = workflow_name.as_str(),
        origin = origin.as_str(),
        "custom command started"
    );

    let tx = slash_tx.clone();
    thread::spawn(move || {
        let mut payloads = Vec::new();

        for step in expanded_steps {
            match step {
                CustomCommandStep::Slash(slash) => {
                    let (cmd, arg) = parse_command_parts(&slash);
                    let Some(spec_meta) = resolve_builtin_command(&cmd) else {
                        let _ = tx.send(SlashJobOutcome::Error {
                            failed_trace: make_trace(
                                ProgressStatus::Failed,
                                format!("custom command failed: {workflow_name}"),
                                false,
                            ),
                            message: format!("custom workflow references unsupported step: {cmd}"),
                        });
                        return;
                    };

                    match spec_meta.kind {
                        BuiltinKind::Context => {
                            let Some(spec) = build_context_spec(&cmd, arg) else {
                                let _ = tx.send(SlashJobOutcome::Error {
                                    failed_trace: make_trace(
                                        ProgressStatus::Failed,
                                        format!("custom command failed: {workflow_name}"),
                                        false,
                                    ),
                                    message: format!("invalid usage for workflow step `{slash}`"),
                                });
                                return;
                            };

                            let _ = tx.send(SlashJobOutcome::Trace(make_trace(
                                ProgressStatus::Started,
                                spec.started_trace.clone(),
                                false,
                            )));
                            match (spec.work)() {
                                Ok(output) => {
                                    let safe = sanitize_for_display(&output);
                                    payloads.push(InjectedContextPayload {
                                        content: format!("{}\n\n{safe}", spec.context_prefix),
                                        metadata: spec.metadata.clone(),
                                    });
                                    let _ = tx.send(SlashJobOutcome::Trace(make_trace(
                                        ProgressStatus::Finished,
                                        spec.finished_trace,
                                        false,
                                    )));
                                }
                                Err(error) => {
                                    let _ = tx.send(SlashJobOutcome::Trace(make_trace(
                                        ProgressStatus::Failed,
                                        spec.failed_trace,
                                        false,
                                    )));
                                    let _ = tx.send(SlashJobOutcome::Error {
                                        failed_trace: make_trace(
                                            ProgressStatus::Failed,
                                            format!("custom command failed: {workflow_name}"),
                                            false,
                                        ),
                                        message: error,
                                    });
                                    return;
                                }
                            }
                        }
                        BuiltinKind::Mutating => {
                            let final_trace = make_trace(
                                ProgressStatus::Finished,
                                format!("completed {workflow_name}"),
                                false,
                            );
                            match spec_meta.canonical {
                                "/run" => {
                                    let _ = tx.send(SlashJobOutcome::WorkflowShell {
                                        finished_trace: final_trace,
                                        payloads,
                                        command: arg.to_string(),
                                    });
                                }
                                "/write" => {
                                    let Some((path, raw_content)) = arg.split_once(' ') else {
                                        let _ = tx.send(SlashJobOutcome::Error {
                                            failed_trace: make_trace(
                                                ProgressStatus::Failed,
                                                format!("custom command failed: {workflow_name}"),
                                                false,
                                            ),
                                            message: "workflow /write step must use `/write <path> <content>`".to_string(),
                                        });
                                        return;
                                    };
                                    let _ = tx.send(SlashJobOutcome::WorkflowWrite {
                                        finished_trace: final_trace,
                                        payloads,
                                        path: path.trim().to_string(),
                                        content: decode_slash_write_content(raw_content.trim()),
                                    });
                                }
                                "/edit" => {
                                    let Some((path, edits)) = parse_slash_edit_body(arg) else {
                                        let _ = tx.send(SlashJobOutcome::Error {
                                            failed_trace: make_trace(
                                                ProgressStatus::Failed,
                                                format!("custom command failed: {workflow_name}"),
                                                false,
                                            ),
                                            message: "workflow /edit step must use `/edit <path>` followed by a multiline ```params-edit block".to_string(),
                                        });
                                        return;
                                    };
                                    let _ = tx.send(SlashJobOutcome::WorkflowEdit {
                                        finished_trace: final_trace,
                                        payloads,
                                        path,
                                        edits,
                                    });
                                }
                                _ => {}
                            }
                            return;
                        }
                        BuiltinKind::Session | BuiltinKind::Discovery => {
                            let _ = tx.send(SlashJobOutcome::Error {
                                failed_trace: make_trace(
                                    ProgressStatus::Failed,
                                    format!("custom command failed: {workflow_name}"),
                                    false,
                                ),
                                message: format!(
                                    "workflow step `{}` is not supported in custom commands",
                                    spec_meta.canonical
                                ),
                            });
                            return;
                        }
                    }
                }
                CustomCommandStep::Prompt(prompt) => {
                    let _ = tx.send(SlashJobOutcome::WorkflowPrompt {
                        finished_trace: make_trace(
                            ProgressStatus::Finished,
                            format!("completed {workflow_name}"),
                            false,
                        ),
                        payloads,
                        prompt,
                    });
                    return;
                }
            }
        }

        let _ = tx.send(SlashJobOutcome::ContextBatch {
            finished_trace: make_trace(
                ProgressStatus::Finished,
                format!("completed {workflow_name}"),
                false,
            ),
            payloads,
        });
    });
}

pub(crate) fn handle_command_input(
    input: &str,
    state: &mut AppState,
    prompt_tx: &mpsc::Sender<SessionCommand>,
    slash_tx: &mpsc::Sender<SlashJobOutcome>,
    command_registry: &mut CommandRegistry,
) {
    let (cmd, arg) = parse_command_parts(input);
    info!(command = cmd.as_str(), "slash command received");

    if resolve_builtin_command(&cmd).is_some() {
        handle_builtin_slash_command(&cmd, arg, state, prompt_tx, slash_tx, command_registry);
        return;
    }

    let Some(command) = command_registry.resolve(&cmd).cloned() else {
        state.add_system_message(&format!(
            "unknown command: {cmd}. Type /help for available commands."
        ));
        return;
    };

    let args = if arg.is_empty() {
        Vec::new()
    } else {
        arg.split_whitespace().collect::<Vec<_>>()
    };
    match &command.body {
        CustomCommandBody::Prompt(_) => execute_custom_template(&command, &args, state, prompt_tx),
        CustomCommandBody::Workflow(_) => execute_custom_workflow(command, args, state, slash_tx),
    }
}

fn handle_builtin_slash_command(
    cmd: &str,
    arg: &str,
    state: &mut AppState,
    prompt_tx: &mpsc::Sender<SessionCommand>,
    slash_tx: &mpsc::Sender<SlashJobOutcome>,
    command_registry: &mut CommandRegistry,
) {
    let canonical = resolve_builtin_command(cmd)
        .map(|spec| spec.canonical)
        .unwrap_or(cmd);

    match canonical {
        "/read" | "/search" | "/diag" | "/hover" | "/def" | "/fetch" => {
            let Some(spec) = build_context_spec(canonical, arg) else {
                let usage = resolve_builtin_command(canonical)
                    .map(|spec| spec.usage)
                    .unwrap_or("/help");
                state.add_system_message(&format!("Usage: {usage}"));
                return;
            };
            spawn_slash_context_job(state, slash_tx, spec, true);
        }
        "/ls" | "/git" | "/lcheck" => {
            if let Some(spec) = build_context_spec(canonical, arg) {
                spawn_slash_context_job(state, slash_tx, spec, true);
            }
        }
        "/run" => {
            if arg.is_empty() {
                state.add_system_message("Usage: /run <command>");
                return;
            }
            let _ = prompt_tx.send(SessionCommand::RequestShellCommand(arg.to_string()));
        }
        "/write" => {
            let Some((path, raw_content)) = arg.split_once(' ') else {
                state
                    .add_system_message("Usage: /write <path> <content>. Use \\n for line breaks.");
                return;
            };
            if path.trim().is_empty() || raw_content.trim().is_empty() {
                state
                    .add_system_message("Usage: /write <path> <content>. Use \\n for line breaks.");
                return;
            }
            let _ = prompt_tx.send(SessionCommand::RequestFileWrite {
                path: path.trim().to_string(),
                content: decode_slash_write_content(raw_content.trim()),
            });
        }
        "/edit" => {
            let Some((path, edits)) = parse_slash_edit_body(arg) else {
                state.add_system_message(
                    "Usage: /edit <path> followed by a multiline ```params-edit block.",
                );
                return;
            };
            let _ = prompt_tx.send(SessionCommand::RequestFileEdit { path, edits });
        }
        "/reflect" => {
            let mode = arg.to_ascii_lowercase();
            match mode.as_str() {
                "on" => {
                    if state.eco_enabled {
                        state.add_system_message(
                            "reflection stays off while eco mode is enabled. Use /eco off first.",
                        );
                        return;
                    }
                    state.set_reflection_enabled(true);
                    state.add_system_message("reflection enabled");
                    let _ = prompt_tx.send(SessionCommand::SetReflection(true));
                }
                "off" => {
                    state.set_reflection_enabled(false);
                    state.add_system_message("reflection disabled");
                    let _ = prompt_tx.send(SessionCommand::SetReflection(false));
                }
                "" | "status" => {
                    state.add_system_message(&format!(
                        "reflection is {}",
                        if state.reflection_enabled {
                            "on"
                        } else {
                            "off"
                        }
                    ));
                }
                _ => state.add_system_message("Usage: /reflect <on|off|status>"),
            }
        }
        "/eco" => {
            let mode = arg.to_ascii_lowercase();
            match mode.as_str() {
                "on" => {
                    state.set_eco_enabled(true);
                    state.set_reflection_enabled(false);
                    state.add_system_message("eco mode enabled");
                    let _ = prompt_tx.send(SessionCommand::SetEco(true));
                }
                "off" => {
                    state.set_eco_enabled(false);
                    state.add_system_message("eco mode disabled");
                    let _ = prompt_tx.send(SessionCommand::SetEco(false));
                }
                "" | "status" => {
                    state.add_system_message(&format!(
                        "eco mode is {}",
                        if state.eco_enabled { "on" } else { "off" }
                    ));
                }
                _ => state.add_system_message("Usage: /eco <on|off|status>"),
            }
        }
        "/debug-log" => {
            let mode = arg.to_ascii_lowercase();
            match mode.as_str() {
                "on" => {
                    state.set_debug_logging_enabled(true);
                    state.add_system_message("separate debug content logging enabled");
                    let _ = prompt_tx.send(SessionCommand::SetDebugLogging(true));
                }
                "off" => {
                    state.set_debug_logging_enabled(false);
                    state.add_system_message("separate debug content logging disabled");
                    let _ = prompt_tx.send(SessionCommand::SetDebugLogging(false));
                }
                "" | "status" => {
                    state.add_system_message(&format!(
                        "debug content logging is {}",
                        if state.debug_logging_enabled {
                            "on"
                        } else {
                            "off"
                        }
                    ));
                }
                _ => state.add_system_message("Usage: /debug-log <on|off|status>"),
            }
        }
        "/approve" => match state.pending_action_id() {
            Some(id) => {
                let _ = prompt_tx.send(SessionCommand::ApproveAction(id));
                state.mark_pending_action_submitted("processing approval");
            }
            None => state.add_system_message("No pending action to approve."),
        },
        "/reject" => match state.pending_action_id() {
            Some(id) => {
                let _ = prompt_tx.send(SessionCommand::RejectAction(id));
                state.mark_pending_action_submitted("processing rejection");
            }
            None => state.add_system_message("No pending action to reject."),
        },
        "/clear" => {
            state.clear_messages();
            state.add_system_message("conversation cleared");
            let _ = prompt_tx.send(SessionCommand::ClearSession);
        }
        "/clear-cache" => {
            state.add_system_message("clearing exact cache...");
            let _ = prompt_tx.send(SessionCommand::ClearCache);
        }
        "/clear-debug-log" => {
            state.add_system_message("clearing separate debug content log...");
            let _ = prompt_tx.send(SessionCommand::ClearDebugLog);
        }
        "/sessions" => {
            let trimmed = arg.trim();
            let (subcommand, rest) = if trimmed.is_empty() {
                ("list", "")
            } else if let Some((sub, remainder)) = trimmed.split_once(' ') {
                (sub, remainder.trim())
            } else {
                (trimmed, "")
            };

            match subcommand {
                "list" => {
                    let _ = prompt_tx.send(SessionCommand::ListSessions);
                }
                "new" => {
                    let name = if rest.is_empty() {
                        None
                    } else {
                        Some(rest.to_string())
                    };
                    let _ = prompt_tx.send(SessionCommand::NewSession(name));
                }
                "rename" => {
                    if rest.is_empty() {
                        state.add_system_message("Usage: /sessions rename <name>");
                        return;
                    }
                    let _ = prompt_tx.send(SessionCommand::RenameSession(rest.to_string()));
                }
                "resume" => {
                    if rest.is_empty() {
                        state.add_system_message("Usage: /sessions resume <name-or-id>");
                        return;
                    }
                    let _ = prompt_tx.send(SessionCommand::ResumeSession(rest.to_string()));
                }
                "delete" => {
                    if rest.is_empty() {
                        state.add_system_message("Usage: /sessions delete <name-or-id>");
                        return;
                    }
                    let _ = prompt_tx.send(SessionCommand::DeleteSession(rest.to_string()));
                }
                "export" => {
                    let Some((selector, format)) = parse_sessions_export_args(rest) else {
                        state.add_system_message(
                            "Usage: /sessions export <name-or-id> [markdown|json]",
                        );
                        return;
                    };
                    let _ = prompt_tx.send(SessionCommand::ExportSession { selector, format });
                }
                _ => {
                    state.add_system_message(
                        "Usage: /sessions <list|new|rename|resume|delete|export>",
                    );
                }
            }
        }
        "/memory" => {
            let trimmed = arg.trim();
            let subcommand = if trimmed.is_empty() {
                "status"
            } else {
                trimmed
            };
            match subcommand {
                "status" => state.add_system_message(&format_memory_status(&state.memory_snapshot)),
                "facts" => state.add_system_message(&format_memory_facts(&state.memory_snapshot)),
                "last" => state.add_system_message(&format_memory_last(&state.memory_snapshot)),
                "prune" => {
                    let _ = prompt_tx.send(SessionCommand::PruneMemory);
                }
                _ if subcommand.starts_with("recall ") => {
                    let query = subcommand["recall ".len()..].trim();
                    if query.is_empty() {
                        state.add_system_message("Usage: /memory recall <query>");
                    } else {
                        let _ = prompt_tx.send(SessionCommand::RecallMemory(query.to_string()));
                    }
                }
                _ => state
                    .add_system_message("Usage: /memory [status|facts|last|recall <query>|prune]"),
            }
        }
        "/transcript" => {
            let subcommand = if arg.is_empty() { "status" } else { arg.trim() };
            match subcommand {
                "status" => {
                    let (total, collapsed) = state.transcript_item_counts();
                    state.add_system_message(&format_transcript_status(total, collapsed));
                }
                "collapse" => {
                    let changed = state.collapse_all_transcript_items();
                    state.add_system_message(&format!("collapsed {changed} transcript block(s)"));
                }
                "expand" => {
                    let changed = state.expand_all_transcript_items();
                    state.add_system_message(&format!("expanded {changed} transcript block(s)"));
                }
                "toggle" => {
                    let changed = state.toggle_all_transcript_items();
                    state.add_system_message(&format!("toggled {changed} transcript block(s)"));
                }
                _ => state.add_system_message("Usage: /transcript [status|collapse|expand|toggle]"),
            }
        }
        "/display" => {
            let subcommand = if arg.is_empty() { "status" } else { arg.trim() };
            match subcommand {
                "status" => state.add_system_message(&format_display_status(state)),
                "tokens on" => {
                    state.set_show_top_bar_tokens(true);
                    state.add_system_message("display: tokens on");
                }
                "tokens off" => {
                    state.set_show_top_bar_tokens(false);
                    state.add_system_message("display: tokens off");
                }
                "time on" => {
                    state.set_show_top_bar_time(true);
                    state.add_system_message("display: time on");
                }
                "time off" => {
                    state.set_show_top_bar_time(false);
                    state.add_system_message("display: time off");
                }
                _ => state
                    .add_system_message("Usage: /display [status|tokens <on|off>|time <on|off>]"),
            }
        }
        "/help" => {
            state.add_system_message(&custom_help_text());
        }
        "/commands" => {
            let subcommand = if arg.is_empty() { "list" } else { arg };
            match subcommand {
                "list" => {
                    state.add_system_message(&format_custom_commands_list(command_registry));
                }
                "reload" => {
                    let report = CommandRegistry::load_report();
                    *command_registry = report.registry;
                    state.add_system_message(&format!(
                        "reloaded custom commands: {} loaded, {} invalid, {} source file(s)",
                        report.loaded, report.invalid, report.sources_loaded
                    ));
                }
                _ => {
                    state.add_system_message("Usage: /commands [list|reload]");
                }
            }
        }
        _ => {
            state.add_system_message(&format!(
                "unknown command: {cmd}. Type /help for available commands."
            ));
        }
    }
}

#[cfg(test)]
mod tests;
