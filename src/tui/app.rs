use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};

use crate::app::config::{AllowedCommandTool, Config};
use crate::app::paths::AppPaths;
use crate::app::AppContext;
use crate::app::Result;
use crate::runtime::{AnswerSource, RuntimeEvent, RuntimeRequest};

use super::commands;
use super::render::render;
use super::state::AppState;

pub(crate) fn run_app(
    stdout: &mut io::Stdout,
    config: &Config,
    paths: &AppPaths,
    app: &mut AppContext,
) -> Result<()> {
    let mut state = AppState::new(config, paths);

    loop {
        render(stdout, &state)?;

        if state.should_quit {
            return Ok(());
        }

        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) => handle_key_event(stdout, &mut state, app, config, key)?,
                Event::Paste(text) => state.insert_str(&text),
                Event::Resize(_, _) => {}
                _ => {}
            }
        }
    }
}

fn handle_key_event(
    stdout: &mut io::Stdout,
    state: &mut AppState,
    app: &mut AppContext,
    config: &Config,
    key: KeyEvent,
) -> Result<()> {
    match (key.code, key.modifiers) {
        (KeyCode::Char('c'), KeyModifiers::CONTROL)
        | (KeyCode::Char('q'), KeyModifiers::CONTROL) => {
            state.should_quit = true;
        }
        (KeyCode::Enter, _) => {
            if let Some(input) = state.submit_input() {
                match commands::parse(&input) {
                    None => submit_to_app(stdout, state, app, input)?,
                    Some(Ok(cmd)) => handle_command(stdout, state, app, cmd)?,
                    Some(Err(commands::ParseError::UnknownCommand)) => {
                        match resolve_custom_command(config, &input) {
                            None => state.add_system_message(
                                commands::ParseError::UnknownCommand.user_message(),
                            ),
                            Some(Err(msg)) => state.add_system_message(msg),
                            Some(Ok(req)) => {
                                dispatch_command_runtime_request(stdout, state, app, req)?
                            }
                        }
                    }
                    Some(Err(e)) => state.add_system_message(e.user_message()),
                }
            }
        }
        (KeyCode::Backspace, _) => state.delete_char_before(),
        (KeyCode::Left, _) => state.cursor_left(),
        (KeyCode::Right, _) => state.cursor_right(),
        (KeyCode::Home, _) => state.cursor_home(),
        (KeyCode::End, _) => state.cursor_end(),
        (KeyCode::Char(c), KeyModifiers::NONE | KeyModifiers::SHIFT) => state.insert_char(c),
        _ => {}
    }

    Ok(())
}

// Used by Approve and Reject: applies Failed event before propagating render errors.
// submit_to_app has a different post-handle ordering and is kept separate.
fn dispatch_command_runtime_request(
    stdout: &mut io::Stdout,
    state: &mut AppState,
    app: &mut AppContext,
    req: RuntimeRequest,
) -> Result<()> {
    let mut render_error = None;
    if let Err(e) = app.handle(req, &mut |event| {
        if render_error.is_some() {
            return;
        }
        apply_runtime_event(state, event);
        if let Err(e) = render(stdout, state) {
            render_error = Some(e);
        }
    }) {
        apply_runtime_event(
            state,
            RuntimeEvent::Failed {
                message: e.to_string(),
            },
        );
    }
    if let Some(e) = render_error {
        return Err(e);
    }
    Ok(())
}

fn submit_to_app(
    stdout: &mut io::Stdout,
    state: &mut AppState,
    app: &mut AppContext,
    prompt: String,
) -> Result<()> {
    state.add_user_message(prompt.clone());
    let mut render_error = None;

    let handle_result = app.handle(RuntimeRequest::Submit { text: prompt }, &mut |event| {
        if render_error.is_some() {
            return;
        }
        apply_runtime_event(state, event);
        if let Err(e) = render(stdout, state) {
            render_error = Some(e);
        }
    });

    if let Some(e) = render_error {
        return Err(e);
    }

    if let Err(e) = handle_result {
        apply_runtime_event(
            state,
            RuntimeEvent::Failed {
                message: e.to_string(),
            },
        );
    }

    Ok(())
}

enum CommandAction {
    Quit,
    ShowHelp,
    ClearSession,
    Runtime(RuntimeRequest),
}

fn resolve_command(cmd: commands::Command) -> CommandAction {
    match cmd {
        commands::Command::Help => CommandAction::ShowHelp,
        commands::Command::Quit => CommandAction::Quit,
        commands::Command::Clear => CommandAction::ClearSession,
        commands::Command::Approve => CommandAction::Runtime(RuntimeRequest::Approve),
        commands::Command::Reject => CommandAction::Runtime(RuntimeRequest::Reject),
        commands::Command::Last => CommandAction::Runtime(RuntimeRequest::QueryLast),
        commands::Command::Anchors => CommandAction::Runtime(RuntimeRequest::QueryAnchors),
        commands::Command::History => CommandAction::Runtime(RuntimeRequest::QueryHistory),
        commands::Command::Read(path) => CommandAction::Runtime(RuntimeRequest::ReadFile { path }),
        commands::Command::Search(query) => {
            CommandAction::Runtime(RuntimeRequest::SearchCode { query })
        }
    }
}

fn handle_command(
    stdout: &mut io::Stdout,
    state: &mut AppState,
    app: &mut AppContext,
    cmd: commands::Command,
) -> Result<()> {
    match resolve_command(cmd) {
        CommandAction::ShowHelp => {
            state.add_system_message(
                "Commands: /help — show this message  |  /clear — clear history  |  /quit — exit  |  /approve — confirm pending action  |  /reject — cancel pending action  |  /read <path> — read file  |  /search <query> — search code  |  /last — last response  |  /anchors — anchor state  |  /history — conversation history",
            );
        }
        CommandAction::Quit => {
            state.should_quit = true;
        }
        CommandAction::ClearSession => {
            state.clear_messages();
            if let Err(e) = app.reset() {
                state.add_system_message(format!("session reset failed: {e}"));
            }
        }
        CommandAction::Runtime(req) => {
            dispatch_command_runtime_request(stdout, state, app, req)?;
        }
    }
    Ok(())
}

/// Resolves a raw input string against the custom command definitions in config.
///
/// Returns:
/// - `None`           — no custom command with this name; caller shows "unknown command"
/// - `Some(Err(msg))` — command found but argument is missing
/// - `Some(Ok(req))`  — resolved to a RuntimeRequest ready for dispatch
fn resolve_custom_command(
    config: &Config,
    input: &str,
) -> Option<std::result::Result<RuntimeRequest, String>> {
    let trimmed = input.trim();
    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let slash_name = parts.next()?;
    let name = slash_name.strip_prefix('/')?;
    let def = config.commands.get(name)?;

    let arg = parts.next().map(str::trim).filter(|s| !s.is_empty());
    let arg_str = match arg {
        Some(a) => a.to_string(),
        None => return Some(Err(format!("/{name}: argument required"))),
    };

    let value = def.template.replace("{input}", &arg_str);
    let req = match def.tool {
        AllowedCommandTool::ReadFile => RuntimeRequest::ReadFile { path: value },
        AllowedCommandTool::SearchCode => RuntimeRequest::SearchCode { query: value },
    };
    Some(Ok(req))
}

/// Converts a raw tool_result InfoMessage into a compact human-readable summary.
/// Non-tool-result InfoMessages (query output, error text, etc.) pass through unchanged.
fn summarize_command_output(text: &str) -> String {
    let Some(after_prefix) = text.strip_prefix("=== tool_result: ") else {
        return text.to_string();
    };
    let Some(name_end) = after_prefix.find(" ===\n") else {
        return text.to_string();
    };
    let tool_name = &after_prefix[..name_end];
    let header_len = "=== tool_result: ".len() + name_end + " ===\n".len();
    let raw_body = text.get(header_len..).unwrap_or("").trim_end();
    let body = raw_body
        .strip_suffix("=== /tool_result ===")
        .unwrap_or(raw_body)
        .trim_end();

    match tool_name {
        "read_file" => {
            let first = body.lines().next().unwrap_or("");
            match parse_read_file_header(first) {
                Some((n, false)) => format!("read: {n} lines"),
                Some((n, true)) => format!("read: {n} lines (truncated)"),
                None => "read: done".to_string(),
            }
        }
        "search_code" => {
            if body.starts_with("No matches found.") {
                return "search: no matches".to_string();
            }
            let first = body.lines().next().unwrap_or("");
            // Truncated header: "[showing first M of N matches — ...]"
            if let Some(inner) = first.strip_prefix("[showing first ") {
                if let Some(of_pos) = inner.find(" of ") {
                    let m = &inner[..of_pos];
                    let after_of = &inner[of_pos + " of ".len()..];
                    let n = after_of.split_whitespace().next().unwrap_or("?");
                    return format!("search: {n} matches (showing {m})");
                }
            }
            // Untruncated: match lines are indented "  <line_num>: <content>"
            let count = body
                .lines()
                .filter(|l| {
                    l.starts_with("  ")
                        && l.trim_start()
                            .chars()
                            .next()
                            .map(|c| c.is_ascii_digit())
                            .unwrap_or(false)
                })
                .count();
            if count > 0 {
                format!("search: {count} matches")
            } else {
                "search: done".to_string()
            }
        }
        _ => text.to_string(),
    }
}

/// Parses the first line of a read_file body: "[N lines]" or "[N lines — showing first M]".
/// Returns `(total_lines, is_truncated)` or `None` if the format is not recognised.
fn parse_read_file_header(line: &str) -> Option<(usize, bool)> {
    let inner = line.strip_prefix('[')?.strip_suffix(']')?;
    let truncated = inner.contains(" — ");
    let count_str = inner.split(" — ").next()?.split_whitespace().next()?;
    let n: usize = count_str.parse().ok()?;
    Some((n, truncated))
}

fn apply_runtime_event(state: &mut AppState, event: RuntimeEvent) {
    match event {
        RuntimeEvent::ActivityChanged(activity) => state.set_status(activity.label()),
        RuntimeEvent::AssistantMessageStarted => state.begin_assistant_message(),
        RuntimeEvent::AssistantMessageChunk(chunk) => state.append_assistant_chunk(&chunk),
        RuntimeEvent::AssistantMessageFinished => {}
        RuntimeEvent::ToolCallStarted { name } => {
            state.add_tool_message(format!("tool: {name}"));
        }
        RuntimeEvent::ToolCallFinished { name, summary } => match summary {
            Some(s) => state.add_tool_message(s),
            None => state.add_tool_message(format!("tool failed: {name}")),
        },
        RuntimeEvent::AnswerReady(source) => {
            if let AnswerSource::ToolLimitReached = source {
                state.add_system_message("Tool limit reached. Response may be incomplete.");
            }
        }
        RuntimeEvent::Failed { message } => {
            state.set_status("error");
            state.add_system_message(message);
        }
        RuntimeEvent::ApprovalRequired(pending) => {
            state.add_system_message(format!(
                "[approval required] {} — type /approve to confirm or /reject to cancel",
                pending.summary
            ));
            state.set_status("awaiting approval");
        }
        RuntimeEvent::InfoMessage(text) => {
            state.add_system_message(summarize_command_output(&text))
        }
        // Advisory only — absorbed by the logging layer before reaching here.
        RuntimeEvent::BackendTiming { .. } => {}
        RuntimeEvent::RuntimeTrace(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_read_file_header, summarize_command_output};

    fn tool_result(name: &str, body: &str) -> String {
        format!("=== tool_result: {name} ===\n{body}\n=== /tool_result ===\n\n")
    }

    // parse_read_file_header

    #[test]
    fn parses_untruncated_header() {
        assert_eq!(parse_read_file_header("[42 lines]"), Some((42, false)));
    }

    #[test]
    fn parses_truncated_header() {
        assert_eq!(
            parse_read_file_header("[300 lines — showing first 200]"),
            Some((300, true))
        );
    }

    #[test]
    fn rejects_malformed_header() {
        assert_eq!(parse_read_file_header("no brackets here"), None);
        assert_eq!(parse_read_file_header("[not a number lines]"), None);
    }

    // summarize_command_output — pass-through cases

    #[test]
    fn non_tool_result_passes_through_unchanged() {
        let msg = "no conversation history";
        assert_eq!(summarize_command_output(msg), msg);
    }

    #[test]
    fn query_output_passes_through_unchanged() {
        let msg = "last search: fn handle";
        assert_eq!(summarize_command_output(msg), msg);
    }

    // summarize_command_output — read_file

    #[test]
    fn read_file_untruncated_shows_line_count() {
        let body = "[42 lines]\nfn main() {}\n";
        let summary = summarize_command_output(&tool_result("read_file", body));
        assert_eq!(summary, "read: 42 lines");
    }

    #[test]
    fn read_file_truncated_shows_line_count_and_truncated() {
        let body =
            "[300 lines — showing first 200]\nfn main() {}\n[truncated: 100 lines not shown]";
        let summary = summarize_command_output(&tool_result("read_file", body));
        assert_eq!(summary, "read: 300 lines (truncated)");
    }

    // summarize_command_output — search_code

    #[test]
    fn search_no_matches_shows_no_matches() {
        let body = "No matches found.";
        let summary = summarize_command_output(&tool_result("search_code", body));
        assert_eq!(summary, "search: no matches");
    }

    #[test]
    fn search_truncated_shows_total_and_shown() {
        let body = "[showing first 15 of 42 matches — read a specific matched file with read_file]\nsrc/main.rs (3 matches)\n  12: fn handle()";
        let summary = summarize_command_output(&tool_result("search_code", body));
        assert_eq!(summary, "search: 42 matches (showing 15)");
    }

    #[test]
    fn search_untruncated_counts_match_lines() {
        let body =
            "src/main.rs (2 matches)\n  12: fn handle_request() {}\n  45: fn handle_response() {}";
        let summary = summarize_command_output(&tool_result("search_code", body));
        assert_eq!(summary, "search: 2 matches");
    }

    #[test]
    fn unknown_tool_passes_through_raw() {
        let raw = tool_result("git_status", "clean");
        assert_eq!(summarize_command_output(&raw), raw);
    }
}
