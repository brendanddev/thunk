/// A parsed slash command entered by the user.
/// Command parsing is a pure transformation — no runtime calls, no side effects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Help,
    Quit,
    Clear,
    Approve,
    Reject,
    Last,
    Anchors,
    History,
    Read(String),
    Search(String),
}

/// A parse-level error for slash commands. Returned when input begins with `/`
/// but is structurally invalid — not forwarded to the runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// The slash prefix was present but the command name is not recognized.
    UnknownCommand,
    /// The command requires an argument that was not provided.
    MissingArgument { command: &'static str },
}

impl ParseError {
    pub fn user_message(&self) -> String {
        match self {
            Self::UnknownCommand => "unknown command".to_string(),
            Self::MissingArgument { command } => format!("{command}: argument required"),
        }
    }
}

/// Parses user input into a command result.
///
/// - `None`          — no `/` prefix: route to runtime as a normal prompt
/// - `Some(Ok(cmd))` — valid recognized command: execute
/// - `Some(Err(e))`  — slash command attempted but invalid: surface error, do not route to runtime
///
/// Parsing is pure — no runtime calls, no side effects.
pub fn parse(input: &str) -> Option<Result<Command, ParseError>> {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return None;
    }

    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or(trimmed);
    let arg = parts.next().map(str::trim).filter(|s| !s.is_empty());

    match name {
        "/help"           => Some(Ok(Command::Help)),
        "/quit" | "/exit" => Some(Ok(Command::Quit)),
        "/clear"          => Some(Ok(Command::Clear)),
        "/approve"        => Some(Ok(Command::Approve)),
        "/reject"         => Some(Ok(Command::Reject)),
        "/last"           => Some(Ok(Command::Last)),
        "/anchors"        => Some(Ok(Command::Anchors)),
        "/history"        => Some(Ok(Command::History)),
        "/read" => match arg {
            Some(path)  => Some(Ok(Command::Read(path.to_string()))),
            None        => Some(Err(ParseError::MissingArgument { command: "/read" })),
        },
        "/search" => match arg {
            Some(query) => Some(Ok(Command::Search(query.to_string()))),
            None        => Some(Err(ParseError::MissingArgument { command: "/search" })),
        },
        _                 => Some(Err(ParseError::UnknownCommand)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_help() {
        assert_eq!(parse("/help"), Some(Ok(Command::Help)));
    }

    #[test]
    fn parses_quit_aliases() {
        assert_eq!(parse("/quit"), Some(Ok(Command::Quit)));
        assert_eq!(parse("/exit"), Some(Ok(Command::Quit)));
    }

    #[test]
    fn parses_clear() {
        assert_eq!(parse("/clear"), Some(Ok(Command::Clear)));
    }

    #[test]
    fn parses_approve() {
        assert_eq!(parse("/approve"), Some(Ok(Command::Approve)));
    }

    #[test]
    fn parses_reject() {
        assert_eq!(parse("/reject"), Some(Ok(Command::Reject)));
    }

    #[test]
    fn parses_last() {
        assert_eq!(parse("/last"), Some(Ok(Command::Last)));
    }

    #[test]
    fn parses_anchors() {
        assert_eq!(parse("/anchors"), Some(Ok(Command::Anchors)));
    }

    #[test]
    fn parses_history() {
        assert_eq!(parse("/history"), Some(Ok(Command::History)));
    }

    #[test]
    fn ignores_whitespace() {
        assert_eq!(parse("  /help  "), Some(Ok(Command::Help)));
    }

    #[test]
    fn extra_args_on_no_arg_command_are_ignored() {
        assert_eq!(parse("/help extra stuff"), Some(Ok(Command::Help)));
        assert_eq!(parse("/history some arg"), Some(Ok(Command::History)));
    }

    #[test]
    fn non_command_returns_none() {
        assert_eq!(parse("hello"), None);
        assert_eq!(parse("how do I fix this bug?"), None);
    }

    #[test]
    fn empty_input_returns_none() {
        assert_eq!(parse(""), None);
        assert_eq!(parse("   "), None);
    }

    #[test]
    fn unknown_slash_command_returns_error() {
        assert_eq!(parse("/unknown"), Some(Err(ParseError::UnknownCommand)));
        assert_eq!(parse("/foo"), Some(Err(ParseError::UnknownCommand)));
    }

    #[test]
    fn unknown_command_error_message() {
        let e = ParseError::UnknownCommand;
        assert_eq!(e.user_message(), "unknown command");
    }

    #[test]
    fn missing_argument_error_message() {
        let e = ParseError::MissingArgument { command: "/read" };
        assert_eq!(e.user_message(), "/read: argument required");
    }

    #[test]
    fn parses_read_with_path() {
        assert_eq!(
            parse("/read src/main.rs"),
            Some(Ok(Command::Read("src/main.rs".to_string())))
        );
    }

    #[test]
    fn parses_search_with_query() {
        assert_eq!(
            parse("/search fn handle"),
            Some(Ok(Command::Search("fn handle".to_string())))
        );
    }

    #[test]
    fn read_without_arg_returns_missing_argument() {
        assert_eq!(
            parse("/read"),
            Some(Err(ParseError::MissingArgument { command: "/read" }))
        );
        assert_eq!(
            parse("/read   "),
            Some(Err(ParseError::MissingArgument { command: "/read" }))
        );
    }

    #[test]
    fn search_without_arg_returns_missing_argument() {
        assert_eq!(
            parse("/search"),
            Some(Err(ParseError::MissingArgument { command: "/search" }))
        );
        assert_eq!(
            parse("/search   "),
            Some(Err(ParseError::MissingArgument { command: "/search" }))
        );
    }
}
