/// A parsed slash command entered by the user.
/// Command parsing is a pure transformation — no runtime calls, no side effects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Help,
    Quit,
    Clear,
}

/// Returns a `Command` if `input` is a recognized slash command, or `None` if
/// the input should be forwarded to the runtime as a normal prompt.
pub fn parse(input: &str) -> Option<Command> {
    match input.trim() {
        "/help" => Some(Command::Help),
        "/quit" | "/exit" => Some(Command::Quit),
        "/clear" => Some(Command::Clear),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_help() {
        assert_eq!(parse("/help"), Some(Command::Help));
    }

    #[test]
    fn parses_quit_aliases() {
        assert_eq!(parse("/quit"), Some(Command::Quit));
        assert_eq!(parse("/exit"), Some(Command::Quit));
    }

    #[test]
    fn parses_clear() {
        assert_eq!(parse("/clear"), Some(Command::Clear));
    }

    #[test]
    fn ignores_whitespace() {
        assert_eq!(parse("  /help  "), Some(Command::Help));
    }

    #[test]
    fn non_command_returns_none() {
        assert_eq!(parse("hello"), None);
        assert_eq!(parse("how do I fix this bug?"), None);
    }

    #[test]
    fn unknown_slash_command_returns_none() {
        assert_eq!(parse("/unknown"), None);
    }

    #[test]
    fn empty_input_returns_none() {
        assert_eq!(parse(""), None);
        assert_eq!(parse("   "), None);
    }
}
