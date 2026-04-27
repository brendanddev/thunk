/// Parsed command-line arguments for the application.
pub struct Cli {
    /// Provider name supplied via `--model <provider>`, if present.
    pub model: Option<String>,
}

impl Cli {
    pub fn parse() -> Self {
        let mut args = std::env::args().skip(1);
        let mut model = None;
        while let Some(arg) = args.next() {
            if arg == "--model" {
                model = args.next().map(|v| v.trim().to_string());
            }
        }
        Self { model }
    }
}

#[cfg(test)]
mod tests {
    use super::Cli;

    fn parse_from(args: &[&str]) -> Cli {
        // Simulate Cli::parse() against a fixed list rather than std::env::args().
        let mut iter = args.iter().map(|s| s.to_string());
        let mut model = None;
        while let Some(arg) = iter.next() {
            if arg == "--model" {
                model = iter.next().map(|v| v.trim().to_string());
            }
        }
        Cli { model }
    }

    #[test]
    fn model_flag_is_captured() {
        let cli = parse_from(&["--model", "llama_cpp"]);
        assert_eq!(cli.model.as_deref(), Some("llama_cpp"));
    }

    #[test]
    fn model_value_is_trimmed() {
        let cli = parse_from(&["--model", "  mock  "]);
        assert_eq!(cli.model.as_deref(), Some("mock"));
    }

    #[test]
    fn absent_flag_yields_none() {
        let cli = parse_from(&[]);
        assert!(cli.model.is_none());
    }

    #[test]
    fn unknown_flags_are_ignored() {
        let cli = parse_from(&["--verbose", "--model", "mock"]);
        assert_eq!(cli.model.as_deref(), Some("mock"));
    }
}
