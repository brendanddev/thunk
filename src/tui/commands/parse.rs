pub(super) fn parse_command_parts(input: &str) -> (String, &str) {
    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let cmd = parts[0].to_lowercase();
    let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");
    (cmd, arg)
}

pub(crate) fn decode_slash_write_content(raw: &str) -> String {
    let mut output = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.peek().copied() {
                Some('n') => {
                    chars.next();
                    output.push('\n');
                }
                Some('t') => {
                    chars.next();
                    output.push('\t');
                }
                Some('\\') => {
                    chars.next();
                    output.push('\\');
                }
                Some('"') => {
                    chars.next();
                    output.push('"');
                }
                _ => output.push(ch),
            }
        } else {
            output.push(ch);
        }
    }

    output
}

pub(super) fn parse_slash_edit_body(arg: &str) -> Option<(String, String)> {
    let normalized = arg.trim_start();
    let first_newline = normalized.find('\n')?;
    let path = normalized[..first_newline].trim();
    let body = normalized[first_newline + 1..].trim_start_matches('\r');
    if path.is_empty() || body.trim().is_empty() {
        return None;
    }
    Some((path.to_string(), body.to_string()))
}

pub(super) fn parse_sessions_export_args(arg: &str) -> Option<(String, Option<String>)> {
    let trimmed = arg.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Some((selector, maybe_format)) = trimmed.rsplit_once(' ') {
        let format = maybe_format.trim().to_ascii_lowercase();
        if matches!(format.as_str(), "markdown" | "md" | "json") {
            return Some((selector.trim().to_string(), Some(format)));
        }
    }

    Some((trimmed.to_string(), None))
}
