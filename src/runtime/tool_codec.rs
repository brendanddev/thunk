/// tool_codec owns the complete wire protocol between the model and the tool layer.
///
/// Responsibilities:
///   - Parse model output text into typed ToolInput values (inbound)
///   - Format ToolOutput values into conversation text for the model (outbound)
///   - Describe the wire format to the model via format_instructions()
///
/// When the protocol format changes (e.g. to structured JSON or grammar-constrained
/// output in Phase 4), only this module changes. engine.rs and prompt.rs are unaffected.

use std::collections::HashMap;

use crate::tools::{EntryKind, ToolInput, ToolOutput};

const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";

// ── Inbound: model text → ToolInput ──────────────────────────────────────────

/// Scans model output for all `<tool_call>...</tool_call>` blocks and returns
/// a typed `ToolInput` for each one that is valid and recognized.
/// Unknown tool names and malformed blocks are silently skipped.
pub fn parse_tool_calls(text: &str) -> Vec<ToolInput> {
    let mut calls = Vec::new();
    let mut remaining = text;

    while let Some(open_pos) = remaining.find(CALL_OPEN) {
        let after_open = &remaining[open_pos + CALL_OPEN.len()..];
        match after_open.find(CALL_CLOSE) {
            Some(close_pos) => {
                let block = &after_open[..close_pos];
                if let Some(input) = parse_block(block) {
                    calls.push(input);
                }
                remaining = &after_open[close_pos + CALL_CLOSE.len()..];
            }
            None => break, // unclosed tag — stop scanning
        }
    }

    calls
}

fn parse_block(block: &str) -> Option<ToolInput> {
    let params = parse_kvs(block);
    match params.get("name")?.as_str() {
        "read_file" => Some(ToolInput::ReadFile {
            path: params.get("path")?.clone(),
        }),
        "list_dir" => Some(ToolInput::ListDir {
            path: params.get("path").cloned().unwrap_or_else(|| ".".to_string()),
        }),
        "search_code" => Some(ToolInput::SearchCode {
            query: params.get("query")?.clone(),
            path: params.get("path").cloned(),
        }),
        _ => None,
    }
}

/// Parses `key: value` lines into a map. The first `:` on each line is the separator;
/// values may contain further colons. Whitespace around key and value is trimmed.
fn parse_kvs(text: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(colon) = line.find(':') {
            let key = line[..colon].trim();
            let value = line[colon + 1..].trim();
            if !key.is_empty() {
                map.insert(key.to_string(), value.to_string());
            }
        }
    }
    map
}

// ── Outbound: ToolOutput → conversation text ──────────────────────────────────

/// Formats a successful tool result for insertion into the conversation.
/// The model reads this before deciding whether to continue or give a final answer.
pub fn format_tool_result(name: &str, output: &ToolOutput) -> String {
    let body = render_output(output);
    format!("[tool_result: {name}]\n{body}\n[/tool_result]\n\n")
}

/// Formats a tool dispatch error for insertion into the conversation.
pub fn format_tool_error(name: &str, error: &str) -> String {
    format!("[tool_error: {name}]\n{error}\n[/tool_error]\n\n")
}

fn render_output(output: &ToolOutput) -> String {
    match output {
        ToolOutput::FileContents(f) => {
            if f.truncated {
                format!("{}\n[file truncated at read limit]", f.contents)
            } else {
                f.contents.clone()
            }
        }
        ToolOutput::DirectoryListing(d) => {
            if d.entries.is_empty() {
                "(empty directory)".to_string()
            } else {
                d.entries
                    .iter()
                    .map(|e| {
                        let kind = match e.kind {
                            EntryKind::Dir => "dir ",
                            EntryKind::File => "file",
                            EntryKind::Symlink => "link",
                        };
                        format!("{kind}  {}", e.name)
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
        ToolOutput::SearchResults(s) => {
            if s.matches.is_empty() {
                "No matches found.".to_string()
            } else {
                let mut lines: Vec<String> = s
                    .matches
                    .iter()
                    .map(|m| format!("{}:{}: {}", m.file, m.line_number, m.line))
                    .collect();
                if s.truncated {
                    lines.push("[results truncated at match limit]".to_string());
                }
                lines.join("\n")
            }
        }
    }
}

// ── Protocol description ──────────────────────────────────────────────────────

/// Returns the format instructions block that prompt.rs includes in the system prompt.
/// Keeping this here ensures the prompt's description of the format always matches
/// the actual format that parse_tool_calls expects and format_tool_result produces.
pub fn format_instructions() -> &'static str {
    r#"To use a tool, output a tool call block in exactly this format:

<tool_call>
name: <tool_name>
<param_name>: <param_value>
</tool_call>

The tool result will be returned to you as a user message wrapped in [tool_result: name]...[/tool_result].
You may then continue your response or make further tool calls.
Only call tools when they are needed to answer the question. When you have enough information, respond directly without a tool call."#
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // — Parsing —

    #[test]
    fn parses_read_file_call() {
        let text = "<tool_call>\nname: read_file\npath: src/main.rs\n</tool_call>";
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert!(matches!(&calls[0], ToolInput::ReadFile { path } if path == "src/main.rs"));
    }

    #[test]
    fn parses_list_dir_call() {
        let text = "<tool_call>\nname: list_dir\npath: src/\n</tool_call>";
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert!(matches!(&calls[0], ToolInput::ListDir { path } if path == "src/"));
    }

    #[test]
    fn list_dir_defaults_path_when_missing() {
        let text = "<tool_call>\nname: list_dir\n</tool_call>";
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert!(matches!(&calls[0], ToolInput::ListDir { path } if path == "."));
    }

    #[test]
    fn parses_search_code_with_optional_path() {
        let text = "<tool_call>\nname: search_code\nquery: fn main\npath: src/\n</tool_call>";
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert!(matches!(&calls[0], ToolInput::SearchCode { query, path: Some(p) }
            if query == "fn main" && p == "src/"));
    }

    #[test]
    fn parses_search_code_without_path() {
        let text = "<tool_call>\nname: search_code\nquery: use std\n</tool_call>";
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert!(matches!(&calls[0], ToolInput::SearchCode { path: None, .. }));
    }

    #[test]
    fn parses_multiple_calls_in_one_response() {
        let text = "Let me check.\n\
                    <tool_call>\nname: read_file\npath: a.rs\n</tool_call>\n\
                    And also:\n\
                    <tool_call>\nname: list_dir\npath: src/\n</tool_call>";
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn skips_unknown_tool_names() {
        let text = "<tool_call>\nname: nonexistent_tool\narg: value\n</tool_call>";
        assert!(parse_tool_calls(text).is_empty());
    }

    #[test]
    fn returns_empty_on_no_tool_calls() {
        assert!(parse_tool_calls("Just a normal response with no tool calls.").is_empty());
    }

    #[test]
    fn ignores_unclosed_tag() {
        let text = "Some text <tool_call>\nname: read_file\npath: x.rs\n(no closing tag)";
        assert!(parse_tool_calls(text).is_empty());
    }

    #[test]
    fn value_may_contain_colon() {
        let text =
            "<tool_call>\nname: read_file\npath: /home/user/project/src/main.rs\n</tool_call>";
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert!(
            matches!(&calls[0], ToolInput::ReadFile { path } if path == "/home/user/project/src/main.rs")
        );
    }

    // — Formatting —

    #[test]
    fn format_tool_result_wraps_body() {
        use crate::tools::{FileContentsOutput, ToolOutput};
        let output = ToolOutput::FileContents(FileContentsOutput {
            path: "x.rs".into(),
            contents: "fn main() {}".into(),
            line_count: 1,
            truncated: false,
        });
        let result = format_tool_result("read_file", &output);
        assert!(result.starts_with("[tool_result: read_file]"));
        assert!(result.contains("fn main() {}"));
        assert!(result.contains("[/tool_result]"));
    }

    #[test]
    fn format_tool_error_wraps_message() {
        let result = format_tool_error("read_file", "file not found");
        assert!(result.starts_with("[tool_error: read_file]"));
        assert!(result.contains("file not found"));
        assert!(result.contains("[/tool_error]"));
    }

    #[test]
    fn format_instructions_mentions_both_tags() {
        let instructions = format_instructions();
        assert!(instructions.contains("<tool_call>"));
        assert!(instructions.contains("</tool_call>"));
        assert!(instructions.contains("[tool_result:"));
    }
}
