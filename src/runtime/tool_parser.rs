use std::collections::HashMap;

use crate::tools::ToolInput;

const OPEN_TAG: &str = "<tool_call>";
const CLOSE_TAG: &str = "</tool_call>";

/// Scans model output for all `<tool_call>...</tool_call>` blocks and returns
/// a typed `ToolInput` for each one that is valid and recognized.
/// Unknown tool names and malformed blocks are silently skipped.
pub fn parse_tool_calls(text: &str) -> Vec<ToolInput> {
    let mut calls = Vec::new();
    let mut remaining = text;

    while let Some(open_pos) = remaining.find(OPEN_TAG) {
        let after_open = &remaining[open_pos + OPEN_TAG.len()..];
        match after_open.find(CLOSE_TAG) {
            Some(close_pos) => {
                let block = &after_open[..close_pos];
                if let Some(input) = parse_block(block) {
                    calls.push(input);
                }
                remaining = &after_open[close_pos + CLOSE_TAG.len()..];
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

/// Parses `key: value` lines into a map. Whitespace around key and value is trimmed.
/// The first `:` on each line is the separator; values may contain colons.
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let calls = parse_tool_calls(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn returns_empty_on_no_tool_calls() {
        let calls = parse_tool_calls("Just a normal response with no tool calls.");
        assert!(calls.is_empty());
    }

    #[test]
    fn ignores_unclosed_tag() {
        let text = "Some text <tool_call>\nname: read_file\npath: x.rs\n(no closing tag)";
        let calls = parse_tool_calls(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn value_may_contain_colon() {
        let text = "<tool_call>\nname: read_file\npath: /home/user/project/src/main.rs\n</tool_call>";
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert!(matches!(&calls[0], ToolInput::ReadFile { path } if path == "/home/user/project/src/main.rs"));
    }
}
