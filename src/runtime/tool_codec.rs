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
const EDIT_OPEN: &str = "<edit_file>";
const EDIT_CLOSE: &str = "</edit_file>";
const WRITE_OPEN: &str = "<write_file>";
const WRITE_CLOSE: &str = "</write_file>";

const SEARCH_DELIM: &str = "---search---";
const REPLACE_DELIM: &str = "---replace---";
const CONTENT_DELIM: &str = "---content---";
// Line-anchored form: require delimiter to appear at the start of a line
// so occurrences embedded mid-line in content are not mistaken for delimiters.
const REPLACE_LINE: &str = "\n---replace---";

// ── Inbound: model text → ToolInput ──────────────────────────────────────────

/// Scans model output for all tool block types and returns typed ToolInput values
/// in document order. Handles `<tool_call>`, `<edit_file>`, and `<write_file>` blocks.
/// Malformed or unrecognized blocks are silently skipped.
pub fn parse_all_tool_inputs(text: &str) -> Vec<ToolInput> {
    let mut all: Vec<(usize, ToolInput)> = Vec::new();
    all.extend(scan_tool_call_blocks(text));
    all.extend(scan_edit_blocks(text));
    all.extend(scan_write_blocks(text));
    all.sort_by_key(|(pos, _)| *pos);
    all.into_iter().map(|(_, input)| input).collect()
}

/// Scans model output for `<tool_call>...</tool_call>` blocks only.
/// Preserved as-is for the existing engine.rs call site; will be replaced by
/// parse_all_tool_inputs in the engine once the approval flow lands (Step 5).
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

fn scan_tool_call_blocks(text: &str) -> Vec<(usize, ToolInput)> {
    let mut results = Vec::new();
    let mut remaining = text;
    let mut offset = 0usize;

    while let Some(open_pos) = remaining.find(CALL_OPEN) {
        let after_open = &remaining[open_pos + CALL_OPEN.len()..];
        match after_open.find(CALL_CLOSE) {
            Some(close_pos) => {
                let block = &after_open[..close_pos];
                if let Some(input) = parse_block(block) {
                    results.push((offset + open_pos, input));
                }
                let advance = open_pos + CALL_OPEN.len() + close_pos + CALL_CLOSE.len();
                offset += advance;
                remaining = &remaining[advance..];
            }
            None => break,
        }
    }

    results
}

fn scan_edit_blocks(text: &str) -> Vec<(usize, ToolInput)> {
    let mut results = Vec::new();
    let mut remaining = text;
    let mut offset = 0usize;

    while let Some(open_pos) = remaining.find(EDIT_OPEN) {
        let after_open = &remaining[open_pos + EDIT_OPEN.len()..];
        match after_open.find(EDIT_CLOSE) {
            Some(close_pos) => {
                let block = &after_open[..close_pos];
                if let Some(input) = parse_edit_block(block) {
                    results.push((offset + open_pos, input));
                }
                let advance = open_pos + EDIT_OPEN.len() + close_pos + EDIT_CLOSE.len();
                offset += advance;
                remaining = &remaining[advance..];
            }
            None => break,
        }
    }

    results
}

fn scan_write_blocks(text: &str) -> Vec<(usize, ToolInput)> {
    let mut results = Vec::new();
    let mut remaining = text;
    let mut offset = 0usize;

    while let Some(open_pos) = remaining.find(WRITE_OPEN) {
        let after_open = &remaining[open_pos + WRITE_OPEN.len()..];
        match after_open.find(WRITE_CLOSE) {
            Some(close_pos) => {
                let block = &after_open[..close_pos];
                if let Some(input) = parse_write_block(block) {
                    results.push((offset + open_pos, input));
                }
                let advance = open_pos + WRITE_OPEN.len() + close_pos + WRITE_CLOSE.len();
                offset += advance;
                remaining = &remaining[advance..];
            }
            None => break,
        }
    }

    results
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

fn parse_edit_block(block: &str) -> Option<ToolInput> {
    let search_pos = block.find(SEARCH_DELIM)?;
    let after_search = &block[search_pos + SEARCH_DELIM.len()..];
    // Use the line-anchored form so ---replace--- embedded mid-line in the search
    // content (e.g. inside a comment) is not mistaken for the actual delimiter.
    let replace_nl_offset = after_search.find(REPLACE_LINE)?;
    let replace_pos = search_pos + SEARCH_DELIM.len() + replace_nl_offset + 1;

    let path = parse_kvs(&block[..search_pos]).get("path")?.clone();
    let search = trim_block_content(&after_search[..replace_nl_offset]);
    let replace = trim_block_content(&block[replace_pos + REPLACE_DELIM.len()..]);

    Some(ToolInput::EditFile { path, search, replace })
}

fn parse_write_block(block: &str) -> Option<ToolInput> {
    let content_pos = block.find(CONTENT_DELIM)?;

    let path = parse_kvs(&block[..content_pos]).get("path")?.clone();
    let content = trim_block_content(&block[content_pos + CONTENT_DELIM.len()..]);

    Some(ToolInput::WriteFile { path, content })
}

/// Strips exactly one leading newline and one trailing newline from block content.
/// This removes the newlines that immediately follow a delimiter line and precede
/// the next delimiter or closing tag, without touching internal whitespace.
fn trim_block_content(s: &str) -> String {
    let s = s.strip_prefix('\n').unwrap_or(s);
    let s = s.strip_suffix('\n').unwrap_or(s);
    s.to_string()
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
/// Keeping this here ensures the prompt's description always matches the actual
/// formats that the scanners expect and format_tool_result produces.
pub fn format_instructions() -> &'static str {
    r#"To read files or search code, use a tool call block:

<tool_call>
name: <tool_name>
<param_name>: <param_value>
</tool_call>

To edit an existing file, use an edit block with the exact text to find and its replacement:

<edit_file>
path: path/to/file
---search---
exact text to find
---replace---
replacement text
</edit_file>

To create or overwrite a file, use a write block:

<write_file>
path: path/to/file
---content---
full file content
</write_file>

Tool results are returned as [tool_result: name]...[/tool_result].
You may continue your response or make further tool calls after receiving a result.
Only call tools when needed. When you have enough information, respond directly."#
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // — Existing <tool_call> parsing (unchanged) —

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

    // — <edit_file> parsing —

    #[test]
    fn parses_valid_edit_block() {
        let text = "<edit_file>\npath: src/lib.rs\n---search---\nfn old() {}\n---replace---\nfn new() {}\n</edit_file>";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        assert!(matches!(&inputs[0], ToolInput::EditFile { path, search, replace }
            if path == "src/lib.rs" && search == "fn old() {}" && replace == "fn new() {}"));
    }

    #[test]
    fn edit_block_missing_search_delimiter_is_skipped() {
        let text = "<edit_file>\npath: src/lib.rs\n---replace---\nfn new() {}\n</edit_file>";
        assert!(parse_all_tool_inputs(text).is_empty());
    }

    #[test]
    fn edit_block_replace_delim_inside_search_content_is_handled_correctly() {
        // ---replace--- appearing inside the search text must not be treated as the delimiter.
        let text = "<edit_file>\npath: src/lib.rs\n---search---\n// see ---replace--- below\n---replace---\n// fixed\n</edit_file>";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        let ToolInput::EditFile { search, replace, .. } = &inputs[0] else {
            panic!("expected EditFile");
        };
        assert_eq!(search, "// see ---replace--- below");
        assert_eq!(replace, "// fixed");
    }

    #[test]
    fn edit_block_missing_replace_delimiter_is_skipped() {
        let text = "<edit_file>\npath: src/lib.rs\n---search---\nfn old() {}\n</edit_file>";
        assert!(parse_all_tool_inputs(text).is_empty());
    }

    #[test]
    fn edit_block_preserves_multiline_content() {
        let text = "<edit_file>\npath: src/lib.rs\n---search---\nfn old() {\n    println!(\"old\");\n}\n---replace---\nfn new() {\n    println!(\"new\");\n}\n</edit_file>";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        let ToolInput::EditFile { search, replace, .. } = &inputs[0] else {
            panic!("expected EditFile");
        };
        assert!(search.contains("println!(\"old\")"));
        assert!(search.contains('\n'));
        assert!(replace.contains("println!(\"new\")"));
        assert!(replace.contains('\n'));
    }

    // — <write_file> parsing —

    #[test]
    fn parses_valid_write_block() {
        let text = "<write_file>\npath: src/new.rs\n---content---\npub fn hello() {}\n</write_file>";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        assert!(matches!(&inputs[0], ToolInput::WriteFile { path, content }
            if path == "src/new.rs" && content == "pub fn hello() {}"));
    }

    #[test]
    fn write_block_missing_content_delimiter_is_skipped() {
        let text = "<write_file>\npath: src/new.rs\npub fn hello() {}\n</write_file>";
        assert!(parse_all_tool_inputs(text).is_empty());
    }

    #[test]
    fn write_block_preserves_multiline_content() {
        let text = "<write_file>\npath: src/new.rs\n---content---\nuse std::fs;\n\npub fn hello() {\n    println!(\"hi\");\n}\n</write_file>";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        let ToolInput::WriteFile { content, .. } = &inputs[0] else {
            panic!("expected WriteFile");
        };
        assert!(content.contains("use std::fs;"));
        assert!(content.contains("println!(\"hi\")"));
        assert!(content.contains('\n'));
    }

    // — Document order across mixed block types —

    #[test]
    fn mixed_blocks_preserve_document_order() {
        let text = "\
<tool_call>\nname: read_file\npath: a.rs\n</tool_call>\n\
<edit_file>\npath: b.rs\n---search---\nold\n---replace---\nnew\n</edit_file>\n\
<write_file>\npath: c.rs\n---content---\nhello\n</write_file>";

        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 3);
        assert!(matches!(&inputs[0], ToolInput::ReadFile { path } if path == "a.rs"));
        assert!(matches!(&inputs[1], ToolInput::EditFile { path, .. } if path == "b.rs"));
        assert!(matches!(&inputs[2], ToolInput::WriteFile { path, .. } if path == "c.rs"));
    }

    #[test]
    fn mixed_blocks_order_is_by_position_not_type() {
        // write comes before tool_call in document order
        let text = "\
<write_file>\npath: first.rs\n---content---\nhello\n</write_file>\n\
<tool_call>\nname: read_file\npath: second.rs\n</tool_call>";

        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 2);
        assert!(matches!(&inputs[0], ToolInput::WriteFile { path, .. } if path == "first.rs"));
        assert!(matches!(&inputs[1], ToolInput::ReadFile { path } if path == "second.rs"));
    }

    // — Formatting —

    #[test]
    fn format_tool_result_wraps_body() {
        use crate::tools::ToolOutput;
        use crate::tools::types::FileContentsOutput;
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
    fn format_instructions_mentions_all_block_types() {
        let instructions = format_instructions();
        assert!(instructions.contains("<tool_call>"));
        assert!(instructions.contains("</tool_call>"));
        assert!(instructions.contains("<edit_file>"));
        assert!(instructions.contains("</edit_file>"));
        assert!(instructions.contains("<write_file>"));
        assert!(instructions.contains("</write_file>"));
        assert!(instructions.contains("---search---"));
        assert!(instructions.contains("---replace---"));
        assert!(instructions.contains("---content---"));
        assert!(instructions.contains("[tool_result:"));
    }
}
