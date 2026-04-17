/// tool_codec owns the complete wire protocol between the model and the tool layer.
///
/// Responsibilities:
///   - Parse model output text into typed ToolInput values (inbound)
///   - Format ToolOutput values into conversation text for the model (outbound)
///   - Describe the wire format to the model via format_instructions()
///
/// When the protocol format changes, only this module changes.
/// engine.rs and prompt.rs are unaffected.

use std::collections::HashMap;

use crate::tools::{EntryKind, ToolInput, ToolOutput};

// Outer tags for multi-line block tools
const WRITE_OPEN: &str = "[write_file]";
const WRITE_CLOSE: &str = "[/write_file]";
const EDIT_OPEN: &str = "[edit_file]";
const EDIT_CLOSE: &str = "[/edit_file]";

const SEARCH_DELIM: &str = "---search---";
const REPLACE_DELIM: &str = "---replace---";
const CONTENT_DELIM: &str = "---content---";
// Line-anchored form: require delimiter to appear at the start of a line
// so occurrences embedded mid-line in content are not mistaken for delimiters.
const REPLACE_LINE: &str = "\n---replace---";

// Inbound: model text -> ToolInput

/// Scans model output for all tool call types and returns typed ToolInput values
/// in document order. Malformed or unrecognized blocks are silently skipped.
pub fn parse_all_tool_inputs(text: &str) -> Vec<ToolInput> {
    let mut all: Vec<(usize, ToolInput)> = Vec::new();
    all.extend(scan_bracket_calls(text));
    all.extend(scan_edit_blocks(text));
    all.extend(scan_write_blocks(text));
    all.sort_by_key(|(pos, _)| *pos);
    all.into_iter().map(|(_, input)| input).collect()
}

/// Scans for single-line bracket calls: [read_file: path], [list_dir: path],
/// [search_code: query], [write_file: path].
/// The closing ] must appear on the same line as the opening [.
/// Note: [write_file: path] creates an empty file. Files with content use the block form.
fn scan_bracket_calls(text: &str) -> Vec<(usize, ToolInput)> {
    let mut results = Vec::new();
    let named_tools: &[(&str, &str)] = &[
        ("read_file", "[read_file:"),
        ("list_dir", "[list_dir:"),
        ("search_code", "[search_code:"),
        ("write_file", "[write_file:"),
    ];

    for (tool_name, prefix) in named_tools {
        let mut search_start = 0;
        while search_start < text.len() {
            let Some(rel) = text[search_start..].find(prefix) else { break };
            let open_abs = search_start + rel;
            let after_colon = open_abs + prefix.len();

            let Some(bracket_rel) = text[after_colon..].find(']') else { break };
            let bracket_abs = after_colon + bracket_rel;

            let arg_text = &text[after_colon..bracket_abs];
            // Reject if a newline appears before ]
            if arg_text.contains('\n') {
                search_start = after_colon;
                continue;
            }

            let arg = arg_text.trim();
            if let Some(input) = make_bracket_input(tool_name, arg) {
                results.push((open_abs, input));
            }
            search_start = bracket_abs + 1;
        }
    }

    results
}

fn make_bracket_input(tool_name: &str, arg: &str) -> Option<ToolInput> {
    match tool_name {
        "read_file" if !arg.is_empty() => Some(ToolInput::ReadFile { path: arg.to_string() }),
        "list_dir" => Some(ToolInput::ListDir {
            path: if arg.is_empty() { ".".to_string() } else { arg.to_string() },
        }),
        "search_code" if !arg.is_empty() => Some(ToolInput::SearchCode {
            query: arg.to_string(),
            path: None,
        }),
        "write_file" if !arg.is_empty() => {
            let path = arg.strip_prefix("path=").unwrap_or(arg).trim().to_string();
            if path.is_empty() {
                return None;
            }
            Some(ToolInput::WriteFile { path, content: String::new() })
        }
        _ => None,
    }
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

// Outbound: ToolOutput -> conversation text

/// Returns a compact one-line summary of a tool result for TUI display.
/// This is separate from format_tool_result, which produces the full conversation text.
pub fn render_compact_summary(output: &ToolOutput) -> String {
    match output {
        ToolOutput::FileContents(f) => {
            if f.truncated {
                format!("read {} ({} lines, truncated)", f.path, f.total_lines)
            } else {
                format!("read {} ({} lines)", f.path, f.total_lines)
            }
        }
        ToolOutput::DirectoryListing(d) => {
            format!("listed {} ({} entries)", d.path, d.entries.len())
        }
        ToolOutput::SearchResults(s) => {
            let count = s.matches.len();
            if count == 0 {
                format!("no matches for '{}'", s.query)
            } else {
                format!("found {} match(es) for '{}'", count, s.query)
            }
        }
        ToolOutput::EditFile(e) => {
            format!("replaced {} line(s) in {}", e.lines_replaced, e.path)
        }
        ToolOutput::WriteFile(w) => {
            let verb = if w.created { "created" } else { "overwrote" };
            format!("{} {} ({} bytes)", verb, w.path, w.bytes_written)
        }
    }
}

/// Formats a successful tool result for insertion into the conversation.
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
                let shown = f.contents.lines().count();
                let remaining = f.total_lines.saturating_sub(shown);
                format!(
                    "[{total} lines — showing first {shown}]\n{contents}\n[truncated: {remaining} lines not shown]",
                    total = f.total_lines,
                    contents = f.contents,
                )
            } else {
                format!("[{} lines]\n{}", f.total_lines, f.contents)
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
        ToolOutput::EditFile(e) => {
            format!("replaced {} line(s) in {}", e.lines_replaced, e.path)
        }
        ToolOutput::WriteFile(w) => {
            let verb = if w.created { "created" } else { "overwrote" };
            format!("{} {} ({} bytes)", verb, w.path, w.bytes_written)
        }
    }
}

// Protocol guard

/// Returns true if the text contains a fabricated tool result or error block.
/// Assistant output must never contain these — they are runtime-injected only.
/// Used by the engine to detect and surface model misbehavior rather than
/// silently accepting a fabricated result as a valid direct answer.
pub fn contains_fabricated_exchange(text: &str) -> bool {
    text.contains("[tool_result:") || text.contains("[tool_error:")
}

// Protocol description

/// Returns the format instructions block that prompt.rs includes in the system prompt.
/// Keeping this here ensures the prompt's description always matches the actual
/// formats that the scanners expect and format_tool_result produces.
pub fn format_instructions() -> &'static str {
    r#"TOOL USE RULES — read carefully:

Your role: emit tool CALL TAGS only. The system executes them and returns results.
You do NOT produce file contents, directory listings, or search results yourself.
You do NOT write result blocks. Result blocks are written by the system, not you.

When a tool is needed, your ENTIRE response must be the call tag only — no prose, no prefix, no explanation.

Request a file read:
[read_file: path/to/file.rs]

List a directory:
[list_dir: src/]

Search code:
[search_code: pattern]

Edit a file:
[edit_file]
path: path/to/file.rs
---search---
exact text to find
---replace---
replacement text
[/edit_file]

Create an empty file (simple form):
[write_file: path/to/file.rs]

Create or overwrite a file with content:
[write_file]
path: path/to/file.rs
---content---
full file content
[/write_file]

When you have enough information, respond directly in plain text with no tool tags."#
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    // Single-line bracket calls

    #[test]
    fn parses_read_file_call() {
        let text = "[read_file: src/main.rs]";
        let calls = parse_all_tool_inputs(text);
        assert_eq!(calls.len(), 1);
        assert!(matches!(&calls[0], ToolInput::ReadFile { path } if path == "src/main.rs"));
    }

    #[test]
    fn parses_list_dir_call() {
        let text = "[list_dir: src/]";
        let calls = parse_all_tool_inputs(text);
        assert_eq!(calls.len(), 1);
        assert!(matches!(&calls[0], ToolInput::ListDir { path } if path == "src/"));
    }

    #[test]
    fn list_dir_defaults_path_when_empty() {
        let text = "[list_dir: ]";
        let calls = parse_all_tool_inputs(text);
        assert_eq!(calls.len(), 1);
        assert!(matches!(&calls[0], ToolInput::ListDir { path } if path == "."));
    }

    #[test]
    fn parses_search_code_call() {
        let text = "[search_code: fn main]";
        let calls = parse_all_tool_inputs(text);
        assert_eq!(calls.len(), 1);
        assert!(matches!(&calls[0], ToolInput::SearchCode { query, path: None }
            if query == "fn main"));
    }

    #[test]
    fn parses_multiple_bracket_calls_in_response() {
        let text = "Let me check.\n[read_file: a.rs]\nAnd also:\n[list_dir: src/]";
        let calls = parse_all_tool_inputs(text);
        assert_eq!(calls.len(), 2);
        assert!(matches!(&calls[0], ToolInput::ReadFile { path } if path == "a.rs"));
        assert!(matches!(&calls[1], ToolInput::ListDir { path } if path == "src/"));
    }

    #[test]
    fn read_file_missing_arg_is_skipped() {
        let text = "[read_file: ]";
        assert!(parse_all_tool_inputs(text).is_empty());
    }

    #[test]
    fn bracket_call_newline_before_close_is_rejected() {
        let text = "[read_file: src/main.rs\n]";
        assert!(parse_all_tool_inputs(text).is_empty());
    }

    #[test]
    fn path_may_contain_colon() {
        let text = "[read_file: /home/user/project/src/main.rs]";
        let calls = parse_all_tool_inputs(text);
        assert_eq!(calls.len(), 1);
        assert!(
            matches!(&calls[0], ToolInput::ReadFile { path } if path == "/home/user/project/src/main.rs")
        );
    }

    #[test]
    fn returns_empty_on_no_tool_calls() {
        assert!(parse_all_tool_inputs("Just a normal response.").is_empty());
    }

    // [write_file] blocks

    #[test]
    fn parses_valid_write_block() {
        let text = "[write_file]\npath: src/new.rs\n---content---\npub fn hello() {}\n[/write_file]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        assert!(matches!(&inputs[0], ToolInput::WriteFile { path, content }
            if path == "src/new.rs" && content == "pub fn hello() {}"));
    }

    #[test]
    fn write_block_missing_content_delimiter_is_skipped() {
        let text = "[write_file]\npath: src/new.rs\npub fn hello() {}\n[/write_file]";
        assert!(parse_all_tool_inputs(text).is_empty());
    }

    #[test]
    fn write_block_missing_close_tag_is_skipped() {
        let text = "[write_file]\npath: src/new.rs\n---content---\ncontent";
        assert!(parse_all_tool_inputs(text).is_empty());
    }

    #[test]
    fn write_block_preserves_multiline_content() {
        let text = "[write_file]\npath: src/new.rs\n---content---\nuse std::fs;\n\npub fn hello() {\n    println!(\"hi\");\n}\n[/write_file]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        let ToolInput::WriteFile { content, .. } = &inputs[0] else {
            panic!("expected WriteFile");
        };
        assert!(content.contains("use std::fs;"));
        assert!(content.contains("println!(\"hi\")"));
        assert!(content.contains('\n'));
    }

    #[test]
    fn parses_write_file_bracket_form() {
        let text = "[write_file: src/new.rs]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        assert!(matches!(&inputs[0], ToolInput::WriteFile { path, content }
            if path == "src/new.rs" && content.is_empty()));
    }

    #[test]
    fn parses_write_file_bracket_form_with_path_prefix() {
        let text = "[write_file: path=src/new.rs]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        assert!(matches!(&inputs[0], ToolInput::WriteFile { path, content }
            if path == "src/new.rs" && content.is_empty()));
    }

    #[test]
    fn write_file_bracket_empty_arg_is_skipped() {
        let text = "[write_file: ]";
        assert!(parse_all_tool_inputs(text).is_empty());
    }

    #[test]
    fn write_file_bracket_path_prefix_only_is_skipped() {
        let text = "[write_file: path=]";
        assert!(parse_all_tool_inputs(text).is_empty());
    }

    #[test]
    fn write_file_bracket_and_block_coexist() {
        let text = "[write_file: empty.rs]\n[write_file]\npath: full.rs\n---content---\nhello\n[/write_file]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 2);
        assert!(matches!(&inputs[0], ToolInput::WriteFile { path, content }
            if path == "empty.rs" && content.is_empty()));
        assert!(matches!(&inputs[1], ToolInput::WriteFile { path, content }
            if path == "full.rs" && content == "hello"));
    }

    #[test]
    fn write_block_absolute_path_is_accepted() {
        // Regression: model was observed emitting absolute paths.
        let text = "[write_file]\npath: /Users/user/project/test.txt\n---content---\nhello\n[/write_file]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        assert!(matches!(&inputs[0], ToolInput::WriteFile { path, .. }
            if path == "/Users/user/project/test.txt"));
    }

    // [edit_file] blocks

    #[test]
    fn parses_valid_edit_block() {
        let text = "[edit_file]\npath: src/lib.rs\n---search---\nfn old() {}\n---replace---\nfn new() {}\n[/edit_file]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        assert!(matches!(&inputs[0], ToolInput::EditFile { path, search, replace }
            if path == "src/lib.rs" && search == "fn old() {}" && replace == "fn new() {}"));
    }

    #[test]
    fn edit_block_missing_search_delimiter_is_skipped() {
        let text = "[edit_file]\npath: src/lib.rs\n---replace---\nfn new() {}\n[/edit_file]";
        assert!(parse_all_tool_inputs(text).is_empty());
    }

    #[test]
    fn edit_block_missing_replace_delimiter_is_skipped() {
        let text = "[edit_file]\npath: src/lib.rs\n---search---\nfn old() {}\n[/edit_file]";
        assert!(parse_all_tool_inputs(text).is_empty());
    }

    #[test]
    fn edit_block_missing_close_tag_is_skipped() {
        let text = "[edit_file]\npath: src/lib.rs\n---search---\nold\n---replace---\nnew";
        assert!(parse_all_tool_inputs(text).is_empty());
    }

    #[test]
    fn edit_block_replace_delim_inside_search_content_is_handled_correctly() {
        // ---replace--- appearing mid-line inside the search text must not be treated as the delimiter.
        let text = "[edit_file]\npath: src/lib.rs\n---search---\n// see ---replace--- below\n---replace---\n// fixed\n[/edit_file]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        let ToolInput::EditFile { search, replace, .. } = &inputs[0] else {
            panic!("expected EditFile");
        };
        assert_eq!(search, "// see ---replace--- below");
        assert_eq!(replace, "// fixed");
    }

    #[test]
    fn edit_block_preserves_multiline_content() {
        let text = "[edit_file]\npath: src/lib.rs\n---search---\nfn old() {\n    println!(\"old\");\n}\n---replace---\nfn new() {\n    println!(\"new\");\n}\n[/edit_file]";
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

    // Document order across mixed call types

    #[test]
    fn mixed_blocks_preserve_document_order() {
        let text = "\
[read_file: a.rs]\n\
[edit_file]\npath: b.rs\n---search---\nold\n---replace---\nnew\n[/edit_file]\n\
[write_file]\npath: c.rs\n---content---\nhello\n[/write_file]";

        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 3);
        assert!(matches!(&inputs[0], ToolInput::ReadFile { path } if path == "a.rs"));
        assert!(matches!(&inputs[1], ToolInput::EditFile { path, .. } if path == "b.rs"));
        assert!(matches!(&inputs[2], ToolInput::WriteFile { path, .. } if path == "c.rs"));
    }

    #[test]
    fn write_before_read_in_document_order() {
        let text = "[write_file]\npath: first.rs\n---content---\nhello\n[/write_file]\n[read_file: second.rs]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 2);
        assert!(matches!(&inputs[0], ToolInput::WriteFile { path, .. } if path == "first.rs"));
        assert!(matches!(&inputs[1], ToolInput::ReadFile { path } if path == "second.rs"));
    }

    // Outbound formatting

    #[test]
    fn format_tool_result_wraps_body() {
        use crate::tools::ToolOutput;
        use crate::tools::types::FileContentsOutput;
        let output = ToolOutput::FileContents(FileContentsOutput {
            path: "x.rs".into(),
            contents: "fn main() {}".into(),
            total_lines: 1,
            truncated: false,
        });
        let result = format_tool_result("read_file", &output);
        assert!(result.starts_with("[tool_result: read_file]"));
        assert!(result.contains("[1 lines]"));
        assert!(result.contains("fn main() {}"));
        assert!(result.contains("[/tool_result]"));
    }

    #[test]
    fn render_output_includes_metadata_line_for_untruncated_file() {
        use crate::tools::ToolOutput;
        use crate::tools::types::FileContentsOutput;
        let output = ToolOutput::FileContents(FileContentsOutput {
            path: "x.rs".into(),
            contents: "line 1\nline 2".into(),
            total_lines: 2,
            truncated: false,
        });
        let body = render_output(&output);
        assert_eq!(body, "[2 lines]\nline 1\nline 2");
    }

    #[test]
    fn render_output_includes_truncation_notice_for_large_file() {
        use crate::tools::ToolOutput;
        use crate::tools::types::FileContentsOutput;
        // Simulate a 412-line file where only 200 lines are in contents
        let shown_content: String = (0..200).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
        let output = ToolOutput::FileContents(FileContentsOutput {
            path: "big.rs".into(),
            contents: shown_content,
            total_lines: 412,
            truncated: true,
        });
        let body = render_output(&output);
        assert!(body.starts_with("[412 lines — showing first 200]"), "got: {body}");
        assert!(body.contains("line 0"));
        assert!(body.ends_with("[truncated: 212 lines not shown]"), "got: {body}");
    }

    #[test]
    fn format_tool_error_wraps_message() {
        let result = format_tool_error("read_file", "file not found");
        assert!(result.starts_with("[tool_error: read_file]"));
        assert!(result.contains("file not found"));
        assert!(result.contains("[/tool_error]"));
    }

    #[test]
    fn format_instructions_mentions_all_formats() {
        let instructions = format_instructions();
        assert!(instructions.contains("[read_file:"));
        assert!(instructions.contains("[list_dir:"));
        assert!(instructions.contains("[search_code:"));
        assert!(instructions.contains("[edit_file]"));
        assert!(instructions.contains("[/edit_file]"));
        assert!(instructions.contains("[write_file:"));
        assert!(instructions.contains("[write_file]"));
        assert!(instructions.contains("[/write_file]"));
        assert!(instructions.contains("---search---"));
        assert!(instructions.contains("---replace---"));
        assert!(instructions.contains("---content---"));
    }

    #[test]
    fn format_instructions_does_not_describe_tool_result_format() {
        // [tool_result:] and [tool_error:] are runtime-only. Describing their format
        // causes the model to fabricate completed exchanges instead of issuing real calls.
        let instructions = format_instructions();
        assert!(
            !instructions.contains("Tool results are returned as"),
            "must not document the tool_result format for the model"
        );
        assert!(
            !instructions.contains("[tool_result:"),
            "must not show [tool_result:] syntax anywhere — even in a prohibition"
        );
        assert!(
            !instructions.contains("[tool_error:"),
            "must not show [tool_error:] syntax anywhere"
        );
        // Role framing must be present.
        assert!(
            instructions.contains("You do NOT produce"),
            "must include explicit role framing"
        );
    }

    #[test]
    fn contains_fabricated_exchange_detects_tool_result_blocks() {
        assert!(contains_fabricated_exchange("[tool_result: read_file]\nsome content\n[/tool_result]"));
        assert!(contains_fabricated_exchange("[tool_error: read_file]\nfailed\n[/tool_error]"));
        assert!(!contains_fabricated_exchange("[read_file: src/main.rs]"));
        assert!(!contains_fabricated_exchange("Here is my answer."));
    }
}
