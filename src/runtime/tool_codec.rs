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
const SEARCH_CODE_OPEN: &str = "[search_code]";
const SEARCH_CODE_CLOSE: &str = "[/search_code]";

const SEARCH_DELIM: &str = "---search---";
const REPLACE_DELIM: &str = "---replace---";
const CONTENT_DELIM: &str = "---content---";
const OLD_CONTENT_LABEL: &str = "old content:";
const NEW_CONTENT_LABEL: &str = "new content:";
// Line-anchored form: require delimiter to appear at the start of a line
// so occurrences embedded mid-line in content are not mistaken for delimiters.
const REPLACE_LINE: &str = "\n---replace---";

// Inbound: model text -> ToolInput

/// Scans model output for all tool call types and returns typed ToolInput values
/// in document order. Malformed or unrecognized blocks are silently skipped.
/// Tool syntax found inside markdown code fences (``` ... ```) is excluded — those
/// are illustrative examples, not real invocations.
pub fn parse_all_tool_inputs(text: &str) -> Vec<ToolInput> {
    let fences = code_fence_ranges(text);
    let mut all: Vec<(usize, ToolInput)> = Vec::new();
    all.extend(scan_bracket_calls(text));
    all.extend(scan_edit_blocks(text));
    all.extend(scan_write_blocks(text));
    all.extend(scan_search_code_blocks(text));
    if !fences.is_empty() {
        all.retain(|(pos, _)| !fences.iter().any(|&(s, e)| *pos >= s && *pos < e));
    }
    all.sort_by_key(|(pos, _)| *pos);
    all.into_iter().map(|(_, input)| input).collect()
}

/// Returns the byte ranges (start, exclusive end) of markdown code fence blocks (``` ... ```).
/// Used to exclude tool syntax inside fences from being treated as real invocations.
fn code_fence_ranges(text: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut pos = 0;
    while pos < text.len() {
        let Some(rel) = text[pos..].find("```") else {
            break;
        };
        let open = pos + rel;
        let after_marker = open + 3;
        // Skip the optional language tag on the opening fence line (e.g. ```rust)
        let content_start = text[after_marker..]
            .find('\n')
            .map(|r| after_marker + r + 1)
            .unwrap_or(text.len());
        // Find the closing ``` — take the first one after content_start
        let Some(close_rel) = text[content_start..].find("```") else {
            break;
        };
        let close_end = content_start + close_rel + 3;
        ranges.push((open, close_end));
        pos = close_end;
    }
    ranges
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
            let Some(rel) = text[search_start..].find(prefix) else {
                break;
            };
            let open_abs = search_start + rel;
            let after_colon = open_abs + prefix.len();

            let Some(bracket_rel) = text[after_colon..].find(']') else {
                break;
            };
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
        "read_file" if !arg.is_empty() => Some(ToolInput::ReadFile {
            path: arg.to_string(),
        }),
        "list_dir" => Some(ToolInput::ListDir {
            path: if arg.is_empty() {
                ".".to_string()
            } else {
                arg.to_string()
            },
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
            Some(ToolInput::WriteFile {
                path,
                content: String::new(),
            })
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

/// Handles the block form `[search_code]\n...\n[/search_code]` that the model
/// sometimes emits when following the edit/write block pattern.
/// Extracts the query from `pattern=X`, `query=X`, or the first non-empty line.
fn scan_search_code_blocks(text: &str) -> Vec<(usize, ToolInput)> {
    let mut results = Vec::new();
    let mut remaining = text;
    let mut offset = 0usize;

    while let Some(open_pos) = remaining.find(SEARCH_CODE_OPEN) {
        let after_open = &remaining[open_pos + SEARCH_CODE_OPEN.len()..];
        match after_open.find(SEARCH_CODE_CLOSE) {
            Some(close_pos) => {
                let block = &after_open[..close_pos];
                if let Some(input) = parse_search_code_block(block) {
                    results.push((offset + open_pos, input));
                }
                let advance =
                    open_pos + SEARCH_CODE_OPEN.len() + close_pos + SEARCH_CODE_CLOSE.len();
                offset += advance;
                remaining = &remaining[advance..];
            }
            None => break,
        }
    }

    results
}

fn parse_search_code_block(block: &str) -> Option<ToolInput> {
    for line in block.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Accept `pattern=X`, `pattern: X`, `query=X`, `query: X`, or bare text.
        // Models commonly emit the colon-space form (matching kv-style formatting),
        // so both separators are tolerated.
        let query = if let Some(rest) = line.strip_prefix("pattern=") {
            rest.trim()
        } else if let Some(rest) = line.strip_prefix("pattern:") {
            rest.trim()
        } else if let Some(rest) = line.strip_prefix("query=") {
            rest.trim()
        } else if let Some(rest) = line.strip_prefix("query:") {
            rest.trim()
        } else {
            line
        };
        if !query.is_empty() {
            return Some(ToolInput::SearchCode {
                query: query.to_string(),
                path: None,
            });
        }
    }
    None
}

fn parse_edit_block(block: &str) -> Option<ToolInput> {
    if let Some(search_pos) = block.find(SEARCH_DELIM) {
        // Full form: both ---search--- and ---replace--- present.
        let after_search = &block[search_pos + SEARCH_DELIM.len()..];
        // Use the line-anchored form so ---replace--- embedded mid-line in the search
        // content (e.g. inside a comment) is not mistaken for the actual delimiter.
        let replace_nl_offset = after_search.find(REPLACE_LINE)?;
        let replace_pos = search_pos + SEARCH_DELIM.len() + replace_nl_offset + 1;

        let path = parse_kvs(&block[..search_pos]).get("path")?.clone();
        let search = trim_block_content(&after_search[..replace_nl_offset]);
        let replace = trim_block_content(&block[replace_pos + REPLACE_DELIM.len()..]);

        Some(ToolInput::EditFile {
            path,
            search,
            replace,
        })
    } else if let Some(replace_nl_pos) = block.find(REPLACE_LINE) {
        // Partial form: ---replace--- present but ---search--- absent.
        // Parse what we can and produce an empty search string. The empty-search
        // validation in edit_file.run() will surface a clear error into the conversation
        // rather than silently discarding the block as a non-tool-call.
        let path = parse_kvs(&block[..replace_nl_pos]).get("path")?.clone();
        let replace = trim_block_content(&block[replace_nl_pos + REPLACE_LINE.len()..]);
        Some(ToolInput::EditFile {
            path,
            search: String::new(),
            replace,
        })
    } else if let Some(input) = parse_edit_block_conflict_style(block) {
        // <<<<<<< SEARCH / ======= / >>>>>>> REPLACE (Aider/git conflict style)
        Some(input)
    } else if let Some(input) = parse_edit_block_labeled_content(block) {
        // old content: ... / new content: ... (observed local-model drift)
        Some(input)
    } else {
        // Generic fallback: any ---xxx--- / ---yyy--- delimiter pair.
        // Models sometimes derive delimiter names from the prompt's placeholder text
        // (e.g. ---text to find--- / ---replacement text---). Accept any valid
        // ---word(s)--- pair rather than silently falling through as a Direct response.
        parse_edit_block_generic_delimiters(block)
    }
}

/// Parses the conflict-marker style that many models emit instead of ---search---/---replace---:
///
///   <<<<<<< SEARCH
///   text to find
///   =======
///   replacement text
///   >>>>>>> REPLACE
fn parse_edit_block_conflict_style(block: &str) -> Option<ToolInput> {
    let search_marker = block.find("<<<<<<<")?;
    let path = parse_kvs(&block[..search_marker]).get("path")?.clone();

    // Skip the rest of the <<<<<<< ... opening line to reach content
    let after_marker = &block[search_marker + "<<<<<<<".len()..];
    let content_start = after_marker
        .find('\n')
        .map(|p| &after_marker[p + 1..])
        .unwrap_or(after_marker);

    // ======= separator must appear at the start of a line
    let sep_pos = content_start.find("\n=======")?;
    let search_text = trim_block_content(&content_start[..sep_pos]);

    let after_sep = &content_start[sep_pos + "\n=======".len()..];
    let after_sep = after_sep.strip_prefix('\n').unwrap_or(after_sep);

    // >>>>>>> end marker — stop before it; trailing text after >>>>>>> is ignored
    let replace_end = after_sep.find("\n>>>>>>>").unwrap_or(after_sep.len());
    let replace_text = trim_block_content(&after_sep[..replace_end]);

    Some(ToolInput::EditFile {
        path,
        search: search_text,
        replace: replace_text,
    })
}

/// Parses the narrow label style observed from local models:
///
///   old content: text to find
///   new content: replacement text
///
/// This is intentionally scoped to `edit_file` and these exact labels. It is not a
/// general key/value edit parser.
fn parse_edit_block_labeled_content(block: &str) -> Option<ToolInput> {
    let (old_line_start, old_value_start) = find_label_line(block, OLD_CONTENT_LABEL, 0)?;
    let (new_line_start, new_value_start) =
        find_label_line(block, NEW_CONTENT_LABEL, old_value_start)?;
    let path = parse_kvs(&block[..old_line_start]).get("path")?.clone();
    let search_text = trim_labeled_content(&block[old_value_start..new_line_start]);
    let replace_text = trim_labeled_content(&block[new_value_start..]);
    Some(ToolInput::EditFile {
        path,
        search: search_text,
        replace: replace_text,
    })
}

fn find_label_line(block: &str, label: &str, start_at: usize) -> Option<(usize, usize)> {
    let mut pos = 0usize;
    for raw_line in block.split_inclusive('\n') {
        if pos < start_at {
            pos += raw_line.len();
            continue;
        }

        let line = raw_line.strip_suffix('\n').unwrap_or(raw_line);
        let trimmed = line.trim_start();
        let leading = line.len() - trimmed.len();
        if trimmed.starts_with(label) {
            return Some((pos, pos + leading + label.len()));
        }
        pos += raw_line.len();
    }
    None
}

fn trim_labeled_content(s: &str) -> String {
    let s = s.trim_start_matches(|c| c == ' ' || c == '\t');
    trim_block_content(s)
}

/// Returns true for lines of the form `---word(s)---` that are not the canonical
/// `---search---`, `---replace---`, or `---content---` delimiters (those are handled
/// by the primary branches of `parse_edit_block`). The inner text must be non-empty
/// and must not itself contain `---`, which would indicate a nested or malformed marker.
fn is_triple_dash_delimiter(line: &str) -> bool {
    if !line.starts_with("---") || !line.ends_with("---") || line.len() <= 6 {
        return false;
    }
    let inner = &line[3..line.len() - 3];
    !inner.trim().is_empty() && !inner.contains("---")
}

/// Fallback parser for edit blocks that use arbitrary `---xxx---` / `---yyy---` delimiters.
///
/// Models sometimes derive delimiter names from the prompt's placeholder text rather than
/// using the canonical `---search---`/`---replace---` markers exactly as shown. For example,
/// a model might emit `---text to find---` / `---replacement text---` after reading the
/// `exact text to find` / `replacement text` examples in the instructions. This function
/// accepts any valid `---word(s)---` pair as search/replace delimiters so those blocks
/// are not silently dropped as Direct responses.
fn parse_edit_block_generic_delimiters(block: &str) -> Option<ToolInput> {
    // Collect (line_start, line_end_excl_newline) for each triple-dash delimiter line.
    let mut delimiters: Vec<(usize, usize)> = Vec::new();
    let mut pos = 0usize;
    for line in block.split('\n') {
        if is_triple_dash_delimiter(line.trim()) {
            delimiters.push((pos, pos + line.len()));
        }
        pos += line.len() + 1; // +1 for the '\n' consumed by split
    }
    if delimiters.len() < 2 {
        return None;
    }
    let (d1_start, d1_end) = delimiters[0];
    let (d2_start, d2_end) = delimiters[1];
    let path = parse_kvs(&block[..d1_start]).get("path")?.clone();
    let search_start = (d1_end + 1).min(block.len());
    let search_text = trim_block_content(&block[search_start..d2_start]);
    let replace_start = (d2_end + 1).min(block.len());
    let replace_text = trim_block_content(&block[replace_start..]);
    Some(ToolInput::EditFile {
        path,
        search: search_text,
        replace: replace_text,
    })
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
            if s.total_matches == 0 {
                format!("no matches for '{}'", s.query)
            } else if s.truncated {
                format!(
                    "found {} match(es) for '{}' (showing {})",
                    s.total_matches,
                    s.query,
                    s.matches.len()
                )
            } else {
                format!("found {} match(es) for '{}'", s.total_matches, s.query)
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
/// Uses === delimiters rather than bracket tags so the result format is visually
/// distinct from executable call syntax ([tool_name: arg]).
pub fn format_tool_result(name: &str, output: &ToolOutput) -> String {
    let body = render_output(output);
    format!("=== tool_result: {name} ===\n{body}\n=== /tool_result ===\n\n")
}

/// Formats a tool dispatch error for insertion into the conversation.
pub fn format_tool_error(name: &str, error: &str) -> String {
    format!("=== tool_error: {name} ===\n{error}\n=== /tool_error ===\n\n")
}

/// Maximum number of match lines shown per file in grouped search output.
/// Files with more hits show this many lines plus a "(N more not shown)" note.
/// Kept small so a single high-match file cannot crowd out other files in the window.
const MAX_LINES_PER_FILE: usize = 3;

/// Returns true if the file path belongs to the source tier.
/// Mirrors the tier-0 set from search_code::file_class_priority without importing it.
fn is_source_tier(path: &str) -> bool {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    matches!(
        ext,
        "rs" | "go"
            | "ts"
            | "tsx"
            | "js"
            | "jsx"
            | "py"
            | "h"
            | "hpp"
            | "sh"
            | "bash"
            | "zsh"
            | "fish"
            | "html"
            | "css"
            | "scss"
            | "sql"
            | "xml"
    )
}

/// Returns true if the line (after stripping leading whitespace) looks like a symbol definition.
/// Coverage: Rust, Python, Go, TypeScript, JavaScript.
/// C/C++ patterns are excluded — too many false positives without a type parser.
/// No regex, no scoring — prefix matching only.
pub fn looks_like_definition(line: &str) -> bool {
    let t = line.trim_start();
    // Rust
    t.starts_with("pub enum ")
        || t.starts_with("pub struct ")
        || t.starts_with("pub fn ")
        || t.starts_with("pub type ")
        || t.starts_with("pub trait ")
        || t.starts_with("pub const ")
        || t.starts_with("pub static ")
        || t.starts_with("enum ")
        || t.starts_with("struct ")
        || t.starts_with("fn ")
        || t.starts_with("type ")
        || t.starts_with("const ")
        || t.starts_with("trait ")
        || t.starts_with("impl ")
        // Python / TypeScript / JavaScript (shared keywords)
        || t.starts_with("class ")
        // Python
        || t.starts_with("def ")
        // Go
        || t.starts_with("func ")
        // TypeScript / JavaScript
        || t.starts_with("function ")
        || t.starts_with("interface ")
}

/// Returns the path of the one source-tier file whose match lines contain a definition,
/// or None if zero or more than one such file exists.
/// The single-file guard keeps the hint conservative: suppress when ambiguous.
fn definition_site_file<'a>(
    groups: &[(&'a str, Vec<&crate::tools::types::SearchMatch>)],
) -> Option<&'a str> {
    let mut found: Option<&'a str> = None;
    for (file, matches) in groups {
        if !is_source_tier(file) {
            continue;
        }
        if matches.iter().any(|m| looks_like_definition(&m.line)) {
            if found.is_some() {
                // More than one candidate — ambiguous; suppress the hint.
                return None;
            }
            found = Some(file);
        }
    }
    found
}

fn render_search_results_grouped(s: &crate::tools::types::SearchResultsOutput) -> String {
    use crate::tools::types::SearchMatch;

    let mut lines: Vec<String> = Vec::new();

    if s.truncated {
        lines.push(format!(
            "[showing first {} of {} matches — read a specific matched file with read_file]",
            s.matches.len(),
            s.total_matches
        ));
    }

    // Group consecutive matches that share the same file path.
    // Matches arrive in tier-sorted, then within-tier alphabetical order (Phase 9.0.1),
    // so same-file matches are already adjacent — a single linear pass suffices.
    let mut groups: Vec<(&str, Vec<&SearchMatch>)> = Vec::new();
    for m in &s.matches {
        match groups.last_mut() {
            Some((file, group_matches)) if *file == m.file.as_str() => {
                group_matches.push(m);
            }
            _ => groups.push((m.file.as_str(), vec![m])),
        }
    }

    // If exactly one source-tier file has a definition-like line, prepend a directive
    // so the model reads the definition site rather than a high-match usage file.
    if let Some(def_file) = definition_site_file(&groups) {
        lines.push(format!(
            "[definition found in {} — read this file first]",
            def_file
        ));
    }

    for (file, group_matches) in groups {
        let total_in_file = group_matches.len();
        let shown = total_in_file.min(MAX_LINES_PER_FILE);
        if shown < total_in_file {
            lines.push(format!(
                "{} ({} matches, showing {})",
                file, total_in_file, shown
            ));
        } else {
            lines.push(format!("{} ({} matches)", file, total_in_file));
        }
        for m in &group_matches[..shown] {
            lines.push(format!("  {}: {}", m.line_number, m.line));
        }
    }

    lines.join("\n")
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
                render_search_results_grouped(s)
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
    text.contains("=== tool_result:") || text.contains("=== tool_error:")
}

/// Returns true when an assistant response contains edit_file tag syntax (both open and close
/// tags are present) but the block could not be parsed into a valid ToolInput. This fingerprints
/// garbled edit repair attempts where the model included `[edit_file]...[/edit_file]` but used
/// unrecognized delimiter names or no delimiters at all. Used by the engine to inject a targeted
/// correction rather than silently accepting the response as a Direct answer.
pub fn contains_edit_attempt(text: &str) -> bool {
    text.contains("[edit_file]") && text.contains("[/edit_file]")
}

/// Returns true if the text contains a known tool CLOSE tag without a matching open tag.
/// This fingerprints the common drift case where the model uses a wrong opening tag
/// (e.g. `[test_file]...[/write_file]`) — the open fails to match, the close is present.
/// Used by the engine to trigger a correction instead of silently accepting the response
/// as a direct text answer.
pub fn contains_malformed_block(text: &str) -> bool {
    (text.contains("[/write_file]") && !text.contains("[write_file]"))
        || (text.contains("[/edit_file]") && !text.contains("[edit_file]"))
        || (text.contains("[/search_code]") && !text.contains("[search_code]"))
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

When a tool is needed, your ENTIRE response must be the call tag only — no prose, no fences, no explanation.

Tag names are EXACT. Do not rename, abbreviate, or invent tag names. Use only the tags shown below.

Request a file read:
[read_file: path/to/file.rs]

List a directory:
[list_dir: src/]

Search code:
[search_code: keyword]

Use search_code for any question about where something is, how something works, or what something is called in this project. Do not ask the user — search first.
Use exactly one plain literal keyword or identifier, such as logging, write_file, SessionLog, or sessions.
Do not use phrases, dots, parentheses, backslashes, regex syntax, or method-call syntax.
Emit only one search call at a time. If the results point to a specific file but do not show enough detail, read that file once with read_file, then respond. Never emit a second search_code.
Only if results are completely empty, try one different single keyword, then stop searching and respond.

Edit a file:
[edit_file]
path: path/to/file.rs
---search---
old content
---replace---
new content
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

    // Code fence filtering

    #[test]
    fn tool_call_inside_code_fence_is_not_executed() {
        // Model reproduces protocol syntax inside a code fence as an example.
        // Must not be treated as a real invocation.
        let text = "Here is how you use it:\n```\n[write_file: path/to/file.rs]\n```\nThat creates a file.";
        let calls = parse_all_tool_inputs(text);
        assert!(
            calls.is_empty(),
            "tool syntax inside code fence must not execute: {calls:?}"
        );
    }

    #[test]
    fn tool_call_inside_fenced_code_block_with_language_tag_is_not_executed() {
        let text = "Example:\n```rust\n[read_file: src/main.rs]\n```\nDone.";
        let calls = parse_all_tool_inputs(text);
        assert!(
            calls.is_empty(),
            "tool syntax inside fenced block must not execute: {calls:?}"
        );
    }

    #[test]
    fn block_tool_inside_code_fence_is_not_executed() {
        let text = "Use this form:\n```\n[write_file]\npath: foo.rs\n---content---\nhello\n[/write_file]\n```";
        let calls = parse_all_tool_inputs(text);
        assert!(
            calls.is_empty(),
            "block tool syntax inside code fence must not execute: {calls:?}"
        );
    }

    #[test]
    fn tool_call_outside_code_fence_still_executes() {
        // A real tool call that appears outside any code fence must still work.
        let text = "Let me check.\n[read_file: src/main.rs]";
        let calls = parse_all_tool_inputs(text);
        assert_eq!(calls.len(), 1, "real tool call outside fence must execute");
        assert!(matches!(&calls[0], ToolInput::ReadFile { path } if path == "src/main.rs"));
    }

    #[test]
    fn tool_call_after_code_fence_executes() {
        // Tool call appears AFTER a code fence block — not inside it.
        let text = "Some example:\n```\nfoo bar\n```\nNow for real:\n[list_dir: src/]";
        let calls = parse_all_tool_inputs(text);
        assert_eq!(calls.len(), 1, "tool call after fence must execute");
        assert!(matches!(&calls[0], ToolInput::ListDir { path } if path == "src/"));
    }

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
        assert!(
            matches!(&calls[0], ToolInput::SearchCode { query, path: None }
            if query == "fn main")
        );
    }

    #[test]
    fn parses_multiple_bracket_calls_in_response() {
        let text = "Let me check.\n[read_file: a.rs]\nAnd also:\n[list_dir: src/]";
        let calls = parse_all_tool_inputs(text);
        assert_eq!(calls.len(), 2);
        assert!(matches!(&calls[0], ToolInput::ReadFile { path } if path == "a.rs"));
        assert!(matches!(&calls[1], ToolInput::ListDir { path } if path == "src/"));
    }

    // [search_code] block form (model-drift tolerance)

    #[test]
    fn parses_search_code_block_with_pattern_prefix() {
        let text = "[search_code]\npattern=logging\n[/search_code]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        assert!(
            matches!(&inputs[0], ToolInput::SearchCode { query, path: None }
            if query == "logging")
        );
    }

    #[test]
    fn parses_search_code_block_with_pattern_colon_prefix() {
        // Model emits `pattern: log` (colon-space form) rather than `pattern=log`.
        let text = "[search_code]\npattern: log\n[/search_code]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        assert!(
            matches!(&inputs[0], ToolInput::SearchCode { query, path: None }
            if query == "log")
        );
    }

    #[test]
    fn parses_search_code_block_with_query_colon_prefix() {
        let text = "[search_code]\nquery: fn main\n[/search_code]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        assert!(
            matches!(&inputs[0], ToolInput::SearchCode { query, path: None }
            if query == "fn main")
        );
    }

    #[test]
    fn parses_search_code_block_with_query_prefix() {
        let text = "[search_code]\nquery=fn main\n[/search_code]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        assert!(
            matches!(&inputs[0], ToolInput::SearchCode { query, path: None }
            if query == "fn main")
        );
    }

    #[test]
    fn parses_search_code_block_bare_text() {
        let text = "[search_code]\nfn main\n[/search_code]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        assert!(
            matches!(&inputs[0], ToolInput::SearchCode { query, path: None }
            if query == "fn main")
        );
    }

    #[test]
    fn search_code_block_empty_body_is_skipped() {
        let text = "[search_code]\n   \n[/search_code]";
        assert!(parse_all_tool_inputs(text).is_empty());
    }

    #[test]
    fn search_code_block_missing_close_tag_is_skipped() {
        let text = "[search_code]\npattern=logging";
        assert!(parse_all_tool_inputs(text).is_empty());
    }

    #[test]
    fn search_code_bracket_and_block_both_parse() {
        let text = "[search_code: logging]\n[search_code]\npattern=tracing\n[/search_code]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 2);
        assert!(matches!(&inputs[0], ToolInput::SearchCode { query, .. } if query == "logging"));
        assert!(matches!(&inputs[1], ToolInput::SearchCode { query, .. } if query == "tracing"));
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
        let text =
            "[write_file]\npath: src/new.rs\n---content---\npub fn hello() {}\n[/write_file]";
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
        let text =
            "[write_file]\npath: /Users/user/project/test.txt\n---content---\nhello\n[/write_file]";
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
        assert!(
            matches!(&inputs[0], ToolInput::EditFile { path, search, replace }
            if path == "src/lib.rs" && search == "fn old() {}" && replace == "fn new() {}")
        );
    }

    #[test]
    fn edit_block_missing_search_delimiter_produces_empty_search() {
        // When ---search--- is absent but ---replace--- is present, the block is parsed
        // with an empty search string. The tool's run() then returns a clear error
        // ("search text must not be empty") rather than silently discarding the block.
        let text = "[edit_file]\npath: src/lib.rs\n---replace---\nfn new() {}\n[/edit_file]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        assert!(
            matches!(&inputs[0], ToolInput::EditFile { path, search, replace }
            if path == "src/lib.rs" && search.is_empty() && replace == "fn new() {}")
        );
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
        let ToolInput::EditFile {
            search, replace, ..
        } = &inputs[0]
        else {
            panic!("expected EditFile");
        };
        assert_eq!(search, "// see ---replace--- below");
        assert_eq!(replace, "// fixed");
    }

    #[test]
    fn edit_block_conflict_style_markers_are_accepted() {
        // Model emits <<<<<<< SEARCH / ======= / >>>>>>> REPLACE instead of ---search---/---replace---.
        // The parser must accept this and extract search/replace correctly.
        let text = "[edit_file]\npath: src/lib.rs\n<<<<<<< SEARCH\nfn old() {}\n=======\nfn new() {}\n>>>>>>> REPLACE\n[/edit_file]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(
            inputs.len(),
            1,
            "conflict-style edit block must parse: {inputs:?}"
        );
        assert!(
            matches!(&inputs[0], ToolInput::EditFile { path, search, replace }
            if path == "src/lib.rs" && search == "fn old() {}" && replace == "fn new() {}")
        );
    }

    #[test]
    fn edit_block_conflict_style_multiline() {
        let text = "[edit_file]\npath: src/lib.rs\n<<<<<<< SEARCH\nfn old() {\n    1\n}\n=======\nfn new() {\n    2\n}\n>>>>>>> REPLACE\n[/edit_file]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        let ToolInput::EditFile {
            search, replace, ..
        } = &inputs[0]
        else {
            panic!()
        };
        assert!(search.contains("fn old()") && search.contains("1"));
        assert!(replace.contains("fn new()") && replace.contains("2"));
    }

    #[test]
    fn edit_block_old_new_content_labels_are_accepted() {
        let text = "[edit_file]\npath: test_phase82.txt\nold content: hello world\nnew content: hello params\n[/edit_file]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        assert!(
            matches!(&inputs[0], ToolInput::EditFile { path, search, replace }
            if path == "test_phase82.txt" && search == "hello world" && replace == "hello params")
        );
    }

    #[test]
    fn edit_block_old_new_content_labels_support_multiline_values() {
        let text = "[edit_file]\npath: src/lib.rs\nold content:\nfn old() {\n    println!(\"old\");\n}\nnew content:\nfn new() {\n    println!(\"new\");\n}\n[/edit_file]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        assert!(
            matches!(&inputs[0], ToolInput::EditFile { path, search, replace }
            if path == "src/lib.rs" && search.contains("println!(\"old\")") && replace.contains("println!(\"new\")"))
        );
    }

    #[test]
    fn edit_block_generic_delimiters_accepted() {
        // Model derived delimiter names from prompt placeholder text instead of using
        // the canonical ---search---/---replace--- markers. Must still parse correctly.
        let text = "[edit_file]\npath: test_phase82.txt\n---text to find---\nhello world\n---replacement text---\nhello params\n[/edit_file]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(
            inputs.len(),
            1,
            "generic delimiter edit block must parse: {inputs:?}"
        );
        assert!(
            matches!(&inputs[0], ToolInput::EditFile { path, search, replace }
            if path == "test_phase82.txt" && search == "hello world" && replace == "hello params")
        );
    }

    #[test]
    fn edit_block_generic_delimiters_multiline_content() {
        let text = "[edit_file]\npath: src/lib.rs\n---find---\nfn old() {\n    1\n}\n---with---\nfn new() {\n    2\n}\n[/edit_file]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        let ToolInput::EditFile {
            search, replace, ..
        } = &inputs[0]
        else {
            panic!()
        };
        assert!(search.contains("fn old()") && search.contains("1"));
        assert!(replace.contains("fn new()") && replace.contains("2"));
    }

    #[test]
    fn edit_block_generic_delimiters_single_delimiter_is_skipped() {
        // Only one triple-dash delimiter — cannot determine search vs replace boundary.
        let text = "[edit_file]\npath: src/lib.rs\n---find---\nhello\n[/edit_file]";
        assert!(parse_all_tool_inputs(text).is_empty());
    }

    #[test]
    fn edit_block_preserves_multiline_content() {
        let text = "[edit_file]\npath: src/lib.rs\n---search---\nfn old() {\n    println!(\"old\");\n}\n---replace---\nfn new() {\n    println!(\"new\");\n}\n[/edit_file]";
        let inputs = parse_all_tool_inputs(text);
        assert_eq!(inputs.len(), 1);
        let ToolInput::EditFile {
            search, replace, ..
        } = &inputs[0]
        else {
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
        use crate::tools::types::FileContentsOutput;
        use crate::tools::ToolOutput;
        let output = ToolOutput::FileContents(FileContentsOutput {
            path: "x.rs".into(),
            contents: "fn main() {}".into(),
            total_lines: 1,
            truncated: false,
        });
        let result = format_tool_result("read_file", &output);
        assert!(result.starts_with("=== tool_result: read_file ==="));
        assert!(result.contains("[1 lines]"));
        assert!(result.contains("fn main() {}"));
        assert!(result.contains("=== /tool_result ==="));
    }

    #[test]
    fn render_output_includes_metadata_line_for_untruncated_file() {
        use crate::tools::types::FileContentsOutput;
        use crate::tools::ToolOutput;
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
        use crate::tools::types::FileContentsOutput;
        use crate::tools::ToolOutput;
        // Simulate a 412-line file where only 200 lines are in contents
        let shown_content: String = (0..200)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let output = ToolOutput::FileContents(FileContentsOutput {
            path: "big.rs".into(),
            contents: shown_content,
            total_lines: 412,
            truncated: true,
        });
        let body = render_output(&output);
        assert!(
            body.starts_with("[412 lines — showing first 200]"),
            "got: {body}"
        );
        assert!(body.contains("line 0"));
        assert!(
            body.ends_with("[truncated: 212 lines not shown]"),
            "got: {body}"
        );
    }

    #[test]
    fn format_tool_error_wraps_message() {
        let result = format_tool_error("read_file", "file not found");
        assert!(result.starts_with("=== tool_error: read_file ==="));
        assert!(result.contains("file not found"));
        assert!(result.contains("=== /tool_error ==="));
    }

    // SearchResults grouped rendering

    fn make_match(file: &str, line_number: usize, line: &str) -> crate::tools::types::SearchMatch {
        crate::tools::types::SearchMatch {
            file: file.to_string(),
            line_number,
            line: line.to_string(),
        }
    }

    fn make_search_output(
        matches: Vec<crate::tools::types::SearchMatch>,
        total_matches: usize,
    ) -> ToolOutput {
        use crate::tools::types::SearchResultsOutput;
        let truncated = total_matches > matches.len();
        ToolOutput::SearchResults(SearchResultsOutput {
            query: "q".into(),
            matches,
            total_matches,
            truncated,
        })
    }

    #[test]
    fn search_results_grouped_single_file_within_cap() {
        let output = make_search_output(
            vec![
                make_match("src/lib.rs", 10, "fn needle() {}"),
                make_match("src/lib.rs", 20, "let needle = 1;"),
            ],
            2,
        );
        let body = render_output(&output);
        // Header shows file with count, no "showing K" because within cap.
        assert!(
            body.contains("src/lib.rs (2 matches)"),
            "expected file header with count; got:\n{body}"
        );
        assert!(!body.contains("showing"), "no 'showing' annotation needed");
        assert!(body.contains("  10: fn needle() {}"));
        assert!(body.contains("  20: let needle = 1;"));
    }

    #[test]
    fn search_results_grouped_single_file_exceeds_per_file_cap() {
        let matches = (1..=5)
            .map(|i| make_match("src/lib.rs", i, &format!("needle line {i}")))
            .collect();
        let output = make_search_output(matches, 5);
        let body = render_output(&output);
        // 5 matches, cap is MAX_LINES_PER_FILE (3) → header says "showing 3".
        assert!(
            body.contains("src/lib.rs (5 matches, showing 3)"),
            "got:\n{body}"
        );
        assert!(body.contains("  1: needle line 1"));
        assert!(body.contains("  3: needle line 3"));
        assert!(!body.contains("  4: needle line 4"), "lines beyond cap must not appear");
    }

    #[test]
    fn search_results_grouped_multiple_files_separate_groups() {
        let output = make_search_output(
            vec![
                make_match("src/types.rs", 47, "pub enum TaskStatus {"),
                make_match("src/engine.rs", 312, "let task = Task::new();"),
                make_match("README.md", 12, "A task is a unit"),
            ],
            3,
        );
        let body = render_output(&output);
        assert!(body.contains("src/types.rs (1 matches)"), "got:\n{body}");
        assert!(body.contains("src/engine.rs (1 matches)"), "got:\n{body}");
        assert!(body.contains("README.md (1 matches)"), "got:\n{body}");
        // types.rs must appear before engine.rs, which must appear before README.md.
        let pos_types = body.find("src/types.rs").unwrap();
        let pos_engine = body.find("src/engine.rs").unwrap();
        let pos_readme = body.find("README.md").unwrap();
        assert!(pos_types < pos_engine, "file order must be preserved");
        assert!(pos_engine < pos_readme, "file order must be preserved");
    }

    #[test]
    fn search_results_grouped_truncation_notice_present_when_truncated() {
        let matches: Vec<_> = (1..=3)
            .map(|i| make_match("src/lib.rs", i, "needle"))
            .collect();
        // total_matches = 20 but only 3 are in the shown set → truncated.
        let output = make_search_output(matches, 20);
        let body = render_output(&output);
        assert!(
            body.contains("[showing first 3 of 20 matches"),
            "truncation notice must appear; got:\n{body}"
        );
        assert!(body.contains("src/lib.rs (3 matches)"));
    }

    #[test]
    fn search_results_grouped_no_matches_returns_sentinel() {
        use crate::tools::types::SearchResultsOutput;
        let output = ToolOutput::SearchResults(SearchResultsOutput {
            query: "q".into(),
            matches: vec![],
            total_matches: 0,
            truncated: false,
        });
        let body = render_output(&output);
        assert_eq!(body, "No matches found.");
    }

    #[test]
    fn search_results_grouped_within_file_line_order_preserved() {
        // Lines must appear in ascending line_number order within the file.
        let output = make_search_output(
            vec![
                make_match("src/lib.rs", 5, "first"),
                make_match("src/lib.rs", 12, "second"),
                make_match("src/lib.rs", 99, "third"),
            ],
            3,
        );
        let body = render_output(&output);
        let pos_first = body.find("  5: first").unwrap();
        let pos_second = body.find("  12: second").unwrap();
        let pos_third = body.find("  99: third").unwrap();
        assert!(pos_first < pos_second && pos_second < pos_third);
    }

    // Phase 9.2.1 — Definition Lookup Mode

    #[test]
    fn looks_like_definition_matches_rust_keywords() {
        assert!(looks_like_definition("pub enum TaskStatus {"));
        assert!(looks_like_definition("pub struct Config {"));
        assert!(looks_like_definition("pub fn run_turns("));
        assert!(looks_like_definition("pub type Result<T> ="));
        assert!(looks_like_definition("pub trait Backend {"));
        assert!(looks_like_definition("pub const MAX: usize = 10;"));
        assert!(looks_like_definition("pub static INSTANCE: Lazy<Foo>"));
        assert!(looks_like_definition("enum State {"));
        assert!(looks_like_definition("struct Inner {"));
        assert!(looks_like_definition("fn helper("));
        assert!(looks_like_definition("type Alias = u32;"));
        assert!(looks_like_definition("const CAP: usize = 50;"));
        assert!(looks_like_definition("trait Render {"));
        assert!(looks_like_definition("impl TaskStatus {"));
        // leading whitespace stripped
        assert!(looks_like_definition("    pub fn method("));
        assert!(looks_like_definition("\tfn indented("));
    }

    #[test]
    fn looks_like_definition_matches_other_languages() {
        assert!(looks_like_definition("def my_function(self):"));
        assert!(looks_like_definition("class MyService:"));
        assert!(looks_like_definition("func HandleRequest("));
        assert!(looks_like_definition("function onClick("));
        assert!(looks_like_definition("interface UserRepo {"));
        assert!(looks_like_definition("class Component extends React.Component {"));
        assert!(looks_like_definition("type Config = {"));
        assert!(looks_like_definition("const handler = ("));
    }

    #[test]
    fn looks_like_definition_rejects_usage_lines() {
        assert!(!looks_like_definition("let x = TaskStatus::Running;"));
        assert!(!looks_like_definition("task.execute();"));
        assert!(!looks_like_definition("use crate::tools::types::SearchMatch;"));
        assert!(!looks_like_definition("// pub fn commented_out("));
        assert!(!looks_like_definition("println!(\"fn not a definition\");"));
        assert!(!looks_like_definition("x.fn_call()"));
        assert!(!looks_like_definition("result = some_fn(a, b)"));
    }

    #[test]
    fn definition_preamble_fires_for_single_definition_file() {
        // One source file has a definition line; one has only usage lines.
        // Preamble must fire for the definition file.
        let output = make_search_output(
            vec![
                make_match("src/types.rs", 47, "pub enum TaskStatus {"),
                make_match("src/engine.rs", 312, "let status = TaskStatus::Running;"),
                make_match("src/engine.rs", 415, "task.set_status(status);"),
            ],
            3,
        );
        let body = render_output(&output);
        assert!(
            body.contains("[definition found in src/types.rs — read this file first]"),
            "preamble must fire for the single definition file; got:\n{body}"
        );
    }

    #[test]
    fn definition_preamble_suppressed_when_multiple_definition_files() {
        // Two source files both have definition lines → ambiguous → no preamble.
        let output = make_search_output(
            vec![
                make_match("src/types.rs", 10, "pub enum TaskStatus {"),
                make_match("src/models.rs", 5, "pub struct Task {"),
            ],
            2,
        );
        let body = render_output(&output);
        assert!(
            !body.contains("[definition found in"),
            "preamble must be suppressed when multiple files have definitions; got:\n{body}"
        );
    }

    #[test]
    fn definition_preamble_suppressed_when_no_definition_lines() {
        // All match lines are usage sites — no definition patterns → no preamble.
        let output = make_search_output(
            vec![
                make_match("src/engine.rs", 100, "task.execute();"),
                make_match("src/engine.rs", 200, "let t = Task::new();"),
            ],
            2,
        );
        let body = render_output(&output);
        assert!(
            !body.contains("[definition found in"),
            "preamble must not fire when no definition lines exist; got:\n{body}"
        );
    }

    #[test]
    fn definition_preamble_suppressed_for_docs_tier_only() {
        // README.md mentions a class definition in prose — docs tier, not source tier.
        // Preamble must not fire.
        let output = make_search_output(
            vec![make_match(
                "README.md",
                5,
                "class MyService handles all requests",
            )],
            1,
        );
        let body = render_output(&output);
        assert!(
            !body.contains("[definition found in"),
            "preamble must not fire for docs-tier files; got:\n{body}"
        );
    }

    #[test]
    fn definition_preamble_correct_when_definition_and_usage_in_source_tier() {
        // types.rs: definition line. engine.rs: usage lines only. Both source tier.
        // Preamble must name types.rs, not engine.rs.
        let output = make_search_output(
            vec![
                make_match("src/types.rs", 47, "pub struct Config {"),
                make_match("src/engine.rs", 88, "let c = Config::default();"),
                make_match("src/engine.rs", 200, "config.apply();"),
                make_match("README.md", 3, "Config is the main configuration type."),
            ],
            4,
        );
        let body = render_output(&output);
        assert!(
            body.contains("[definition found in src/types.rs — read this file first]"),
            "preamble must name the definition file, not usage files; got:\n{body}"
        );
        assert!(!body.contains("src/engine.rs — read this file first"));
    }

    #[test]
    fn definition_preamble_present_with_truncated_results() {
        // total_matches > shown → truncation notice fires.
        // If the shown set includes a definition, preamble must still fire.
        let matches = vec![
            make_match("src/types.rs", 10, "pub fn important()"),
            make_match("src/engine.rs", 50, "important();"),
            make_match("src/engine.rs", 60, "important();"),
        ];
        let output = make_search_output(matches, 50); // total=50, shown=3 → truncated
        let body = render_output(&output);
        assert!(
            body.contains("[showing first 3 of 50 matches"),
            "truncation notice must be present"
        );
        assert!(
            body.contains("[definition found in src/types.rs — read this file first]"),
            "preamble must fire even when results are truncated; got:\n{body}"
        );
    }

    #[test]
    fn definition_preamble_does_not_alter_group_rendering() {
        // The preamble is additive — existing group headers and match lines must still appear.
        let output = make_search_output(
            vec![
                make_match("src/types.rs", 47, "pub enum Status {"),
                make_match("src/engine.rs", 10, "status.run();"),
            ],
            2,
        );
        let body = render_output(&output);
        assert!(body.contains("src/types.rs (1 matches)"), "group header must still appear");
        assert!(body.contains("src/engine.rs (1 matches)"), "group header must still appear");
        assert!(body.contains("  47: pub enum Status {"), "match line must still appear");
        assert!(body.contains("  10: status.run();"), "match line must still appear");
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
        // Result/error wrappers are runtime-only. Describing their format causes the model
        // to fabricate completed exchanges instead of issuing real calls.
        let instructions = format_instructions();
        assert!(
            !instructions.contains("Tool results are returned as"),
            "must not document the result format for the model"
        );
        assert!(
            !instructions.contains("[tool_result:"),
            "must not show old bracket tool_result syntax in instructions"
        );
        assert!(
            !instructions.contains("[tool_error:"),
            "must not show old bracket tool_error syntax in instructions"
        );
        assert!(
            !instructions.contains("=== tool_result:"),
            "must not show result wrapper format in instructions — model must not learn to fabricate it"
        );
        assert!(
            !instructions.contains("=== tool_error:"),
            "must not show error wrapper format in instructions"
        );
        // Role framing must be present.
        assert!(
            instructions.contains("You do NOT produce"),
            "must include explicit role framing"
        );
    }

    #[test]
    fn contains_fabricated_exchange_detects_tool_result_blocks() {
        assert!(contains_fabricated_exchange(
            "=== tool_result: read_file ===\nsome content\n=== /tool_result ==="
        ));
        assert!(contains_fabricated_exchange(
            "=== tool_error: read_file ===\nfailed\n=== /tool_error ==="
        ));
        assert!(!contains_fabricated_exchange("[read_file: src/main.rs]"));
        assert!(!contains_fabricated_exchange("Here is my answer."));
    }

    // contains_malformed_block

    #[test]
    fn malformed_block_detected_when_close_tag_has_no_matching_open() {
        // The drift case: model used wrong opening tag, correct closing tag
        assert!(contains_malformed_block(
            "[test_file]\npath: f.txt\n---content---\nhello\n[/write_file]"
        ));
        assert!(contains_malformed_block(
            "[wrong]\npath: f.rs\n---search---\nx\n---replace---\ny\n[/edit_file]"
        ));
        assert!(contains_malformed_block(
            "[unknown]\npattern: log\n[/search_code]"
        ));
    }

    #[test]
    fn malformed_block_not_triggered_by_correct_blocks() {
        // Correctly formed blocks have both open and close tags — not malformed
        assert!(!contains_malformed_block(
            "[write_file]\npath: f.txt\n---content---\nhello\n[/write_file]"
        ));
        assert!(!contains_malformed_block(
            "[edit_file]\npath: f.rs\n---search---\nx\n---replace---\ny\n[/edit_file]"
        ));
        assert!(!contains_malformed_block(
            "[search_code]\npattern=log\n[/search_code]"
        ));
    }

    #[test]
    fn malformed_block_not_triggered_by_plain_responses() {
        assert!(!contains_malformed_block("Here is my answer."));
        assert!(!contains_malformed_block("[read_file: src/main.rs]"));
    }

    #[test]
    fn format_instructions_contains_exact_tag_warning() {
        let instructions = format_instructions();
        assert!(
            instructions.contains("Tag names are EXACT"),
            "must warn the model about exact tag names"
        );
    }
}
