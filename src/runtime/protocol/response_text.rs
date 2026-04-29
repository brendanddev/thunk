use super::super::investigation::tool_surface::ToolSurface;

/// Injected into the conversation when a fabricated tool-result block is detected.
/// Shown to the model only; not displayed in the TUI.
/// The [runtime:correction] sentinel prefix lets session restore detect and strip these messages
/// so they do not pollute future conversation context.
pub(crate) const FABRICATION_CORRECTION: &str =
    "[runtime:correction] Your response contained a result block which is forbidden. \
     You must emit ONLY a tool call tag (e.g. [read_file: path]) or answer directly in plain text. \
     Output the tool call tag now, with no other text.";

/// Injected when a search_code call is blocked by the per-turn search budget.
/// The budget allows 1 search, plus 1 retry only if the first returned no results.
pub(crate) const SEARCH_BUDGET_EXCEEDED: &str =
    "[runtime:correction] search budget exceeded — you have already searched once this turn. \
     A second search is only permitted when the first returned no results. \
     Do not search again. Answer based on the information you already have.";

pub(crate) const SEARCH_CLOSED_AFTER_RESULTS: &str =
    "[runtime:correction] Search returned matches. Do not call search_code again this turn. \
     Read one specific matched file with read_file before answering.";

pub(crate) const SEARCH_CLOSED_AFTER_EMPTY_RETRY: &str =
    "[runtime:correction] The allowed search retry also returned no matches. \
     Do not call search_code again this turn. Answer directly that no matching code was found \
     for the searched literal keywords.";

/// Injected when an edit_file failed and the repair response contained [edit_file] tags
/// but could not be parsed (unrecognized delimiters, missing delimiters, etc.).
pub(crate) const EDIT_REPAIR_CORRECTION: &str =
    "[runtime:correction] Your edit_file block could not be parsed. \
     The block requires: path: followed by ---search--- with the exact text to find, \
     then ---replace--- with the replacement text. \
     Emit the corrected [edit_file]...[/edit_file] block now with no other text.";

/// Injected when the model uses a wrong opening tag for a block tool (e.g. [test_file] instead
/// of [write_file]). Tag names are fixed — the model must use the exact names from the protocol.
pub(crate) const MALFORMED_BLOCK_CORRECTION: &str =
    "[runtime:correction] Your response contained a block with an unrecognized opening tag. \
     Tag names are exact — you must use [write_file], [edit_file], etc. exactly as shown. \
     Do not rename or abbreviate them. Emit the correct tool call now with no other text.";

/// Injected when an edit_file block is missing its closing [/edit_file] tag.
/// Shows the exact canonical block format so weak models know how to repair it.
pub(crate) fn malformed_edit_file_correction() -> String {
    "[runtime:correction] Your edit_file block is malformed — it is missing the closing [/edit_file] tag. \
     The exact format is:\n\
     [edit_file]\n\
     path: <file path>\n\
     ---search---\n\
     <exact text to find>\n\
     ---replace---\n\
     <replacement text>\n\
     [/edit_file]\n\
     Emit the corrected block now with no other text."
        .to_string()
}

/// Injected when a write_file block is missing its closing [/write_file] tag.
/// Shows the exact canonical block format so weak models know how to repair it.
pub(crate) fn malformed_write_file_correction() -> String {
    "[runtime:correction] Your write_file block is malformed — it is missing the closing [/write_file] tag. \
     The exact format is:\n\
     [write_file]\n\
     path: <file path>\n\
     ---content---\n\
     <file content>\n\
     [/write_file]\n\
     Emit the corrected block now with no other text."
        .to_string()
}

/// Injected when search returned matches but the model attempts synthesis without reading any file.
/// One correction is allowed per turn; after that, the runtime terminates with insufficient evidence.
pub(crate) const READ_BEFORE_ANSWERING: &str =
    "[runtime:correction] Search returned matches but no matched file has been read this turn. \
     Read one of the matched files with [read_file: path] before answering.";

pub(crate) const EVIDENCE_READY_ANSWER_ONLY: &str =
    "[runtime:correction] Evidence is already ready from the file(s) read this turn. \
     Do not call more tools. Answer using the existing file evidence.";

pub(crate) const TURN_COMPLETE_ANSWER_ONLY: &str =
    "[runtime:correction] The file was already read this turn. \
     Do not call more tools. Provide your final answer now based on what was read.";

pub(crate) fn usage_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is a usage lookup. The file just read only showed definition matches, \
         but a matched usage candidate exists. Read this exact matched usage file next with no other text: \
         [read_file: {path}]"
    )
}

pub(crate) fn import_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] The file just read contained only import matches for this identifier. \
         A matched file with substantive usage or definition exists. \
         Read this exact file next with no other text: \
         [read_file: {path}]"
    )
}

pub(crate) fn config_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is a config lookup. The file just read is a source file, \
         but a matched config file exists. \
         Read this exact config file next with no other text: \
         [read_file: {path}]"
    )
}

pub(crate) fn initialization_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is an initialization lookup. The file just read did not show \
         an initialization match, but a matched initialization candidate exists. \
         Read this exact initialization file next with no other text: \
         [read_file: {path}]"
    )
}

pub(crate) fn create_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is a creation lookup. The file just read did not show \
         a creation match, but a matched creation candidate exists. \
         Read this exact creation file next with no other text: \
         [read_file: {path}]"
    )
}

pub(crate) fn register_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is a registration lookup. The file just read did not show \
         a registration match, but a matched registration candidate exists. \
         Read this exact registration file next with no other text: \
         [read_file: {path}]"
    )
}

pub(crate) fn load_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is a load lookup. The file just read did not show \
         a load match, but a matched load candidate exists. \
         Read this exact load file next with no other text: \
         [read_file: {path}]"
    )
}

pub(crate) fn save_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is a save lookup. The file just read did not show \
         a save match, but a matched save candidate exists. \
         Read this exact save file next with no other text: \
         [read_file: {path}]"
    )
}

pub(crate) fn lockfile_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] The file just read is a lockfile, but a matched source candidate exists. \
         Read this exact matched source file next with no other text: \
         [read_file: {path}]"
    )
}

/// Injected when the question contains a code identifier but the model attempts a Direct answer
/// without any investigation. Fires at most once per turn (see direct_answer_correction_issued).
pub(crate) const SEARCH_BEFORE_ANSWERING: &str =
    "[runtime:correction] This question is about a specific code element. \
     Use search_code with the identifier as the keyword before answering.";

pub(crate)const READ_ONLY_TOOL_POLICY_ERROR: &str =
    "mutating tools are not allowed for this read-only informational request. \
     Do not call write_file or edit_file unless the user explicitly asks to create, write, edit, change, update, or modify a file.";

pub(crate) const READ_REQUEST_TOOL_REQUIRED: &str =
    "[runtime:correction] The user asked to read a specific file. \
     Call read_file for that exact path before answering.";

/// Injected when the model tries to read a file that was already read earlier in the same turn.
/// The file's contents are already in the conversation context; re-reading adds no new evidence
/// and only inflates the prompt.
pub(crate) const DUPLICATE_READ_REJECTED: &str =
    "this file was already read this turn. The contents are already in context — \
     use the existing evidence to answer.";

/// Injected when the model exceeds MAX_READS_PER_TURN in one turn.
pub(crate) const READ_CAP_EXCEEDED: &str =
    "read limit for this turn reached. Answer from the file evidence already in context.";

pub(crate)const CANDIDATE_READ_CAP_EXCEEDED: &str =
    "candidate read limit for this investigation reached. No additional matched files will be read.";

pub(crate) const NO_LAST_READ_FILE_AVAILABLE: &str = "No previous file is available to read.";
pub(crate) const NO_LAST_SEARCH_AVAILABLE: &str = "No previous search is available to repeat.";
pub(crate) const NO_LAST_SCOPED_SEARCH_AVAILABLE: &str =
    "No previous scoped search is available to reuse.";
pub(crate) const LAST_SEARCH_REPLAYED: &str = "Repeated the last search.";
pub(crate) const LAST_SEARCH_REPLAY_FAILED: &str = "Could not repeat the previous search.";

pub(crate)const LIST_DIR_BEFORE_SEARCH_BLOCKED: &str =
    "[runtime: code investigation questions require search_code, not list_dir.\nUse search_code with a keyword from the question — a function name, variable, or concept.]";

pub(crate) fn git_acquisition_answer_section(name: &str, body: &str) -> String {
    format!("{name}:\n{}", body.trim_end())
}

pub(crate) fn render_git_acquisition_answer(sections: Vec<String>) -> Option<String> {
    if sections.is_empty() {
        None
    } else {
        Some(format!(
            "Git read-only result:\n\n{}",
            sections.join("\n\n")
        ))
    }
}

pub(crate) fn surface_policy_correction(surface: ToolSurface) -> &'static str {
    match surface {
        ToolSurface::RetrievalFirst => {
            "[runtime:correction] This turn allows retrieval tools only: search_code, read_file, list_dir. Git tools are not available."
        }
        ToolSurface::GitReadOnly => {
            "[runtime:correction] This turn allows Git read-only tools only: git_status, git_diff, git_log. Retrieval tools are not available."
        }
        ToolSurface::AnswerOnly => {
            "[runtime:correction] No tools are available. Provide your final answer now."
        }
        ToolSurface::MutationEnabled => {
            "[runtime:correction] This turn allows retrieval tools and mutation tools: search_code, read_file, list_dir, edit_file, write_file. Git tools are not available."
        }
    }
}

pub(crate) fn repeated_disallowed_tool_error(surface: ToolSurface) -> &'static str {
    match surface {
        ToolSurface::RetrievalFirst => {
            "repeated unavailable tool use for this retrieval-first turn."
        }
        ToolSurface::GitReadOnly => "repeated unavailable tool use for this Git read-only turn.",
        ToolSurface::AnswerOnly => "no tools are available during answer synthesis.",
        ToolSurface::MutationEnabled => {
            "repeated unavailable tool use for this mutation-enabled turn."
        }
    }
}

pub(crate) fn repeated_disallowed_tool_final_answer() -> &'static str {
    "I could not continue because the model repeatedly tried to use tools that are unavailable for this request."
}

pub(crate) fn repeated_tool_after_evidence_ready_final_answer() -> &'static str {
    "I could not continue because the model kept calling tools after sufficient file evidence was already read."
}

pub(crate) fn repeated_tool_after_answer_phase_final_answer() -> &'static str {
    "I could not continue because the model kept calling tools after the file was already read this turn."
}

pub(crate) fn mutation_complete_final_answer(tool_name: &str, summary: &str) -> String {
    format!("{tool_name} result: {summary}")
}

pub(crate) fn weak_search_query_correction(reason: &str) -> String {
    format!(
        "[runtime:correction] This search query is too broad for an investigation turn ({reason}). Use a specific literal identifier or project term."
    )
}

pub(crate) fn repeated_weak_search_query_final_answer() -> &'static str {
    "I could not continue because the model repeatedly used search queries that are too broad for this investigation."
}

pub(crate) fn rejection_final_answer(tool_name: &str) -> &'static str {
    match tool_name {
        "write_file" => "Canceled. No file was created or changed.",
        "edit_file" => "Canceled. No file was changed.",
        _ => "Canceled. No action was taken.",
    }
}

pub(crate) fn read_failure_final_answer(path: &str, error: &str) -> String {
    format!("I couldn't read `{path}`: {error}. No file contents were read.")
}

pub(crate) fn read_path_mismatch_final_answer(requested: &str, attempted: &str) -> String {
    format!(
        "I couldn't read `{requested}` because the model tried to read `{attempted}` instead. No file contents were read."
    )
}

pub(crate) fn unread_requested_file_final_answer(path: &str) -> String {
    format!(
        "I couldn't read `{path}` because no matching read_file result was produced. No file contents were read."
    )
}

/// Fallback answer for a direct-read turn where the model repeatedly called tools instead of
/// synthesizing. Strips the tool_result wrapper so the user sees clean file content rather
/// than the model-facing protocol block.
pub(crate) fn direct_read_fallback_answer(results: &str) -> String {
    const HDR: &str = "=== tool_result: read_file ===\n";
    const FTR: &str = "=== /tool_result ===";
    let mut inner = results.trim_end_matches('\n');
    if let Some(after_header) = inner.strip_prefix(HDR) {
        inner = after_header;
    }
    if let Some(before_footer) = inner.strip_suffix(FTR) {
        inner = before_footer;
    }
    inner.trim_end_matches('\n').to_string()
}

pub(crate) fn mutation_input_rejected_final_answer(tool_name: &str, error: &str) -> String {
    format!("I couldn't complete {tool_name}: {error}. No changes were made.")
}

pub(crate) fn insufficient_evidence_final_answer() -> &'static str {
    "I searched for relevant code but found no matches. I don't have enough information to answer."
}

pub(crate) fn ungrounded_investigation_final_answer() -> &'static str {
    "I don't have enough grounded file evidence to answer. No final answer was accepted before a matching file was read."
}

/// Injected when a read_file call targets a file that was not returned by the most recent
/// search.  Fires only on investigation turns after search results exist.
/// First offense: model is corrected and may retry with a matched file.
pub(crate) fn non_candidate_read_correction(path: &str) -> String {
    format!(
        "[runtime:correction] `{path}` was not returned by the search — \
         read one of the matched files from the search results instead."
    )
}

pub(crate) fn non_candidate_read_terminal_answer() -> &'static str {
    "I could not continue because the model attempted to read a file that was not in the search results."
}
