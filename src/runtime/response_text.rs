use super::tool_surface::ToolSurface;

/// Injected into the conversation when a fabricated tool-result block is detected.
/// Shown to the model only; not displayed in the TUI.
/// The [runtime:correction] sentinel prefix lets session restore detect and strip these messages
/// so they do not pollute future conversation context.
pub(super) const FABRICATION_CORRECTION: &str =
    "[runtime:correction] Your response contained a result block which is forbidden. \
     You must emit ONLY a tool call tag (e.g. [read_file: path]) or answer directly in plain text. \
     Output the tool call tag now, with no other text.";

/// Injected when a search_code call is blocked by the per-turn search budget.
/// The budget allows 1 search, plus 1 retry only if the first returned no results.
pub(super) const SEARCH_BUDGET_EXCEEDED: &str =
    "[runtime:correction] search budget exceeded — you have already searched once this turn. \
     A second search is only permitted when the first returned no results. \
     Do not search again. Answer based on the information you already have.";

pub(super) const SEARCH_CLOSED_AFTER_RESULTS: &str =
    "[runtime:correction] Search returned matches. Do not call search_code again this turn. \
     Read one specific matched file with read_file before answering.";

pub(super) const SEARCH_CLOSED_AFTER_EMPTY_RETRY: &str =
    "[runtime:correction] The allowed search retry also returned no matches. \
     Do not call search_code again this turn. Answer directly that no matching code was found \
     for the searched literal keywords.";

/// Injected when an edit_file failed and the repair response contained [edit_file] tags
/// but could not be parsed (unrecognized delimiters, missing delimiters, etc.).
pub(super) const EDIT_REPAIR_CORRECTION: &str =
    "[runtime:correction] Your edit_file block could not be parsed. \
     The block requires: path: followed by ---search--- with the exact text to find, \
     then ---replace--- with the replacement text. \
     Emit the corrected [edit_file]...[/edit_file] block now with no other text.";

/// Injected when the model uses a wrong opening tag for a block tool (e.g. [test_file] instead
/// of [write_file]). Tag names are fixed — the model must use the exact names from the protocol.
pub(super) const MALFORMED_BLOCK_CORRECTION: &str =
    "[runtime:correction] Your response contained a block with an unrecognized opening tag. \
     Tag names are exact — you must use [write_file], [edit_file], etc. exactly as shown. \
     Do not rename or abbreviate them. Emit the correct tool call now with no other text.";

/// Injected when search returned matches but the model attempts synthesis without reading any file.
/// One correction is allowed per turn; after that, the runtime terminates with insufficient evidence.
pub(super) const READ_BEFORE_ANSWERING: &str =
    "[runtime:correction] Search returned matches but no matched file has been read this turn. \
     Read one of the matched files with [read_file: path] before answering.";

pub(super) const EVIDENCE_READY_ANSWER_ONLY: &str =
    "[runtime:correction] Evidence is already ready from the file(s) read this turn. \
     Do not call more tools. Answer using the existing file evidence.";

pub(super) const TURN_COMPLETE_ANSWER_ONLY: &str =
    "[runtime:correction] The file was already read this turn. \
     Do not call more tools. Provide your final answer now based on what was read.";

pub(super) fn usage_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is a usage lookup. The file just read only showed definition matches, \
         but a matched usage candidate exists. Read this exact matched usage file next with no other text: \
         [read_file: {path}]"
    )
}

pub(super) fn import_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] The file just read contained only import matches for this identifier. \
         A matched file with substantive usage or definition exists. \
         Read this exact file next with no other text: \
         [read_file: {path}]"
    )
}

pub(super) fn config_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is a config lookup. The file just read is a source file, \
         but a matched config file exists. \
         Read this exact config file next with no other text: \
         [read_file: {path}]"
    )
}

pub(super) fn initialization_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is an initialization lookup. The file just read did not show \
         an initialization match, but a matched initialization candidate exists. \
         Read this exact initialization file next with no other text: \
         [read_file: {path}]"
    )
}

pub(super) fn create_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is a creation lookup. The file just read did not show \
         a creation match, but a matched creation candidate exists. \
         Read this exact creation file next with no other text: \
         [read_file: {path}]"
    )
}

pub(super) fn register_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is a registration lookup. The file just read did not show \
         a registration match, but a matched registration candidate exists. \
         Read this exact registration file next with no other text: \
         [read_file: {path}]"
    )
}

pub(super) fn load_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is a load lookup. The file just read did not show \
         a load match, but a matched load candidate exists. \
         Read this exact load file next with no other text: \
         [read_file: {path}]"
    )
}

pub(super) fn save_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is a save lookup. The file just read did not show \
         a save match, but a matched save candidate exists. \
         Read this exact save file next with no other text: \
         [read_file: {path}]"
    )
}

pub(super) fn lockfile_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] The file just read is a lockfile, but a matched source candidate exists. \
         Read this exact matched source file next with no other text: \
         [read_file: {path}]"
    )
}

/// Injected when the question contains a code identifier but the model attempts a Direct answer
/// without any investigation. Fires at most once per turn (see direct_answer_correction_issued).
pub(super) const SEARCH_BEFORE_ANSWERING: &str =
    "[runtime:correction] This question is about a specific code element. \
     Use search_code with the identifier as the keyword before answering.";

pub(super) const READ_ONLY_TOOL_POLICY_ERROR: &str =
    "mutating tools are not allowed for this read-only informational request. \
     Do not call write_file or edit_file unless the user explicitly asks to create, write, edit, change, update, or modify a file.";

pub(super) const READ_REQUEST_TOOL_REQUIRED: &str =
    "[runtime:correction] The user asked to read a specific file. \
     Call read_file for that exact path before answering.";

/// Injected when the model tries to read a file that was already read earlier in the same turn.
/// The file's contents are already in the conversation context; re-reading adds no new evidence
/// and only inflates the prompt.
pub(super) const DUPLICATE_READ_REJECTED: &str =
    "this file was already read this turn. The contents are already in context — \
     use the existing evidence to answer.";

/// Injected when the model exceeds MAX_READS_PER_TURN in one turn.
pub(super) const READ_CAP_EXCEEDED: &str =
    "read limit for this turn reached. Answer from the file evidence already in context.";

pub(super) const CANDIDATE_READ_CAP_EXCEEDED: &str =
    "candidate read limit for this investigation reached. No additional matched files will be read.";

pub(super) const NO_LAST_READ_FILE_AVAILABLE: &str = "No previous file is available to read.";
pub(super) const NO_LAST_SEARCH_AVAILABLE: &str = "No previous search is available to repeat.";
pub(super) const NO_LAST_SCOPED_SEARCH_AVAILABLE: &str =
    "No previous scoped search is available to reuse.";
pub(super) const LAST_SEARCH_REPLAYED: &str = "Repeated the last search.";
pub(super) const LAST_SEARCH_REPLAY_FAILED: &str = "Could not repeat the previous search.";

pub(super) const LIST_DIR_BEFORE_SEARCH_BLOCKED: &str =
    "[runtime: code investigation questions require search_code, not list_dir.\nUse search_code with a keyword from the question — a function name, variable, or concept.]";

pub(super) fn git_acquisition_answer_section(name: &str, body: &str) -> String {
    format!("{name}:\n{}", body.trim_end())
}

pub(super) fn render_git_acquisition_answer(sections: Vec<String>) -> Option<String> {
    if sections.is_empty() {
        None
    } else {
        Some(format!(
            "Git read-only result:\n\n{}",
            sections.join("\n\n")
        ))
    }
}

pub(super) fn surface_policy_correction(surface: ToolSurface) -> &'static str {
    match surface {
        ToolSurface::RetrievalFirst => {
            "[runtime:correction] This turn allows retrieval tools only: search_code, read_file, list_dir. Git tools are not available."
        }
        ToolSurface::GitReadOnly => {
            "[runtime:correction] This turn allows Git read-only tools only: git_status, git_diff, git_log. Retrieval tools are not available."
        }
    }
}

pub(super) fn repeated_disallowed_tool_error(surface: ToolSurface) -> &'static str {
    match surface {
        ToolSurface::RetrievalFirst => {
            "repeated unavailable tool use for this retrieval-first turn."
        }
        ToolSurface::GitReadOnly => "repeated unavailable tool use for this Git read-only turn.",
    }
}

pub(super) fn repeated_disallowed_tool_final_answer() -> &'static str {
    "I could not continue because the model repeatedly tried to use tools that are unavailable for this request."
}

pub(super) fn repeated_tool_after_evidence_ready_final_answer() -> &'static str {
    "I could not continue because the model kept calling tools after sufficient file evidence was already read."
}

pub(super) fn repeated_tool_after_answer_phase_final_answer() -> &'static str {
    "I could not continue because the model kept calling tools after the file was already read this turn."
}

pub(super) fn mutation_complete_final_answer(tool_name: &str, summary: &str) -> String {
    format!("{tool_name} result: {summary}")
}

pub(super) fn weak_search_query_correction(reason: &str) -> String {
    format!(
        "[runtime:correction] This search query is too broad for an investigation turn ({reason}). Use a specific literal identifier or project term."
    )
}

pub(super) fn repeated_weak_search_query_final_answer() -> &'static str {
    "I could not continue because the model repeatedly used search queries that are too broad for this investigation."
}

pub(super) fn rejection_final_answer(tool_name: &str) -> &'static str {
    match tool_name {
        "write_file" => "Canceled. No file was created or changed.",
        "edit_file" => "Canceled. No file was changed.",
        _ => "Canceled. No action was taken.",
    }
}

pub(super) fn read_failure_final_answer(path: &str, error: &str) -> String {
    format!("I couldn't read `{path}`: {error}. No file contents were read.")
}

pub(super) fn read_path_mismatch_final_answer(requested: &str, attempted: &str) -> String {
    format!(
        "I couldn't read `{requested}` because the model tried to read `{attempted}` instead. No file contents were read."
    )
}

pub(super) fn unread_requested_file_final_answer(path: &str) -> String {
    format!(
        "I couldn't read `{path}` because no matching read_file result was produced. No file contents were read."
    )
}

pub(super) fn insufficient_evidence_final_answer() -> &'static str {
    "I searched for relevant code but found no matches. I don't have enough information to answer."
}

pub(super) fn ungrounded_investigation_final_answer() -> &'static str {
    "I don't have enough grounded file evidence to answer. No final answer was accepted before a matching file was read."
}
