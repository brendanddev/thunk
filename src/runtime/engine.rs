use std::collections::HashSet;
use std::path::Path;

use crate::app::config::Config;
use crate::app::Result;
use crate::llm::backend::{BackendEvent, BackendStatus, GenerateRequest, ModelBackend};
use crate::tools::{ExecutionKind, PendingAction, ToolInput, ToolRegistry, ToolRunResult};

use super::anchors::{
    has_same_scope_reference, is_last_read_file_anchor_prompt, is_last_search_anchor_prompt,
    AnchorState,
};
use super::conversation::Conversation;
#[cfg(test)]
use super::investigation::{
    contains_create_term, contains_initialization_term, contains_load_term, contains_register_term,
    contains_save_term, is_config_file, looks_like_import,
};
use super::investigation::{
    detect_investigation_mode, InvestigationMode, InvestigationState, RecoveryKind,
};
use super::prompt;
use super::tool_codec;
use super::types::{Activity, AnswerSource, RuntimeEvent, RuntimeRequest, RuntimeTerminalReason};

/// Maximum tool rounds per turn. Prevents runaway loops when the model keeps
/// producing tool calls without reaching a final answer.
const MAX_TOOL_ROUNDS: usize = 10;

/// Maximum automatic corrections per turn. One correction is enough — if the
/// model fabricates twice in a row the prompt fix is insufficient and we surface
/// the failure rather than looping silently.
const MAX_CORRECTIONS: usize = 1;

/// Injected into the conversation when a fabricated tool-result block is detected.
/// Shown to the model only; not displayed in the TUI.
/// The [runtime:correction] sentinel prefix lets session restore detect and strip these messages
/// so they do not pollute future conversation context.
const FABRICATION_CORRECTION: &str =
    "[runtime:correction] Your response contained a result block which is forbidden. \
     You must emit ONLY a tool call tag (e.g. [read_file: path]) or answer directly in plain text. \
     Output the tool call tag now, with no other text.";

/// Injected when a search_code call is blocked by the per-turn search budget.
/// The budget allows 1 search, plus 1 retry only if the first returned no results.
const SEARCH_BUDGET_EXCEEDED: &str =
    "[runtime:correction] search budget exceeded — you have already searched once this turn. \
     A second search is only permitted when the first returned no results. \
     Do not search again. Answer based on the information you already have.";

const SEARCH_CLOSED_AFTER_RESULTS: &str =
    "[runtime:correction] Search returned matches. Do not call search_code again this turn. \
     Read one specific matched file with read_file before answering.";

const SEARCH_CLOSED_AFTER_EMPTY_RETRY: &str =
    "[runtime:correction] The allowed search retry also returned no matches. \
     Do not call search_code again this turn. Answer directly that no matching code was found \
     for the searched literal keywords.";

/// Injected when an edit_file failed and the repair response contained [edit_file] tags
/// but could not be parsed (unrecognized delimiters, missing delimiters, etc.).
const EDIT_REPAIR_CORRECTION: &str =
    "[runtime:correction] Your edit_file block could not be parsed. \
     The block requires: path: followed by ---search--- with the exact text to find, \
     then ---replace--- with the replacement text. \
     Emit the corrected [edit_file]...[/edit_file] block now with no other text.";

/// Injected when the model uses a wrong opening tag for a block tool (e.g. [test_file] instead
/// of [write_file]). Tag names are fixed — the model must use the exact names from the protocol.
const MALFORMED_BLOCK_CORRECTION: &str =
    "[runtime:correction] Your response contained a block with an unrecognized opening tag. \
     Tag names are exact — you must use [write_file], [edit_file], etc. exactly as shown. \
     Do not rename or abbreviate them. Emit the correct tool call now with no other text.";

/// Injected when search returned matches but the model attempts synthesis without reading any file.
/// One correction is allowed per turn; after that, the runtime terminates with insufficient evidence.
const READ_BEFORE_ANSWERING: &str =
    "[runtime:correction] Search returned matches but no matched file has been read this turn. \
     Read one of the matched files with [read_file: path] before answering.";

fn usage_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is a usage lookup. The file just read only showed definition matches, \
         but a matched usage candidate exists. Read this exact matched usage file next with no other text: \
         [read_file: {path}]"
    )
}

fn import_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] The file just read contained only import matches for this identifier. \
         A matched file with substantive usage or definition exists. \
         Read this exact file next with no other text: \
         [read_file: {path}]"
    )
}

fn config_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is a config lookup. The file just read is a source file, \
         but a matched config file exists. \
         Read this exact config file next with no other text: \
         [read_file: {path}]"
    )
}

fn initialization_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is an initialization lookup. The file just read did not show \
         an initialization match, but a matched initialization candidate exists. \
         Read this exact initialization file next with no other text: \
         [read_file: {path}]"
    )
}

fn create_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is a creation lookup. The file just read did not show \
         a creation match, but a matched creation candidate exists. \
         Read this exact creation file next with no other text: \
         [read_file: {path}]"
    )
}

fn register_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is a registration lookup. The file just read did not show \
         a registration match, but a matched registration candidate exists. \
         Read this exact registration file next with no other text: \
         [read_file: {path}]"
    )
}

fn load_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is a load lookup. The file just read did not show \
         a load match, but a matched load candidate exists. \
         Read this exact load file next with no other text: \
         [read_file: {path}]"
    )
}

fn save_read_recovery_correction(path: &str) -> String {
    format!(
        "[runtime:correction] This is a save lookup. The file just read did not show \
         a save match, but a matched save candidate exists. \
         Read this exact save file next with no other text: \
         [read_file: {path}]"
    )
}

/// Injected when the question contains a code identifier but the model attempts a Direct answer
/// without any investigation. Fires at most once per turn (see direct_answer_correction_issued).
const SEARCH_BEFORE_ANSWERING: &str =
    "[runtime:correction] This question is about a specific code element. \
     Use search_code with the identifier as the keyword before answering.";

const READ_ONLY_TOOL_POLICY_ERROR: &str =
    "mutating tools are not allowed for this read-only informational request. \
     Do not call write_file or edit_file unless the user explicitly asks to create, write, edit, change, update, or modify a file.";

const READ_REQUEST_TOOL_REQUIRED: &str =
    "[runtime:correction] The user asked to read a specific file. \
     Call read_file for that exact path before answering.";

/// Injected when the model tries to read a file that was already read earlier in the same turn.
/// The file's contents are already in the conversation context; re-reading adds no new evidence
/// and only inflates the prompt.
const DUPLICATE_READ_REJECTED: &str =
    "this file was already read this turn. The contents are already in context — \
     use the existing evidence to answer.";

/// Maximum number of successful read_file calls allowed in a single turn.
/// Each read injects up to MAX_LINES lines into the prompt; this cap bounds worst-case
/// context growth when the model reads speculatively or drifts into repeated reads.
/// 3 is conservative: a correct investigation needs 1 (search → read → answer);
/// 2-3 accommodates a reasonable follow-up read without runaway context expansion.
const MAX_READS_PER_TURN: usize = 3;

/// Maximum number of distinct search-candidate files that may be read in a single
/// investigation turn.  After two candidate reads, if evidence is still not ready,
/// the runtime terminates cleanly rather than allowing another correction cycle.
const MAX_CANDIDATE_READS_PER_INVESTIGATION: usize = 2;

/// Injected when the model exceeds MAX_READS_PER_TURN in one turn.
const READ_CAP_EXCEEDED: &str =
    "read limit for this turn reached. Answer from the file evidence already in context.";

const CANDIDATE_READ_CAP_EXCEEDED: &str =
    "candidate read limit for this investigation reached. No additional matched files will be read.";

const NO_LAST_READ_FILE_AVAILABLE: &str = "No previous file is available to read.";
const NO_LAST_SEARCH_AVAILABLE: &str = "No previous search is available to repeat.";
const NO_LAST_SCOPED_SEARCH_AVAILABLE: &str = "No previous scoped search is available to reuse.";
const LAST_SEARCH_REPLAYED: &str = "Repeated the last search.";
const LAST_SEARCH_REPLAY_FAILED: &str = "Could not repeat the previous search.";

const LIST_DIR_BEFORE_SEARCH_BLOCKED: &str =
    "[runtime: code investigation questions require search_code, not list_dir.\nUse search_code with a keyword from the question — a function name, variable, or concept.]";

const RUNTIME_TRACE_ENV: &str = "PARAMS_TRACE_RUNTIME";

fn trace_runtime_decision(
    on_event: &mut dyn FnMut(RuntimeEvent),
    event: &str,
    fields: &[(&str, String)],
) {
    if std::env::var_os(RUNTIME_TRACE_ENV).is_none() {
        return;
    }

    let mut line = format!("[runtime:trace] event={event}");
    for (key, value) in fields {
        line.push(' ');
        line.push_str(key);
        line.push('=');
        line.push_str(&trace_field_value(value));
    }
    on_event(RuntimeEvent::RuntimeTrace(line));
}

fn trace_field_value(value: &str) -> String {
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '/' | '.' | ':' | '='))
    {
        value.to_string()
    } else {
        format!("{value:?}")
    }
}

fn normalize_evidence_path(path: &str) -> String {
    path.replace('\\', "/").trim_start_matches("./").to_string()
}

fn path_has_parent_component(path: &str) -> bool {
    path.split('/').any(|component| component == "..")
}

/// Returns true when `model_path` is within (equal to or narrower than) `scope`.
///
/// Both paths are normalized before comparison. Trailing slashes are stripped so
/// "sandbox/services/" and "sandbox/services" compare identically. The boundary
/// guard (`get(s.len()) == Some(&b'/')`) prevents "sandbox/service_extra" from
/// falsely matching scope "sandbox/service".
///
/// Absolute paths (e.g. emitted by the model as "/abs/path/") are never within
/// a relative scope and will always return false, causing the caller to clamp.
/// Parent-directory components (`..`) are also rejected structurally before
/// accepting equal-or-child scope relationships.
fn path_is_within_scope(model_path: &str, scope: &str) -> bool {
    let p = normalize_evidence_path(model_path);
    let s = normalize_evidence_path(scope);
    if path_has_parent_component(&p) || path_has_parent_component(&s) {
        return false;
    }
    let p = p.trim_end_matches('/');
    let s = s.trim_end_matches('/');
    p.starts_with(s) && (p.len() == s.len() || p.as_bytes().get(s.len()) == Some(&b'/'))
}

/// Tracks search_code usage within a single turn.
/// Rules: 1 search always permitted; a second search is permitted only when the first
/// returned zero matches; any further searches are blocked.
struct SearchBudget {
    calls: usize,
    last_was_empty: bool,
}

impl SearchBudget {
    fn new() -> Self {
        Self {
            calls: 0,
            last_was_empty: false,
        }
    }

    fn is_allowed(&self) -> bool {
        self.calls == 0 || (self.calls == 1 && self.last_was_empty)
    }

    fn record(&mut self, was_empty: bool) {
        self.calls += 1;
        self.last_was_empty = was_empty;
    }

    fn is_closed(&self) -> bool {
        self.calls >= 2 || (self.calls == 1 && !self.last_was_empty)
    }

    fn closed_message(&self) -> &'static str {
        if self.calls >= 2 && self.last_was_empty {
            SEARCH_CLOSED_AFTER_EMPTY_RETRY
        } else {
            SEARCH_CLOSED_AFTER_RESULTS
        }
    }
}

fn trace_insufficient_evidence_terminal(
    reason: &str,
    tool_rounds: usize,
    search_budget: &SearchBudget,
    investigation: &InvestigationState,
    on_event: &mut dyn FnMut(RuntimeEvent),
) {
    trace_runtime_decision(
        on_event,
        "terminal_insufficient_evidence",
        &[
            ("reason", reason.to_string()),
            ("rounds", tool_rounds.to_string()),
            ("search_calls", search_budget.calls.to_string()),
            (
                "search_produced_results",
                investigation.search_produced_results().to_string(),
            ),
            ("files_read", investigation.files_read_count().to_string()),
            (
                "candidate_reads",
                investigation.candidate_reads_count().to_string(),
            ),
            ("evidence_ready", investigation.evidence_ready().to_string()),
        ],
    );
}

/// Converts model-generated search phrases or regex/method-shaped text into the
/// single literal keyword that search_code actually supports best.
fn simplify_search_query(query: &str) -> String {
    const STOPWORDS: &[&str] = &[
        "a",
        "an",
        "and",
        "are",
        "fn",
        "for",
        "find",
        "how",
        "in",
        "initialize",
        "initialized",
        "initialization",
        "implemented",
        "is",
        "of",
        "or",
        "the",
        "to",
        "what",
        "where",
    ];

    let trimmed = query.trim();
    for raw in trimmed.split(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                '\\' | '(' | ')' | '[' | ']' | '{' | '}' | '.' | ',' | ';' | ':' | '"' | '\'' | '`'
            )
    }) {
        let token = raw.trim_matches(|c: char| !(c.is_ascii_alphanumeric() || c == '_'));
        if token.is_empty() {
            continue;
        }
        let lower = token.to_ascii_lowercase();
        if !STOPWORDS.contains(&lower.as_str()) {
            return token.to_string();
        }
    }

    trimmed.to_string()
}

fn simplify_search_input(input: &mut ToolInput) {
    if let ToolInput::SearchCode { query, .. } = input {
        let simplified = simplify_search_query(query);
        if !simplified.is_empty() && simplified != *query {
            *query = simplified;
        }
    }
}

/// Returns a stable fingerprint for a tool call, used for consecutive-cycle detection.
/// Null bytes separate fields; they cannot appear in paths, queries, or file content
/// on any supported platform, so false matches are impossible.
fn call_fingerprint(input: &ToolInput) -> String {
    match input {
        ToolInput::ReadFile { path } => format!("read_file\x00{path}"),
        ToolInput::ListDir { path } => format!("list_dir\x00{path}"),
        ToolInput::SearchCode { query, path } => {
            format!(
                "search_code\x00{query}\x00{}",
                path.as_deref().unwrap_or("")
            )
        }
        ToolInput::GitStatus => "git_status".to_string(),
        ToolInput::GitDiff => "git_diff".to_string(),
        ToolInput::EditFile {
            path,
            search,
            replace,
        } => {
            format!("edit_file\x00{path}\x00{search}\x00{replace}")
        }
        ToolInput::WriteFile { path, content } => {
            format!("write_file\x00{path}\x00{content}")
        }
    }
}

fn rejection_final_answer(tool_name: &str) -> &'static str {
    match tool_name {
        "write_file" => "Canceled. No file was created or changed.",
        "edit_file" => "Canceled. No file was changed.",
        _ => "Canceled. No action was taken.",
    }
}

fn read_failure_final_answer(path: &str, error: &str) -> String {
    format!("I couldn't read `{path}`: {error}. No file contents were read.")
}

fn read_path_mismatch_final_answer(requested: &str, attempted: &str) -> String {
    format!(
        "I couldn't read `{requested}` because the model tried to read `{attempted}` instead. No file contents were read."
    )
}

fn unread_requested_file_final_answer(path: &str) -> String {
    format!("I couldn't read `{path}` because no matching read_file result was produced. No file contents were read.")
}

fn insufficient_evidence_final_answer() -> &'static str {
    "I searched for relevant code but found no matches. I don't have enough information to answer."
}

fn ungrounded_investigation_final_answer() -> &'static str {
    "I don't have enough grounded file evidence to answer. No final answer was accepted before a matching file was read."
}

/// Returns true if the prompt contains a token that looks like a code identifier.
/// Only two structural patterns are checked — no NLP, no heuristics.
fn prompt_requires_investigation(text: &str) -> bool {
    for raw in text.split(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                ',' | '.'
                    | '?'
                    | '!'
                    | ';'
                    | ':'
                    | '"'
                    | '\''
                    | '`'
                    | '('
                    | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
            )
    }) {
        let token = raw.trim();
        if token.is_empty() {
            continue;
        }
        if is_snake_case_identifier(token) || is_pascal_case_identifier(token) {
            return true;
        }
    }

    natural_language_code_lookup_requires_investigation(text)
}

fn natural_language_code_lookup_requires_investigation(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let has_lookup_verb = contains_word(&lower, "find")
        || contains_word(&lower, "where")
        || contains_word(&lower, "locate")
        || contains_word(&lower, "search");
    if !has_lookup_verb {
        return false;
    }

    // "search" is a self-sufficient trigger — it is an explicit request to run the search tool.
    // "find/where/locate" still require a secondary condition to avoid false positives on
    // conversational phrasing like "find a good approach".
    if contains_word(&lower, "search") {
        return true;
    }

    [
        "defined",
        "implemented",
        "initialize",
        "initialized",
        "initialization",
        "initialised",
        "configured",
        "create",
        "created",
        "creation",
        "register",
        "registered",
        "registration",
        "load",
        "loaded",
        "loading",
        "save",
        "saved",
        "saving",
        "stored",
        "handled",
        "called",
        "used",
        // occurrence/appearance phrasing: "find all occurrences of X", "where it appears"
        "occur",
        "occurs",
        "occurrence",
        "occurrences",
        "appear",
        "appears",
        "filtered",
    ]
    .iter()
    .any(|term| contains_word(&lower, term))
}

fn contains_word(text: &str, needle: &str) -> bool {
    text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .any(|token| token == needle)
}

fn user_requested_mutation(text: &str) -> bool {
    text.split(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                ',' | '.'
                    | '?'
                    | '!'
                    | ';'
                    | ':'
                    | '"'
                    | '\''
                    | '`'
                    | '('
                    | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
                    | '/'
                    | '\\'
            )
    })
    .any(|token| {
        matches!(
            token.to_ascii_lowercase().as_str(),
            "add"
                | "change"
                | "create"
                | "delete"
                | "edit"
                | "modify"
                | "overwrite"
                | "replace"
                | "update"
                | "write"
        )
    })
}

/// Extracts a single relative path scope from an investigation prompt.
///
/// Fires only on the conservative pattern `in <token>` / `within <token>`, with
/// an optional `the` before the token, where the token contains `/`, has no
/// whitespace, and is not a URL. Trailing punctuation
/// that is not part of a path is stripped. Returns `None` when the pattern is absent
/// or ambiguous (multiple qualifying tokens, empty token after stripping, etc.).
///
/// Examples that match:
///   "Where is TaskStatus handled in sandbox/cli/"     → Some("sandbox/cli/")
///   "Find logging in sandbox/services/"               → Some("sandbox/services/")
///   "Find where database is configured in the sandbox/ folder" → Some("sandbox/")
///
/// Examples that do not match:
///   "Find X in the application"  → None  (no `/` in token)
///   "Find X in context"          → None  (no `/`)
///   "Find X in https://…"        → None  (URL rejected)
fn extract_investigation_path_scope(text: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    let words: Vec<&str> = text.split_whitespace().collect();
    let lower_words: Vec<&str> = lower.split_whitespace().collect();

    let mut found: Option<String> = None;

    for (i, lw) in lower_words.iter().enumerate() {
        if (*lw == "in" || *lw == "within") && i + 1 < words.len() {
            let next = i + 1;
            let path_index = if lower_words[next] == "the" && next + 1 < words.len() {
                next + 1
            } else {
                next
            };
            let raw = words[path_index];
            // Strip trailing punctuation that cannot be part of a relative path.
            let stripped = raw.trim_end_matches(|c: char| {
                matches!(
                    c,
                    '.' | ',' | '?' | '!' | ';' | ':' | ')' | ']' | '}' | '"' | '\''
                )
            });
            if stripped.is_empty() {
                continue;
            }
            // Require at least one `/` — distinguishes paths from plain words.
            if !stripped.contains('/') {
                continue;
            }
            // Reject URLs.
            if stripped.starts_with("http://") || stripped.starts_with("https://") {
                continue;
            }
            // Reject anything with embedded whitespace (shouldn't happen after split, but be safe).
            if stripped.contains(|c: char| c.is_whitespace()) {
                continue;
            }
            // More than one qualifying token → ambiguous; return None.
            if found.is_some() {
                return None;
            }
            found = Some(normalize_evidence_path(stripped));
        }
    }

    found
}

fn requested_read_path(text: &str) -> Option<String> {
    let mut tokens = text.split_whitespace();
    let first = tokens.next()?;
    if !first.eq_ignore_ascii_case("read") {
        return None;
    }

    let mut candidate = tokens.next()?;
    if candidate.eq_ignore_ascii_case("file") {
        candidate = tokens.next()?;
    }

    let path = candidate.trim_matches(|c: char| {
        matches!(
            c,
            '`' | '"' | '\'' | ',' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}'
        )
    });
    if looks_like_file_path(path) {
        Some(path.to_string())
    } else {
        None
    }
}

fn looks_like_file_path(path: &str) -> bool {
    !path.is_empty()
        && (path.contains('/')
            || path.contains('\\')
            || path.contains('.')
            || path.eq_ignore_ascii_case("README"))
}

fn is_mutating_tool(input: &ToolInput) -> bool {
    matches!(
        input,
        ToolInput::EditFile { .. } | ToolInput::WriteFile { .. }
    )
}

/// snake_case: contains underscore, ≥2 segments, each segment ≥2 alphanumeric chars.
fn is_snake_case_identifier(token: &str) -> bool {
    if !token.contains('_') {
        return false;
    }
    let segments: Vec<&str> = token.split('_').collect();
    segments.len() >= 2
        && segments
            .iter()
            .all(|s| s.len() >= 2 && s.bytes().all(|b| b.is_ascii_alphanumeric()))
}

/// Matches PascalCase/camelCase identifiers.
/// Note: also intentionally matches ALLCAPS tokens of sufficient length (e.g., DEBUG, README)
/// for Phase 8.4 structural detection.
fn is_pascal_case_identifier(token: &str) -> bool {
    if token.len() < 5 {
        return false;
    }
    let mut chars = token.chars();
    match chars.next() {
        Some(c) if c.is_ascii_uppercase() => {}
        _ => return false,
    }
    token[1..].chars().any(|c| c.is_ascii_uppercase())
}

pub struct Runtime {
    conversation: Conversation,
    backend: Box<dyn ModelBackend>,
    registry: ToolRegistry,
    system_prompt: String,
    anchors: AnchorState,
    /// Holds a mutating tool action that is waiting for user approval.
    /// Set when a tool round suspends; cleared by Approve or Reject.
    /// At most one pending action exists at any time.
    pending_action: Option<PendingAction>,
}

/// Outcome of dispatching one round of tool calls.
enum ToolRoundOutcome {
    /// All tools in this round completed immediately; results are ready to push.
    Completed { results: String },
    /// The runtime has enough information to end the turn without asking the model
    /// for another synthesis pass.
    TerminalAnswer {
        results: String,
        answer: String,
        reason: RuntimeTerminalReason,
    },
    /// A tool requested approval. Results accumulated before it are preserved.
    /// The turn is now suspended; the caller must store pending and fire the event.
    ApprovalRequired {
        accumulated: String,
        pending: PendingAction,
    },
}

impl Runtime {
    pub fn new(
        config: &Config,
        project_root: &Path,
        backend: Box<dyn ModelBackend>,
        registry: ToolRegistry,
    ) -> Self {
        let specs = registry.specs();
        let system_prompt = prompt::build_system_prompt(&config.app.name, project_root, &specs);
        Self {
            conversation: Conversation::new(system_prompt.clone()),
            backend,
            registry,
            system_prompt,
            anchors: AnchorState::default(),
            pending_action: None,
        }
    }

    /// Returns a snapshot of all current conversation messages for persistence.
    pub fn messages_snapshot(&self) -> Vec<crate::llm::backend::Message> {
        self.conversation.snapshot()
    }

    /// Appends historical messages into the conversation after the system prompt.
    /// Called once at startup when restoring a prior session. Not for use mid-turn.
    pub fn load_history(&mut self, messages: Vec<crate::llm::backend::Message>) {
        self.conversation.extend_history(messages);
    }

    /// Handles a RuntimeRequest by updating the conversation, invoking the backend,
    /// and firing RuntimeEvents to drive the UI. Each request type has its own
    /// handler method for clarity.
    pub fn handle(&mut self, request: RuntimeRequest, on_event: &mut dyn FnMut(RuntimeEvent)) {
        match request {
            RuntimeRequest::Submit { text } => self.handle_submit(text, on_event),
            RuntimeRequest::Reset => self.handle_reset(on_event),
            RuntimeRequest::Approve => self.handle_approve(on_event),
            RuntimeRequest::Reject => self.handle_reject(on_event),
        }
    }

    fn handle_reset(&mut self, on_event: &mut dyn FnMut(RuntimeEvent)) {
        self.pending_action = None;
        self.anchors.clear();
        trace_runtime_decision(
            on_event,
            "anchor_cleared",
            &[("kind", "last_read_file".into())],
        );
        trace_runtime_decision(
            on_event,
            "anchor_cleared",
            &[("kind", "last_search".into())],
        );
        self.conversation.reset(self.system_prompt.clone());
        on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
    }

    fn handle_submit(&mut self, text: String, on_event: &mut dyn FnMut(RuntimeEvent)) {
        if self.pending_action.is_some() {
            on_event(RuntimeEvent::Failed {
                message:
                    "Cannot submit while a tool approval is pending. Use /approve or /reject first."
                        .to_string(),
            });
            return;
        }

        let trimmed = text.trim();
        if trimmed.is_empty() {
            on_event(RuntimeEvent::Failed {
                message: "Cannot submit an empty prompt.".to_string(),
            });
            return;
        }

        let is_last_read_file_anchor = is_last_read_file_anchor_prompt(trimmed);
        let is_last_search_anchor = is_last_search_anchor_prompt(trimmed);
        self.conversation.push_user(text);
        on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
        if is_last_read_file_anchor {
            trace_runtime_decision(
                on_event,
                "anchor_prompt_matched",
                &[("kind", "last_read_file".into())],
            );
            if let Some(path) = self.anchors.last_read_file().map(str::to_string) {
                trace_runtime_decision(
                    on_event,
                    "anchor_resolved",
                    &[("kind", "last_read_file".into()), ("path", path.clone())],
                );
                self.run_last_read_file_anchor(path, on_event);
            } else {
                trace_runtime_decision(
                    on_event,
                    "anchor_missing",
                    &[("kind", "last_read_file".into())],
                );
                self.finish_with_runtime_answer(
                    NO_LAST_READ_FILE_AVAILABLE,
                    AnswerSource::RuntimeTerminal {
                        reason: RuntimeTerminalReason::ReadFileFailed,
                        rounds: 0,
                    },
                    on_event,
                );
            }
            return;
        }
        if is_last_search_anchor {
            trace_runtime_decision(
                on_event,
                "anchor_prompt_matched",
                &[("kind", "last_search".into())],
            );
            if let Some((query, scope)) = self.anchors.last_search() {
                trace_runtime_decision(
                    on_event,
                    "anchor_resolved",
                    &[
                        ("kind", "last_search".into()),
                        ("query", query.clone()),
                        ("scope", scope.clone().unwrap_or_else(|| "none".into())),
                    ],
                );
                self.run_last_search_anchor(query, scope, on_event);
            } else {
                trace_runtime_decision(
                    on_event,
                    "anchor_missing",
                    &[("kind", "last_search".into())],
                );
                self.finish_with_runtime_answer(
                    NO_LAST_SEARCH_AVAILABLE,
                    AnswerSource::RuntimeTerminal {
                        reason: RuntimeTerminalReason::InsufficientEvidence,
                        rounds: 0,
                    },
                    on_event,
                );
            }
            return;
        }
        self.run_turns(0, on_event);
    }

    fn run_last_read_file_anchor(&mut self, path: String, on_event: &mut dyn FnMut(RuntimeEvent)) {
        let mut last_call_key: Option<String> = None;
        let mut search_budget = SearchBudget::new();
        let mut investigation = InvestigationState::new();
        let mut reads_this_turn: HashSet<String> = HashSet::new();
        let mut requested_read_completed = false;

        on_event(RuntimeEvent::ActivityChanged(Activity::ExecutingTools));
        match run_tool_round(
            &self.registry,
            vec![ToolInput::ReadFile { path }],
            &mut last_call_key,
            &mut search_budget,
            &mut investigation,
            &mut reads_this_turn,
            &mut self.anchors,
            false,
            false,
            InvestigationMode::General,
            None,
            &mut requested_read_completed,
            None,
            on_event,
        ) {
            ToolRoundOutcome::Completed { results } => {
                self.conversation.push_user(results);
                self.conversation.trim_tool_exchanges_if_needed();
                on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
                self.run_turns_with_initial_reads(1, reads_this_turn, on_event);
            }
            ToolRoundOutcome::TerminalAnswer {
                results,
                answer,
                reason,
            } => {
                self.conversation.push_user(results);
                self.conversation.trim_tool_exchanges_if_needed();
                self.finish_with_runtime_answer(
                    &answer,
                    AnswerSource::RuntimeTerminal { reason, rounds: 1 },
                    on_event,
                );
            }
            ToolRoundOutcome::ApprovalRequired {
                accumulated,
                pending,
            } => {
                if !accumulated.is_empty() {
                    self.conversation.push_user(accumulated);
                    self.conversation.trim_tool_exchanges_if_needed();
                }
                self.pending_action = Some(pending.clone());
                on_event(RuntimeEvent::ApprovalRequired(pending));
                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
            }
        }
    }

    fn run_last_search_anchor(
        &mut self,
        query: String,
        scope: Option<String>,
        on_event: &mut dyn FnMut(RuntimeEvent),
    ) {
        let input = ToolInput::SearchCode {
            query: query.clone(),
            path: scope.clone(),
        };
        let name = input.tool_name().to_string();

        on_event(RuntimeEvent::ActivityChanged(Activity::ExecutingTools));
        on_event(RuntimeEvent::ToolCallStarted { name: name.clone() });

        match self.registry.dispatch(input) {
            Ok(ToolRunResult::Immediate(output)) => {
                debug_assert!(
                    self.registry
                        .spec_for(&name)
                        .map(|s| s.execution_kind == ExecutionKind::Immediate)
                        .unwrap_or(true),
                    "tool '{name}' returned Immediate but spec declares RequiresApproval"
                );
                if let Some((query, scope)) =
                    self.anchors
                        .record_successful_search(&output, query.clone(), scope.clone())
                {
                    trace_runtime_decision(
                        on_event,
                        "anchor_updated",
                        &[
                            ("kind", "last_search".into()),
                            ("query", query),
                            ("scope", scope.unwrap_or_else(|| "none".into())),
                        ],
                    );
                }
                let summary = tool_codec::render_compact_summary(&output);
                on_event(RuntimeEvent::ToolCallFinished {
                    name: name.clone(),
                    summary: Some(summary),
                });
                self.conversation
                    .push_user(tool_codec::format_tool_result(&name, &output));
                self.conversation.trim_tool_exchanges_if_needed();
                self.finish_with_runtime_answer(
                    LAST_SEARCH_REPLAYED,
                    AnswerSource::ToolAssisted { rounds: 1 },
                    on_event,
                );
            }
            Ok(ToolRunResult::Approval(pending)) => {
                debug_assert!(
                    self.registry
                        .spec_for(&name)
                        .map(|s| s.execution_kind == ExecutionKind::RequiresApproval)
                        .unwrap_or(false),
                    "tool '{name}' requested approval but spec declares Immediate"
                );
                self.pending_action = Some(pending.clone());
                on_event(RuntimeEvent::ApprovalRequired(pending));
                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
            }
            Err(e) => {
                on_event(RuntimeEvent::ToolCallFinished {
                    name: name.clone(),
                    summary: None,
                });
                self.conversation
                    .push_user(tool_codec::format_tool_error(&name, &e.to_string()));
                self.conversation.trim_tool_exchanges_if_needed();
                self.finish_with_runtime_answer(
                    LAST_SEARCH_REPLAY_FAILED,
                    AnswerSource::RuntimeTerminal {
                        reason: RuntimeTerminalReason::InsufficientEvidence,
                        rounds: 1,
                    },
                    on_event,
                );
            }
        }
    }

    fn handle_approve(&mut self, on_event: &mut dyn FnMut(RuntimeEvent)) {
        let pending = match self.pending_action.take() {
            Some(p) => p,
            None => {
                on_event(RuntimeEvent::Failed {
                    message: "No pending action to approve.".to_string(),
                });
                return;
            }
        };

        on_event(RuntimeEvent::ActivityChanged(Activity::ExecutingTools));
        let tool_name = pending.tool_name.clone();

        match self.registry.execute_approved(&pending) {
            Ok(output) => {
                let summary = tool_codec::render_compact_summary(&output);
                on_event(RuntimeEvent::ToolCallFinished {
                    name: tool_name.clone(),
                    summary: Some(summary),
                });
                let result_text = tool_codec::format_tool_result(&tool_name, &output);
                self.conversation.push_user(result_text);
                self.conversation.trim_tool_exchanges_if_needed();
                // Re-enter the generation loop so the model synthesizes a response
                // confirming what was done — mirrors the Immediate tool path in run_turns.
                on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
                self.run_turns(1, on_event);
            }
            Err(e) => {
                on_event(RuntimeEvent::ToolCallFinished {
                    name: tool_name.clone(),
                    summary: None,
                });
                let error_text = tool_codec::format_tool_error(&tool_name, &e.to_string());
                self.conversation.push_user(error_text);
                // On failure, let the model respond — it may want to retry.
                on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
                self.run_turns(0, on_event);
            }
        }
    }

    fn handle_reject(&mut self, on_event: &mut dyn FnMut(RuntimeEvent)) {
        let pending = match self.pending_action.take() {
            Some(p) => p,
            None => {
                on_event(RuntimeEvent::Failed {
                    message: "No pending action to reject.".to_string(),
                });
                return;
            }
        };

        let tool_name = pending.tool_name.clone();
        on_event(RuntimeEvent::ToolCallFinished {
            name: tool_name.clone(),
            summary: None,
        });
        let rejection = tool_codec::format_tool_error(
            &tool_name,
            "user rejected this action — do not retry or re-propose it. \
             Acknowledge the cancellation in plain text and wait for the user's next instruction.",
        );
        self.conversation.push_user(rejection);
        self.finish_with_runtime_answer(
            rejection_final_answer(&tool_name),
            AnswerSource::RuntimeTerminal {
                reason: RuntimeTerminalReason::RejectedMutation,
                rounds: 1,
            },
            on_event,
        );
    }

    /// Runs the generate -> tool-round loop until the model produces a final answer,
    /// the tool round limit is reached, or a tool action requires approval.
    /// `tool_rounds` is the count already consumed before this call (0 for a fresh turn).
    fn run_turns(&mut self, tool_rounds: usize, on_event: &mut dyn FnMut(RuntimeEvent)) {
        self.run_turns_with_initial_reads(tool_rounds, HashSet::new(), on_event);
    }

    fn run_turns_with_initial_reads(
        &mut self,
        mut tool_rounds: usize,
        mut reads_this_turn: HashSet<String>,
        on_event: &mut dyn FnMut(RuntimeEvent),
    ) {
        let mut corrections = 0usize;
        let mut last_call_key: Option<String> = None;
        let mut search_budget = SearchBudget::new();
        let mut investigation = InvestigationState::new();
        let mut requested_read_completed = false;
        let mut read_request_correction_issued = false;
        // Computed once from the original user message. Excludes tool result/error injections
        // and correction messages so the approve-failure path (run_turns(0,...)) is safe.
        let original_user_prompt = self.conversation.last_user_content().filter(|c| {
            !c.starts_with("=== tool_result:")
                && !c.starts_with("=== tool_error:")
                && !c.starts_with("[runtime:correction]")
        });
        let requested_read_path = original_user_prompt.and_then(requested_read_path);
        let investigation_required = original_user_prompt
            .map(|prompt| {
                requested_read_path.is_none()
                    && !user_requested_mutation(prompt)
                    && prompt_requires_investigation(prompt)
            })
            .unwrap_or(false);
        let mutation_allowed = original_user_prompt
            .map(user_requested_mutation)
            .unwrap_or(false);
        let investigation_mode = original_user_prompt
            .map(detect_investigation_mode)
            .unwrap_or(InvestigationMode::General);
        let explicit_investigation_path_scope: Option<String> = if investigation_required {
            original_user_prompt.and_then(extract_investigation_path_scope)
        } else {
            None
        };
        let same_scope_reference = investigation_required
            && explicit_investigation_path_scope.is_none()
            && original_user_prompt.is_some_and(has_same_scope_reference);
        let investigation_path_scope: Option<String> =
            if let Some(scope) = explicit_investigation_path_scope {
                Some(scope)
            } else if same_scope_reference {
                trace_runtime_decision(
                    on_event,
                    "anchor_prompt_matched",
                    &[("kind", "same_scope".into())],
                );
                match self.anchors.last_scoped_search_scope().map(str::to_string) {
                    Some(scope) => {
                        trace_runtime_decision(
                            on_event,
                            "anchor_resolved",
                            &[("kind", "same_scope".into()), ("scope", scope.clone())],
                        );
                        Some(scope)
                    }
                    None => {
                        trace_runtime_decision(
                            on_event,
                            "anchor_missing",
                            &[("kind", "same_scope".into())],
                        );
                        self.finish_with_runtime_answer(
                            NO_LAST_SCOPED_SEARCH_AVAILABLE,
                            AnswerSource::RuntimeTerminal {
                                reason: RuntimeTerminalReason::InsufficientEvidence,
                                rounds: tool_rounds,
                            },
                            on_event,
                        );
                        return;
                    }
                }
            } else {
                None
            };
        trace_runtime_decision(
            on_event,
            "investigation_mode_detected",
            &[
                ("mode", investigation_mode.as_str().into()),
                ("required", investigation_required.to_string()),
            ],
        );
        trace_runtime_decision(
            on_event,
            "investigation_path_scope",
            &[(
                "scope",
                investigation_path_scope
                    .as_deref()
                    .unwrap_or("none")
                    .to_string(),
            )],
        );
        loop {
            let response =
                match run_generate_turn(self.backend.as_mut(), &mut self.conversation, on_event) {
                    Ok(Some(r)) => r,
                    Ok(None) => {
                        on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                        on_event(RuntimeEvent::Failed {
                            message: format!("{} returned no output.", self.backend.name()),
                        });
                        return;
                    }
                    Err(e) => {
                        on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                        on_event(RuntimeEvent::Failed {
                            message: e.to_string(),
                        });
                        return;
                    }
                };

            let calls = tool_codec::parse_all_tool_inputs(&response);

            if search_budget.is_closed()
                && calls
                    .iter()
                    .any(|c| matches!(c, ToolInput::SearchCode { .. }))
            {
                if corrections < MAX_CORRECTIONS {
                    corrections += 1;
                    self.conversation.discard_last_if_assistant();
                    self.conversation
                        .push_user(search_budget.closed_message().to_string());
                    continue;
                }
                on_event(RuntimeEvent::Failed {
                    message: "Model kept searching after the search budget was closed.".to_string(),
                });
                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                return;
            }

            if calls.is_empty() {
                // If the previous tool round ended in an edit_file error and the model's repair
                // attempt contains edit_file tag syntax but produced no parseable tool calls,
                // inject a targeted correction rather than silently accepting as Direct.
                if tool_codec::contains_edit_attempt(&response)
                    && last_injected_was_edit_error(&self.conversation)
                    && corrections < MAX_CORRECTIONS
                {
                    corrections += 1;
                    self.conversation.discard_last_if_assistant();
                    self.conversation
                        .push_user(EDIT_REPAIR_CORRECTION.to_string());
                    continue;
                }

                // Fabricated [tool_result:] / [tool_error:] blocks mean the model bypassed the
                // protocol. Attempt one automatic correction before surfacing the error.
                if tool_codec::contains_fabricated_exchange(&response) {
                    if corrections < MAX_CORRECTIONS {
                        corrections += 1;
                        self.conversation.discard_last_if_assistant();
                        self.conversation
                            .push_user(FABRICATION_CORRECTION.to_string());
                        continue;
                    }
                    on_event(RuntimeEvent::Failed {
                        message: "Model repeatedly produced fabricated tool results. Try rephrasing your request.".to_string(),
                    });
                    on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                    return;
                }
                // Malformed block: a known closing tag ([/write_file], [/edit_file], etc.)
                // is present without the matching opening tag. The model used a wrong tag name.
                // Attempt one correction before giving up.
                if tool_codec::contains_malformed_block(&response) {
                    if corrections < MAX_CORRECTIONS {
                        corrections += 1;
                        self.conversation.discard_last_if_assistant();
                        self.conversation
                            .push_user(MALFORMED_BLOCK_CORRECTION.to_string());
                        continue;
                    }
                    on_event(RuntimeEvent::Failed {
                        message:
                            "Model used incorrect tool tag names. Try rephrasing your request."
                                .to_string(),
                    });
                    on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                    return;
                }

                if let Some(path) = requested_read_path.as_deref() {
                    if !requested_read_completed {
                        if !read_request_correction_issued && corrections < MAX_CORRECTIONS {
                            corrections += 1;
                            read_request_correction_issued = true;
                            self.conversation.push_user(format!(
                                "{READ_REQUEST_TOOL_REQUIRED} Requested path: `{path}`"
                            ));
                            continue;
                        }

                        self.finish_with_runtime_answer(
                            &unread_requested_file_final_answer(path),
                            AnswerSource::RuntimeTerminal {
                                reason: RuntimeTerminalReason::ReadFileFailed,
                                rounds: tool_rounds,
                            },
                            on_event,
                        );
                        return;
                    }
                }

                // R4: insufficient-evidence terminal.
                // Search was attempted this turn, all results were empty, and no file
                // was read. The model cannot have any grounded evidence to synthesize from.
                // Discard whatever the model produced and emit the runtime-owned answer.
                if search_budget.calls > 0
                    && !investigation.search_produced_results()
                    && investigation.files_read_count() == 0
                {
                    trace_insufficient_evidence_terminal(
                        "empty_search_no_read",
                        tool_rounds,
                        &search_budget,
                        &investigation,
                        on_event,
                    );
                    self.finish_with_runtime_answer(
                        insufficient_evidence_final_answer(),
                        AnswerSource::RuntimeTerminal {
                            reason: RuntimeTerminalReason::InsufficientEvidence,
                            rounds: tool_rounds,
                        },
                        on_event,
                    );
                    return;
                }

                if investigation_required && !investigation.evidence_ready() {
                    if search_budget.calls == 0 {
                        if investigation.issue_direct_answer_correction() {
                            self.conversation
                                .push_user(SEARCH_BEFORE_ANSWERING.to_string());
                            continue;
                        }

                        trace_insufficient_evidence_terminal(
                            "no_search_after_direct_answer_correction",
                            tool_rounds,
                            &search_budget,
                            &investigation,
                            on_event,
                        );
                        self.finish_with_runtime_answer(
                            ungrounded_investigation_final_answer(),
                            AnswerSource::RuntimeTerminal {
                                reason: RuntimeTerminalReason::InsufficientEvidence,
                                rounds: tool_rounds,
                            },
                            on_event,
                        );
                        return;
                    }

                    if investigation.search_produced_results() {
                        // Both candidate-read slots exhausted and evidence is still not ready.
                        // Do not attempt another correction cycle — terminate cleanly.
                        if investigation.candidate_reads_count()
                            >= MAX_CANDIDATE_READS_PER_INVESTIGATION
                        {
                            trace_insufficient_evidence_terminal(
                                "candidate_read_limit_exhausted",
                                tool_rounds,
                                &search_budget,
                                &investigation,
                                on_event,
                            );
                            self.finish_with_runtime_answer(
                                ungrounded_investigation_final_answer(),
                                AnswerSource::RuntimeTerminal {
                                    reason: RuntimeTerminalReason::InsufficientEvidence,
                                    rounds: tool_rounds,
                                },
                                on_event,
                            );
                            return;
                        }

                        if corrections < MAX_CORRECTIONS
                            && investigation.issue_premature_synthesis_correction()
                        {
                            corrections += 1;
                            self.conversation.discard_last_if_assistant();
                            self.conversation
                                .push_user(READ_BEFORE_ANSWERING.to_string());
                            continue;
                        }

                        trace_insufficient_evidence_terminal(
                            "read_required_correction_unavailable",
                            tool_rounds,
                            &search_budget,
                            &investigation,
                            on_event,
                        );
                        self.finish_with_runtime_answer(
                            ungrounded_investigation_final_answer(),
                            AnswerSource::RuntimeTerminal {
                                reason: RuntimeTerminalReason::InsufficientEvidence,
                                rounds: tool_rounds,
                            },
                            on_event,
                        );
                        return;
                    }
                }

                let source = if tool_rounds == 0 {
                    AnswerSource::Direct
                } else {
                    AnswerSource::ToolAssisted {
                        rounds: tool_rounds,
                    }
                };
                emit_visible_assistant_message(&response, on_event);
                on_event(RuntimeEvent::AnswerReady(source));
                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                return;
            }

            tool_rounds += 1;

            if tool_rounds >= MAX_TOOL_ROUNDS {
                on_event(RuntimeEvent::AnswerReady(AnswerSource::ToolLimitReached));
                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                return;
            }

            on_event(RuntimeEvent::ActivityChanged(Activity::ExecutingTools));

            match run_tool_round(
                &self.registry,
                calls,
                &mut last_call_key,
                &mut search_budget,
                &mut investigation,
                &mut reads_this_turn,
                &mut self.anchors,
                mutation_allowed,
                investigation_required,
                investigation_mode,
                requested_read_path.as_deref(),
                &mut requested_read_completed,
                investigation_path_scope.as_deref(),
                on_event,
            ) {
                ToolRoundOutcome::Completed { results } => {
                    self.conversation.push_user(results);
                    self.conversation.trim_tool_exchanges_if_needed();
                    // Signal re-entry before the next generate so the status bar
                    // transitions cleanly from "executing tools" → "processing" → …
                    on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
                    // Do not return — loop continues so the model is re-invoked
                    // with the tool results in context to produce a synthesis response.
                }
                ToolRoundOutcome::TerminalAnswer {
                    results,
                    answer,
                    reason,
                } => {
                    self.conversation.push_user(results);
                    self.conversation.trim_tool_exchanges_if_needed();
                    self.finish_with_runtime_answer(
                        &answer,
                        AnswerSource::RuntimeTerminal {
                            reason,
                            rounds: tool_rounds,
                        },
                        on_event,
                    );
                    return;
                }
                ToolRoundOutcome::ApprovalRequired {
                    accumulated,
                    pending,
                } => {
                    if !accumulated.is_empty() {
                        self.conversation.push_user(accumulated);
                        self.conversation.trim_tool_exchanges_if_needed();
                    }
                    self.pending_action = Some(pending.clone());
                    on_event(RuntimeEvent::ApprovalRequired(pending));
                    on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                    return;
                }
            }
        }
    }

    fn finish_with_runtime_answer(
        &mut self,
        answer: &str,
        source: AnswerSource,
        on_event: &mut dyn FnMut(RuntimeEvent),
    ) {
        on_event(RuntimeEvent::ActivityChanged(Activity::Responding));
        self.conversation.begin_assistant_reply();
        on_event(RuntimeEvent::AssistantMessageStarted);
        self.conversation.push_assistant_chunk(answer);
        on_event(RuntimeEvent::AssistantMessageChunk(answer.to_string()));
        on_event(RuntimeEvent::AssistantMessageFinished);
        on_event(RuntimeEvent::AnswerReady(source));
        on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
    }

    #[cfg(test)]
    pub(crate) fn set_pending_for_test(&mut self, action: PendingAction) {
        self.pending_action = Some(action);
    }
}

/// Dispatches one round of tool calls, accumulating results.
/// Stops at the first tool that requires approval and returns any results
/// accumulated before it alongside the PendingAction.
/// ToolCallStarted is fired for each tool, but ToolCallFinished is NOT fired
/// for the approval-requiring tool — handle_approve/reject fires it after resolution.
///
/// `last_call_key` carries the fingerprint of the most recently executed call across
/// rounds. If the current call matches it, a cycle error is injected instead of
/// dispatching. The key is updated after every non-cycle, non-approval dispatch.
fn run_tool_round(
    registry: &ToolRegistry,
    calls: Vec<ToolInput>,
    last_call_key: &mut Option<String>,
    search_budget: &mut SearchBudget,
    investigation: &mut InvestigationState,
    reads_this_turn: &mut HashSet<String>,
    anchors: &mut AnchorState,
    mutation_allowed: bool,
    investigation_required: bool,
    investigation_mode: InvestigationMode,
    requested_read_path: Option<&str>,
    requested_read_completed: &mut bool,
    investigation_path_scope: Option<&str>,
    on_event: &mut dyn FnMut(RuntimeEvent),
) -> ToolRoundOutcome {
    let mut accumulated = String::new();

    for mut input in calls {
        simplify_search_input(&mut input);
        // Enforce the prompt-derived path scope as an upper bound on search dispatch.
        // None → inject scope (9.1.2 behavior).
        // Some(p) within scope → keep; model narrowed correctly.
        // Some(p) broader than or orthogonal to scope → clamp silently to scope.
        if let (Some(scope), ToolInput::SearchCode { path, .. }) =
            (investigation_path_scope, &mut input)
        {
            match path {
                None => {
                    trace_runtime_decision(
                        on_event,
                        "search_scope_applied",
                        &[
                            ("action", "inject".into()),
                            ("original_path", "none".into()),
                            ("scope", scope.to_string()),
                            ("final_path", scope.to_string()),
                        ],
                    );
                    *path = Some(scope.to_string());
                }
                Some(ref p) if !path_is_within_scope(p, scope) => {
                    trace_runtime_decision(
                        on_event,
                        "search_scope_applied",
                        &[
                            ("action", "clamp".into()),
                            ("original_path", p.to_string()),
                            ("scope", scope.to_string()),
                            ("final_path", scope.to_string()),
                        ],
                    );
                    *path = Some(scope.to_string());
                }
                _ => {}
            }
        }
        let effective_search_input = match &input {
            ToolInput::SearchCode { query, path } => Some((query.clone(), path.clone())),
            _ => None,
        };
        let read_path = match &input {
            ToolInput::ReadFile { path } => Some(path.clone()),
            _ => None,
        };
        let name = input.tool_name().to_string();
        let key = call_fingerprint(&input);
        on_event(RuntimeEvent::ToolCallStarted { name: name.clone() });

        if is_mutating_tool(&input) && !mutation_allowed {
            on_event(RuntimeEvent::ToolCallFinished {
                name: name.clone(),
                summary: None,
            });
            accumulated.push_str(&tool_codec::format_tool_error(
                &name,
                READ_ONLY_TOOL_POLICY_ERROR,
            ));
            continue;
        }

        if matches!(input, ToolInput::ListDir { .. })
            && investigation_required
            && !investigation.search_attempted()
        {
            on_event(RuntimeEvent::ToolCallFinished {
                name: name.clone(),
                summary: None,
            });
            accumulated.push_str(&tool_codec::format_tool_error(
                &name,
                LIST_DIR_BEFORE_SEARCH_BLOCKED,
            ));
            continue;
        }

        if let (Some(requested), ToolInput::ReadFile { path }) = (requested_read_path, &input) {
            if normalize_evidence_path(path) != normalize_evidence_path(requested) {
                let error = format!(
                    "read_file path `{path}` does not match the requested path `{requested}`"
                );
                on_event(RuntimeEvent::ToolCallFinished {
                    name: name.clone(),
                    summary: None,
                });
                accumulated.push_str(&tool_codec::format_tool_error(&name, &error));
                return ToolRoundOutcome::TerminalAnswer {
                    results: accumulated,
                    answer: read_path_mismatch_final_answer(requested, path),
                    reason: RuntimeTerminalReason::ReadFileFailed,
                };
            }
        }

        // Per-turn search budget: 1 search always allowed; a second only when the first
        // returned no results; further searches are always blocked.
        if matches!(input, ToolInput::SearchCode { .. }) && !search_budget.is_allowed() {
            on_event(RuntimeEvent::ToolCallFinished {
                name: name.clone(),
                summary: None,
            });
            accumulated.push_str(&tool_codec::format_tool_error(
                &name,
                SEARCH_BUDGET_EXCEEDED,
            ));
            continue;
        }

        // Dedup: block re-reads of the same file within the same turn.
        // The file's contents are already in context; re-reading only inflates the prompt.
        if let Some(rp) = read_path.as_deref() {
            let normalized = normalize_evidence_path(rp);
            if reads_this_turn.contains(&normalized) {
                on_event(RuntimeEvent::ToolCallFinished {
                    name: name.clone(),
                    summary: None,
                });
                accumulated.push_str(&tool_codec::format_tool_error(
                    &name,
                    DUPLICATE_READ_REJECTED,
                ));
                continue;
            }
        }

        // Candidate-read cap: once two matched candidates have been read without
        // useful evidence, do not allow the model to keep reading current candidates.
        if investigation_required
            && !investigation.evidence_ready()
            && investigation.candidate_reads_count() >= MAX_CANDIDATE_READS_PER_INVESTIGATION
        {
            if let Some(rp) = read_path.as_deref() {
                if investigation.is_search_candidate_path(rp) {
                    trace_runtime_decision(
                        on_event,
                        "read_evidence",
                        &[
                            ("path", normalize_evidence_path(rp)),
                            ("accepted", "false".into()),
                            ("reason", "candidate_read_limit_exhausted".into()),
                            (
                                "candidate_reads",
                                investigation.candidate_reads_count().to_string(),
                            ),
                        ],
                    );
                    trace_runtime_decision(
                        on_event,
                        "terminal_insufficient_evidence",
                        &[
                            ("reason", "candidate_read_limit_exhausted".into()),
                            (
                                "candidate_reads",
                                investigation.candidate_reads_count().to_string(),
                            ),
                        ],
                    );
                    on_event(RuntimeEvent::ToolCallFinished {
                        name: name.clone(),
                        summary: None,
                    });
                    accumulated.push_str(&tool_codec::format_tool_error(
                        &name,
                        CANDIDATE_READ_CAP_EXCEEDED,
                    ));
                    return ToolRoundOutcome::TerminalAnswer {
                        results: accumulated,
                        answer: ungrounded_investigation_final_answer().to_string(),
                        reason: RuntimeTerminalReason::InsufficientEvidence,
                    };
                }
            }
        }

        // Per-turn read cap: block new reads once MAX_READS_PER_TURN unique files have been read.
        // reads_this_turn.len() counts only successful reads, so the cap is exact.
        if read_path.is_some() && reads_this_turn.len() >= MAX_READS_PER_TURN {
            on_event(RuntimeEvent::ToolCallFinished {
                name: name.clone(),
                summary: None,
            });
            accumulated.push_str(&tool_codec::format_tool_error(&name, READ_CAP_EXCEEDED));
            continue;
        }

        if last_call_key.as_deref() == Some(key.as_str()) {
            let msg = format!("{name} called with identical arguments twice in a row");
            on_event(RuntimeEvent::ToolCallFinished {
                name: name.clone(),
                summary: None,
            });
            accumulated.push_str(&tool_codec::format_tool_error(&name, &msg));
            // Do not update last_call_key: keep the same fingerprint so a third
            // consecutive identical call is also blocked.
            continue;
        }

        match registry.dispatch(input) {
            Ok(ToolRunResult::Immediate(output)) => {
                // Guard: spec must agree that this tool is Immediate.
                // A mismatch means the spec() and run() implementations are out of sync.
                debug_assert!(
                    registry
                        .spec_for(&name)
                        .map(|s| s.execution_kind == ExecutionKind::Immediate)
                        .unwrap_or(true),
                    "tool '{name}' returned Immediate but spec declares RequiresApproval"
                );
                // Record search results against the per-turn budget and investigation state.
                let search_closed_message = if name == "search_code" {
                    if let Some((query, scope)) = effective_search_input.clone() {
                        if let Some((query, scope)) =
                            anchors.record_successful_search(&output, query, scope)
                        {
                            trace_runtime_decision(
                                on_event,
                                "anchor_updated",
                                &[
                                    ("kind", "last_search".into()),
                                    ("query", query),
                                    ("scope", scope.unwrap_or_else(|| "none".into())),
                                ],
                            );
                        }
                    }
                    let was_empty = investigation.record_search_results(&output, on_event);
                    search_budget.record(was_empty);
                    search_budget
                        .is_closed()
                        .then(|| search_budget.closed_message())
                } else {
                    None
                };
                // Track successful file reads for evidence grounding and dedup.
                let read_recovery = if name == "read_file" {
                    if let Some(path) = anchors.record_successful_read(&output) {
                        trace_runtime_decision(
                            on_event,
                            "anchor_updated",
                            &[("kind", "last_read_file".into()), ("path", path)],
                        );
                    }
                    let recovery =
                        investigation.record_read_result(&output, investigation_mode, on_event);
                    if let Some(requested) = requested_read_path {
                        if let Some(rp) = read_path.as_deref() {
                            if normalize_evidence_path(rp) == normalize_evidence_path(requested) {
                                *requested_read_completed = true;
                            }
                        }
                    }
                    // Record path so a repeat read in the same turn is blocked.
                    if let Some(rp) = read_path.as_deref() {
                        reads_this_turn.insert(normalize_evidence_path(rp));
                    }
                    recovery
                } else {
                    None
                };
                let summary = tool_codec::render_compact_summary(&output);
                on_event(RuntimeEvent::ToolCallFinished {
                    name: name.clone(),
                    summary: Some(summary),
                });
                accumulated.push_str(&tool_codec::format_tool_result(&name, &output));
                if let Some((path, kind)) = read_recovery {
                    trace_runtime_decision(
                        on_event,
                        "recovery_issued",
                        &[("kind", kind.as_str().into()), ("path", path.clone())],
                    );
                    let correction = match kind {
                        RecoveryKind::DefinitionOnly => usage_read_recovery_correction(&path),
                        RecoveryKind::ImportOnly => import_read_recovery_correction(&path),
                        RecoveryKind::ConfigFile => config_read_recovery_correction(&path),
                        RecoveryKind::Initialization => {
                            initialization_read_recovery_correction(&path)
                        }
                        RecoveryKind::Create => create_read_recovery_correction(&path),
                        RecoveryKind::Register => register_read_recovery_correction(&path),
                        RecoveryKind::Load => load_read_recovery_correction(&path),
                        RecoveryKind::Save => save_read_recovery_correction(&path),
                    };
                    accumulated.push_str(&correction);
                    accumulated.push_str("\n\n");
                }
                if let Some(message) = search_closed_message {
                    accumulated.push_str(message);
                    accumulated.push_str("\n\n");
                }
                *last_call_key = Some(key);
            }
            Ok(ToolRunResult::Approval(pending)) => {
                // Guard: spec must agree that this tool requires approval.
                debug_assert!(
                    registry
                        .spec_for(&name)
                        .map(|s| s.execution_kind == ExecutionKind::RequiresApproval)
                        .unwrap_or(true),
                    "tool '{name}' returned Approval but spec declares Immediate"
                );
                return ToolRoundOutcome::ApprovalRequired {
                    accumulated,
                    pending,
                };
            }
            Err(e) => {
                let error = e.to_string();
                on_event(RuntimeEvent::ToolCallFinished {
                    name: name.clone(),
                    summary: None,
                });
                accumulated.push_str(&tool_codec::format_tool_error(&name, &error));
                if let Some(path) = read_path {
                    return ToolRoundOutcome::TerminalAnswer {
                        results: accumulated,
                        answer: read_failure_final_answer(&path, &error),
                        reason: RuntimeTerminalReason::ReadFileFailed,
                    };
                }
                // Do NOT update last_call_key on error: a failed call should not block
                // an identical retry. Cycle detection applies only to successful executions.
            }
        }
    }

    ToolRoundOutcome::Completed {
        results: accumulated,
    }
}

/// Returns true when the most recent user message in the conversation is an edit_file
/// tool error injected by the runtime. Used to detect the edit-repair failure pattern:
/// model emits garbled edit syntax after a failed edit, producing zero parsed tool calls.
fn last_injected_was_edit_error(conversation: &Conversation) -> bool {
    conversation
        .last_user_content()
        .map(|c| c.starts_with("=== tool_error: edit_file ==="))
        .unwrap_or(false)
}

/// Runs a single generation turn: sends the current conversation to the backend,
/// buffers the assistant response into conversation history, then returns the
/// complete response text, or None if the backend produced no output. Assistant
/// message events are emitted only after runtime admission.
fn run_generate_turn(
    backend: &mut dyn ModelBackend,
    conversation: &mut Conversation,
    on_event: &mut dyn FnMut(RuntimeEvent),
) -> Result<Option<String>> {
    let request = GenerateRequest::new(conversation.snapshot());
    let mut response = String::new();

    let result = backend.generate(request, &mut |event| match event {
        BackendEvent::StatusChanged(status) => {
            on_event(RuntimeEvent::ActivityChanged(map_backend_status(status)));
        }
        BackendEvent::TextDelta(chunk) => {
            response.push_str(&chunk);
        }
        BackendEvent::Timing { stage, elapsed_ms } => {
            on_event(RuntimeEvent::BackendTiming { stage, elapsed_ms });
        }
        BackendEvent::Finished => {}
    });

    result?;

    if response.is_empty() {
        Ok(None)
    } else {
        conversation.begin_assistant_reply();
        conversation.push_assistant_chunk(&response);
        Ok(Some(response))
    }
}

fn emit_visible_assistant_message(text: &str, on_event: &mut dyn FnMut(RuntimeEvent)) {
    on_event(RuntimeEvent::ActivityChanged(Activity::Responding));
    on_event(RuntimeEvent::AssistantMessageStarted);
    on_event(RuntimeEvent::AssistantMessageChunk(text.to_string()));
    on_event(RuntimeEvent::AssistantMessageFinished);
}

fn map_backend_status(status: BackendStatus) -> Activity {
    match status {
        BackendStatus::LoadingModel => Activity::LoadingModel,
        BackendStatus::Generating => Activity::Generating,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::app::config::Config;
    use crate::tools::{default_registry, RiskLevel};

    struct TestBackend {
        responses: Vec<String>,
        call_count: usize,
    }

    impl TestBackend {
        fn new(responses: Vec<impl Into<String>>) -> Self {
            Self {
                responses: responses.into_iter().map(Into::into).collect(),
                call_count: 0,
            }
        }
    }

    impl ModelBackend for TestBackend {
        fn name(&self) -> &str {
            "test"
        }

        fn generate(
            &mut self,
            _request: GenerateRequest,
            on_event: &mut dyn FnMut(BackendEvent),
        ) -> crate::app::Result<()> {
            let reply = self
                .responses
                .get(self.call_count)
                .cloned()
                .unwrap_or_default();
            self.call_count += 1;
            if !reply.is_empty() {
                on_event(BackendEvent::TextDelta(reply));
            }
            on_event(BackendEvent::Finished);
            Ok(())
        }
    }

    fn make_runtime(responses: Vec<impl Into<String>>) -> Runtime {
        Runtime::new(
            &Config::default(),
            &PathBuf::from("."),
            Box::new(TestBackend::new(responses)),
            default_registry(PathBuf::from(".")),
        )
    }

    fn make_runtime_in(responses: Vec<impl Into<String>>, root: &std::path::Path) -> Runtime {
        Runtime::new(
            &Config::default(),
            root,
            Box::new(TestBackend::new(responses)),
            default_registry(root.to_path_buf()),
        )
    }

    fn collect_events(runtime: &mut Runtime, request: RuntimeRequest) -> Vec<RuntimeEvent> {
        let mut events = Vec::new();
        runtime.handle(request, &mut |e| events.push(e));
        events
    }

    fn init_git_repo(root: &std::path::Path) {
        let status = std::process::Command::new("git")
            .args(["init"])
            .current_dir(root)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "git init must succeed");
    }

    fn has_failed(events: &[RuntimeEvent]) -> bool {
        events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::Failed { .. }))
    }

    fn failed_message(events: &[RuntimeEvent]) -> Option<String> {
        events.iter().find_map(|e| {
            if let RuntimeEvent::Failed { message } = e {
                Some(message.clone())
            } else {
                None
            }
        })
    }

    #[test]
    fn approve_with_no_pending_fires_failed() {
        let mut rt = make_runtime(vec!["hello"]);
        let events = collect_events(&mut rt, RuntimeRequest::Approve);
        assert!(has_failed(&events), "expected Failed, got: {events:?}");
        assert_eq!(
            failed_message(&events).as_deref(),
            Some("No pending action to approve.")
        );
    }

    #[test]
    fn reject_with_no_pending_fires_failed() {
        let mut rt = make_runtime(vec!["hello"]);
        let events = collect_events(&mut rt, RuntimeRequest::Reject);
        assert!(has_failed(&events), "expected Failed, got: {events:?}");
        assert_eq!(
            failed_message(&events).as_deref(),
            Some("No pending action to reject.")
        );
    }

    #[test]
    fn reject_uses_runtime_cancellation_even_if_model_would_claim_success() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let mut rt = make_runtime_in(
            vec![
                "[write_file]\npath: reject_test_phase75.txt\n---content---\nshould not exist\n[/write_file]",
                "I created reject_test_phase75.txt.",
            ],
            tmp.path(),
        );

        let submit_events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Create a file reject_test_phase75.txt with the content should not exist"
                    .into(),
            },
        );
        assert!(
            !has_failed(&submit_events),
            "submit failed: {submit_events:?}"
        );
        assert!(
            submit_events
                .iter()
                .any(|e| matches!(e, RuntimeEvent::ApprovalRequired(_))),
            "write_file must request approval"
        );

        let reject_events = collect_events(&mut rt, RuntimeRequest::Reject);
        assert!(
            !has_failed(&reject_events),
            "reject failed: {reject_events:?}"
        );
        assert!(
            !tmp.path().join("reject_test_phase75.txt").exists(),
            "rejected write must not create the file"
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("Canceled. No file was created")),
            "runtime cancellation answer must be recorded"
        );
        assert!(
            !snapshot
                .iter()
                .any(|m| m.content.contains("I created reject_test_phase75.txt.")),
            "backend response after reject must not be used"
        );
        assert!(fs::read_dir(tmp.path()).unwrap().next().is_none());
    }

    #[test]
    fn submit_while_pending_fires_failed() {
        let mut rt = make_runtime(vec!["hello"]);
        rt.set_pending_for_test(PendingAction {
            tool_name: "edit_file".into(),
            summary: "edit src/lib.rs".into(),
            risk: RiskLevel::Medium,
            payload: "{}".into(),
        });
        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "continue".into(),
            },
        );
        assert!(has_failed(&events), "expected Failed, got: {events:?}");
        assert!(failed_message(&events)
            .as_deref()
            .unwrap_or("")
            .contains("pending"),);
    }

    #[test]
    fn reset_clears_pending_state() {
        let mut rt = make_runtime(vec!["hello"]);
        rt.set_pending_for_test(PendingAction {
            tool_name: "write_file".into(),
            summary: "write src/new.rs".into(),
            risk: RiskLevel::High,
            payload: "{}".into(),
        });
        collect_events(&mut rt, RuntimeRequest::Reset);
        // After reset, approve should fail with "no pending" — not "submit blocked"
        let events = collect_events(&mut rt, RuntimeRequest::Approve);
        assert!(
            has_failed(&events),
            "expected Failed after reset, got: {events:?}"
        );
        assert_eq!(
            failed_message(&events).as_deref(),
            Some("No pending action to approve.")
        );
    }

    #[test]
    fn cycle_detection_blocks_second_identical_call() {
        // Model emits the same list_dir call twice in one response.
        // First call executes; second is blocked with a cycle error.
        // A synthesis response is provided so the loop can complete normally.
        let mut rt = make_runtime(vec!["[list_dir: .][list_dir: .]", "Done."]);
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "list it twice".into(),
            },
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: list_dir ===")),
            "first call must produce a tool result"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_error: list_dir ===")
                    && m.content.contains("identical arguments twice in a row")),
            "second identical call must produce a cycle error"
        );
    }

    #[test]
    fn cycle_detection_allows_different_args() {
        // Two list_dir calls with different paths — neither should be blocked.
        // A synthesis response is provided so the loop can complete normally.
        let mut rt = make_runtime(vec!["[list_dir: .][list_dir: src/]", "Listed both."]);
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "list both".into(),
            },
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            !snapshot
                .iter()
                .any(|m| m.content.contains("identical arguments twice in a row")),
            "different args must not trigger cycle detection"
        );
    }

    #[test]
    fn tool_round_followed_by_synthesized_answer() {
        // After a tool call completes, the model is re-invoked in the same turn.
        // The synthesis response (no tool calls) produces the final AnswerReady event.
        let mut rt = make_runtime(vec!["[list_dir: .]", "The root contains several files."]);
        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "what is in root?".into(),
            },
        );

        assert!(!has_failed(&events), "unexpected failure: {events:?}");

        // AnswerReady must be ToolAssisted from the synthesis response
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(
                answer_source,
                Some(AnswerSource::ToolAssisted { rounds: 1 })
            ),
            "expected ToolAssisted(1), got: {answer_source:?}"
        );

        // Conversation must contain: user prompt, assistant tool call, user tool result,
        // assistant synthesis — i.e. two assistant messages.
        let snapshot = rt.messages_snapshot();
        let assistant_msgs: Vec<_> = snapshot
            .iter()
            .filter(|m| m.role == crate::llm::backend::Role::Assistant)
            .collect();
        assert_eq!(
            assistant_msgs.len(),
            2,
            "expected tool-call + synthesis, got: {assistant_msgs:?}"
        );
        assert!(
            assistant_msgs[1].content.contains("several files"),
            "synthesis must contain model's response text"
        );
    }

    #[test]
    fn multi_tool_round_synthesizes_after_all_rounds() {
        // Model calls list_dir twice across two separate rounds, then synthesizes.
        // tool_rounds must reflect both rounds in the AnswerReady source.
        let mut rt = make_runtime(vec![
            "[list_dir: .]",
            "[list_dir: src/]",
            "Found everything I need.",
        ]);
        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "explore".into(),
            },
        );

        assert!(!has_failed(&events), "unexpected failure: {events:?}");

        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(
                answer_source,
                Some(AnswerSource::ToolAssisted { rounds: 2 })
            ),
            "expected ToolAssisted(2), got: {answer_source:?}"
        );

        let snapshot = rt.messages_snapshot();
        let assistant_msgs: Vec<_> = snapshot
            .iter()
            .filter(|m| m.role == crate::llm::backend::Role::Assistant)
            .collect();
        assert_eq!(
            assistant_msgs.len(),
            3,
            "expected two tool calls + synthesis"
        );
    }

    #[test]
    fn malformed_block_triggers_correction_and_retries() {
        // Model emits [test_file]...[/write_file] — wrong opening tag, correct closing tag.
        // The engine should detect the malformed block, discard the response, inject
        // a correction, and re-invoke the model. The synthesis response closes the loop.
        let malformed = "[test_file]\npath: f.txt\n---content---\nhello\n[/write_file]";
        let mut rt = make_runtime(vec![malformed, "Done."]);
        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "create f.txt".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "must not fail permanently: {events:?}"
        );

        // The final answer must be the synthesis response, not the malformed block.
        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Done."),
            "last assistant message must be synthesis"
        );

        // The correction message must have been injected into the conversation.
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.starts_with("[runtime:correction]")
                    && m.content.contains("unrecognized opening tag")),
            "malformed block correction must be in conversation"
        );
    }

    #[test]
    fn cycle_detection_allows_retry_after_tool_error() {
        // A tool fails (bad path), then the model retries with the same args.
        // The retry must NOT be blocked as a cycle — only successful calls set the key.
        // list_dir returns an IO error for a non-existent path, then succeeds on ".".
        // The synthesis response closes the loop.
        let mut rt = make_runtime(vec!["[list_dir: /nonexistent/path][list_dir: .]", "Done."]);
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "list both".into(),
            },
        );

        let snapshot = rt.messages_snapshot();
        // The error from the first call must appear
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_error: list_dir ===")),
            "first call error must be in conversation"
        );
        // The successful retry must produce a result — no cycle error
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: list_dir ===")),
            "successful retry must not be blocked by cycle detection"
        );
        assert!(
            !snapshot
                .iter()
                .any(|m| m.content.contains("identical arguments")
                    && m.content.contains("=== tool_error: list_dir ===")
                    && m.content.contains(".")),
            "retry with same args after error must not trigger cycle detection"
        );
    }

    #[test]
    fn missing_read_file_error_terminates_without_retry_loop() {
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let mut rt = make_runtime_in(
            vec![
                "[read_file: missing_file_phase75.rs]",
                "[read_file: missing_file_phase75.rs]",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Read missing_file_phase75.rs".into(),
            },
        );
        assert!(
            !has_failed(&events),
            "missing read should terminate cleanly: {events:?}"
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_error: read_file ===")),
            "read_file failure must be surfaced as a tool_error"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("No file contents were read.")),
            "runtime terminal answer must explain that no contents were read"
        );
        let assistant_read_calls = snapshot
            .iter()
            .filter(|m| {
                m.role == crate::llm::backend::Role::Assistant
                    && m.content.contains("[read_file: missing_file_phase75.rs]")
            })
            .count();
        assert_eq!(
            assistant_read_calls, 1,
            "read_file must not be retried in a loop"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, RuntimeEvent::AnswerReady(AnswerSource::ToolLimitReached))),
            "missing read must not hit the tool-round limit"
        );
    }

    #[test]
    fn snake_case_classifier_accepts_valid_identifiers() {
        assert!(is_snake_case_identifier("run_turns"));
        assert!(is_snake_case_identifier("search_code"));
        assert!(is_snake_case_identifier("read_file"));
        assert!(is_snake_case_identifier("tool_rounds"));
        assert!(is_snake_case_identifier("investigation_state"));
        assert!(is_snake_case_identifier("is_snake_case_identifier"));
    }

    #[test]
    fn snake_case_classifier_rejects_non_identifiers() {
        assert!(!is_snake_case_identifier("word")); // no underscore
        assert!(!is_snake_case_identifier("a_b")); // segments too short
        assert!(!is_snake_case_identifier("_leading")); // empty first segment
        assert!(!is_snake_case_identifier("trailing_")); // empty last segment
        assert!(!is_snake_case_identifier("has space")); // whitespace
        assert!(!is_snake_case_identifier("run_turns()")); // non-alphanumeric
    }

    #[test]
    fn pascal_case_classifier_accepts_valid_identifiers() {
        assert!(is_pascal_case_identifier("AnswerSource"));
        assert!(is_pascal_case_identifier("RuntimeTerminalReason"));
        assert!(is_pascal_case_identifier("InvestigationState"));
        assert!(is_pascal_case_identifier("ToolInput"));
        assert!(is_pascal_case_identifier("SearchBudget"));
    }

    #[test]
    fn pascal_case_classifier_rejects_non_identifiers() {
        assert!(!is_pascal_case_identifier("Hi")); // too short
        assert!(!is_pascal_case_identifier("Short")); // no second uppercase after first
        assert!(!is_pascal_case_identifier("allower")); // starts lowercase
        assert!(!is_pascal_case_identifier("Done")); // 4 chars, too short
    }

    #[test]
    fn prompt_requires_investigation_detects_snake_case() {
        assert!(prompt_requires_investigation("What does run_turns do?"));
        assert!(prompt_requires_investigation("Explain search_code to me."));
        assert!(prompt_requires_investigation("Where is read_file defined?"));
    }

    #[test]
    fn prompt_requires_investigation_detects_pascal_case() {
        assert!(prompt_requires_investigation("What is AnswerSource?"));
        assert!(prompt_requires_investigation("Explain InvestigationState"));
        assert!(prompt_requires_investigation(
            "How does RuntimeTerminalReason work?"
        ));
    }

    #[test]
    fn prompt_requires_investigation_detects_natural_language_lookup() {
        assert!(prompt_requires_investigation(
            "Find where logging is initialized"
        ));
        assert!(prompt_requires_investigation(
            "Find where sessions are saved"
        ));
        assert!(prompt_requires_investigation(
            "Where is configuration loaded?"
        ));
        assert!(prompt_requires_investigation("Where are tasks created?"));
        assert!(prompt_requires_investigation(
            "Find where session creation happens"
        ));
    }

    #[test]
    fn prompt_requires_investigation_detects_search_verb() {
        // "search" is a self-sufficient trigger — no secondary condition needed.
        assert!(prompt_requires_investigation(
            "Search for 'task' in sandbox/ and explain what parts of the system use it."
        ));
        assert!(prompt_requires_investigation("Search for task in sandbox/"));
        assert!(prompt_requires_investigation(
            "search the codebase for SessionLog"
        ));
    }

    #[test]
    fn prompt_requires_investigation_detects_occurrence_phrasing() {
        // "find/where" + occurrence/appearance words trigger investigation.
        assert!(prompt_requires_investigation(
            "Find all occurrences of 'logging' in sandbox/ and summarize where it appears."
        ));
        assert!(prompt_requires_investigation(
            "Find all occurrences of logging in sandbox/"
        ));
        assert!(prompt_requires_investigation(
            "Where does TaskStatus appear in the codebase?"
        ));
        assert!(prompt_requires_investigation(
            "Find where the error occurs in this module."
        ));
        assert!(prompt_requires_investigation(
            "Where are completed tasks filtered in sandbox/"
        ));
    }

    #[test]
    fn prompt_requires_investigation_rejects_plain_questions() {
        assert!(!prompt_requires_investigation("How are you?"));
        assert!(!prompt_requires_investigation("What time is it?"));
        assert!(!prompt_requires_investigation(
            "Can you help me with something?"
        ));
        assert!(!prompt_requires_investigation(
            "What is the purpose of this project?"
        ));
        // "find" alone without a secondary condition must not trigger
        assert!(!prompt_requires_investigation(
            "Find a good approach to this problem."
        ));
        assert!(!prompt_requires_investigation(
            "Find the best way to structure this."
        ));
    }

    #[test]
    fn mutation_intent_classifier_ignores_tool_name_mentions() {
        assert!(!user_requested_mutation("Where is write_file implemented?"));
        assert!(!user_requested_mutation("How does edit_file recover?"));
        assert!(user_requested_mutation("Create a file named demo.txt"));
        assert!(user_requested_mutation(
            "Edit src/main.rs and change hello to hi"
        ));
    }

    #[test]
    fn requested_read_path_detects_explicit_file_reads() {
        assert_eq!(
            requested_read_path("Read missing_file_phase84x.rs").as_deref(),
            Some("missing_file_phase84x.rs")
        );
        assert_eq!(
            requested_read_path("read file `src/runtime/engine.rs`").as_deref(),
            Some("src/runtime/engine.rs")
        );
        assert_eq!(requested_read_path("Read about logging"), None);
    }

    #[test]
    fn successful_read_file_updates_last_read_file_anchor() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src/runtime")).unwrap();
        fs::write(
            tmp.path().join("src/runtime/engine.rs"),
            "fn run_turns() {}\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec!["[read_file: src/runtime/engine.rs]", "Read engine.rs."],
            tmp.path(),
        );
        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "read src/runtime/engine.rs".into(),
            },
        );

        assert!(!has_failed(&events), "unexpected failure: {events:?}");
        let expected_path = tmp
            .path()
            .join("src/runtime/engine.rs")
            .to_string_lossy()
            .into_owned();
        assert_eq!(rt.anchors.last_read_file(), Some(expected_path.as_str()));
    }

    #[test]
    fn read_that_file_again_dispatches_one_read_to_anchor() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/anchor.rs"), "fn anchor() {}\n").unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[read_file: src/anchor.rs]",
                "First read complete.",
                "Anchored read complete.",
            ],
            tmp.path(),
        );
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "read src/anchor.rs".into(),
            },
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "read that file again".into(),
            },
        );

        let read_starts = events
            .iter()
            .filter(|e| matches!(e, RuntimeEvent::ToolCallStarted { name } if name == "read_file"))
            .count();
        assert_eq!(read_starts, 1, "anchor prompt must dispatch one read");
        let expected_path = tmp
            .path()
            .join("src/anchor.rs")
            .to_string_lossy()
            .into_owned();
        assert!(
            events.iter().any(|e| {
                matches!(
                    e,
                    RuntimeEvent::ToolCallFinished {
                        name,
                        summary: Some(summary)
                    } if name == "read_file" && summary.contains(&expected_path)
                )
            }),
            "anchored read must target the last successful path: {events:?}"
        );

        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(last_assistant, Some("Anchored read complete."));
    }

    #[test]
    fn open_the_last_file_resolves_to_last_read_file_anchor() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/last.rs"), "fn last() {}\n").unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[read_file: src/last.rs]",
                "First read complete.",
                "Opened last file.",
            ],
            tmp.path(),
        );
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "read src/last.rs".into(),
            },
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "open the last file".into(),
            },
        );

        let expected_path = tmp
            .path()
            .join("src/last.rs")
            .to_string_lossy()
            .into_owned();
        assert!(
            events.iter().any(|e| {
                matches!(
                    e,
                    RuntimeEvent::ToolCallFinished {
                        name,
                        summary: Some(summary)
                    } if name == "read_file" && summary.contains(&expected_path)
                )
            }),
            "open the last file must read the anchored path: {events:?}"
        );
    }

    #[test]
    fn reset_clears_last_read_file_anchor() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/reset.rs"), "fn reset_anchor() {}\n").unwrap();

        let mut rt = make_runtime_in(
            vec!["[read_file: src/reset.rs]", "First read complete."],
            tmp.path(),
        );
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "read src/reset.rs".into(),
            },
        );
        assert!(rt.anchors.last_read_file().is_some());

        collect_events(&mut rt, RuntimeRequest::Reset);
        assert_eq!(rt.anchors.last_read_file(), None);

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "read that file".into(),
            },
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                RuntimeEvent::AssistantMessageChunk(chunk)
                    if chunk == NO_LAST_READ_FILE_AVAILABLE
            )),
            "reset anchor prompt must produce deterministic no-anchor answer: {events:?}"
        );
    }

    #[test]
    fn failed_read_file_does_not_update_last_read_file_anchor() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/good.rs"), "fn good() {}\n").unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[read_file: src/good.rs]",
                "First read complete.",
                "[read_file: src/missing.rs]",
            ],
            tmp.path(),
        );
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "read src/good.rs".into(),
            },
        );
        let anchored_path = rt.anchors.last_read_file().map(str::to_string);
        assert!(anchored_path.is_some());

        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "read src/missing.rs".into(),
            },
        );
        assert_eq!(
            rt.anchors.last_read_file().map(str::to_string),
            anchored_path,
            "failed reads must not replace the last successful read anchor"
        );
    }

    #[test]
    fn no_anchor_followup_returns_deterministic_failure() {
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let mut rt = make_runtime_in(Vec::<String>::new(), tmp.path());
        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "read that file".into(),
            },
        );

        assert!(
            events.iter().any(|e| matches!(
                e,
                RuntimeEvent::AssistantMessageChunk(chunk)
                    if chunk == NO_LAST_READ_FILE_AVAILABLE
            )),
            "no-anchor prompt must produce deterministic runtime answer: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { .. })),
            "no-anchor prompt must not guess or dispatch tools: {events:?}"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(
                answer_source,
                Some(AnswerSource::RuntimeTerminal {
                    reason: RuntimeTerminalReason::ReadFileFailed,
                    ..
                })
            ),
            "no-anchor prompt must terminate as runtime-owned read failure: {answer_source:?}"
        );
    }

    #[test]
    fn unsupported_anchor_phrases_do_not_resolve_last_read_file() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/anchor.rs"), "fn anchor() {}\n").unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[read_file: src/anchor.rs]",
                "First read complete.",
                "Not an anchor.",
                "Still not an anchor.",
                "Also not an anchor.",
            ],
            tmp.path(),
        );
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "read src/anchor.rs".into(),
            },
        );
        assert!(rt.anchors.last_read_file().is_some());

        for phrase in ["open it", "read that", "open the second result"] {
            let events = collect_events(
                &mut rt,
                RuntimeRequest::Submit {
                    text: phrase.into(),
                },
            );
            assert!(
                !events.iter().any(
                    |e| matches!(e, RuntimeEvent::ToolCallStarted { name } if name == "read_file")
                ),
                "unsupported phrase `{phrase}` must not resolve the last-read anchor: {events:?}"
            );
        }
    }

    #[test]
    fn anchored_read_counts_against_same_turn_read_cap() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        for file in ["anchor.rs", "b.rs", "c.rs", "d.rs"] {
            fs::write(
                tmp.path().join("src").join(file),
                format!("fn {}() {{}}\n", file.replace(".rs", "")),
            )
            .unwrap();
        }

        let mut rt = make_runtime_in(
            vec![
                "[read_file: src/anchor.rs]",
                "First read complete.",
                "[read_file: src/b.rs]",
                "[read_file: src/c.rs]",
                "[read_file: src/d.rs]",
                "I have enough file evidence.",
            ],
            tmp.path(),
        );
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "read src/anchor.rs".into(),
            },
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "read that file again".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "turn must complete without failure: {events:?}"
        );
        let snapshot = rt.messages_snapshot();
        let all_user: String = snapshot
            .iter()
            .filter(|m| m.role == crate::llm::backend::Role::User)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(
            all_user.matches("=== tool_result: read_file ===").count(),
            4,
            "first turn read plus three second-turn reads must succeed"
        );
        assert!(
            all_user.contains("=== tool_error: read_file ===")
                && all_user.contains("read limit for this turn"),
            "fourth read in the anchor turn must be blocked by the normal read cap"
        );
    }

    #[test]
    fn successful_search_code_updates_last_search_anchor() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("a.rs"), "fn needle() {}\n").unwrap();

        let mut rt = make_runtime_in(
            vec!["[search_code: needle]", "Search complete."],
            tmp.path(),
        );
        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "tool check".into(),
            },
        );

        assert!(!has_failed(&events), "unexpected failure: {events:?}");
        assert_eq!(rt.anchors.last_search_query(), Some("needle"));
        assert_eq!(rt.anchors.last_search_scope(), None);
    }

    #[test]
    fn repeat_last_search_dispatches_one_search_code() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("a.rs"), "fn needle() {}\n").unwrap();

        let mut rt = make_runtime_in(
            vec!["[search_code: needle]", "Search complete."],
            tmp.path(),
        );
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "tool check".into(),
            },
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "repeat the last search".into(),
            },
        );

        let search_starts = events
            .iter()
            .filter(
                |e| matches!(e, RuntimeEvent::ToolCallStarted { name } if name == "search_code"),
            )
            .count();
        assert_eq!(search_starts, 1, "replay must dispatch exactly one search");
        assert!(
            !events.iter().any(
                |e| matches!(e, RuntimeEvent::ToolCallStarted { name } if name == "read_file")
            ),
            "search replay must not auto-read candidates: {events:?}"
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                RuntimeEvent::AssistantMessageChunk(chunk) if chunk == LAST_SEARCH_REPLAYED
            )),
            "search replay must end with runtime-owned completion: {events:?}"
        );
    }

    #[test]
    fn unscoped_search_replays_with_no_scope() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("one")).unwrap();
        fs::create_dir_all(tmp.path().join("two")).unwrap();
        fs::write(tmp.path().join("one/a.rs"), "fn needle_one() {}\n").unwrap();
        fs::write(tmp.path().join("two/b.rs"), "fn needle_two() {}\n").unwrap();

        let mut rt = make_runtime_in(
            vec!["[search_code: needle]", "Search complete."],
            tmp.path(),
        );
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "tool check".into(),
            },
        );
        assert_eq!(rt.anchors.last_search_scope(), None);

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "search the last query again".into(),
            },
        );

        assert!(
            events.iter().any(|e| {
                matches!(
                    e,
                    RuntimeEvent::ToolCallFinished {
                        name,
                        summary: Some(summary)
                    } if name == "search_code"
                        && summary.contains("found 2 match(es)")
                        && summary.contains("needle")
                )
            }),
            "unscoped replay must search the whole project: {events:?}"
        );
        assert_eq!(rt.anchors.last_search_scope(), None);
    }

    #[test]
    fn scoped_search_replay_uses_effective_prompt_scope() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("sandbox/in_scope.py"), "needle = True\n").unwrap();
        fs::write(tmp.path().join("src/outside.py"), "needle = False\n").unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: needle]",
                "[read_file: sandbox/in_scope.py]",
                "needle is in sandbox/in_scope.py.",
            ],
            tmp.path(),
        );
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is needle used in sandbox/".into(),
            },
        );
        assert_eq!(rt.anchors.last_search_query(), Some("needle"));
        assert_eq!(rt.anchors.last_search_scope(), Some("sandbox/"));

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "run the last search again".into(),
            },
        );

        assert!(
            events.iter().any(|e| {
                matches!(
                    e,
                    RuntimeEvent::ToolCallFinished {
                        name,
                        summary: Some(summary)
                    } if name == "search_code" && summary.contains("found 1 match(es)")
                )
            }),
            "scoped replay must preserve the effective prompt scope: {events:?}"
        );
        assert_eq!(rt.anchors.last_search_scope(), Some("sandbox/"));
    }

    #[test]
    fn search_anchor_stores_effective_clamped_scope() {
        use std::collections::HashSet;
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("sandbox/in_scope.py"), "needle = True\n").unwrap();
        fs::write(tmp.path().join("src/outside.py"), "needle = False\n").unwrap();

        let registry = default_registry(tmp.path().to_path_buf());
        let mut last_call_key = None;
        let mut search_budget = SearchBudget::new();
        let mut investigation = InvestigationState::new();
        let mut reads_this_turn = HashSet::new();
        let mut anchors = AnchorState::default();
        let mut requested_read_completed = false;
        let mut events = Vec::new();

        let outcome = run_tool_round(
            &registry,
            vec![ToolInput::SearchCode {
                query: "needle".into(),
                path: Some("src/".into()),
            }],
            &mut last_call_key,
            &mut search_budget,
            &mut investigation,
            &mut reads_this_turn,
            &mut anchors,
            false,
            true,
            InvestigationMode::UsageLookup,
            None,
            &mut requested_read_completed,
            Some("sandbox/"),
            &mut |e| events.push(e),
        );

        assert!(
            matches!(outcome, ToolRoundOutcome::Completed { .. }),
            "search round must complete"
        );
        assert_eq!(anchors.last_search_query(), Some("needle"));
        assert_eq!(anchors.last_search_scope(), Some("sandbox/"));
    }

    #[test]
    fn failed_search_code_does_not_update_last_search_anchor() {
        use std::collections::HashSet;
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("a.rs"), "fn needle() {}\n").unwrap();
        let registry = default_registry(tmp.path().to_path_buf());
        let mut last_call_key = None;
        let mut search_budget = SearchBudget::new();
        let mut investigation = InvestigationState::new();
        let mut reads_this_turn = HashSet::new();
        let mut anchors = AnchorState::default();
        let mut requested_read_completed = false;
        let mut events = Vec::new();

        let seed_outcome = run_tool_round(
            &registry,
            vec![ToolInput::SearchCode {
                query: "needle".into(),
                path: Some("sandbox/".into()),
            }],
            &mut last_call_key,
            &mut search_budget,
            &mut investigation,
            &mut reads_this_turn,
            &mut anchors,
            false,
            false,
            InvestigationMode::General,
            None,
            &mut requested_read_completed,
            None,
            &mut |e| events.push(e),
        );
        assert!(
            matches!(seed_outcome, ToolRoundOutcome::Completed { .. }),
            "seed search round must complete"
        );
        assert_eq!(anchors.last_search_query(), Some("needle"));
        assert_eq!(anchors.last_search_scope(), Some("sandbox/"));

        let outcome = run_tool_round(
            &registry,
            vec![ToolInput::SearchCode {
                query: "".into(),
                path: None,
            }],
            &mut last_call_key,
            &mut search_budget,
            &mut investigation,
            &mut reads_this_turn,
            &mut anchors,
            false,
            false,
            InvestigationMode::General,
            None,
            &mut requested_read_completed,
            None,
            &mut |e| events.push(e),
        );

        assert!(
            matches!(outcome, ToolRoundOutcome::Completed { .. }),
            "failed non-read tool should return completed with tool error"
        );
        assert_eq!(anchors.last_search_query(), Some("needle"));
        assert_eq!(anchors.last_search_scope(), Some("sandbox/"));
    }

    #[test]
    fn reset_clears_last_search_anchor() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("a.rs"), "fn needle() {}\n").unwrap();

        let mut rt = make_runtime_in(
            vec!["[search_code: needle]", "Search complete."],
            tmp.path(),
        );
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "tool check".into(),
            },
        );
        assert!(rt.anchors.last_search_query().is_some());

        collect_events(&mut rt, RuntimeRequest::Reset);

        assert_eq!(rt.anchors.last_search_query(), None);
        assert_eq!(rt.anchors.last_search_scope(), None);
    }

    #[test]
    fn no_search_anchor_replay_returns_deterministic_failure() {
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let mut rt = make_runtime_in(Vec::<String>::new(), tmp.path());
        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "search that again".into(),
            },
        );

        assert!(
            events.iter().any(|e| matches!(
                e,
                RuntimeEvent::AssistantMessageChunk(chunk)
                    if chunk == NO_LAST_SEARCH_AVAILABLE
            )),
            "no-search-anchor prompt must produce deterministic runtime answer: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { .. })),
            "no-search-anchor prompt must not dispatch tools: {events:?}"
        );
    }

    #[test]
    fn unsupported_search_anchor_phrases_do_not_resolve() {
        assert!(!is_last_search_anchor_prompt("search it again"));
        assert!(!is_last_search_anchor_prompt("search for that thing again"));
        assert!(!is_last_search_anchor_prompt("search again"));
        assert!(is_last_search_anchor_prompt("search that again"));
        assert!(is_last_search_anchor_prompt("repeat the last search"));
    }

    #[test]
    fn git_status_does_not_update_anchors() {
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());
        let mut rt = make_runtime_in(vec!["[git_status]", "Working tree checked."], tmp.path());

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Show git status".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "git_status turn must not fail: {events:?}"
        );
        assert_eq!(rt.anchors.last_read_file(), None);
        assert_eq!(rt.anchors.last_search(), None);
        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: git_status ===")),
            "git_status result must be injected as a normal tool result"
        );
    }

    #[test]
    fn git_status_does_not_satisfy_investigation_evidence() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());
        fs::write(
            tmp.path().join("a.rs"),
            "fn use_task_status() { TaskStatus; }\n",
        )
        .unwrap();
        let mut rt = make_runtime_in(
            vec![
                "[git_status]",
                "TaskStatus appears in git status.",
                "[search_code: TaskStatus]",
                "[read_file: a.rs]",
                "TaskStatus is used in a.rs.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is TaskStatus used?".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "turn must recover through search/read: {events:?}"
        );
        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: git_status ===")),
            "git_status should run as a normal tool result"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("Use search_code")),
            "git_status must not satisfy investigation evidence"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: search_code ===")),
            "model must still search after git_status"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: read_file ===")),
            "model must still read matched code evidence"
        );
    }

    #[test]
    fn git_diff_does_not_update_anchors() {
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());
        let mut rt = make_runtime_in(vec!["[git_diff]", "Working tree diff checked."], tmp.path());

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Show git diff".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "git_diff turn must not fail: {events:?}"
        );
        assert_eq!(rt.anchors.last_read_file(), None);
        assert_eq!(rt.anchors.last_search(), None);
        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: git_diff ===")),
            "git_diff result must be injected as a normal tool result"
        );
    }

    #[test]
    fn git_diff_does_not_satisfy_investigation_evidence() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());
        fs::write(
            tmp.path().join("a.rs"),
            "fn use_task_status() { TaskStatus; }\n",
        )
        .unwrap();
        let mut rt = make_runtime_in(
            vec![
                "[git_diff]",
                "TaskStatus appears in git diff.",
                "[search_code: TaskStatus]",
                "[read_file: a.rs]",
                "TaskStatus is used in a.rs.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is TaskStatus used?".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "turn must recover through search/read: {events:?}"
        );
        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: git_diff ===")),
            "git_diff should run as a normal tool result"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("Use search_code")),
            "git_diff must not satisfy investigation evidence"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: search_code ===")),
            "model must still search after git_diff"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: read_file ===")),
            "model must still read matched code evidence"
        );
    }

    #[test]
    fn same_scope_followup_reuses_last_successful_scoped_search_scope() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(
            tmp.path().join("sandbox/services/logging.py"),
            "def initialize_logging():\n    pass\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("sandbox/services/database.yaml"),
            "database: sqlite:///service.db\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("src/database.yaml"),
            "database: sqlite:///wrong.db\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: logging]",
                "[read_file: sandbox/services/logging.py]",
                "logging is initialized in sandbox/services/logging.py.",
                "[search_code: database]",
                "[read_file: sandbox/services/database.yaml]",
                "database is configured in sandbox/services/database.yaml.",
            ],
            tmp.path(),
        );
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where logging is initialized in sandbox/services/".into(),
            },
        );
        assert_eq!(rt.anchors.last_search_scope(), Some("sandbox/services/"));

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where database is configured in the same folder".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        let search_result = snapshot
            .iter()
            .rev()
            .find(|m| m.content.contains("=== tool_result: search_code ==="))
            .map(|m| m.content.as_str())
            .unwrap_or("");
        assert!(
            search_result.contains("sandbox/services/database.yaml"),
            "same-scope search must include in-scope config: {search_result}"
        );
        assert!(
            !search_result.contains("src/database.yaml"),
            "same-scope search must exclude out-of-scope config: {search_result}"
        );
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("database is configured in sandbox/services/database.yaml.")
        );
    }

    #[test]
    fn same_scope_followup_without_prior_search_fails_deterministically() {
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let mut rt = make_runtime_in(Vec::<String>::new(), tmp.path());
        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where database is configured in the same folder".into(),
            },
        );

        assert!(
            events.iter().any(|e| matches!(
                e,
                RuntimeEvent::AssistantMessageChunk(chunk)
                    if chunk == NO_LAST_SCOPED_SEARCH_AVAILABLE
            )),
            "missing same-scope anchor must produce deterministic answer: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { .. })),
            "missing same-scope anchor must not dispatch tools: {events:?}"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(
                answer_source,
                Some(AnswerSource::RuntimeTerminal {
                    reason: RuntimeTerminalReason::InsufficientEvidence,
                    ..
                })
            ),
            "missing same-scope anchor must terminate as insufficient evidence: {answer_source:?}"
        );
    }

    #[test]
    fn same_scope_followup_after_unscoped_search_fails_deterministically() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("a.rs"), "fn needle() {}\n").unwrap();

        let mut rt = make_runtime_in(
            vec!["[search_code: needle]", "Search complete."],
            tmp.path(),
        );
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "tool check".into(),
            },
        );
        assert_eq!(rt.anchors.last_search_query(), Some("needle"));
        assert_eq!(rt.anchors.last_search_scope(), None);

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where database is configured within the same scope".into(),
            },
        );

        assert!(
            events.iter().any(|e| matches!(
                e,
                RuntimeEvent::AssistantMessageChunk(chunk)
                    if chunk == NO_LAST_SCOPED_SEARCH_AVAILABLE
            )),
            "unscoped last search must not provide same-scope continuity: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { .. })),
            "unscoped last search must not fall back to global search: {events:?}"
        );
    }

    #[test]
    fn same_scope_followup_after_empty_scope_search_fails_deterministically() {
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let mut rt = make_runtime_in(Vec::<String>::new(), tmp.path());
        let output =
            crate::tools::ToolOutput::SearchResults(crate::tools::types::SearchResultsOutput {
                query: "needle".into(),
                matches: Vec::new(),
                total_matches: 0,
                truncated: false,
            });

        rt.anchors
            .record_successful_search(&output, "needle".into(), Some("   ".into()));
        assert_eq!(rt.anchors.last_search_query(), Some("needle"));
        assert_eq!(rt.anchors.last_search_scope(), None);
        assert_eq!(rt.anchors.last_scoped_search_scope(), None);

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where database is configured in the same folder".into(),
            },
        );

        assert!(
            events.iter().any(|e| matches!(
                e,
                RuntimeEvent::AssistantMessageChunk(chunk)
                    if chunk == NO_LAST_SCOPED_SEARCH_AVAILABLE
            )),
            "empty stored scope must not provide same-scope continuity: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { .. })),
            "empty stored scope must not dispatch tools: {events:?}"
        );
    }

    #[test]
    fn same_scope_followup_explicit_concrete_path_takes_precedence() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/config")).unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
        fs::write(
            tmp.path().join("sandbox/config/database.yaml"),
            "database: sqlite:///config.db\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("sandbox/services/database.yaml"),
            "database: sqlite:///service.db\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: database]",
                "[read_file: sandbox/config/database.yaml]",
                "database is configured in sandbox/config/database.yaml.",
            ],
            tmp.path(),
        );
        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where database is configured in sandbox/config/ and in the same folder"
                    .into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        let search_result = snapshot
            .iter()
            .find(|m| m.content.contains("=== tool_result: search_code ==="))
            .map(|m| m.content.as_str())
            .unwrap_or("");
        assert!(
            search_result.contains("sandbox/config/database.yaml"),
            "explicit scope must be used even with same-scope phrase: {search_result}"
        );
        assert!(
            !search_result.contains("sandbox/services/database.yaml"),
            "same-scope phrase must not override explicit concrete scope: {search_result}"
        );
    }

    #[test]
    fn unsupported_same_scope_phrases_do_not_match() {
        assert!(!has_same_scope_reference("Find database in the same place"));
        assert!(!has_same_scope_reference("Find it there"));
        assert!(!has_same_scope_reference("Search the same place"));
        assert!(!has_same_scope_reference("Find database in this folder"));
        assert!(!has_same_scope_reference(
            "Find database in the same folderish"
        ));
        assert!(!has_same_scope_reference(
            "Find database within the same scopekeeper"
        ));
        assert!(has_same_scope_reference("Find database in the same folder"));
        assert!(has_same_scope_reference(
            "Find database within the same directory"
        ));
        assert!(has_same_scope_reference(
            "Find database within the same scope"
        ));
    }

    #[test]
    fn same_scope_forced_broader_path_clamps_to_prior_scoped_search() {
        use std::collections::HashSet;
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(
            tmp.path().join("sandbox/services/logging.py"),
            "def initialize_logging():\n    pass\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("sandbox/services/database.yaml"),
            "database: sqlite:///service.db\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("src/database.yaml"),
            "database: sqlite:///wrong.db\n",
        )
        .unwrap();

        let registry = default_registry(tmp.path().to_path_buf());
        let mut anchors = AnchorState::default();
        let mut events = Vec::new();

        let mut seed_last_call_key = None;
        let mut seed_search_budget = SearchBudget::new();
        let mut seed_investigation = InvestigationState::new();
        let mut seed_reads_this_turn = HashSet::new();
        let mut seed_requested_read_completed = false;
        let seed_outcome = run_tool_round(
            &registry,
            vec![ToolInput::SearchCode {
                query: "logging".into(),
                path: Some("sandbox/services/".into()),
            }],
            &mut seed_last_call_key,
            &mut seed_search_budget,
            &mut seed_investigation,
            &mut seed_reads_this_turn,
            &mut anchors,
            false,
            true,
            InvestigationMode::InitializationLookup,
            None,
            &mut seed_requested_read_completed,
            None,
            &mut |e| events.push(e),
        );
        assert!(
            matches!(seed_outcome, ToolRoundOutcome::Completed { .. }),
            "seed scoped search must complete"
        );
        assert_eq!(
            anchors.last_scoped_search_scope(),
            Some("sandbox/services/")
        );

        let same_scope = anchors
            .last_scoped_search_scope()
            .map(str::to_string)
            .expect("seeded scoped search");
        let mut last_call_key = None;
        let mut search_budget = SearchBudget::new();
        let mut investigation = InvestigationState::new();
        let mut reads_this_turn = HashSet::new();
        let mut requested_read_completed = false;
        let outcome = run_tool_round(
            &registry,
            vec![ToolInput::SearchCode {
                query: "database".into(),
                path: Some("src/".into()),
            }],
            &mut last_call_key,
            &mut search_budget,
            &mut investigation,
            &mut reads_this_turn,
            &mut anchors,
            false,
            true,
            InvestigationMode::ConfigLookup,
            None,
            &mut requested_read_completed,
            Some(&same_scope),
            &mut |e| events.push(e),
        );

        let results = match outcome {
            ToolRoundOutcome::Completed { results } => results,
            _ => panic!("forced same-scope clamp should complete"),
        };
        assert!(
            results.contains("sandbox/services/database.yaml"),
            "clamped same-scope search must include prior scoped path: {results}"
        );
        assert!(
            !results.contains("src/database.yaml"),
            "broader model path must be clamped away from src/: {results}"
        );
        assert_eq!(
            anchors.last_scoped_search_scope(),
            Some("sandbox/services/")
        );
    }

    #[test]
    fn premature_investigation_answer_is_not_admitted() {
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let mut rt = make_runtime_in(
            vec!["run_turns drives the loop.", "It still drives the loop."],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "What does run_turns do?".into(),
            },
        );

        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(
                answer_source,
                Some(AnswerSource::RuntimeTerminal {
                    reason: RuntimeTerminalReason::InsufficientEvidence,
                    ..
                })
            ),
            "premature direct answers must not be admitted: {answer_source:?}"
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content == "run_turns drives the loop."),
            "pre-evidence prose is kept in the trace"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("No final answer was accepted")),
            "runtime terminal must explain that no grounded answer was accepted"
        );
    }

    #[test]
    fn search_results_require_matched_read_before_synthesis() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: run_turns]",
                "run_turns is in engine.rs.",
                "It is definitely in engine.rs.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "What does run_turns do?".into(),
            },
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| {
                m.content.starts_with("[runtime:correction]")
                    && m.content.contains("no matched file has been read")
            }),
            "runtime must require read_file after non-empty search"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(
                answer_source,
                Some(AnswerSource::RuntimeTerminal {
                    reason: RuntimeTerminalReason::InsufficientEvidence,
                    ..
                })
            ),
            "unread search results must not admit synthesis: {answer_source:?}"
        );
    }

    #[test]
    fn read_before_answering_correction_discards_premature_synthesis() {
        // After search returns matches, the model synthesizes without reading (premature).
        // The READ_BEFORE_ANSWERING correction must fire AND discard the premature synthesis
        // from context before injecting the correction message.
        // Verified by checking: no premature synthesis message remains in the conversation.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: run_turns]",
                // Round 2: model synthesizes without reading — premature.
                "run_turns is the main driver.",
                // Round 3: after correction, model reads and then synthesizes.
                "[read_file: engine.rs]",
                "run_turns drives the main event loop.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "What does run_turns do?".into(),
            },
        );

        let snapshot = rt.messages_snapshot();

        // The correction must have fired.
        assert!(
            snapshot.iter().any(|m| {
                m.content.starts_with("[runtime:correction]")
                    && m.content.contains("no matched file has been read")
            }),
            "READ_BEFORE_ANSWERING correction must be injected: {snapshot:?}"
        );

        // The premature synthesis must NOT remain in context — it was discarded.
        assert!(
            !snapshot
                .iter()
                .any(|m| m.content == "run_turns is the main driver."),
            "premature synthesis must be discarded from context before correction"
        );

        // The grounded final answer must be present.
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("run_turns drives the main event loop."),
            "grounded synthesis must be the last assistant message"
        );

        // Turn completes successfully as ToolAssisted.
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "turn must complete as ToolAssisted after evidence-ready synthesis: {answer_source:?}"
        );
    }

    #[test]
    fn read_must_come_from_current_search_results() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();
        fs::write(tmp.path().join("notes.rs"), "fn unrelated() {}\n").unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: run_turns]",
                "[read_file: notes.rs]",
                "notes.rs explains it.",
                "Still enough.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "What does run_turns do?".into(),
            },
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: read_file ===")),
            "unmatched read still executes as normal context"
        );
        assert!(
            snapshot.iter().any(|m| {
                m.content.starts_with("[runtime:correction]")
                    && m.content.contains("no matched file has been read")
            }),
            "unmatched read must not satisfy evidence readiness"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(
                answer_source,
                Some(AnswerSource::RuntimeTerminal {
                    reason: RuntimeTerminalReason::InsufficientEvidence,
                    ..
                })
            ),
            "read outside search candidates must not admit synthesis: {answer_source:?}"
        );
    }

    #[test]
    fn list_dir_before_search_is_blocked_for_filtered_investigation() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("task_service.py"),
            "def completed_tasks(tasks):\n    filtered = [task for task in tasks if task.done]\n    return filtered\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[list_dir: .]",
                "[search_code: filtered]",
                "[read_file: task_service.py]",
                "Completed tasks are filtered in task_service.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are completed tasks filtered?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| {
                m.content.contains("=== tool_error: list_dir ===")
                    && m.content.contains("require search_code")
            }),
            "list_dir before search must be blocked on investigation-required turns"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: search_code ===")),
            "model must recover by searching"
        );
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: read_file ===")),
            "model must read a matched file before answering"
        );
    }

    #[test]
    fn usage_lookup_definition_only_read_does_not_satisfy_evidence_when_usage_candidates_exist() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("models")).unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("models").join("enums.py"),
            "from enum import Enum\n\nclass TaskStatus(str, Enum):\n    TODO = \"todo\"\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("task_service.py"),
            "from models.enums import TaskStatus\n\nif task.status == TaskStatus.TODO:\n    pass\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: TaskStatus]",
                "[read_file: models/enums.py]",
                "[read_file: services/task_service.py]",
                "TaskStatus is used in services/task_service.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is TaskStatus used?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| {
                m.content
                    .contains("[runtime:correction] This is a usage lookup")
                    && m.content.contains("services/task_service.py]")
            }),
            "definition-only read must trigger a targeted usage-file recovery correction"
        );

        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "turn should complete after the usage file is read: {answer_source:?}"
        );
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("TaskStatus is used in services/task_service.py.")
        );
    }

    #[test]
    fn usage_lookup_all_definition_candidates_fallback_allows_definition_read() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("models")).unwrap();
        fs::write(
            tmp.path().join("models").join("enums.py"),
            "from enum import Enum\n\nclass TaskStatus(str, Enum):\n    TODO = \"todo\"\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: TaskStatus]",
                "[read_file: models/enums.py]",
                "Only the TaskStatus definition was found.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is TaskStatus used?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        assert!(
            !snapshot
                .iter()
                .any(|m| m.content.starts_with("[runtime:correction]")),
            "definition-only fallback should not inject a correction when no usage candidates exist"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "definition-only fallback must admit synthesis: {answer_source:?}"
        );
    }

    #[test]
    fn usage_lookup_mixed_definition_and_usage_file_is_useful_immediately() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("models")).unwrap();
        fs::write(
            tmp.path().join("models").join("task_status.py"),
            "class TaskStatus:\n    TODO = \"todo\"\n\nDEFAULT_STATUS = TaskStatus.TODO\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: TaskStatus]",
                "[read_file: models/task_status.py]",
                "TaskStatus is defined and used in models/task_status.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is TaskStatus used?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        assert!(
            !snapshot
                .iter()
                .any(|m| m.content.starts_with("[runtime:correction]")),
            "mixed definition+usage file should satisfy usage evidence immediately"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "mixed candidate read must admit synthesis: {answer_source:?}"
        );
    }

    #[test]
    fn definition_lookup_accepts_definition_read_when_usage_candidates_exist() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("models")).unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("models").join("enums.py"),
            "from enum import Enum\n\nclass TaskStatus(str, Enum):\n    TODO = \"todo\"\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("task_service.py"),
            "from models.enums import TaskStatus\n\nif task.status == TaskStatus.TODO:\n    pass\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: TaskStatus]",
                "[read_file: models/enums.py]",
                "TaskStatus is defined in models/enums.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is TaskStatus defined?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        assert!(
            !snapshot.iter().any(|m| {
                m.content.starts_with("[runtime:correction]")
                    && m.content.contains("no matched file has been read")
            }),
            "definition lookup must accept the definition read as useful evidence"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "definition lookup should complete after reading the definition: {answer_source:?}"
        );
    }

    #[test]
    fn mutating_tool_is_blocked_on_informational_turn() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("engine.rs"), "fn write_file() {}\n").unwrap();
        let blocked_path = tmp.path().join("should_not_exist.txt");

        let mut rt = make_runtime_in(
            vec![
                "[write_file]\npath: should_not_exist.txt\n---content---\nnope\n[/write_file]",
                "[search_code: write_file]",
                "[read_file: engine.rs]",
                "write_file is implemented in engine.rs.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is write_file implemented?".into(),
            },
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, RuntimeEvent::ApprovalRequired(_))),
            "read-only informational turn must not create a pending mutation"
        );
        assert!(
            !blocked_path.exists(),
            "blocked write_file must not create a file"
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| {
                m.content.contains("=== tool_error: write_file ===")
                    && m.content.contains("mutating tools are not allowed")
            }),
            "blocked mutation must be surfaced as a tool error"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "turn should continue with allowed read-only tools: {answer_source:?}"
        );
    }

    #[test]
    fn mixed_prose_and_tool_call_does_not_admit_prose() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();
        let mut rt = make_runtime_in(
            vec![
                "It is probably in engine.rs.\n[search_code: run_turns]",
                "[read_file: engine.rs]",
                "run_turns is in engine.rs.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is run_turns defined?".into(),
            },
        );

        let answer_ready_count = events
            .iter()
            .filter(|e| matches!(e, RuntimeEvent::AnswerReady(_)))
            .count();
        assert_eq!(
            answer_ready_count, 1,
            "only final synthesis may be admitted"
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("It is probably in engine.rs.")),
            "mixed pre-evidence prose remains trace context but is not admitted"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "final grounded answer should be tool-assisted: {answer_source:?}"
        );
    }

    #[test]
    fn repeated_pre_evidence_synthesis_is_suppressed_until_read() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();

        let mut rt = make_runtime_in(
            vec![
                "run_turns drives the loop.",
                "[search_code: run_turns]",
                "run_turns is in engine.rs.",
                "[read_file: engine.rs]",
                "run_turns is grounded in engine.rs.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "What does run_turns do?".into(),
            },
        );

        let answer_sources: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let RuntimeEvent::AnswerReady(src) = e {
                    Some(src.clone())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(answer_sources.len(), 1, "only one answer may be admitted");
        assert!(
            matches!(answer_sources[0], AnswerSource::ToolAssisted { .. }),
            "the single admitted answer must be after evidence-ready"
        );

        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(last_assistant, Some("run_turns is grounded in engine.rs."));
    }

    #[test]
    fn search_query_simplification_prefers_single_literal_keyword() {
        assert_eq!(simplify_search_query("logging initialization"), "logging");
        assert_eq!(simplify_search_query("logger initialization"), "logger");
        assert_eq!(simplify_search_query("write_file()"), "write_file");
        assert_eq!(simplify_search_query("sessions saved"), "sessions");
        assert_eq!(simplify_search_query("fn main"), "main");
        assert_eq!(simplify_search_query(r"logging\.init\(\)"), "logging");
    }

    #[test]
    fn search_budget_blocks_second_search_when_first_had_results() {
        // Both searches in one response. "ToolInput" is present in many source files,
        // so the first search will produce matches — the second must be budget-blocked.
        let mut rt = make_runtime(vec![
            "[search_code: ToolInput][search_code: EditFile]",
            "Done.",
        ]);
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "search".into(),
            },
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_error: search_code ===")
                    && m.content.contains("search budget exceeded")),
            "second search must be blocked with budget error when first had results"
        );
    }

    #[test]
    fn search_budget_closes_after_first_search_with_results_across_rounds() {
        use std::fs;
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("logging.rs"), "fn logging() {}").unwrap();
        let synthesis = "Logging appears in logging.rs.";

        let mut rt = make_runtime_in(
            vec![
                "[search_code: logging initialization]",
                "Let me try another search.\n[search_code: logger initialization]",
                synthesis,
            ],
            tmp.path(),
        );
        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "find logging".into(),
            },
        );
        assert!(
            !has_failed(&events),
            "must not fail permanently: {events:?}"
        );

        let snapshot = rt.messages_snapshot();
        let all_user: String = snapshot
            .iter()
            .filter(|m| m.role == crate::llm::backend::Role::User)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            all_user.matches("=== tool_result: search_code ===").count(),
            1,
            "second search must be intercepted before another tool result"
        );
        assert!(
            all_user.contains("Search returned matches"),
            "closed-search guidance must be injected after the first successful search"
        );
        assert!(
            !snapshot
                .iter()
                .any(|m| m.content.contains("Let me try another search")),
            "narrated retry assistant message must be discarded from model context"
        );
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(last_assistant, Some(synthesis));
    }

    #[test]
    fn search_budget_closes_after_empty_retry_across_rounds() {
        // Phase 8.3: after two empty searches and the third attempt discarded, the runtime
        // now emits the insufficient-evidence terminal answer rather than letting the model
        // synthesize without any grounded evidence.
        use std::fs;
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("file.rs"), "fn unrelated() {}").unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: logging initialization]",
                "[search_code: logger initialization]",
                "Trying one more.\n[search_code: tracing]",
                // This response is never consumed — R4 fires before invoking the backend.
                "No matching code was found for those searches.",
            ],
            tmp.path(),
        );
        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "find logging".into(),
            },
        );
        assert!(
            !has_failed(&events),
            "must not fail permanently: {events:?}"
        );

        // Search budget behavior is unchanged.
        let snapshot = rt.messages_snapshot();
        let all_user: String = snapshot
            .iter()
            .filter(|m| m.role == crate::llm::backend::Role::User)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            all_user.matches("=== tool_result: search_code ===").count(),
            2,
            "first empty search and one retry should execute"
        );
        assert!(
            all_user.contains("allowed search retry also returned no matches"),
            "empty-retry terminal guidance must be injected"
        );
        assert!(
            !snapshot
                .iter()
                .any(|m| m.content.contains("Trying one more")),
            "third narrated search must be discarded from model context"
        );

        // Phase 8.3: runtime-owned insufficient-evidence terminal fires instead of model synthesis.
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::RuntimeTerminal {
                reason: RuntimeTerminalReason::InsufficientEvidence, ..
            })),
            "empty-search no-read turn must produce InsufficientEvidence terminal: {answer_source:?}"
        );
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some(insufficient_evidence_final_answer()),
            "last assistant message must be the runtime terminal, not model synthesis"
        );
    }

    #[test]
    fn search_budget_allows_second_search_when_first_empty() {
        // Controlled temp dir: no file matches "no_match_here" but one matches "find_me".
        // First search returns empty → second search must be allowed.
        use std::fs;
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("file.rs"), "fn find_me() {}").unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: no_match_here][search_code: find_me]",
                "Found it.",
            ],
            tmp.path(),
        );
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "search".into(),
            },
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            !snapshot
                .iter()
                .any(|m| m.content.contains("search budget exceeded")),
            "second search must be allowed when first returned empty"
        );
        // Both results land in the same accumulated user message, so count occurrences.
        let all_user: String = snapshot
            .iter()
            .filter(|m| m.role == crate::llm::backend::Role::User)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let result_count = all_user.matches("=== tool_result: search_code ===").count();
        assert_eq!(result_count, 2, "both searches must have tool results");
    }

    #[test]
    fn search_budget_blocks_third_search_regardless() {
        // Controlled temp dir: first two searches return empty, third must still be blocked.
        use std::fs;
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("file.rs"), "fn find_me() {}").unwrap();

        // first=empty, second=empty (allowed by budget), third=blocked regardless
        let mut rt = make_runtime_in(
            vec![
                "[search_code: no_match_a][search_code: no_match_b][search_code: find_me]",
                "Done.",
            ],
            tmp.path(),
        );
        collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "triple search".into(),
            },
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("search budget exceeded")),
            "third search must always be blocked regardless of prior results"
        );
    }

    #[test]
    fn read_cap_blocks_reads_beyond_limit() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("a.rs"), "fn a() {}\n").unwrap();
        fs::write(tmp.path().join("b.rs"), "fn b() {}\n").unwrap();
        fs::write(tmp.path().join("c.rs"), "fn c() {}\n").unwrap();
        fs::write(tmp.path().join("d.rs"), "fn d() {}\n").unwrap();

        // Reads a, b, c all succeed (within MAX_READS_PER_TURN = 3).
        // Read d is the first beyond the cap and must be blocked.
        let mut rt = make_runtime_in(
            vec![
                "[read_file: a.rs]",
                "[read_file: b.rs]",
                "[read_file: c.rs]",
                "[read_file: d.rs]",
                "I have read enough files.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "explore the files".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "turn must complete without failure: {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, RuntimeEvent::AnswerReady(_))),
            "turn must complete with AnswerReady: {events:?}"
        );

        let snapshot = rt.messages_snapshot();
        let all_user: String = snapshot
            .iter()
            .filter(|m| m.role == crate::llm::backend::Role::User)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(
            all_user.matches("=== tool_result: read_file ===").count(),
            3,
            "exactly three reads must succeed (a, b, c)"
        );
        assert!(
            all_user.contains("=== tool_error: read_file ===")
                && all_user.contains("read limit for this turn"),
            "fourth read must be blocked with the cap tool error"
        );
    }

    #[test]
    fn duplicate_read_is_blocked_within_same_turn() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();

        // Round 1: first read succeeds.
        // Round 2: model tries to read the same path again — must be blocked.
        // Round 3: model synthesizes from the evidence already in context.
        let mut rt = make_runtime_in(
            vec![
                "[read_file: engine.rs]",
                "[read_file: engine.rs]",
                "I already have the file contents in context.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "What is in engine.rs?".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "turn must complete without failure: {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, RuntimeEvent::AnswerReady(_))),
            "turn must complete with AnswerReady: {events:?}"
        );

        let snapshot = rt.messages_snapshot();

        // First read must produce a tool result.
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: read_file ===")),
            "first read must succeed and inject a tool result"
        );

        // Second read must be blocked with a tool error containing the dedup message.
        assert!(
            snapshot.iter().any(|m| {
                m.content.contains("=== tool_error: read_file ===")
                    && m.content.contains("already read this turn")
            }),
            "duplicate read must be blocked with the dedup tool error"
        );
    }

    #[test]
    fn edit_repair_correction_injected_on_garbled_repair_after_failure() {
        // First response: edit_file with empty search text — produces an Immediate tool error.
        // Second response: [edit_file] tags present but unrecognized delimiters (zero parse).
        // Engine must inject EDIT_REPAIR_CORRECTION rather than accepting as Direct.
        // Third response: synthesis after correction.
        let bad_edit = "[edit_file]\npath: foo.rs\n---replace---\nnew text\n[/edit_file]";
        let garbled_repair =
            "[edit_file]\npath: foo.rs\nFind: old text\nReplace: new text\n[/edit_file]";
        let synthesis = "I was unable to apply the edit.";

        let mut rt = make_runtime(vec![bad_edit, garbled_repair, synthesis]);
        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "edit foo.rs".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "must not fail permanently: {events:?}"
        );

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.starts_with("[runtime:correction]")
                    && m.content.contains("edit_file")),
            "edit repair correction must be injected: {snapshot:?}"
        );
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(last_assistant, Some(synthesis));
    }

    #[test]
    fn edit_old_new_content_format_requests_approval_and_executes() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("test_phase82.txt");
        fs::write(&file, "hello world").unwrap();

        let edit = "[edit_file]\npath: test_phase82.txt\nold content: hello world\nnew content: hello params\n[/edit_file]";
        let mut rt = make_runtime_in(vec![edit, "Updated."], tmp.path());

        let submit_events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Edit test_phase82.txt and change hello world to hello params".into(),
            },
        );
        assert!(
            !has_failed(&submit_events),
            "submit failed: {submit_events:?}"
        );
        assert!(
            submit_events
                .iter()
                .any(|e| matches!(e, RuntimeEvent::ApprovalRequired(p)
                if p.tool_name == "edit_file")),
            "edit must request approval instead of falling back to Direct: {submit_events:?}"
        );
        assert_eq!(fs::read_to_string(&file).unwrap(), "hello world");

        let approve_events = collect_events(&mut rt, RuntimeRequest::Approve);
        assert!(
            !has_failed(&approve_events),
            "approve failed: {approve_events:?}"
        );
        assert_eq!(fs::read_to_string(&file).unwrap(), "hello params");

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: edit_file ===")),
            "approved edit result must be injected: {snapshot:?}"
        );
    }

    #[test]
    fn approve_synthesizes_after_successful_mutation() {
        // After approving a write_file call, the model must be re-invoked for synthesis.
        // The synthesis response closes the loop and becomes the final assistant message.
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Write an existing file so edit_file has something to edit.
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "hello").unwrap();
        let path = f.path().to_string_lossy().into_owned();

        // edit_file payload: path\x00search\x00replace (null-byte separated)
        let payload = format!("{}\x00hello\x00world", path);

        // The synthesis response — model confirms what was done.
        let mut rt = make_runtime(vec!["Done, the edit has been applied."]);
        let before_count = rt.messages_snapshot().len();

        rt.set_pending_for_test(PendingAction {
            tool_name: "edit_file".into(),
            summary: format!("edit {path}"),
            risk: RiskLevel::Medium,
            payload,
        });

        let events = collect_events(&mut rt, RuntimeRequest::Approve);
        assert!(!has_failed(&events), "approve must not fail: {events:?}");

        // Synthesis must have fired — AssistantMessageChunk is emitted during synthesis.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, RuntimeEvent::AssistantMessageChunk(_))),
            "approve must trigger synthesis: no AssistantMessageChunk in events"
        );

        let snapshot = rt.messages_snapshot();
        // Snapshot must have grown: at minimum tool result + synthesis assistant message.
        assert!(
            snapshot.len() > before_count,
            "snapshot must grow after approve + synthesis"
        );
        // Tool result must be present.
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: edit_file ===")),
            "tool result must be in conversation after approve"
        );
        // The final assistant message must be the synthesis response, not a tool call.
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant);
        assert!(
            last_assistant
                .map(|m| m.content.contains("Done"))
                .unwrap_or(false),
            "last assistant message must be the synthesis response"
        );
    }

    // Phase 9.1.1 — bounded multi-step investigation

    #[test]
    fn two_candidate_reads_second_satisfies_evidence_admits_synthesis() {
        // Usage lookup: two search candidates (definition + usage).
        // First read is definition-only → recovery correction fires.
        // Second read is a usage candidate → evidence ready → synthesis admitted.
        // Validates that candidate_reads_count reaching 2 does not prematurely terminate
        // when the second read satisfies evidence.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("models")).unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("models").join("enums.py"),
            "class TaskStatus(str, Enum):\n    PENDING = \"pending\"\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("runner.py"),
            "from models.enums import TaskStatus\nif task.status == TaskStatus.PENDING:\n    run()\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: TaskStatus]",
                // Round 2: reads definition file first (definition-only candidate).
                // Runtime injects recovery correction pointing to runner.py.
                "[read_file: models/enums.py]",
                // Round 3: model follows correction and reads the usage file.
                "[read_file: services/runner.py]",
                "TaskStatus is used in services/runner.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is TaskStatus used?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "second candidate read satisfying evidence must admit synthesis: {answer_source:?}"
        );
        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("TaskStatus is used in services/runner.py.")
        );
    }

    #[test]
    fn two_candidate_reads_both_insufficient_terminates_cleanly() {
        // Usage lookup: three search candidates (two definition-only + one usage).
        // First read is definition-only → recovery correction fires pointing to usage file.
        // Model ignores correction and reads a second definition-only file.
        // After two candidate reads with evidence still not ready the runtime must
        // terminate cleanly with InsufficientEvidence — no further correction cycles.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("models")).unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("models").join("enums.py"),
            "class TaskStatus(str, Enum):\n    TODO = \"todo\"\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("models").join("alt_enums.py"),
            "class TaskStatus:\n    DONE = \"done\"\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("task_service.py"),
            "from models.enums import TaskStatus\nif task.status == TaskStatus.TODO:\n    pass\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: TaskStatus]",
                // Round 2: reads first definition file.
                // Runtime injects recovery correction pointing to task_service.py.
                "[read_file: models/enums.py]",
                // Round 3: model ignores correction and reads second definition file.
                // candidate_reads_count reaches 2, evidence still not ready.
                "[read_file: models/alt_enums.py]",
                // Round 4: model tries to synthesize without reading usage evidence.
                // Runtime must terminate with InsufficientEvidence — not fire another correction.
                "TaskStatus is defined in models/enums.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is TaskStatus used?".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "turn must terminate cleanly: {events:?}"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(
                answer_source,
                Some(AnswerSource::RuntimeTerminal {
                    reason: RuntimeTerminalReason::InsufficientEvidence,
                    ..
                })
            ),
            "two insufficient candidate reads must produce InsufficientEvidence: {answer_source:?}"
        );

        // The model's premature synthesis must not appear as the last assistant message.
        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some(ungrounded_investigation_final_answer()),
            "last assistant must be the runtime terminal, not model synthesis"
        );
    }

    #[test]
    fn third_candidate_read_after_two_insufficient_reads_is_blocked_pre_dispatch() {
        // Usage lookup: two definition-only reads exhaust the candidate-read budget
        // without useful evidence. If the model then tries a third distinct matched
        // candidate read instead of synthesizing, runtime must stop before dispatch.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("models")).unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("models").join("enums.py"),
            "class TaskStatus(str, Enum):\n    TODO = \"todo\"\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("models").join("alt_enums.py"),
            "class TaskStatus:\n    DONE = \"done\"\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("task_service.py"),
            "from models.enums import TaskStatus\nif task.status == TaskStatus.TODO:\n    pass\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: TaskStatus]",
                // First insufficient candidate read: recovery points to task_service.py.
                "[read_file: models/enums.py]",
                // Second insufficient candidate read: budget is now exhausted.
                "[read_file: models/alt_enums.py]",
                // Third distinct search candidate: must be blocked before dispatch.
                "[read_file: services/task_service.py]",
                "TaskStatus is used in services/task_service.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is TaskStatus used?".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "turn must terminate cleanly: {events:?}"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(
                answer_source,
                Some(AnswerSource::RuntimeTerminal {
                    reason: RuntimeTerminalReason::InsufficientEvidence,
                    ..
                })
            ),
            "third candidate read must terminate with InsufficientEvidence: {answer_source:?}"
        );

        let snapshot = rt.messages_snapshot();
        let all_user: String = snapshot
            .iter()
            .filter(|m| m.role == crate::llm::backend::Role::User)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            all_user.matches("=== tool_result: read_file ===").count(),
            2,
            "third candidate read must not dispatch"
        );
        assert!(
            all_user.contains("candidate read limit for this investigation reached"),
            "blocked third read must be recorded as a runtime tool error"
        );
        assert!(
            !all_user.contains("=== tool_result: read_file ===\npath: services/task_service.py"),
            "usage candidate must not be read after the two-candidate cap"
        );
    }

    // Phase 9.1.2 — Path-Scoped Investigation

    #[test]
    fn extract_investigation_path_scope_detects_in_pattern() {
        assert_eq!(
            extract_investigation_path_scope("Where is TaskStatus handled in sandbox/cli/"),
            Some("sandbox/cli/".into())
        );
        assert_eq!(
            extract_investigation_path_scope(
                "Find where logging is initialized in sandbox/services/"
            ),
            Some("sandbox/services/".into())
        );
        assert_eq!(
            extract_investigation_path_scope("Where is TaskStatus used in sandbox/"),
            Some("sandbox/".into())
        );
    }

    #[test]
    fn extract_investigation_path_scope_detects_the_before_path() {
        assert_eq!(
            extract_investigation_path_scope(
                "Find where database is configured in the sandbox/ folder"
            ),
            Some("sandbox/".into())
        );
    }

    #[test]
    fn extract_investigation_path_scope_detects_within_pattern() {
        assert_eq!(
            extract_investigation_path_scope("Find TaskStatus within sandbox/cli/"),
            Some("sandbox/cli/".into())
        );
    }

    #[test]
    fn extract_investigation_path_scope_rejects_plain_words() {
        // "in" followed by a token with no "/" → None
        assert_eq!(
            extract_investigation_path_scope("Find X in the application"),
            None
        );
        assert_eq!(
            extract_investigation_path_scope("Where is X used in context"),
            None
        );
        assert_eq!(
            extract_investigation_path_scope("What does run_turns do?"),
            None
        );
    }

    #[test]
    fn extract_investigation_path_scope_rejects_urls() {
        assert_eq!(
            extract_investigation_path_scope("Find X in https://example.com/path"),
            None
        );
    }

    #[test]
    fn extract_investigation_path_scope_returns_none_for_ambiguous_multiple_paths() {
        // Two qualifying path tokens → ambiguous → None
        assert_eq!(
            extract_investigation_path_scope("Find X in sandbox/a/ and in sandbox/b/"),
            None
        );
    }

    #[test]
    fn extract_investigation_path_scope_strips_trailing_punctuation() {
        assert_eq!(
            extract_investigation_path_scope("Where is TaskStatus in sandbox/cli?"),
            Some("sandbox/cli".into())
        );
        assert_eq!(
            extract_investigation_path_scope("Find X in sandbox/services/."),
            Some("sandbox/services/".into())
        );
    }

    #[test]
    fn path_scope_narrows_search_to_specified_directory() {
        // Files exist both inside and outside sandbox/cli/.
        // The query scopes to sandbox/cli/.
        // search_code must only receive candidates from sandbox/cli/,
        // so the file outside that directory never becomes a candidate.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/cli")).unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/models")).unwrap();
        fs::write(
            tmp.path().join("sandbox/cli/handler.py"),
            "if task.status == TaskStatus.PENDING:\n    handle(task)\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("sandbox/models/enums.py"),
            "class TaskStatus(str, Enum):\n    PENDING = \"pending\"\n",
        )
        .unwrap();

        // Model searches (no path in tool call — runtime injects sandbox/cli/).
        // Only sandbox/cli/handler.py matches; sandbox/models/enums.py is outside scope.
        // Model reads handler.py → evidence ready → synthesis admitted.
        let mut rt = make_runtime_in(
            vec![
                "[search_code: TaskStatus]",
                "[read_file: sandbox/cli/handler.py]",
                "TaskStatus is handled in sandbox/cli/handler.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is TaskStatus handled in sandbox/cli/".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");

        let snapshot = rt.messages_snapshot();

        // Search result must be present.
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_result: search_code ===")),
            "search must have executed"
        );

        // The scoped search must not surface the out-of-scope enums.py as a candidate.
        // Verify by checking that the read of handler.py satisfied evidence (ToolAssisted).
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "scoped search + read must admit synthesis: {answer_source:?}"
        );

        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("TaskStatus is handled in sandbox/cli/handler.py.")
        );
    }

    #[test]
    fn path_scope_after_list_dir_failure_keeps_search_candidates_inside_scope() {
        // Manual regression: "in the sandbox/ folder" must still produce sandbox/
        // as the prompt-derived upper bound after an initial list_dir failure.
        // The model later reads an out-of-scope matched-looking file; that read must
        // not satisfy evidence because it was never a scoped search candidate.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox")).unwrap();
        fs::create_dir_all(tmp.path().join("src/app")).unwrap();
        fs::write(
            tmp.path().join("sandbox").join("database.yaml"),
            "database:\n  url: sqlite:///sandbox.db\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("src/app").join("session.rs"),
            "/// Owns the active database handle and current session ID.\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[list_dir: .]",
                "[search_code: database]",
                "[read_file: src/app/session.rs]",
                "The database is configured in src/app/session.rs.",
                "[read_file: sandbox/database.yaml]",
                "The database is configured in sandbox/database.yaml.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where database is configured in the sandbox/ folder".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| {
                m.content.contains("=== tool_error: list_dir ===")
                    && m.content.contains("require search_code")
            }),
            "list_dir before scoped search must be blocked"
        );

        let search_result = snapshot
            .iter()
            .find(|m| m.content.contains("=== tool_result: search_code ==="))
            .map(|m| m.content.as_str())
            .unwrap_or("");
        assert!(
            search_result.contains("sandbox/database.yaml"),
            "scoped search must include the sandbox config candidate: {search_result}"
        );
        assert!(
            !search_result.contains("src/app/session.rs"),
            "scoped search must not include out-of-scope candidates: {search_result}"
        );

        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("The database is configured in sandbox/database.yaml.")
        );
    }

    // Phase 9.1.4 — Prompt Scope as Search Upper Bound

    #[test]
    fn path_is_within_scope_exact_match() {
        assert!(path_is_within_scope(
            "sandbox/services/",
            "sandbox/services/"
        ));
        assert!(path_is_within_scope("sandbox/services", "sandbox/services"));
        // trailing-slash variants must compare identically
        assert!(path_is_within_scope(
            "sandbox/services/",
            "sandbox/services"
        ));
        assert!(path_is_within_scope(
            "sandbox/services",
            "sandbox/services/"
        ));
    }

    #[test]
    fn path_is_within_scope_narrower_path_accepted() {
        assert!(path_is_within_scope(
            "sandbox/services/tasks/",
            "sandbox/services/"
        ));
        assert!(path_is_within_scope(
            "sandbox/cli/handlers/",
            "sandbox/cli/"
        ));
        assert!(path_is_within_scope("sandbox/", "sandbox/"));
    }

    #[test]
    fn path_is_within_scope_broader_path_rejected() {
        assert!(!path_is_within_scope("sandbox/", "sandbox/services/"));
        assert!(!path_is_within_scope("src/", "sandbox/services/"));
        assert!(!path_is_within_scope(".", "sandbox/services/"));
    }

    #[test]
    fn path_is_within_scope_orthogonal_path_rejected() {
        assert!(!path_is_within_scope("src/runtime/", "sandbox/services/"));
        assert!(!path_is_within_scope("models/", "services/"));
    }

    #[test]
    fn path_is_within_scope_boundary_guard_prevents_prefix_collision() {
        // "sandbox/service_extra" must NOT match scope "sandbox/service"
        assert!(!path_is_within_scope(
            "sandbox/service_extra/",
            "sandbox/service/"
        ));
        assert!(!path_is_within_scope(
            "sandbox/services_extended/",
            "sandbox/services/"
        ));
        // Legitimate subdirectory still passes
        assert!(path_is_within_scope(
            "sandbox/services/sub/",
            "sandbox/services/"
        ));
    }

    #[test]
    fn path_is_within_scope_absolute_path_rejected() {
        // Absolute paths can never be within a relative scope — always clamped.
        assert!(!path_is_within_scope(
            "/Users/project/sandbox/services/",
            "sandbox/services/"
        ));
        assert!(!path_is_within_scope("/abs/path/", "sandbox/"));
    }

    #[test]
    fn path_is_within_scope_parent_components_rejected() {
        assert!(!path_is_within_scope(
            "sandbox/services/../",
            "sandbox/services/"
        ));
        assert!(!path_is_within_scope(
            "sandbox/services/../../src/",
            "sandbox/services/"
        ));
        assert!(!path_is_within_scope(
            "sandbox/services/tasks/",
            "sandbox/services/../"
        ));
    }

    #[test]
    fn path_is_within_scope_dotslash_normalization() {
        // ./sandbox/services/ normalizes to sandbox/services/ — should match.
        assert!(path_is_within_scope(
            "./sandbox/services/",
            "sandbox/services/"
        ));
        assert!(path_is_within_scope(
            "sandbox/services/",
            "./sandbox/services/"
        ));
    }

    #[test]
    fn scope_upper_bound_clamps_broader_model_path() {
        // Verifies end-to-end that when a prompt scope is extracted and the search
        // produces results only within the scope, synthesis is admitted (ToolAssisted).
        // The injection (9.1.2 None arm) and the clamping guard (9.1.4 Some arm) both
        // live in the same match block; this test exercises the combined path.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::create_dir_all(tmp.path().join("models")).unwrap();
        fs::write(
            tmp.path().join("services").join("task_service.py"),
            "if task.status == TaskStatus.TODO:\n    pass\n",
        )
        .unwrap();
        // This file exists but is outside the prompt scope — must not appear as candidate.
        fs::write(
            tmp.path().join("models").join("enums.py"),
            "class TaskStatus(str, Enum):\n    TODO = \"todo\"\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                // Model issues search without a path; scope injection fires.
                "[search_code: TaskStatus]",
                "[read_file: services/task_service.py]",
                "TaskStatus is used in services/task_service.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is TaskStatus used in services/".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "scoped search must admit synthesis: {answer_source:?}"
        );
        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("TaskStatus is used in services/task_service.py.")
        );
    }

    // Scope enforcement: direct unit tests for the enforcement block.
    //
    // The codec never produces path: Some(...) for search calls, so the clamping arm
    // of the enforcement block cannot fire end-to-end. These tests exercise it directly
    // by constructing ToolInput values and applying the same enforcement logic inline.
    // path_is_within_scope is the underlying predicate; these tests verify the branch
    // outcomes of the match block (inject / clamp / preserve) as a whole.

    #[test]
    fn scope_enforcement_clamps_broader_parent_path() {
        // Scope: sandbox/services/, model path: sandbox/ (parent — broader).
        // Enforcement must clamp to sandbox/services/.
        let scope = "sandbox/services/";
        let mut path: Option<String> = Some("sandbox/".into());
        if let Some(ref p) = path.clone() {
            if !path_is_within_scope(p, scope) {
                path = Some(scope.to_string());
            }
        }
        assert_eq!(path.as_deref(), Some("sandbox/services/"));
    }

    #[test]
    fn scope_enforcement_clamps_parent_component_path() {
        // Scope: sandbox/services/, model path: sandbox/services/../ (parent escape).
        // Enforcement must clamp to sandbox/services/.
        let scope = "sandbox/services/";
        let mut path: Option<String> = Some("sandbox/services/../".into());
        if let Some(ref p) = path.clone() {
            if !path_is_within_scope(p, scope) {
                path = Some(scope.to_string());
            }
        }
        assert_eq!(path.as_deref(), Some("sandbox/services/"));
    }

    #[test]
    fn scope_enforcement_clamps_unrelated_path() {
        // Scope: sandbox/services/, model path: src/ (unrelated — orthogonal).
        // Enforcement must clamp to sandbox/services/.
        let scope = "sandbox/services/";
        let mut path: Option<String> = Some("src/".into());
        if let Some(ref p) = path.clone() {
            if !path_is_within_scope(p, scope) {
                path = Some(scope.to_string());
            }
        }
        assert_eq!(path.as_deref(), Some("sandbox/services/"));
    }

    #[test]
    fn scope_enforcement_preserves_exact_scope_path() {
        // Scope: sandbox/services/, model path: sandbox/services/ (exact).
        // Enforcement must preserve.
        let scope = "sandbox/services/";
        let mut path: Option<String> = Some("sandbox/services/".into());
        if let Some(ref p) = path.clone() {
            if !path_is_within_scope(p, scope) {
                path = Some(scope.to_string());
            }
        }
        assert_eq!(path.as_deref(), Some("sandbox/services/"));
    }

    #[test]
    fn scope_enforcement_preserves_child_path() {
        // Scope: sandbox/services/, model path: sandbox/services/tasks/ (child — narrower).
        // Enforcement must preserve the narrower model path.
        let scope = "sandbox/services/";
        let mut path: Option<String> = Some("sandbox/services/tasks/".into());
        if let Some(ref p) = path.clone() {
            if !path_is_within_scope(p, scope) {
                path = Some(scope.to_string());
            }
        }
        assert_eq!(path.as_deref(), Some("sandbox/services/tasks/"));
    }

    #[test]
    fn scope_enforcement_injects_when_path_absent() {
        // Scope present, path None → inject scope.
        let scope = "sandbox/services/";
        let mut path: Option<String> = None;
        if path.is_none() {
            path = Some(scope.to_string());
        }
        assert_eq!(path.as_deref(), Some("sandbox/services/"));
    }

    #[test]
    fn no_scope_search_behavior_unchanged() {
        // Prompt has no path scope (no "in dir/" pattern).
        // Runtime must not inject or clamp — standard search behavior applies.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::create_dir_all(tmp.path().join("models")).unwrap();
        fs::write(
            tmp.path().join("services").join("task_service.py"),
            "if task.status == TaskStatus.TODO:\n    pass\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("models").join("enums.py"),
            "class TaskStatus(str, Enum):\n    TODO = \"todo\"\n",
        )
        .unwrap();

        // Prompt has no scope — no injection or clamping.
        // Both files match; model reads the usage file → ToolAssisted.
        let mut rt = make_runtime_in(
            vec![
                "[search_code: TaskStatus]",
                "[read_file: services/task_service.py]",
                "TaskStatus is used in services/task_service.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is TaskStatus used?".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "no-scope turn must not fail: {events:?}"
        );

        let snapshot = rt.messages_snapshot();
        // Both in-scope and out-of-scope files must appear as candidates (no clamping).
        let search_result = snapshot
            .iter()
            .find(|m| m.content.contains("=== tool_result: search_code ==="))
            .map(|m| m.content.as_str())
            .unwrap_or("");
        assert!(
            search_result.contains("services/task_service.py"),
            "unscoped search must include services/task_service.py: {search_result}"
        );
        assert!(
            search_result.contains("models/enums.py"),
            "unscoped search must include models/enums.py (no clamping without scope): {search_result}"
        );

        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "unscoped search + read must admit synthesis: {answer_source:?}"
        );
    }

    #[test]
    fn scope_upper_bound_forced_broader_path_clamped_end_to_end() {
        // Forced failure-path validation from the spec:
        // Prompt: "Find where logging is initialized in sandbox/services/"
        // The scope extracts to sandbox/services/.
        // Model issues search without path (codec limitation — path always None),
        // runtime injects sandbox/services/ → only in-scope files become candidates.
        // Out-of-scope src/ files are never candidates.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(
            tmp.path().join("sandbox/services").join("logger.py"),
            "def initialize_logging():\n    logging.basicConfig()\n",
        )
        .unwrap();
        // Out-of-scope file that would match if scope was not enforced.
        fs::write(
            tmp.path().join("src").join("logger.py"),
            "def initialize_logging():\n    setup_logger()\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: logging]",
                "[read_file: sandbox/services/logger.py]",
                "Logging is initialized in sandbox/services/logger.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where logging is initialized in sandbox/services/".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");

        let snapshot = rt.messages_snapshot();
        let search_result = snapshot
            .iter()
            .find(|m| m.content.contains("=== tool_result: search_code ==="))
            .map(|m| m.content.as_str())
            .unwrap_or("");

        // Scope enforcement: only sandbox/services/ candidates must appear.
        assert!(
            search_result.contains("sandbox/services/logger.py"),
            "scoped search must include in-scope candidate: {search_result}"
        );
        assert!(
            !search_result.contains("src/logger.py"),
            "scoped search must exclude out-of-scope src/ candidate: {search_result}"
        );

        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Logging is initialized in sandbox/services/logger.py.")
        );
    }

    #[test]
    fn scope_upper_bound_clamped_to_cli_not_sandbox() {
        // Forced failure-path validation case 2:
        // Prompt: "Where is TaskStatus used in sandbox/cli/"
        // Model would search sandbox/ (broader) — clamp must restrict to sandbox/cli/.
        // Since codec produces path: None, injection fires and restricts to sandbox/cli/.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/cli")).unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/models")).unwrap();
        fs::write(
            tmp.path().join("sandbox/cli").join("handler.py"),
            "if task.status == TaskStatus.PENDING:\n    handle(task)\n",
        )
        .unwrap();
        // Out-of-scope: outside sandbox/cli/.
        fs::write(
            tmp.path().join("sandbox/models").join("enums.py"),
            "class TaskStatus(str, Enum):\n    PENDING = \"pending\"\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: TaskStatus]",
                "[read_file: sandbox/cli/handler.py]",
                "TaskStatus is used in sandbox/cli/handler.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is TaskStatus used in sandbox/cli/".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");

        let snapshot = rt.messages_snapshot();
        let search_result = snapshot
            .iter()
            .find(|m| m.content.contains("=== tool_result: search_code ==="))
            .map(|m| m.content.as_str())
            .unwrap_or("");

        assert!(
            search_result.contains("sandbox/cli/handler.py"),
            "scoped search must include in-scope candidate: {search_result}"
        );
        assert!(
            !search_result.contains("sandbox/models/enums.py"),
            "scoped search must exclude out-of-scope sandbox/models/ candidate: {search_result}"
        );
    }

    // Phase 9.1.3 — Candidate Selection Quality (import-only weak candidate rejection)

    #[test]
    fn looks_like_import_accepts_simple_import() {
        assert!(looks_like_import("import logging"));
        assert!(looks_like_import("import os, sys"));
        assert!(looks_like_import("  import logging"));
    }

    #[test]
    fn looks_like_import_accepts_from_import() {
        assert!(looks_like_import("from models.enums import TaskStatus"));
        assert!(looks_like_import("from . import utils"));
        assert!(looks_like_import("  from models.enums import TaskStatus"));
    }

    #[test]
    fn looks_like_import_rejects_usage_lines() {
        assert!(!looks_like_import(
            "if task.status == TaskStatus.TODO: pass"
        ));
        assert!(!looks_like_import("result = TaskStatus.COMPLETED"));
        assert!(!looks_like_import("logger = logging.getLogger(__name__)"));
    }

    #[test]
    fn looks_like_import_rejects_definition_lines() {
        assert!(!looks_like_import("class TaskStatus(str, Enum):"));
        assert!(!looks_like_import("def get_status(task):"));
    }

    #[test]
    fn import_only_candidate_rejected_when_non_import_candidate_exists() {
        // File A: only an import line → classified import-only.
        // File B: a usage line → classified as non-import candidate.
        // Model reads A first → correction fires pointing to B.
        // Model reads B → evidence ready → ToolAssisted.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("init")).unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("init").join("header.py"),
            "from models.enums import TaskStatus\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("task_service.py"),
            "if task.status == TaskStatus.TODO:\n    pass\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: TaskStatus]",
                // Round 2: reads import-only file first.
                // Runtime injects import-only correction pointing to task_service.py.
                "[read_file: init/header.py]",
                // Round 3: model follows correction and reads the usage file.
                "[read_file: services/task_service.py]",
                "TaskStatus is used in services/task_service.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is TaskStatus used?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "import-only rejection + non-import read must admit synthesis: {answer_source:?}"
        );
        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("TaskStatus is used in services/task_service.py.")
        );
    }

    #[test]
    fn import_only_fallback_accepts_when_all_candidates_are_import_only() {
        // Single candidate: only an import line.
        // has_non_import_candidates == false → import-only gate does not fire.
        // File is accepted as evidence → ToolAssisted.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("models")).unwrap();
        fs::write(
            tmp.path().join("models").join("enums.py"),
            "from models.enums import TaskStatus\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: TaskStatus]",
                "[read_file: models/enums.py]",
                "TaskStatus is imported from models.enums.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is TaskStatus used?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "all-import-only candidates must fall back to accepting the read: {answer_source:?}"
        );
        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("TaskStatus is imported from models.enums.")
        );
    }

    // Phase 9.2.1 — InvestigationMode enum + Config lookup mode

    #[test]
    fn detect_investigation_mode_returns_usage_lookup() {
        assert!(matches!(
            detect_investigation_mode("Where is TaskStatus used?"),
            InvestigationMode::UsageLookup
        ));
        assert!(matches!(
            detect_investigation_mode("Find all references to build_report"),
            InvestigationMode::UsageLookup
        ));
        assert!(matches!(
            detect_investigation_mode("Where does TaskStatus appear?"),
            InvestigationMode::UsageLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_returns_config_lookup() {
        assert!(matches!(
            detect_investigation_mode("Where is the database configured?"),
            InvestigationMode::ConfigLookup
        ));
        assert!(matches!(
            detect_investigation_mode("Find where logging configuration lives"),
            InvestigationMode::ConfigLookup
        ));
        assert!(matches!(
            detect_investigation_mode("How is the connection configured?"),
            InvestigationMode::ConfigLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_returns_initialization_lookup() {
        assert!(matches!(
            detect_investigation_mode("Find where logging is initialized"),
            InvestigationMode::InitializationLookup
        ));
        assert!(matches!(
            detect_investigation_mode("Find logging initialization"),
            InvestigationMode::InitializationLookup
        ));
        assert!(matches!(
            detect_investigation_mode("Find code that can initialize logging"),
            InvestigationMode::InitializationLookup
        ));
        assert!(matches!(
            detect_investigation_mode("Find where logging is initialised"),
            InvestigationMode::General
        ));
    }

    #[test]
    fn detect_investigation_mode_returns_definition_lookup() {
        assert!(matches!(
            detect_investigation_mode("Where is TaskStatus defined?"),
            InvestigationMode::DefinitionLookup
        ));
        assert!(matches!(
            detect_investigation_mode("Where is the TaskRunner declared?"),
            InvestigationMode::DefinitionLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_returns_general() {
        assert!(matches!(
            detect_investigation_mode("What does run_turns do?"),
            InvestigationMode::General
        ));
        assert!(matches!(
            detect_investigation_mode("Explain the TaskRunner"),
            InvestigationMode::General
        ));
    }

    #[test]
    fn detect_investigation_mode_usage_priority_over_config() {
        // "configured" + "used" in same prompt — UsageLookup wins (higher priority).
        assert!(matches!(
            detect_investigation_mode("Where is the configured value used?"),
            InvestigationMode::UsageLookup
        ));
        assert!(matches!(
            detect_investigation_mode("Where is configuration used?"),
            InvestigationMode::UsageLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_usage_priority_over_initialization() {
        assert!(matches!(
            detect_investigation_mode("Where is logging initialization used?"),
            InvestigationMode::UsageLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_config_priority_over_definition() {
        assert!(matches!(
            detect_investigation_mode("Where is config defined?"),
            InvestigationMode::ConfigLookup
        ));
        assert!(matches!(
            detect_investigation_mode("Find config for logging"),
            InvestigationMode::ConfigLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_config_priority_over_initialization() {
        assert!(matches!(
            detect_investigation_mode("Find where logging configuration is initialized"),
            InvestigationMode::ConfigLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_initialization_priority_over_definition() {
        assert!(matches!(
            detect_investigation_mode("Where is initialization defined?"),
            InvestigationMode::InitializationLookup
        ));
    }

    #[test]
    fn contains_initialization_term_matches_exact_allowed_substrings_only() {
        assert!(contains_initialization_term("def initialize_logging():"));
        assert!(contains_initialization_term(
            "# logging is initialized here"
        ));
        assert!(contains_initialization_term("logging initialization entry"));
        assert!(!contains_initialization_term("setup_logging()"));
        assert!(!contains_initialization_term("bootstrap logging"));
        assert!(!contains_initialization_term("logging is initialised here"));
    }

    #[test]
    fn is_config_file_accepts_standard_extensions() {
        assert!(is_config_file("config/database.yaml"));
        assert!(is_config_file("config/app.yml"));
        assert!(is_config_file("Cargo.toml"));
        assert!(is_config_file("config/settings.json"));
        assert!(is_config_file("config/app.ini"));
        assert!(is_config_file("deploy/app.cfg"));
        assert!(is_config_file("config/logging.conf"));
        assert!(is_config_file("config/db.properties"));
    }

    #[test]
    fn is_config_file_accepts_env_dotfiles() {
        assert!(is_config_file(".env"));
        assert!(is_config_file("config/.env"));
        assert!(!is_config_file(".env.local"));
        assert!(!is_config_file(".env.production"));
    }

    #[test]
    fn is_config_file_rejects_source_files() {
        assert!(!is_config_file("services/task_service.py"));
        assert!(!is_config_file("src/runtime/engine.rs"));
        assert!(!is_config_file("models/enums.py"));
        assert!(!is_config_file("main.go"));
    }

    #[test]
    fn config_lookup_non_config_read_triggers_recovery_to_config_file() {
        // Config lookup: two candidates — a source file and a config file.
        // Model reads the source file first → runtime injects config recovery pointing to YAML.
        // Model follows recovery and reads the config file → evidence ready → ToolAssisted.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::create_dir_all(tmp.path().join("config")).unwrap();
        fs::write(
            tmp.path().join("services").join("database.py"),
            "DATABASE_URL = os.getenv(\"DATABASE_URL\")\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("config").join("database.yaml"),
            "database:\n  url: postgres://localhost/mydb\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: database]",
                // Round 2: model reads source file first.
                // Runtime injects config recovery pointing to database.yaml.
                "[read_file: services/database.py]",
                // Round 3: model follows recovery and reads the config file.
                "[read_file: config/database.yaml]",
                "The database is configured in config/database.yaml.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is the database configured?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "config recovery + config read must admit synthesis: {answer_source:?}"
        );
        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("The database is configured in config/database.yaml.")
        );
    }

    #[test]
    fn config_lookup_second_non_config_candidate_after_recovery_is_not_accepted() {
        // Config lookup: config candidate exists, but the model ignores the config recovery
        // and reads a second non-config candidate. The second read must remain insufficient;
        // after two candidate reads the bounded investigation terminates cleanly.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::create_dir_all(tmp.path().join("config")).unwrap();
        fs::write(
            tmp.path().join("services").join("database.py"),
            "database = os.getenv(\"DATABASE_URL\")\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("database_alt.py"),
            "database = load_from_environment()\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("config").join("database.yaml"),
            "database:\n  url: postgres://localhost/mydb\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: database]",
                "[read_file: services/database.py]",
                "[read_file: services/database_alt.py]",
                "The database is configured in services/database_alt.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is the database configured?".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "turn must terminate cleanly: {events:?}"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(
                answer_source,
                Some(AnswerSource::RuntimeTerminal {
                    reason: RuntimeTerminalReason::InsufficientEvidence,
                    ..
                })
            ),
            "second non-config candidate must not satisfy config evidence: {answer_source:?}"
        );

        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some(ungrounded_investigation_final_answer()),
            "last assistant must be the runtime terminal, not model synthesis"
        );
    }

    #[test]
    fn config_lookup_no_config_candidates_degrades_cleanly() {
        // Config lookup triggered, but no config-file candidates exist (source files only).
        // has_non_config_candidates = true, config_file_candidates is empty.
        // Gate 2 does not fire — source file read is accepted → ToolAssisted.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("services").join("database.py"),
            "DATABASE_URL = os.getenv(\"DATABASE_URL\")\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: database]",
                "[read_file: services/database.py]",
                "The database connection is set up in services/database.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is the database configured?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "config lookup with no config candidates must degrade to acceptance: {answer_source:?}"
        );
        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("The database connection is set up in services/database.py.")
        );
    }

    // Phase 9.2.2 — Narrow Action-Specific Lookup Satisfaction: Initialization Lookup

    #[test]
    fn initialization_lookup_non_initialization_read_triggers_recovery() {
        // Initialization lookup: two source candidates, but only one matched line
        // contains an exact initialization term. Reading the other candidate first
        // must trigger one bounded recovery to the initialization candidate.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("services").join("logging_factory.py"),
            "logger = logging.getLogger(__name__)\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("logging_setup.py"),
            "def initialize_logging():\n    logging.basicConfig(level=logging.INFO)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: logging]",
                "[read_file: services/logging_factory.py]",
                "[read_file: services/logging_setup.py]",
                "Logging is initialized in services/logging_setup.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where logging is initialized".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "initialization recovery + initialization read must admit synthesis: {answer_source:?}"
        );

        let snapshot = rt.messages_snapshot();
        let expected_recovery_path = tmp
            .path()
            .join("services")
            .join("logging_setup.py")
            .to_string_lossy()
            .into_owned();
        assert!(
            snapshot.iter().any(|m| {
                m.content.contains("This is an initialization lookup")
                    && m.content
                        .contains(&format!("[read_file: {expected_recovery_path}]"))
            }),
            "runtime must inject bounded initialization recovery"
        );
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Logging is initialized in services/logging_setup.py.")
        );
    }

    #[test]
    fn initialization_lookup_second_non_initialization_after_recovery_is_not_accepted() {
        // Initialization lookup: initialization candidate exists, but the model ignores
        // recovery and reads a second non-initialization candidate. That second read must
        // remain insufficient; after two candidate reads the runtime terminates cleanly.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("services").join("logging_factory.py"),
            "logger = logging.getLogger(__name__)\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("logging_reader.py"),
            "logging.getLogger(\"reader\")\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("logging_setup.py"),
            "def initialize_logging():\n    logging.basicConfig(level=logging.INFO)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: logging]",
                "[read_file: services/logging_factory.py]",
                "[read_file: services/logging_reader.py]",
                "Logging is initialized in services/logging_reader.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where logging is initialized".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "turn must terminate cleanly: {events:?}"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(
                answer_source,
                Some(AnswerSource::RuntimeTerminal {
                    reason: RuntimeTerminalReason::InsufficientEvidence,
                    ..
                })
            ),
            "second non-initialization candidate must not satisfy evidence: {answer_source:?}"
        );

        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some(ungrounded_investigation_final_answer()),
            "last assistant must be the runtime terminal, not model synthesis"
        );
    }

    #[test]
    fn initialization_lookup_no_initialization_candidates_degrades_cleanly() {
        // Initialization lookup triggered, but no matched line contains an exact
        // initialization term. Gate 3 does not fire — existing candidate-read
        // behavior is preserved.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("services").join("logging_factory.py"),
            "logger = logging.getLogger(__name__)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: logging]",
                "[read_file: services/logging_factory.py]",
                "Logging is handled in services/logging_factory.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where logging is initialized".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "initialization lookup with no initialization candidates must degrade: {answer_source:?}"
        );
        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Logging is handled in services/logging_factory.py.")
        );
    }

    #[test]
    fn initialization_lookup_path_scope_keeps_candidates_inside_scope() {
        // Prompt scope must remain the upper bound. The out-of-scope initialization
        // file is stronger-looking but must not appear in search candidates.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/other")).unwrap();
        fs::write(
            tmp.path()
                .join("sandbox/services")
                .join("logging_factory.py"),
            "logger = logging.getLogger(__name__)\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("sandbox/services").join("logging_setup.py"),
            "def initialize_logging():\n    logging.basicConfig(level=logging.INFO)\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("sandbox/other").join("logging_setup.py"),
            "def initialize_logging():\n    logging.basicConfig(level=logging.DEBUG)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: logging]",
                "[read_file: sandbox/services/logging_factory.py]",
                "[read_file: sandbox/services/logging_setup.py]",
                "Logging is initialized in sandbox/services/logging_setup.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where logging is initialized in sandbox/services/".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        let search_result = snapshot
            .iter()
            .find(|m| m.content.contains("=== tool_result: search_code ==="))
            .map(|m| m.content.as_str())
            .unwrap_or("");
        assert!(
            search_result.contains("sandbox/services/logging_factory.py"),
            "scoped search must include in-scope non-initialization candidate: {search_result}"
        );
        assert!(
            search_result.contains("sandbox/services/logging_setup.py"),
            "scoped search must include in-scope initialization candidate: {search_result}"
        );
        assert!(
            !search_result.contains("sandbox/other/logging_setup.py"),
            "scoped search must exclude out-of-scope initialization candidate: {search_result}"
        );

        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Logging is initialized in sandbox/services/logging_setup.py.")
        );
    }

    // Phase 9.2.3 — CreateLookup

    #[test]
    fn detect_investigation_mode_returns_create_lookup() {
        assert!(matches!(
            detect_investigation_mode("Where is the session created?"),
            InvestigationMode::CreateLookup
        ));
        assert!(matches!(
            detect_investigation_mode("Find where tasks are created"),
            InvestigationMode::CreateLookup
        ));
        assert!(matches!(
            detect_investigation_mode("Where does task creation happen?"),
            InvestigationMode::CreateLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_create_priority_over_definition() {
        // "created" + "defined" in same prompt — CreateLookup wins (higher priority).
        assert!(matches!(
            detect_investigation_mode("Where is the session created and defined?"),
            InvestigationMode::CreateLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_initialization_priority_over_create() {
        // "initialized" + "created" in same prompt — InitializationLookup wins.
        assert!(matches!(
            detect_investigation_mode("Find where the session is initialized and created"),
            InvestigationMode::InitializationLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_usage_priority_over_create() {
        // "used" + "created" in same prompt — UsageLookup wins.
        assert!(matches!(
            detect_investigation_mode("Where is the session used and created?"),
            InvestigationMode::UsageLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_config_priority_over_create() {
        // "configured" + "created" in same prompt — ConfigLookup wins.
        assert!(matches!(
            detect_investigation_mode("Where is the session configured and created?"),
            InvestigationMode::ConfigLookup
        ));
    }

    #[test]
    fn contains_create_term_matches_exact_allowed_substrings_only() {
        // Exact allowed terms.
        assert!(contains_create_term("db.create(session)"));
        assert!(contains_create_term("session was created here"));
        assert!(contains_create_term("handles session creation"));
        // Case insensitive.
        assert!(contains_create_term("Session.Create()"));
        assert!(contains_create_term("CREATED_AT timestamp"));
        // Noisy: substring of longer word — these DO match (substring semantics, same as initialization).
        assert!(contains_create_term("recreate the session"));
        assert!(contains_create_term("createTable migration"));
        // Not a create term.
        assert!(!contains_create_term("def handle_session(s):"));
        assert!(!contains_create_term("return session_id"));
    }

    #[test]
    fn create_lookup_non_create_read_triggers_recovery_to_create_file() {
        // File A: no create-term matches → non-create candidate.
        // File B: a create-term match → create candidate.
        // Model reads A first → recovery fires pointing to B.
        // Model reads B → evidence ready → ToolAssisted.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::create_dir_all(tmp.path().join("storage")).unwrap();
        fs::write(
            tmp.path().join("services").join("task_handler.py"),
            "def handle_task(task):\n    task.run()\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("storage").join("task_store.py"),
            "def store_task(task):\n    db.create(task)\n    return task.id\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: task]",
                // Reads the non-create file first — recovery fires.
                "[read_file: services/task_handler.py]",
                // Follows recovery, reads the create file.
                "[read_file: storage/task_store.py]",
                "Tasks are created in storage/task_store.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are tasks created?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");

        let snapshot = rt.messages_snapshot();
        // Recovery correction must appear.
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("creation lookup")
                    && m.content.contains("storage/task_store.py")),
            "create recovery correction must point to the create candidate"
        );

        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "create lookup + recovery + create read must admit synthesis: {answer_source:?}"
        );
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Tasks are created in storage/task_store.py.")
        );
    }

    #[test]
    fn create_lookup_no_create_candidates_degrades_cleanly() {
        // All candidates have no create-term matches.
        // Gate does not fire — any read is accepted (fallback behavior).
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("services").join("task_handler.py"),
            "def handle_task(task):\n    task.run()\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: task]",
                "[read_file: services/task_handler.py]",
                "Tasks are handled in services/task_handler.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are tasks created?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "create lookup with no create candidates must degrade to acceptance: {answer_source:?}"
        );
        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Tasks are handled in services/task_handler.py.")
        );
    }

    #[test]
    fn create_lookup_second_non_create_candidate_after_recovery_is_not_accepted() {
        // After one recovery the correction flag is set.
        // A second non-create read falls through the gate without accepting.
        // With candidate_reads_count == 2 and evidence_ready false, the runtime
        // terminates with InsufficientEvidence.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::create_dir_all(tmp.path().join("handlers")).unwrap();
        fs::create_dir_all(tmp.path().join("storage")).unwrap();
        fs::write(
            tmp.path().join("services").join("runner.py"),
            "def run_task(task):\n    task.execute()\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("handlers").join("task_handler.py"),
            "def handle_task(task):\n    pass\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("storage").join("task_store.py"),
            "def store_task(task):\n    db.create(task)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: task]",
                // First read: non-create → recovery fires pointing to task_store.py.
                "[read_file: services/runner.py]",
                // Second read: another non-create (ignores recovery, reads wrong file).
                "[read_file: handlers/task_handler.py]",
                // Model attempts synthesis — candidate limit hit; runtime terminates.
                "Tasks run in services/runner.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are tasks created?".into(),
            },
        );

        // Runtime terminates with InsufficientEvidence (not a runtime Failed).
        assert!(!has_failed(&events), "must terminate cleanly: {events:?}");
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(
                answer_source,
                Some(AnswerSource::RuntimeTerminal {
                    reason: RuntimeTerminalReason::InsufficientEvidence,
                    ..
                })
            ),
            "two non-create reads must terminate with InsufficientEvidence: {answer_source:?}"
        );
    }

    #[test]
    fn create_lookup_noisy_create_term_in_comment_still_classifies_as_create() {
        // A line like "# TODO: create session handling" contains "create" as substring.
        // The classification is structural/substring — comments match the same as code.
        // This tests the known noisy behavior described in the spec.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::create_dir_all(tmp.path().join("models")).unwrap();
        fs::write(
            tmp.path().join("services").join("session_service.py"),
            "# TODO: create session handling\ndef get_session(sid):\n    return db.get(sid)\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("models").join("session.py"),
            "class Session:\n    pass\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: session]",
                // Comment-containing file is a create candidate (substring match).
                // Model reads it directly — evidence accepted.
                "[read_file: services/session_service.py]",
                "Sessions are handled in services/session_service.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are sessions created?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        // No recovery should fire — the comment-line file is a create candidate.
        let snapshot = rt.messages_snapshot();
        assert!(
            !snapshot
                .iter()
                .any(|m| m.content.contains("creation lookup")),
            "no recovery expected when create candidate is read first"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "create candidate read must admit synthesis: {answer_source:?}"
        );
    }

    // Phase 9.2.4 — RegisterLookup

    #[test]
    fn detect_investigation_mode_returns_register_lookup() {
        assert!(matches!(
            detect_investigation_mode("Where is the command registered?"),
            InvestigationMode::RegisterLookup
        ));
        assert!(matches!(
            detect_investigation_mode("Find where handlers register commands"),
            InvestigationMode::RegisterLookup
        ));
        assert!(matches!(
            detect_investigation_mode("Where does command registration happen?"),
            InvestigationMode::RegisterLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_create_priority_over_register() {
        // "created" + "registered" in same prompt — CreateLookup wins (higher priority).
        assert!(matches!(
            detect_investigation_mode("Where is the command created and registered?"),
            InvestigationMode::CreateLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_register_priority_over_definition() {
        // "registered" + "defined" in same prompt — RegisterLookup wins.
        assert!(matches!(
            detect_investigation_mode("Where is the command registered and defined?"),
            InvestigationMode::RegisterLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_usage_priority_over_register() {
        assert!(matches!(
            detect_investigation_mode("Where is the registered command used?"),
            InvestigationMode::UsageLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_config_priority_over_register() {
        assert!(matches!(
            detect_investigation_mode("Where is command registration configured?"),
            InvestigationMode::ConfigLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_initialization_priority_over_register() {
        assert!(matches!(
            detect_investigation_mode("Find where command registration is initialized"),
            InvestigationMode::InitializationLookup
        ));
    }

    #[test]
    fn contains_register_term_matches_exact_allowed_substrings_only() {
        // Exact allowed terms.
        assert!(contains_register_term("registry.register(command)"));
        assert!(contains_register_term("command was registered here"));
        assert!(contains_register_term("command registration lives here"));
        // Case insensitive.
        assert!(contains_register_term("Registry.Register(command)"));
        assert!(contains_register_term("REGISTERED_COMMANDS"));
        // Noisy: substring of longer word — these DO match (substring semantics).
        assert!(contains_register_term("reregister command handlers"));
        assert!(contains_register_term("registration_notes = []"));
        // Not a register term.
        assert!(!contains_register_term("def handle_command(command):"));
        assert!(!contains_register_term("return command_id"));
    }

    #[test]
    fn register_lookup_non_register_read_triggers_recovery_to_register_file() {
        // File A: no register-term matches → non-register candidate.
        // File B: a register-term match → register candidate.
        // Model reads A first → recovery fires pointing to B.
        // Model reads B → evidence ready → ToolAssisted.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("cli")).unwrap();
        fs::write(
            tmp.path().join("cli").join("handlers.py"),
            "def handle_command(command):\n    return command.run()\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("cli").join("registry.py"),
            "def wire_command(command):\n    registry.register(command)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: command]",
                // Reads the non-register file first — recovery fires.
                "[read_file: cli/handlers.py]",
                // Follows recovery, reads the register file.
                "[read_file: cli/registry.py]",
                "Commands are registered in cli/registry.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are commands registered?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("registration lookup")
                    && m.content.contains("cli/registry.py")),
            "register recovery correction must point to the register candidate"
        );

        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "register lookup + recovery + register read must admit synthesis: {answer_source:?}"
        );
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Commands are registered in cli/registry.py.")
        );
    }

    #[test]
    fn register_lookup_no_register_candidates_degrades_cleanly() {
        // All candidates have no register-term matches.
        // Gate does not fire — any read is accepted (fallback behavior).
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("cli")).unwrap();
        fs::write(
            tmp.path().join("cli").join("handlers.py"),
            "def handle_command(command):\n    return command.run()\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: command]",
                "[read_file: cli/handlers.py]",
                "Commands are handled in cli/handlers.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are commands registered?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "register lookup with no register candidates must degrade to acceptance: {answer_source:?}"
        );
        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Commands are handled in cli/handlers.py.")
        );
    }

    #[test]
    fn register_lookup_second_non_register_candidate_after_recovery_is_not_accepted() {
        // After one recovery the correction flag is set.
        // A second non-register read falls through the gate without accepting.
        // With candidate_reads_count == 2 and evidence_ready false, the runtime
        // terminates with InsufficientEvidence.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("cli")).unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("cli").join("handlers.py"),
            "def handle_command(command):\n    return command.run()\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("command_runner.py"),
            "def run_command(command):\n    command.run()\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("cli").join("registry.py"),
            "def wire_command(command):\n    registry.register(command)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: command]",
                // First read: non-register → recovery fires pointing to registry.py.
                "[read_file: cli/handlers.py]",
                // Second read: another non-register (ignores recovery, reads wrong file).
                "[read_file: services/command_runner.py]",
                // Model attempts synthesis — candidate limit hit; runtime terminates.
                "Commands are registered in cli/handlers.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are commands registered?".into(),
            },
        );

        assert!(!has_failed(&events), "must terminate cleanly: {events:?}");
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(
                answer_source,
                Some(AnswerSource::RuntimeTerminal {
                    reason: RuntimeTerminalReason::InsufficientEvidence,
                    ..
                })
            ),
            "two non-register reads must terminate with InsufficientEvidence: {answer_source:?}"
        );
    }

    #[test]
    fn register_lookup_noisy_register_term_in_comment_still_classifies_as_register() {
        // A line like "# TODO: register command handler" contains "register".
        // The classification is structural/substring — comments match the same as code.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("cli")).unwrap();
        fs::write(
            tmp.path().join("cli").join("commands.py"),
            "# TODO: register command handler\ndef command_handler(command):\n    return command.run()\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: command]",
                // Comment-containing file is a register candidate (substring match).
                // Model reads it directly — evidence accepted.
                "[read_file: cli/commands.py]",
                "Commands are handled in cli/commands.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are commands registered?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        assert!(
            !snapshot
                .iter()
                .any(|m| m.content.contains("registration lookup")),
            "no recovery expected when register candidate is read first"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "register candidate read must admit synthesis: {answer_source:?}"
        );
    }

    #[test]
    fn register_lookup_path_scope_keeps_candidates_inside_scope() {
        // Prompt scope must remain the upper bound. The out-of-scope registration
        // file is stronger-looking but must not appear in search candidates.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/cli")).unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
        fs::write(
            tmp.path().join("sandbox/cli").join("commands.py"),
            "def command_handler(command):\n    return command.run()\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("sandbox/cli").join("registry.py"),
            "def wire_command(command):\n    registry.register(command)\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("sandbox/services").join("registry.py"),
            "def wire_command(command):\n    registry.register(command)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: command]",
                "[read_file: sandbox/cli/commands.py]",
                "[read_file: sandbox/cli/registry.py]",
                "Commands are registered in sandbox/cli/registry.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where commands are registered in sandbox/cli/".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        let search_result = snapshot
            .iter()
            .find(|m| m.content.contains("=== tool_result: search_code ==="))
            .map(|m| m.content.as_str())
            .unwrap_or("");
        assert!(
            search_result.contains("sandbox/cli/commands.py"),
            "scoped search must include in-scope non-register candidate: {search_result}"
        );
        assert!(
            search_result.contains("sandbox/cli/registry.py"),
            "scoped search must include in-scope register candidate: {search_result}"
        );
        assert!(
            !search_result.contains("sandbox/services/registry.py"),
            "scoped search must exclude out-of-scope register candidate: {search_result}"
        );

        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Commands are registered in sandbox/cli/registry.py.")
        );
    }

    // Phase 9.2.5 — LoadLookup

    #[test]
    fn detect_investigation_mode_returns_load_lookup() {
        assert!(matches!(
            detect_investigation_mode("Where is the session loaded?"),
            InvestigationMode::LoadLookup
        ));
        assert!(matches!(
            detect_investigation_mode("Find where session loading happens"),
            InvestigationMode::LoadLookup
        ));
        assert!(matches!(
            detect_investigation_mode("Where do handlers load sessions?"),
            InvestigationMode::LoadLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_register_priority_over_load() {
        // "registered" + "loaded" in same prompt — RegisterLookup wins (higher priority).
        assert!(matches!(
            detect_investigation_mode("Where is the command registered and loaded?"),
            InvestigationMode::RegisterLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_load_priority_over_definition() {
        // "loaded" + "defined" in same prompt — LoadLookup wins.
        assert!(matches!(
            detect_investigation_mode("Where is the session loaded and defined?"),
            InvestigationMode::LoadLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_usage_priority_over_load() {
        assert!(matches!(
            detect_investigation_mode("Where is the loaded session used?"),
            InvestigationMode::UsageLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_config_priority_over_load() {
        assert!(matches!(
            detect_investigation_mode("Where is loaded config configured?"),
            InvestigationMode::ConfigLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_initialization_priority_over_load() {
        assert!(matches!(
            detect_investigation_mode("Find where session loading is initialized"),
            InvestigationMode::InitializationLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_create_priority_over_load() {
        assert!(matches!(
            detect_investigation_mode("Find where the loaded session is created"),
            InvestigationMode::CreateLookup
        ));
    }

    #[test]
    fn contains_load_term_matches_exact_allowed_substrings_only() {
        // Exact allowed terms.
        assert!(contains_load_term("session = load_session(session_id)"));
        assert!(contains_load_term("session was loaded here"));
        assert!(contains_load_term("session loading happens here"));
        // Case insensitive.
        assert!(contains_load_term("Session.Load()"));
        assert!(contains_load_term("LOADED_SESSION"));
        // Noisy: substring of longer word — these DO match (substring semantics).
        assert!(contains_load_term("session loader"));
        assert!(contains_load_term("reload session"));
        assert!(contains_load_term("autoload session"));
        // Not a load term.
        assert!(!contains_load_term("def handle_session(session):"));
        assert!(!contains_load_term("return session_id"));
    }

    #[test]
    fn load_lookup_non_load_read_triggers_recovery_to_load_file() {
        // File A: no load-term matches → non-load candidate.
        // File B: a load-term match → load candidate.
        // Model reads A first → recovery fires pointing to B.
        // Model reads B → evidence ready → ToolAssisted.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("services").join("session_handler.py"),
            "def handle_session(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("session_loader.py"),
            "def get_session(session_id):\n    return load_session(session_id)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: session]",
                // Reads the non-load file first — recovery fires.
                "[read_file: services/session_handler.py]",
                // Follows recovery, reads the load file.
                "[read_file: services/session_loader.py]",
                "Sessions are loaded in services/session_loader.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are sessions loaded?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| m.content.contains("load lookup")
                && m.content.contains("services/session_loader.py")),
            "load recovery correction must point to the load candidate"
        );

        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "load lookup + recovery + load read must admit synthesis: {answer_source:?}"
        );
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Sessions are loaded in services/session_loader.py.")
        );
    }

    #[test]
    fn load_lookup_no_load_candidates_degrades_cleanly() {
        // All candidates have no load-term matches.
        // Gate does not fire — any read is accepted (fallback behavior).
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("services").join("session_handler.py"),
            "def handle_session(session):\n    return session.id\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: session]",
                "[read_file: services/session_handler.py]",
                "Sessions are handled in services/session_handler.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are sessions loaded?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "load lookup with no load candidates must degrade to acceptance: {answer_source:?}"
        );
        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Sessions are handled in services/session_handler.py.")
        );
    }

    #[test]
    fn load_lookup_second_non_load_candidate_after_recovery_is_not_accepted() {
        // After one recovery the correction flag is set.
        // A second non-load read falls through the gate without accepting.
        // With candidate_reads_count == 2 and evidence_ready false, the runtime
        // terminates with InsufficientEvidence.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::create_dir_all(tmp.path().join("controllers")).unwrap();
        fs::write(
            tmp.path().join("services").join("session_handler.py"),
            "def handle_session(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("controllers").join("session_controller.py"),
            "def show_session(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("session_loader.py"),
            "def get_session(session_id):\n    return load_session(session_id)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: session]",
                // First read: non-load → recovery fires pointing to session_loader.py.
                "[read_file: services/session_handler.py]",
                // Second read: another non-load (ignores recovery, reads wrong file).
                "[read_file: controllers/session_controller.py]",
                // Model attempts synthesis — candidate limit hit; runtime terminates.
                "Sessions are loaded in services/session_handler.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are sessions loaded?".into(),
            },
        );

        assert!(!has_failed(&events), "must terminate cleanly: {events:?}");
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(
                answer_source,
                Some(AnswerSource::RuntimeTerminal {
                    reason: RuntimeTerminalReason::InsufficientEvidence,
                    ..
                })
            ),
            "two non-load reads must terminate with InsufficientEvidence: {answer_source:?}"
        );
    }

    #[test]
    fn load_lookup_noisy_load_term_in_comment_still_classifies_as_load() {
        // A line like "# TODO: load session data" contains "load".
        // The classification is structural/substring — comments match the same as code.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("services").join("session_service.py"),
            "# TODO: load session data\ndef handle_session(session):\n    return session.id\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: session]",
                // Comment-containing file is a load candidate (substring match).
                // Model reads it directly — evidence accepted.
                "[read_file: services/session_service.py]",
                "Sessions are handled in services/session_service.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are sessions loaded?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        assert!(
            !snapshot.iter().any(|m| m.content.contains("load lookup")),
            "no recovery expected when load candidate is read first"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "load candidate read must admit synthesis: {answer_source:?}"
        );
    }

    #[test]
    fn load_lookup_path_scope_keeps_candidates_inside_scope() {
        // Prompt scope must remain the upper bound. The out-of-scope load
        // file is stronger-looking but must not appear in search candidates.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/controllers")).unwrap();
        fs::write(
            tmp.path()
                .join("sandbox/services")
                .join("session_handler.py"),
            "def handle_session(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path()
                .join("sandbox/services")
                .join("session_loader.py"),
            "def get_session(session_id):\n    return load_session(session_id)\n",
        )
        .unwrap();
        fs::write(
            tmp.path()
                .join("sandbox/controllers")
                .join("session_loader.py"),
            "def get_session(session_id):\n    return load_session(session_id)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: session]",
                "[read_file: sandbox/services/session_handler.py]",
                "[read_file: sandbox/services/session_loader.py]",
                "Sessions are loaded in sandbox/services/session_loader.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where sessions are loaded in sandbox/services/".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        let search_result = snapshot
            .iter()
            .find(|m| m.content.contains("=== tool_result: search_code ==="))
            .map(|m| m.content.as_str())
            .unwrap_or("");
        assert!(
            search_result.contains("sandbox/services/session_handler.py"),
            "scoped search must include in-scope non-load candidate: {search_result}"
        );
        assert!(
            search_result.contains("sandbox/services/session_loader.py"),
            "scoped search must include in-scope load candidate: {search_result}"
        );
        assert!(
            !search_result.contains("sandbox/controllers/session_loader.py"),
            "scoped search must exclude out-of-scope load candidate: {search_result}"
        );

        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Sessions are loaded in sandbox/services/session_loader.py.")
        );
    }

    #[test]
    fn load_lookup_read_cap_still_applies() {
        // MaxReadsPerTurn must still apply under LoadLookup.
        // After 3 reads the runtime blocks further reads regardless of mode.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        for dir in &["a", "b", "c", "d"] {
            fs::create_dir_all(tmp.path().join(dir)).unwrap();
        }
        fs::write(
            tmp.path().join("a").join("session.py"),
            "def session_a(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("b").join("session.py"),
            "def session_b(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("c").join("session.py"),
            "def session_c(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("d").join("session.py"),
            "session = load_session(session_id)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: session]",
                // Reads 3 non-load files — hits cap before reaching load file.
                "[read_file: a/session.py]",
                "[read_file: b/session.py]",
                "[read_file: c/session.py]",
                "[read_file: d/session.py]",
                "Sessions are loaded in d/session.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are sessions loaded?".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "must not fail (cap is a correction): {events:?}"
        );
        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_error: read_file ===")
                    && m.content.contains("read limit")),
            "read cap must block the 4th read"
        );
    }

    // Phase 9.2.6 — SaveLookup

    #[test]
    fn detect_investigation_mode_returns_save_lookup() {
        assert!(matches!(
            detect_investigation_mode("Where is the session saved?"),
            InvestigationMode::SaveLookup
        ));
        assert!(matches!(
            detect_investigation_mode("Find where session saving happens"),
            InvestigationMode::SaveLookup
        ));
        assert!(matches!(
            detect_investigation_mode("Where do handlers save sessions?"),
            InvestigationMode::SaveLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_load_priority_over_save() {
        // "loaded" + "saved" in same prompt — LoadLookup wins (higher priority).
        assert!(matches!(
            detect_investigation_mode("Where is the session loaded and saved?"),
            InvestigationMode::LoadLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_save_priority_over_definition() {
        // "saved" + "defined" in same prompt — SaveLookup wins.
        assert!(matches!(
            detect_investigation_mode("Where is the session saved and defined?"),
            InvestigationMode::SaveLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_usage_priority_over_save() {
        assert!(matches!(
            detect_investigation_mode("Where is the saved session used?"),
            InvestigationMode::UsageLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_config_priority_over_save() {
        assert!(matches!(
            detect_investigation_mode("Where is saved config configured?"),
            InvestigationMode::ConfigLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_initialization_priority_over_save() {
        assert!(matches!(
            detect_investigation_mode("Find where session saving is initialized"),
            InvestigationMode::InitializationLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_create_priority_over_save() {
        assert!(matches!(
            detect_investigation_mode("Find where the saved session is created"),
            InvestigationMode::CreateLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_register_priority_over_save() {
        assert!(matches!(
            detect_investigation_mode("Find where the saved command is registered"),
            InvestigationMode::RegisterLookup
        ));
    }

    #[test]
    fn contains_save_term_matches_exact_allowed_substrings_only() {
        // Exact allowed terms.
        assert!(contains_save_term("save_session(session)"));
        assert!(contains_save_term("session was saved here"));
        assert!(contains_save_term("session saving happens here"));
        // Case insensitive.
        assert!(contains_save_term("Session.Save()"));
        assert!(contains_save_term("SAVED_SESSION"));
        // Noisy: substring of longer word — these DO match (substring semantics).
        assert!(contains_save_term("autosave session"));
        assert!(contains_save_term("savepoint created"));
        assert!(contains_save_term("saved_at timestamp"));
        // Not a save term.
        assert!(!contains_save_term("def handle_session(session):"));
        assert!(!contains_save_term("return session_id"));
    }

    #[test]
    fn save_lookup_non_save_read_triggers_recovery_to_save_file() {
        // File A: no save-term matches → non-save candidate.
        // File B: a save-term match → save candidate.
        // Model reads A first → recovery fires pointing to B.
        // Model reads B → evidence ready → ToolAssisted.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("services").join("session_handler.py"),
            "def handle_session(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("session_store.py"),
            "def store_session(session):\n    save_session(session)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: session]",
                // Reads the non-save file first — recovery fires.
                "[read_file: services/session_handler.py]",
                // Follows recovery, reads the save file.
                "[read_file: services/session_store.py]",
                "Sessions are saved in services/session_store.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are sessions saved?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");

        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| m.content.contains("save lookup")
                && m.content.contains("services/session_store.py")),
            "save recovery correction must point to the save candidate"
        );

        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "save lookup + recovery + save read must admit synthesis: {answer_source:?}"
        );
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Sessions are saved in services/session_store.py.")
        );
    }

    #[test]
    fn save_lookup_no_save_candidates_degrades_cleanly() {
        // All candidates have no save-term matches.
        // Gate does not fire — any read is accepted (fallback behavior).
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("services").join("session_handler.py"),
            "def handle_session(session):\n    return session.id\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: session]",
                "[read_file: services/session_handler.py]",
                "Sessions are handled in services/session_handler.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are sessions saved?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "save lookup with no save candidates must degrade to acceptance: {answer_source:?}"
        );
        let snapshot = rt.messages_snapshot();
        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Sessions are handled in services/session_handler.py.")
        );
    }

    #[test]
    fn save_lookup_second_non_save_candidate_after_recovery_is_not_accepted() {
        // After one recovery the correction flag is set.
        // A second non-save read falls through the gate without accepting.
        // With candidate_reads_count == 2 and evidence_ready false, the runtime
        // terminates with InsufficientEvidence.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::create_dir_all(tmp.path().join("controllers")).unwrap();
        fs::write(
            tmp.path().join("services").join("session_handler.py"),
            "def handle_session(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("controllers").join("session_controller.py"),
            "def show_session(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("services").join("session_store.py"),
            "def store_session(session):\n    save_session(session)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: session]",
                // First read: non-save → recovery fires pointing to session_store.py.
                "[read_file: services/session_handler.py]",
                // Second read: another non-save (ignores recovery, reads wrong file).
                "[read_file: controllers/session_controller.py]",
                // Model attempts synthesis — candidate limit hit; runtime terminates.
                "Sessions are saved in services/session_handler.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are sessions saved?".into(),
            },
        );

        assert!(!has_failed(&events), "must terminate cleanly: {events:?}");
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(
                answer_source,
                Some(AnswerSource::RuntimeTerminal {
                    reason: RuntimeTerminalReason::InsufficientEvidence,
                    ..
                })
            ),
            "two non-save reads must terminate with InsufficientEvidence: {answer_source:?}"
        );
    }

    #[test]
    fn save_lookup_noisy_save_term_in_comment_still_classifies_as_save() {
        // A line like "# TODO: save session data" contains "save".
        // The classification is structural/substring — comments match the same as code.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::write(
            tmp.path().join("services").join("session_service.py"),
            "# TODO: save session data\ndef handle_session(session):\n    return session.id\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: session]",
                // Comment-containing file is a save candidate (substring match).
                // Model reads it directly — evidence accepted.
                "[read_file: services/session_service.py]",
                "Sessions are handled in services/session_service.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are sessions saved?".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        assert!(
            !snapshot.iter().any(|m| m.content.contains("save lookup")),
            "no recovery expected when save candidate is read first"
        );
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "save candidate read must admit synthesis: {answer_source:?}"
        );
    }

    #[test]
    fn save_lookup_path_scope_keeps_candidates_inside_scope() {
        // Prompt scope must remain the upper bound. The out-of-scope save
        // file is stronger-looking but must not appear in search candidates.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/services")).unwrap();
        fs::create_dir_all(tmp.path().join("sandbox/controllers")).unwrap();
        fs::write(
            tmp.path()
                .join("sandbox/services")
                .join("session_handler.py"),
            "def handle_session(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("sandbox/services").join("session_store.py"),
            "def store_session(session):\n    save_session(session)\n",
        )
        .unwrap();
        fs::write(
            tmp.path()
                .join("sandbox/controllers")
                .join("session_store.py"),
            "def store_session(session):\n    save_session(session)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: session]",
                "[read_file: sandbox/services/session_handler.py]",
                "[read_file: sandbox/services/session_store.py]",
                "Sessions are saved in sandbox/services/session_store.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Find where sessions are saved in sandbox/services/".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        let search_result = snapshot
            .iter()
            .find(|m| m.content.contains("=== tool_result: search_code ==="))
            .map(|m| m.content.as_str())
            .unwrap_or("");
        assert!(
            search_result.contains("sandbox/services/session_handler.py"),
            "scoped search must include in-scope non-save candidate: {search_result}"
        );
        assert!(
            search_result.contains("sandbox/services/session_store.py"),
            "scoped search must include in-scope save candidate: {search_result}"
        );
        assert!(
            !search_result.contains("sandbox/controllers/session_store.py"),
            "scoped search must exclude out-of-scope save candidate: {search_result}"
        );

        let last_assistant = snapshot
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::backend::Role::Assistant)
            .map(|m| m.content.as_str());
        assert_eq!(
            last_assistant,
            Some("Sessions are saved in sandbox/services/session_store.py.")
        );
    }

    #[test]
    fn save_lookup_read_cap_still_applies() {
        // MaxReadsPerTurn must still apply under SaveLookup.
        // After 3 reads the runtime blocks further reads regardless of mode.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        for dir in &["a", "b", "c", "d"] {
            fs::create_dir_all(tmp.path().join(dir)).unwrap();
        }
        fs::write(
            tmp.path().join("a").join("session.py"),
            "def session_a(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("b").join("session.py"),
            "def session_b(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("c").join("session.py"),
            "def session_c(session):\n    return session.id\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("d").join("session.py"),
            "save_session(session)\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: session]",
                // Reads 3 non-save files — hits cap before reaching save file.
                "[read_file: a/session.py]",
                "[read_file: b/session.py]",
                "[read_file: c/session.py]",
                "[read_file: d/session.py]",
                "Sessions are saved in d/session.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are sessions saved?".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "must not fail (cap is a correction): {events:?}"
        );
        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_error: read_file ===")
                    && m.content.contains("read limit")),
            "read cap must block the 4th read"
        );
    }

    // Phase 9.2.3 — regression tests for earlier modes/invariants

    #[test]
    fn create_lookup_does_not_affect_usage_lookup_regression() {
        // A usage-lookup prompt with create terms must remain UsageLookup (higher priority).
        // The create gate must not activate for UsageLookup turns.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("services")).unwrap();
        fs::create_dir_all(tmp.path().join("models")).unwrap();
        fs::write(
            tmp.path().join("services").join("task_service.py"),
            "if task.status == TaskStatus.DONE:\n    pass\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("models").join("enums.py"),
            "class TaskStatus(Enum):\n    DONE = 'done'\n",
        )
        .unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: TaskStatus]",
                "[read_file: services/task_service.py]",
                "TaskStatus is used in services/task_service.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                // "used" triggers UsageLookup; "created" present but must not win.
                text: "Where is TaskStatus used and created?".into(),
            },
        );

        assert!(!has_failed(&events), "regression: {events:?}");
        let answer_source = events.iter().find_map(|e| {
            if let RuntimeEvent::AnswerReady(src) = e {
                Some(src.clone())
            } else {
                None
            }
        });
        assert!(
            matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
            "UsageLookup must not be disrupted by create terms: {answer_source:?}"
        );
    }

    #[test]
    fn create_lookup_read_cap_still_applies() {
        // MaxReadsPerTurn must still apply under CreateLookup.
        // After 3 reads the runtime blocks further reads regardless of mode.
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        for dir in &["a", "b", "c", "d"] {
            fs::create_dir_all(tmp.path().join(dir)).unwrap();
        }
        fs::write(
            tmp.path().join("a").join("task.py"),
            "def task_a():\n    pass\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("b").join("task.py"),
            "def task_b():\n    pass\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("c").join("task.py"),
            "def task_c():\n    pass\n",
        )
        .unwrap();
        fs::write(tmp.path().join("d").join("task.py"), "db.create(task)\n").unwrap();

        let mut rt = make_runtime_in(
            vec![
                "[search_code: task]",
                // Reads 3 non-create files — hits cap before reaching create file.
                "[read_file: a/task.py]",
                "[read_file: b/task.py]",
                "[read_file: c/task.py]",
                "[read_file: d/task.py]",
                "Tasks are created in d/task.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where are tasks created?".into(),
            },
        );

        assert!(
            !has_failed(&events),
            "must not fail (cap is a correction): {events:?}"
        );
        let snapshot = rt.messages_snapshot();
        // The 4th read must be blocked by the cap.
        assert!(
            snapshot
                .iter()
                .any(|m| m.content.contains("=== tool_error: read_file ===")
                    && m.content.contains("read limit")),
            "read cap must block the 4th read"
        );
    }
}
