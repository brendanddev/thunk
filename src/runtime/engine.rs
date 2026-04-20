use std::collections::HashSet;
use std::path::Path;

use crate::app::config::Config;
use crate::app::Result;
use crate::llm::backend::{BackendEvent, BackendStatus, GenerateRequest, ModelBackend};
use crate::tools::{
    ExecutionKind, PendingAction, ToolInput, ToolOutput, ToolRegistry, ToolRunResult,
};

use super::conversation::Conversation;
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

/// Injected when the model exceeds MAX_READS_PER_TURN in one turn.
const READ_CAP_EXCEEDED: &str =
    "read limit for this turn reached. Answer from the file evidence already in context.";

/// Tracks per-turn search → read investigation state.
/// Resets at the start of each call to run_turns, exactly like SearchBudget.
struct InvestigationState {
    /// True once any search_code call this turn returned at least one match.
    search_produced_results: bool,
    /// Count of read_file calls that completed successfully this turn.
    files_read_count: usize,
    /// File paths from the current non-empty search results.
    search_candidate_paths: Vec<String>,
    /// Candidate paths where every matched line looks like a definition site.
    /// Populated during record_search_results alongside search_candidate_paths.
    definition_only_candidates: HashSet<String>,
    /// True if at least one candidate in the current search results has a
    /// non-definition match line (i.e. a usage file is available).
    has_non_definition_candidates: bool,
    /// True once a search candidate has been read that provides useful evidence.
    /// Definition-only files are useful only when no non-definition candidates
    /// exist in the current result set.
    read_useful_candidate: bool,
    /// True after the read-before-answering correction has been issued once.
    /// Prevents the correction from firing more than once per turn.
    premature_synthesis_correction_issued: bool,
    /// True after the search-before-answering correction has been issued once.
    /// R1 uses its own flag and does NOT increment the shared corrections counter,
    /// so R1 and R2 can compose sequentially in the same turn.
    direct_answer_correction_issued: bool,
}

impl InvestigationState {
    fn new() -> Self {
        Self {
            search_produced_results: false,
            files_read_count: 0,
            search_candidate_paths: Vec::new(),
            definition_only_candidates: HashSet::new(),
            has_non_definition_candidates: false,
            read_useful_candidate: false,
            premature_synthesis_correction_issued: false,
            direct_answer_correction_issued: false,
        }
    }

    fn evidence_ready(&self) -> bool {
        self.search_produced_results && self.read_useful_candidate
    }

    fn record_search_results(&mut self, output: &ToolOutput) -> bool {
        let ToolOutput::SearchResults(results) = output else {
            return false;
        };

        let was_empty = results.matches.is_empty();
        if !was_empty {
            self.search_produced_results = true;
            self.search_candidate_paths.clear();
            self.definition_only_candidates.clear();
            self.has_non_definition_candidates = false;
            self.read_useful_candidate = false;

            for result in &results.matches {
                push_unique_path(&mut self.search_candidate_paths, &result.file);
            }

            // Classify each candidate file: definition-only if every matched line looks
            // like a definition site, usage-bearing otherwise.
            // Uses tool_codec::looks_like_definition as a predicate only — no rendering.
            let mut file_has_non_def: HashSet<String> = HashSet::new();
            for m in &results.matches {
                if !tool_codec::looks_like_definition(&m.line) {
                    file_has_non_def.insert(m.file.clone());
                }
            }
            for path in &self.search_candidate_paths {
                if file_has_non_def.contains(path) {
                    self.has_non_definition_candidates = true;
                } else {
                    self.definition_only_candidates.insert(path.clone());
                }
            }
        }
        was_empty
    }

    fn record_read_result(&mut self, output: &ToolOutput) {
        let ToolOutput::FileContents(file) = output else {
            return;
        };

        self.files_read_count += 1;
        let read_path = normalize_evidence_path(&file.path);

        let is_search_candidate = self
            .search_candidate_paths
            .iter()
            .any(|candidate| normalize_evidence_path(candidate) == read_path);

        if is_search_candidate {
            let is_def_only = self
                .definition_only_candidates
                .iter()
                .any(|c| normalize_evidence_path(c) == read_path);
            // A definition-only file satisfies evidence only when no usage candidates
            // exist in the current result set.
            if !is_def_only || !self.has_non_definition_candidates {
                self.read_useful_candidate = true;
            }
        }
    }
}

fn push_unique_path(paths: &mut Vec<String>, path: &str) {
    if !paths.iter().any(|existing| existing == path) {
        paths.push(path.to_string());
    }
}

fn normalize_evidence_path(path: &str) -> String {
    path.replace('\\', "/").trim_start_matches("./").to_string()
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
        "initialized",
        "initialised",
        "configured",
        "saved",
        "stored",
        "loaded",
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

        self.conversation.push_user(text);
        on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
        self.run_turns(0, on_event);
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
    fn run_turns(&mut self, mut tool_rounds: usize, on_event: &mut dyn FnMut(RuntimeEvent)) {
        let mut corrections = 0usize;
        let mut last_call_key: Option<String> = None;
        let mut search_budget = SearchBudget::new();
        let mut investigation = InvestigationState::new();
        let mut reads_this_turn: HashSet<String> = HashSet::new();
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
                    && !investigation.search_produced_results
                    && investigation.files_read_count == 0
                {
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
                        if !investigation.direct_answer_correction_issued {
                            investigation.direct_answer_correction_issued = true;
                            self.conversation
                                .push_user(SEARCH_BEFORE_ANSWERING.to_string());
                            continue;
                        }

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

                    if investigation.search_produced_results {
                        if !investigation.premature_synthesis_correction_issued
                            && corrections < MAX_CORRECTIONS
                        {
                            corrections += 1;
                            investigation.premature_synthesis_correction_issued = true;
                            self.conversation.discard_last_if_assistant();
                            self.conversation
                                .push_user(READ_BEFORE_ANSWERING.to_string());
                            continue;
                        }

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
                mutation_allowed,
                requested_read_path.as_deref(),
                &mut requested_read_completed,
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
    mutation_allowed: bool,
    requested_read_path: Option<&str>,
    requested_read_completed: &mut bool,
    on_event: &mut dyn FnMut(RuntimeEvent),
) -> ToolRoundOutcome {
    let mut accumulated = String::new();

    for mut input in calls {
        simplify_search_input(&mut input);
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
                    let was_empty = investigation.record_search_results(&output);
                    search_budget.record(was_empty);
                    search_budget
                        .is_closed()
                        .then(|| search_budget.closed_message())
                } else {
                    None
                };
                // Track successful file reads for evidence grounding and dedup.
                if name == "read_file" {
                    investigation.record_read_result(&output);
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
                }
                let summary = tool_codec::render_compact_summary(&output);
                on_event(RuntimeEvent::ToolCallFinished {
                    name: name.clone(),
                    summary: Some(summary),
                });
                accumulated.push_str(&tool_codec::format_tool_result(&name, &output));
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
                "TaskStatus is defined in models/enums.py.",
                "[read_file: services/task_service.py]",
                "TaskStatus is used in services/task_service.py.",
            ],
            tmp.path(),
        );

        let events = collect_events(
            &mut rt,
            RuntimeRequest::Submit {
                text: "Where is TaskStatus used in sandbox/".into(),
            },
        );

        assert!(!has_failed(&events), "turn must not fail: {events:?}");
        let snapshot = rt.messages_snapshot();
        assert!(
            snapshot.iter().any(|m| {
                m.content.starts_with("[runtime:correction]")
                    && m.content.contains("no matched file has been read")
            }),
            "definition-only read must not satisfy usage evidence before a usage file is read"
        );
        assert!(
            !snapshot
                .iter()
                .any(|m| m.content == "TaskStatus is defined in models/enums.py."),
            "premature synthesis after definition-only read must be discarded"
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
                text: "Where is TaskStatus used in sandbox/".into(),
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
                text: "Where is TaskStatus used in sandbox/".into(),
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
            events.iter().any(|e| matches!(e, RuntimeEvent::AnswerReady(_))),
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
            events.iter().any(|e| matches!(e, RuntimeEvent::AnswerReady(_))),
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
}
