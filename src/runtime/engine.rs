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

/// Structural mode for the current investigation turn.
/// Computed once from the user prompt before the tool loop starts.
/// Controls which evidence-acceptance gates are active for this turn.
#[derive(Copy, Clone)]
enum InvestigationMode {
    /// No mode-specific gating. Any search-candidate read satisfies evidence.
    General,
    /// Prompt signals a usage lookup (where X is used/referenced/appears).
    /// Definition-only reads are structurally insufficient when usage candidates exist.
    UsageLookup,
    /// Prompt signals a definition lookup (where X is defined/declared).
    /// No mode-specific gating beyond General — definition reads are always accepted.
    DefinitionLookup,
    /// Prompt signals a config lookup (where X is configured/configuration).
    /// Source-file reads are structurally insufficient when config-file candidates exist.
    ConfigLookup,
    /// Prompt signals a narrow initialization lookup.
    /// Non-initialization reads are structurally insufficient when initialization candidates exist.
    InitializationLookup,
    /// Prompt signals a narrow creation lookup (where X is created/creation).
    /// Non-create reads are structurally insufficient when create candidates exist.
    CreateLookup,
    /// Prompt signals a narrow registration lookup.
    /// Non-register reads are structurally insufficient when register candidates exist.
    RegisterLookup,
    /// Prompt signals a narrow load lookup.
    /// Non-load reads are structurally insufficient when load candidates exist.
    LoadLookup,
}

impl InvestigationMode {
    fn as_str(self) -> &'static str {
        match self {
            InvestigationMode::General => "General",
            InvestigationMode::UsageLookup => "UsageLookup",
            InvestigationMode::DefinitionLookup => "DefinitionLookup",
            InvestigationMode::ConfigLookup => "ConfigLookup",
            InvestigationMode::InitializationLookup => "InitializationLookup",
            InvestigationMode::CreateLookup => "CreateLookup",
            InvestigationMode::RegisterLookup => "RegisterLookup",
            InvestigationMode::LoadLookup => "LoadLookup",
        }
    }
}

/// Distinguishes which structural insufficiency caused a candidate read to be rejected.
/// Used by the caller in run_tool_round to select the appropriate correction message.
enum RecoveryKind {
    /// The file was definition-only on a usage lookup with usage candidates available.
    DefinitionOnly,
    /// The file had only import-declaration matches with substantive candidates available.
    ImportOnly,
    /// The file was a non-config source file on a config lookup when config-file candidates exist.
    ConfigFile,
    /// The file lacked initialization matches when initialization candidates exist.
    Initialization,
    /// The file lacked create-term matches when create candidates exist.
    Create,
    /// The file lacked register-term matches when register candidates exist.
    Register,
    /// The file lacked load-term matches when load candidates exist.
    Load,
}

impl RecoveryKind {
    fn as_str(&self) -> &'static str {
        match self {
            RecoveryKind::DefinitionOnly => "DefinitionOnly",
            RecoveryKind::ImportOnly => "ImportOnly",
            RecoveryKind::ConfigFile => "ConfigFile",
            RecoveryKind::Initialization => "Initialization",
            RecoveryKind::Create => "Create",
            RecoveryKind::Register => "Register",
            RecoveryKind::Load => "Load",
        }
    }
}

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
    /// True once any search_code call has completed this turn, even with no matches.
    /// Used to block list_dir before search on investigation-required turns.
    search_attempted: bool,
    /// Count of distinct search-candidate files successfully read this turn.
    /// Bounded investigation: a second candidate read is allowed when the first was
    /// insufficient; after two candidate reads the runtime terminates cleanly if
    /// evidence_ready() is still false.
    candidate_reads_count: usize,
    /// Candidate paths where every matched line looks like an import declaration.
    /// Populated during record_search_results alongside search_candidate_paths.
    import_only_candidates: HashSet<String>,
    /// True if at least one candidate in the current search results has a non-import
    /// match line (i.e. a file with substantive usage or definition is available).
    has_non_import_candidates: bool,
    /// True after the import-only recovery correction has been issued once this turn.
    /// Uses its own flag so it does not consume the premature_synthesis correction slot.
    import_correction_issued: bool,
    /// Candidate paths whose file extension identifies them as a config file
    /// (e.g. .yaml, .toml, .json, .env).  Populated during record_search_results.
    config_file_candidates: HashSet<String>,
    /// True if at least one candidate in the current search results is NOT a config file
    /// (i.e. a source or other non-config file was also matched).
    has_non_config_candidates: bool,
    /// True after the config-file recovery correction has been issued once this turn.
    config_correction_issued: bool,
    /// Candidate paths where at least one matched line contains an initialization term.
    /// Populated during record_search_results alongside search_candidate_paths.
    initialization_candidates: HashSet<String>,
    /// True if at least one candidate in the current search results has no matched
    /// initialization line.
    has_non_initialization_candidates: bool,
    /// True after the initialization recovery correction has been issued once this turn.
    initialization_correction_issued: bool,
    /// Candidate paths where at least one matched line contains a create term.
    /// Populated during record_search_results alongside search_candidate_paths.
    create_candidates: HashSet<String>,
    /// True if at least one candidate in the current search results has no matched
    /// create line.
    has_non_create_candidates: bool,
    /// True after the create recovery correction has been issued once this turn.
    create_correction_issued: bool,
    /// Candidate paths where at least one matched line contains a register term.
    /// Populated during record_search_results alongside search_candidate_paths.
    register_candidates: HashSet<String>,
    /// True if at least one candidate in the current search results has no matched
    /// register line.
    has_non_register_candidates: bool,
    /// True after the register recovery correction has been issued once this turn.
    register_correction_issued: bool,
    /// Candidate paths where at least one matched line contains a load term.
    /// Populated during record_search_results alongside search_candidate_paths.
    load_candidates: HashSet<String>,
    /// True if at least one candidate in the current search results has no matched
    /// load line.
    has_non_load_candidates: bool,
    /// True after the load recovery correction has been issued once this turn.
    load_correction_issued: bool,
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
            search_attempted: false,
            candidate_reads_count: 0,
            import_only_candidates: HashSet::new(),
            has_non_import_candidates: false,
            import_correction_issued: false,
            config_file_candidates: HashSet::new(),
            has_non_config_candidates: false,
            config_correction_issued: false,
            initialization_candidates: HashSet::new(),
            has_non_initialization_candidates: false,
            initialization_correction_issued: false,
            create_candidates: HashSet::new(),
            has_non_create_candidates: false,
            create_correction_issued: false,
            register_candidates: HashSet::new(),
            has_non_register_candidates: false,
            register_correction_issued: false,
            load_candidates: HashSet::new(),
            has_non_load_candidates: false,
            load_correction_issued: false,
        }
    }

    fn evidence_ready(&self) -> bool {
        self.search_produced_results && self.read_useful_candidate
    }

    fn record_search_results(
        &mut self,
        output: &ToolOutput,
        on_event: &mut dyn FnMut(RuntimeEvent),
    ) -> bool {
        let ToolOutput::SearchResults(results) = output else {
            return false;
        };

        self.search_attempted = true;
        let was_empty = results.matches.is_empty();
        if !was_empty {
            self.search_produced_results = true;
            self.search_candidate_paths.clear();
            self.definition_only_candidates.clear();
            self.has_non_definition_candidates = false;
            self.import_only_candidates.clear();
            self.has_non_import_candidates = false;
            self.config_file_candidates.clear();
            self.has_non_config_candidates = false;
            self.initialization_candidates.clear();
            self.has_non_initialization_candidates = false;
            self.create_candidates.clear();
            self.has_non_create_candidates = false;
            self.register_candidates.clear();
            self.has_non_register_candidates = false;
            self.load_candidates.clear();
            self.has_non_load_candidates = false;
            self.read_useful_candidate = false;

            for result in &results.matches {
                push_unique_path(&mut self.search_candidate_paths, &result.file);
            }

            // Classify each candidate file along structural axes.
            // definition-only: every matched line looks like a definition site (line-content).
            // import-only: every matched line looks like an import declaration (line-content).
            // config-file: the file's extension identifies it as a config file (path-based).
            // initialization: at least one matched line contains an exact initialization term.
            // create: at least one matched line contains an exact create term.
            // register: at least one matched line contains an exact register term.
            // load: at least one matched line contains an exact load term.
            let mut file_has_non_def: HashSet<String> = HashSet::new();
            let mut file_has_non_import: HashSet<String> = HashSet::new();
            let mut file_has_initialization: HashSet<String> = HashSet::new();
            let mut file_has_create: HashSet<String> = HashSet::new();
            let mut file_has_register: HashSet<String> = HashSet::new();
            let mut file_has_load: HashSet<String> = HashSet::new();
            for m in &results.matches {
                if !tool_codec::looks_like_definition(&m.line) {
                    file_has_non_def.insert(m.file.clone());
                }
                if !looks_like_import(&m.line) {
                    file_has_non_import.insert(m.file.clone());
                }
                if contains_initialization_term(&m.line) {
                    file_has_initialization.insert(m.file.clone());
                }
                if contains_create_term(&m.line) {
                    file_has_create.insert(m.file.clone());
                }
                if contains_register_term(&m.line) {
                    file_has_register.insert(m.file.clone());
                }
                if contains_load_term(&m.line) {
                    file_has_load.insert(m.file.clone());
                }
            }
            for path in &self.search_candidate_paths {
                if file_has_non_def.contains(path) {
                    self.has_non_definition_candidates = true;
                } else {
                    self.definition_only_candidates.insert(path.clone());
                }
                if file_has_non_import.contains(path) {
                    self.has_non_import_candidates = true;
                } else {
                    self.import_only_candidates.insert(path.clone());
                }
                if is_config_file(path) {
                    self.config_file_candidates.insert(path.clone());
                } else {
                    self.has_non_config_candidates = true;
                }
                if file_has_initialization.contains(path) {
                    self.initialization_candidates.insert(path.clone());
                } else {
                    self.has_non_initialization_candidates = true;
                }
                if file_has_create.contains(path) {
                    self.create_candidates.insert(path.clone());
                } else {
                    self.has_non_create_candidates = true;
                }
                if file_has_register.contains(path) {
                    self.register_candidates.insert(path.clone());
                } else {
                    self.has_non_register_candidates = true;
                }
                if file_has_load.contains(path) {
                    self.load_candidates.insert(path.clone());
                } else {
                    self.has_non_load_candidates = true;
                }
            }
        }
        trace_runtime_decision(
            on_event,
            "search_candidates_classified",
            &[
                ("shown_matches", results.matches.len().to_string()),
                ("total_matches", results.total_matches.to_string()),
                ("truncated", results.truncated.to_string()),
                (
                    "candidate_files",
                    self.search_candidate_paths.len().to_string(),
                ),
                (
                    "definition_only",
                    self.definition_only_candidates.len().to_string(),
                ),
                (
                    "has_non_definition",
                    self.has_non_definition_candidates.to_string(),
                ),
                ("import_only", self.import_only_candidates.len().to_string()),
                ("has_non_import", self.has_non_import_candidates.to_string()),
                (
                    "config_files",
                    self.config_file_candidates.len().to_string(),
                ),
                ("has_non_config", self.has_non_config_candidates.to_string()),
                (
                    "initialization_files",
                    self.initialization_candidates.len().to_string(),
                ),
                (
                    "has_non_initialization",
                    self.has_non_initialization_candidates.to_string(),
                ),
                ("create_files", self.create_candidates.len().to_string()),
                ("has_non_create", self.has_non_create_candidates.to_string()),
                ("register_files", self.register_candidates.len().to_string()),
                (
                    "has_non_register",
                    self.has_non_register_candidates.to_string(),
                ),
                ("load_files", self.load_candidates.len().to_string()),
                ("has_non_load", self.has_non_load_candidates.to_string()),
            ],
        );
        was_empty
    }

    fn record_read_result(
        &mut self,
        output: &ToolOutput,
        mode: InvestigationMode,
        on_event: &mut dyn FnMut(RuntimeEvent),
    ) -> Option<(String, RecoveryKind)> {
        let ToolOutput::FileContents(file) = output else {
            return None;
        };

        self.files_read_count += 1;
        let read_path = normalize_evidence_path(&file.path);

        let is_search_candidate = self
            .search_candidate_paths
            .iter()
            .any(|candidate| normalize_evidence_path(candidate) == read_path);

        if is_search_candidate {
            self.candidate_reads_count += 1;
            let is_def_only = self
                .definition_only_candidates
                .iter()
                .any(|c| normalize_evidence_path(c) == read_path);
            let is_import_only = self
                .import_only_candidates
                .iter()
                .any(|c| normalize_evidence_path(c) == read_path);
            let is_config_candidate = self
                .config_file_candidates
                .iter()
                .any(|c| normalize_evidence_path(c) == read_path);
            let is_initialization_candidate = self
                .initialization_candidates
                .iter()
                .any(|c| normalize_evidence_path(c) == read_path);
            let is_create_candidate = self
                .create_candidates
                .iter()
                .any(|c| normalize_evidence_path(c) == read_path);
            let is_register_candidate = self
                .register_candidates
                .iter()
                .any(|c| normalize_evidence_path(c) == read_path);
            let is_load_candidate = self
                .load_candidates
                .iter()
                .any(|c| normalize_evidence_path(c) == read_path);

            // Gate 1 (UsageLookup): definition-only reads are structurally insufficient
            // when usage candidates exist. Fire once; subsequent reads fall through ungated.
            if matches!(mode, InvestigationMode::UsageLookup)
                && is_def_only
                && self.has_non_definition_candidates
            {
                if !self.premature_synthesis_correction_issued {
                    let suggested_path = self.first_non_definition_candidate().map(str::to_string);
                    if suggested_path.is_some() {
                        self.premature_synthesis_correction_issued = true;
                    }
                    trace_runtime_decision(
                        on_event,
                        "read_evidence",
                        &[
                            ("path", read_path.clone()),
                            ("accepted", "false".into()),
                            ("reason", "usage_definition_only_candidate".into()),
                            (
                                "recovery_path",
                                suggested_path.clone().unwrap_or_else(|| "none".into()),
                            ),
                        ],
                    );
                    return suggested_path.map(|p| (p, RecoveryKind::DefinitionOnly));
                }
                trace_runtime_decision(
                    on_event,
                    "read_evidence",
                    &[
                        ("path", read_path.clone()),
                        ("accepted", "false".into()),
                        (
                            "reason",
                            "usage_definition_only_recovery_already_issued".into(),
                        ),
                    ],
                );
                // Correction already issued: fall through without accepting.
            }
            // Gate 2 (ConfigLookup): non-config reads are structurally insufficient when
            // config-file candidates exist. Fire once; fallback accepts if no config candidates.
            else if matches!(mode, InvestigationMode::ConfigLookup)
                && !is_config_candidate
                && !self.config_file_candidates.is_empty()
            {
                if !self.config_correction_issued {
                    self.config_correction_issued = true;
                    let suggested_path = self.first_config_candidate().map(str::to_string);
                    trace_runtime_decision(
                        on_event,
                        "read_evidence",
                        &[
                            ("path", read_path.clone()),
                            ("accepted", "false".into()),
                            ("reason", "config_non_config_candidate".into()),
                            (
                                "recovery_path",
                                suggested_path.clone().unwrap_or_else(|| "none".into()),
                            ),
                        ],
                    );
                    return suggested_path.map(|p| (p, RecoveryKind::ConfigFile));
                }
                trace_runtime_decision(
                    on_event,
                    "read_evidence",
                    &[
                        ("path", read_path.clone()),
                        ("accepted", "false".into()),
                        ("reason", "config_non_config_recovery_already_issued".into()),
                    ],
                );
                // Correction already issued: fall through without accepting.
            }
            // Gate 3 (InitializationLookup): non-initialization reads are structurally
            // insufficient when initialization candidates exist. Fire once; fallback
            // accepts if no initialization candidates exist.
            else if matches!(mode, InvestigationMode::InitializationLookup)
                && !is_initialization_candidate
                && !self.initialization_candidates.is_empty()
            {
                if !self.initialization_correction_issued {
                    self.initialization_correction_issued = true;
                    let suggested_path = self.first_initialization_candidate().map(str::to_string);
                    trace_runtime_decision(
                        on_event,
                        "read_evidence",
                        &[
                            ("path", read_path.clone()),
                            ("accepted", "false".into()),
                            (
                                "reason",
                                "initialization_non_initialization_candidate".into(),
                            ),
                            (
                                "recovery_path",
                                suggested_path.clone().unwrap_or_else(|| "none".into()),
                            ),
                        ],
                    );
                    return suggested_path.map(|p| (p, RecoveryKind::Initialization));
                }
                trace_runtime_decision(
                    on_event,
                    "read_evidence",
                    &[
                        ("path", read_path.clone()),
                        ("accepted", "false".into()),
                        ("reason", "initialization_recovery_already_issued".into()),
                    ],
                );
                // Correction already issued: fall through without accepting.
            }
            // Gate 4 (CreateLookup): non-create reads are structurally insufficient when
            // create candidates exist. Fire once; fallback accepts if no create candidates.
            else if matches!(mode, InvestigationMode::CreateLookup)
                && !is_create_candidate
                && !self.create_candidates.is_empty()
            {
                if !self.create_correction_issued {
                    self.create_correction_issued = true;
                    let suggested_path = self.first_create_candidate().map(str::to_string);
                    trace_runtime_decision(
                        on_event,
                        "read_evidence",
                        &[
                            ("path", read_path.clone()),
                            ("accepted", "false".into()),
                            ("reason", "create_non_create_candidate".into()),
                            (
                                "recovery_path",
                                suggested_path.clone().unwrap_or_else(|| "none".into()),
                            ),
                        ],
                    );
                    return suggested_path.map(|p| (p, RecoveryKind::Create));
                }
                trace_runtime_decision(
                    on_event,
                    "read_evidence",
                    &[
                        ("path", read_path.clone()),
                        ("accepted", "false".into()),
                        ("reason", "create_recovery_already_issued".into()),
                    ],
                );
                // Correction already issued: fall through without accepting.
            }
            // Gate 5 (RegisterLookup): non-register reads are structurally insufficient when
            // register candidates exist. Fire once; fallback accepts if no register candidates.
            else if matches!(mode, InvestigationMode::RegisterLookup)
                && !is_register_candidate
                && !self.register_candidates.is_empty()
            {
                if !self.register_correction_issued {
                    self.register_correction_issued = true;
                    let suggested_path = self.first_register_candidate().map(str::to_string);
                    trace_runtime_decision(
                        on_event,
                        "read_evidence",
                        &[
                            ("path", read_path.clone()),
                            ("accepted", "false".into()),
                            ("reason", "register_non_register_candidate".into()),
                            (
                                "recovery_path",
                                suggested_path.clone().unwrap_or_else(|| "none".into()),
                            ),
                        ],
                    );
                    return suggested_path.map(|p| (p, RecoveryKind::Register));
                }
                trace_runtime_decision(
                    on_event,
                    "read_evidence",
                    &[
                        ("path", read_path.clone()),
                        ("accepted", "false".into()),
                        ("reason", "register_recovery_already_issued".into()),
                    ],
                );
                // Correction already issued: fall through without accepting.
            }
            // Gate 6 (LoadLookup): non-load reads are structurally insufficient when
            // load candidates exist. Fire once; fallback accepts if no load candidates.
            else if matches!(mode, InvestigationMode::LoadLookup)
                && !is_load_candidate
                && !self.load_candidates.is_empty()
            {
                if !self.load_correction_issued {
                    self.load_correction_issued = true;
                    let suggested_path = self.first_load_candidate().map(str::to_string);
                    trace_runtime_decision(
                        on_event,
                        "read_evidence",
                        &[
                            ("path", read_path.clone()),
                            ("accepted", "false".into()),
                            ("reason", "load_non_load_candidate".into()),
                            (
                                "recovery_path",
                                suggested_path.clone().unwrap_or_else(|| "none".into()),
                            ),
                        ],
                    );
                    return suggested_path.map(|p| (p, RecoveryKind::Load));
                }
                trace_runtime_decision(
                    on_event,
                    "read_evidence",
                    &[
                        ("path", read_path.clone()),
                        ("accepted", "false".into()),
                        ("reason", "load_recovery_already_issued".into()),
                    ],
                );
                // Correction already issued: fall through without accepting.
            } else {
                // Candidate would normally be accepted. Check import-only before committing.
                // Import-only candidates are structurally insufficient when substantive
                // (non-import) candidates exist in the current result set.
                if is_import_only
                    && self.has_non_import_candidates
                    && !self.import_correction_issued
                {
                    self.import_correction_issued = true;
                    let suggested_path = self.first_non_import_candidate().map(str::to_string);
                    trace_runtime_decision(
                        on_event,
                        "read_evidence",
                        &[
                            ("path", read_path.clone()),
                            ("accepted", "false".into()),
                            ("reason", "import_only_candidate".into()),
                            (
                                "recovery_path",
                                suggested_path.clone().unwrap_or_else(|| "none".into()),
                            ),
                        ],
                    );
                    return suggested_path.map(|p| (p, RecoveryKind::ImportOnly));
                }
                self.read_useful_candidate = true;
                trace_runtime_decision(
                    on_event,
                    "read_evidence",
                    &[
                        ("path", read_path.clone()),
                        ("accepted", "true".into()),
                        (
                            "reason",
                            self.acceptance_reason(mode, is_def_only, is_import_only),
                        ),
                        ("candidate_reads", self.candidate_reads_count.to_string()),
                    ],
                );
            }
        } else {
            trace_runtime_decision(
                on_event,
                "read_evidence",
                &[
                    ("path", read_path),
                    ("accepted", "false".into()),
                    ("reason", "not_search_candidate".into()),
                ],
            );
        }
        None
    }

    fn acceptance_reason(
        &self,
        mode: InvestigationMode,
        is_def_only: bool,
        is_import_only: bool,
    ) -> String {
        if matches!(mode, InvestigationMode::ConfigLookup) && self.config_file_candidates.is_empty()
        {
            "config_fallback_no_config_candidates".into()
        } else if matches!(mode, InvestigationMode::InitializationLookup)
            && self.initialization_candidates.is_empty()
        {
            "initialization_fallback_no_initialization_candidates".into()
        } else if matches!(mode, InvestigationMode::CreateLookup)
            && self.create_candidates.is_empty()
        {
            "create_fallback_no_create_candidates".into()
        } else if matches!(mode, InvestigationMode::RegisterLookup)
            && self.register_candidates.is_empty()
        {
            "register_fallback_no_register_candidates".into()
        } else if matches!(mode, InvestigationMode::LoadLookup) && self.load_candidates.is_empty() {
            "load_fallback_no_load_candidates".into()
        } else if matches!(mode, InvestigationMode::UsageLookup)
            && is_def_only
            && !self.has_non_definition_candidates
        {
            "usage_fallback_no_usage_candidates".into()
        } else if is_import_only && !self.has_non_import_candidates {
            "import_fallback_all_import_candidates".into()
        } else {
            "search_candidate".into()
        }
    }

    fn first_non_definition_candidate(&self) -> Option<&str> {
        self.search_candidate_paths
            .iter()
            .find(|path| !self.definition_only_candidates.contains(*path))
            .map(String::as_str)
    }

    fn first_non_import_candidate(&self) -> Option<&str> {
        self.search_candidate_paths
            .iter()
            .find(|path| !self.import_only_candidates.contains(*path))
            .map(String::as_str)
    }

    fn first_config_candidate(&self) -> Option<&str> {
        self.search_candidate_paths
            .iter()
            .find(|path| self.config_file_candidates.contains(*path))
            .map(String::as_str)
    }

    fn first_initialization_candidate(&self) -> Option<&str> {
        self.search_candidate_paths
            .iter()
            .find(|path| self.initialization_candidates.contains(*path))
            .map(String::as_str)
    }

    fn first_create_candidate(&self) -> Option<&str> {
        self.search_candidate_paths
            .iter()
            .find(|path| self.create_candidates.contains(*path))
            .map(String::as_str)
    }

    fn first_register_candidate(&self) -> Option<&str> {
        self.search_candidate_paths
            .iter()
            .find(|path| self.register_candidates.contains(*path))
            .map(String::as_str)
    }

    fn first_load_candidate(&self) -> Option<&str> {
        self.search_candidate_paths
            .iter()
            .find(|path| self.load_candidates.contains(*path))
            .map(String::as_str)
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

/// Returns true when `model_path` is within (equal to or narrower than) `scope`.
///
/// Both paths are normalized before comparison. Trailing slashes are stripped so
/// "sandbox/services/" and "sandbox/services" compare identically. The boundary
/// guard (`get(s.len()) == Some(&b'/')`) prevents "sandbox/service_extra" from
/// falsely matching scope "sandbox/service".
///
/// Absolute paths (e.g. emitted by the model as "/abs/path/") are never within
/// a relative scope and will always return false, causing the caller to clamp.
fn path_is_within_scope(model_path: &str, scope: &str) -> bool {
    let p = normalize_evidence_path(model_path);
    let s = normalize_evidence_path(scope);
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
                investigation.search_produced_results.to_string(),
            ),
            ("files_read", investigation.files_read_count.to_string()),
            (
                "candidate_reads",
                investigation.candidate_reads_count.to_string(),
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
        "saved",
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

/// Detects the structural investigation mode from the prompt text.
/// Evaluated in priority order so each prompt maps to exactly one mode.
/// Priority: UsageLookup > ConfigLookup > InitializationLookup > CreateLookup > RegisterLookup > LoadLookup > DefinitionLookup > General.
fn detect_investigation_mode(text: &str) -> InvestigationMode {
    let lower = text.to_ascii_lowercase();
    if [
        "use",
        "used",
        "uses",
        "usage",
        "reference",
        "referenced",
        "references",
        "occur",
        "occurs",
        "occurrence",
        "occurrences",
        "appear",
        "appears",
    ]
    .iter()
    .any(|term| contains_word(&lower, term))
    {
        return InvestigationMode::UsageLookup;
    }
    if ["config", "configured", "configuration", "configure"]
        .iter()
        .any(|term| contains_word(&lower, term))
    {
        return InvestigationMode::ConfigLookup;
    }
    if contains_initialization_term(&lower) {
        return InvestigationMode::InitializationLookup;
    }
    if contains_create_term(&lower) {
        return InvestigationMode::CreateLookup;
    }
    if contains_register_term(&lower) {
        return InvestigationMode::RegisterLookup;
    }
    if contains_load_term(&lower) {
        return InvestigationMode::LoadLookup;
    }
    if [
        "defined",
        "definition",
        "declared",
        "declares",
        "declaration",
    ]
    .iter()
    .any(|term| contains_word(&lower, term))
    {
        return InvestigationMode::DefinitionLookup;
    }
    InvestigationMode::General
}

const INITIALIZATION_TERMS: &[&str] = &["initialize", "initialized", "initialization"];

fn contains_initialization_term(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    INITIALIZATION_TERMS.iter().any(|term| lower.contains(term))
}

const CREATE_TERMS: &[&str] = &["create", "created", "creation"];

fn contains_create_term(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    CREATE_TERMS.iter().any(|term| lower.contains(term))
}

const REGISTER_TERMS: &[&str] = &["register", "registered", "registration"];

fn contains_register_term(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    REGISTER_TERMS.iter().any(|term| lower.contains(term))
}

const LOAD_TERMS: &[&str] = &["load", "loaded", "loading"];

fn contains_load_term(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    LOAD_TERMS.iter().any(|term| lower.contains(term))
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

/// Returns true if the path's file extension identifies it as a config file.
/// Classification is purely extension-based — no content analysis or filename heuristics.
/// Handles the exact `.env` dotfile explicitly since `Path::extension()` returns None for it.
fn is_config_file(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    let p = Path::new(&lower);
    if matches!(
        p.extension().and_then(|e| e.to_str()),
        Some("yaml" | "yml" | "toml" | "json" | "ini" | "cfg" | "conf" | "properties")
    ) {
        return true;
    }
    // `.env` is part of the allowed config extension set; `.env.*` is intentionally
    // excluded because its actual extension is something else.
    if let Some(filename) = p.file_name().and_then(|f| f.to_str()) {
        if filename == ".env" {
            return true;
        }
    }
    false
}

/// Returns true if the line (after stripping leading whitespace) is an import declaration.
/// Coverage: Python (`import X`, `from X import Y`) and Java/Go/TypeScript (`import X`).
/// Rust `use` statements and C `#include` are intentionally excluded — too many false positives
/// from identifiers like `use` appearing in natural language or in assertion-style code.
/// No regex, no scoring — prefix matching only, same style as looks_like_definition.
fn looks_like_import(line: &str) -> bool {
    let t = line.trim_start();
    // `import X` — Python, Java, Go, TypeScript, JavaScript
    t.starts_with("import ")
        // `from X import Y` — Python
        || (t.starts_with("from ") && t.contains(" import "))
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
        let investigation_mode = original_user_prompt
            .map(detect_investigation_mode)
            .unwrap_or(InvestigationMode::General);
        let investigation_path_scope: Option<String> = if investigation_required {
            original_user_prompt.and_then(extract_investigation_path_scope)
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
                    && !investigation.search_produced_results
                    && investigation.files_read_count == 0
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
                        if !investigation.direct_answer_correction_issued {
                            investigation.direct_answer_correction_issued = true;
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

                    if investigation.search_produced_results {
                        // Both candidate-read slots exhausted and evidence is still not ready.
                        // Do not attempt another correction cycle — terminate cleanly.
                        if investigation.candidate_reads_count
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
            && !investigation.search_attempted
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
