use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::tools::ToolOutput;

use super::super::paths::normalize_evidence_path;
use super::super::types::RuntimeEvent;

const RUNTIME_TRACE_ENV: &str = "THUNK_TRACE_RUNTIME";

// Exact substring triggers used for structured investigation modes.
// Keep these narrow: broad matching increases false positives for small local models.
const INITIALIZATION_TERMS: &[&str] = &["initialize", "initialized", "initialization"];
const CREATE_TERMS: &[&str] = &["create", "created", "creation"];
const REGISTER_TERMS: &[&str] = &["register", "registered", "registration"];
const LOAD_TERMS: &[&str] = &["load", "loaded", "loading"];
const SAVE_TERMS: &[&str] = &["save", "saved", "saving"];

// Lockfiles are useful project metadata, but usually poor evidence for code-location answers.
const LOCKFILE_NAMES: &[&str] = &[
    "Cargo.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "Pipfile.lock",
];

// Source extensions used to prefer implementation files over generated or metadata matches.
const SOURCE_EXTENSIONS: &[&str] = &[
    "rs", "py", "ts", "tsx", "js", "jsx", "go", "java", "c", "cpp", "h", "hpp",
];

// Advisory runtime tracing only. Trace events must not influence control flow.
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

fn push_unique_path(paths: &mut Vec<String>, path: &str) {
    if !paths.iter().any(|existing| existing == path) {
        paths.push(path.to_string());
    }
}

pub(crate) fn contains_initialization_term(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    INITIALIZATION_TERMS.iter().any(|term| lower.contains(term))
}

pub(crate) fn contains_create_term(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    CREATE_TERMS.iter().any(|term| lower.contains(term))
}

pub(crate) fn contains_register_term(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    REGISTER_TERMS.iter().any(|term| lower.contains(term))
}

pub(crate) fn contains_load_term(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    LOAD_TERMS.iter().any(|term| lower.contains(term))
}

pub(crate) fn contains_save_term(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    SAVE_TERMS.iter().any(|term| lower.contains(term))
}

fn contains_word(text: &str, needle: &str) -> bool {
    text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .any(|token| token == needle)
}

/// Returns true if the path's file extension identifies it as a config file.
/// Classification is purely extension-based — no content analysis or filename heuristics.
/// Handles the exact `.env` dotfile explicitly since `Path::extension()` returns None for it.
pub(crate) fn is_config_file(path: &str) -> bool {
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

fn is_lockfile_path(path: &str) -> bool {
    Path::new(path)
        .file_name()
        .and_then(|f| f.to_str())
        .is_some_and(|filename| LOCKFILE_NAMES.contains(&filename))
}

fn is_source_candidate_path(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|ext| SOURCE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Returns true if the line (after stripping leading whitespace) is an import declaration.
/// Coverage: Python (`import X`, `from X import Y`) and Java/Go/TypeScript (`import X`).
/// Rust `use` statements and C `#include` are intentionally excluded — too many false positives
/// from identifiers like `use` appearing in natural language or in assertion-style code.
/// No regex, no scoring — prefix matching only, same style as looks_like_definition.
pub(crate) fn looks_like_import(line: &str) -> bool {
    let t = line.trim_start();
    // `import X` — Python, Java, Go, TypeScript, JavaScript
    t.starts_with("import ")
        // `from X import Y` — Python
        || (t.starts_with("from ") && t.contains(" import "))
}

/// Returns true if the line defines the exact identifier `symbol`.
/// Strips each known definition prefix, extracts the first alphanumeric+underscore token,
/// and requires exact equality — so "class TaskStatus:" does not match symbol "Task".
/// Coverage mirrors `looks_like_definition`.
pub(crate) fn looks_like_definition_of_symbol(line: &str, symbol: &str) -> bool {
    let t = line.trim_start();
    const PREFIXES: &[&str] = &[
        "pub enum ",
        "pub struct ",
        "pub fn ",
        "pub type ",
        "pub trait ",
        "pub const ",
        "pub static ",
        "enum ",
        "struct ",
        "fn ",
        "type ",
        "const ",
        "trait ",
        "impl ",
        "class ",
        "def ",
        "func ",
        "function ",
        "interface ",
    ];
    for prefix in PREFIXES {
        if let Some(rest) = t.strip_prefix(prefix) {
            let ident = rest
                .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .next()
                .unwrap_or("");
            if ident == symbol {
                return true;
            }
        }
    }
    false
}

/// Returns true if the line (after stripping leading whitespace) looks like a symbol definition.
/// Coverage: Rust, Python, Go, TypeScript, JavaScript.
/// C/C++ patterns are excluded — too many false positives without a type parser.
/// No regex, no scoring — prefix matching only.
fn looks_like_definition(line: &str) -> bool {
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

/// Structural mode for the current investigation turn.
/// Computed once from the user prompt before the tool loop starts.
/// Controls which evidence-acceptance gates are active for this turn.
#[derive(Copy, Clone)]
pub(crate) enum InvestigationMode {
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
    /// Prompt signals a narrow save lookup.
    /// Non-save reads are structurally insufficient when save candidates exist.
    SaveLookup,
}

impl InvestigationMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            InvestigationMode::General => "General",
            InvestigationMode::UsageLookup => "UsageLookup",
            InvestigationMode::DefinitionLookup => "DefinitionLookup",
            InvestigationMode::ConfigLookup => "ConfigLookup",
            InvestigationMode::InitializationLookup => "InitializationLookup",
            InvestigationMode::CreateLookup => "CreateLookup",
            InvestigationMode::RegisterLookup => "RegisterLookup",
            InvestigationMode::LoadLookup => "LoadLookup",
            InvestigationMode::SaveLookup => "SaveLookup",
        }
    }
}

/// Detects the structural investigation mode from the prompt text.
/// Evaluated in priority order so each prompt maps to exactly one mode.
/// Priority: UsageLookup > ConfigLookup > InitializationLookup > CreateLookup > RegisterLookup > LoadLookup > SaveLookup > DefinitionLookup > General.
pub(crate) fn detect_investigation_mode(text: &str) -> InvestigationMode {
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
    if contains_save_term(&lower) {
        return InvestigationMode::SaveLookup;
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

/// Distinguishes which structural insufficiency caused a candidate read to be rejected.
/// Used by the caller in run_tool_round to select the appropriate correction message.
pub(crate) enum RecoveryKind {
    /// The file was definition-only on a usage lookup with usage candidates available.
    DefinitionOnly,
    /// The file was not a definition-site candidate on a definition lookup when definition
    /// candidates exist. Runtime dispatches the correct definition file directly.
    NonDefinitionSite,
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
    /// The file lacked save-term matches when save candidates exist.
    Save,
    /// The file was a lockfile when a matched source candidate exists.
    Lockfile,
}

impl RecoveryKind {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            RecoveryKind::DefinitionOnly => "DefinitionOnly",
            RecoveryKind::NonDefinitionSite => "NonDefinitionSite",
            RecoveryKind::ImportOnly => "ImportOnly",
            RecoveryKind::ConfigFile => "ConfigFile",
            RecoveryKind::Initialization => "Initialization",
            RecoveryKind::Create => "Create",
            RecoveryKind::Register => "Register",
            RecoveryKind::Load => "Load",
            RecoveryKind::Save => "Save",
            RecoveryKind::Lockfile => "Lockfile",
        }
    }
}

/// Tracks per-turn search → read investigation state.
/// Resets at the start of each call to run_turns, exactly like SearchBudget.
pub(crate) struct InvestigationState {
    /// True once any search_code call this turn returned at least one match.
    search_produced_results: bool,
    /// Count of read_file calls that completed successfully this turn.
    files_read_count: usize,
    /// File paths from the current non-empty search results.
    search_candidate_paths: Vec<String>,
    /// Candidate paths where every matched line looks like a definition site.
    /// Populated during record_search_results alongside search_candidate_paths.
    definition_only_candidates: HashSet<String>,
    /// Count of matched lines per candidate that are not definition sites.
    /// Preserves only search-result-local evidence and is used for UsageLookup
    /// candidate quality ranking after search_code succeeds.
    non_definition_match_counts: HashMap<String, usize>,
    /// Candidate paths where at least one matched line is an exact definition of the
    /// queried symbol (weaker than definition_only_candidates, which requires all lines).
    /// Used exclusively by DefinitionLookup: Gate 8 and first_definition_candidate.
    definition_site_candidates: HashSet<String>,
    /// True if at least one candidate in the current search results has a
    /// non-definition match line (i.e. a usage file is available).
    has_non_definition_candidates: bool,
    /// Number of accepted matched-candidate reads that counted as useful evidence.
    /// Kept separate from candidate_reads_count so the runtime can distinguish
    /// broad UsageLookup happy-path reads from rejected or fallback reads.
    useful_accepted_candidate_reads: usize,
    /// Normalized paths of accepted matched-candidate reads that counted as useful evidence.
    /// Used to deterministically exclude already-read candidates when broad UsageLookup
    /// requires a second runtime-owned evidence read.
    useful_accepted_candidate_paths: HashSet<String>,
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
    /// True when this turn is a broad UsageLookup prompt eligible for the
    /// multi-candidate evidence policy.
    broad_usage_lookup: bool,
    /// Runtime-owned target for how many accepted useful matched-candidate reads
    /// are required before evidence is ready. Defaults to 1; broad UsageLookup
    /// may raise it to 2 when two substantive candidates are available.
    useful_candidate_reads_target: usize,
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
    /// Candidate paths where at least one matched line contains a save term.
    /// Populated during record_search_results alongside search_candidate_paths.
    save_candidates: HashSet<String>,
    /// True after the save recovery correction has been issued once this turn.
    save_correction_issued: bool,
    /// Candidate paths whose basename is an exact known lockfile name.
    lockfile_candidates: HashSet<String>,
    /// True after the lockfile recovery correction has been issued once this turn.
    lockfile_correction_issued: bool,
    /// Number of times a non-candidate read_file was attempted this turn.
    /// Persists across run_tool_round calls so the repeated-offense terminal fires
    /// even when the first offense and second offense are in separate model responses.
    non_candidate_read_attempts: usize,
}

impl InvestigationState {
    pub(crate) fn new() -> Self {
        Self {
            search_produced_results: false,
            files_read_count: 0,
            search_candidate_paths: Vec::new(),
            definition_only_candidates: HashSet::new(),
            non_definition_match_counts: HashMap::new(),
            definition_site_candidates: HashSet::new(),
            has_non_definition_candidates: false,
            useful_accepted_candidate_reads: 0,
            useful_accepted_candidate_paths: HashSet::new(),
            premature_synthesis_correction_issued: false,
            direct_answer_correction_issued: false,
            search_attempted: false,
            candidate_reads_count: 0,
            broad_usage_lookup: false,
            useful_candidate_reads_target: 1,
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
            save_candidates: HashSet::new(),
            save_correction_issued: false,
            lockfile_candidates: HashSet::new(),
            lockfile_correction_issued: false,
            non_candidate_read_attempts: 0,
        }
    }

    pub(crate) fn configure_usage_evidence_policy(&mut self, broad_usage_lookup: bool) {
        self.broad_usage_lookup = broad_usage_lookup;
    }

    pub(crate) fn evidence_ready(&self) -> bool {
        self.search_produced_results
            && self.useful_accepted_candidate_reads >= self.useful_candidate_reads_target
    }

    pub(crate) fn all_useful_accepted_reads_are_definition_only(&self) -> bool {
        self.useful_accepted_candidate_reads > 0
            && self.useful_accepted_candidate_paths.iter().all(|p| {
                self.definition_only_candidates
                    .iter()
                    .any(|d| normalize_evidence_path(d) == *p)
            })
    }

    pub(crate) fn has_non_definition_candidates(&self) -> bool {
        self.has_non_definition_candidates
    }

    pub(crate) fn search_produced_results(&self) -> bool {
        self.search_produced_results
    }

    pub(crate) fn files_read_count(&self) -> usize {
        self.files_read_count
    }

    pub(crate) fn candidate_reads_count(&self) -> usize {
        self.candidate_reads_count
    }

    pub(crate) fn useful_candidate_reads_count(&self) -> usize {
        self.useful_accepted_candidate_reads
    }

    pub(crate) fn search_attempted(&self) -> bool {
        self.search_attempted
    }

    /// Increments the non-candidate read attempt counter and returns the new count.
    /// Called in run_tool_round before dispatch; persists across rounds within a turn.
    pub(crate) fn increment_non_candidate_read_attempts(&mut self) -> usize {
        self.non_candidate_read_attempts += 1;
        self.non_candidate_read_attempts
    }

    pub(crate) fn search_candidate_count(&self) -> usize {
        self.search_candidate_paths.len()
    }

    /// Returns the best candidate path for the given investigation mode.
    /// Routes to the mode-specific classifier first; falls back to the first search
    /// candidate if the mode has no dedicated set or that set is empty.
    pub(crate) fn best_candidate_for_mode(&self, mode: InvestigationMode) -> Option<&str> {
        let mode_specific = match mode {
            InvestigationMode::InitializationLookup => self.first_initialization_candidate(),
            InvestigationMode::ConfigLookup => self.first_config_candidate(),
            InvestigationMode::CreateLookup => self.first_create_candidate(),
            InvestigationMode::RegisterLookup => self.first_register_candidate(),
            InvestigationMode::LoadLookup => self.first_load_candidate(),
            InvestigationMode::SaveLookup => self.first_save_candidate(),
            InvestigationMode::DefinitionLookup => self.first_definition_candidate(),
            InvestigationMode::UsageLookup => self.preferred_usage_candidate(),
            InvestigationMode::General => self.first_source_candidate(),
        };
        mode_specific.or_else(|| self.search_candidate_paths.first().map(String::as_str))
    }

    pub(crate) fn issue_direct_answer_correction(&mut self) -> bool {
        if self.direct_answer_correction_issued {
            return false;
        }
        self.direct_answer_correction_issued = true;
        true
    }

    pub(crate) fn issue_premature_synthesis_correction(&mut self) -> bool {
        if self.premature_synthesis_correction_issued {
            return false;
        }
        self.premature_synthesis_correction_issued = true;
        true
    }

    pub(crate) fn is_search_candidate_path(&self, path: &str) -> bool {
        let read_path = normalize_evidence_path(path);
        let relative_suffix = read_path.contains('/').then(|| format!("/{read_path}"));
        self.search_candidate_paths.iter().any(|candidate| {
            let candidate = normalize_evidence_path(candidate);
            candidate == read_path
                || relative_suffix
                    .as_ref()
                    .is_some_and(|suffix| candidate.ends_with(suffix))
        })
    }

    pub(crate) fn record_search_results(
        &mut self,
        output: &ToolOutput,
        query: Option<&str>,
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
            self.non_definition_match_counts.clear();
            self.definition_site_candidates.clear();
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
            self.save_candidates.clear();
            self.lockfile_candidates.clear();
            self.useful_accepted_candidate_reads = 0;
            self.useful_accepted_candidate_paths.clear();
            self.useful_candidate_reads_target = 1;

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
            // save: at least one matched line contains an exact save term.
            // lockfile: exact filename match against known lockfile basenames.
            let mut file_has_non_def: HashSet<String> = HashSet::new();
            let mut file_non_definition_counts: HashMap<String, usize> = HashMap::new();
            let mut file_has_exact_def: HashSet<String> = HashSet::new();
            let mut file_has_non_import: HashSet<String> = HashSet::new();
            let mut file_has_initialization: HashSet<String> = HashSet::new();
            let mut file_has_create: HashSet<String> = HashSet::new();
            let mut file_has_register: HashSet<String> = HashSet::new();
            let mut file_has_load: HashSet<String> = HashSet::new();
            let mut file_has_save: HashSet<String> = HashSet::new();
            for m in &results.matches {
                if match query {
                    Some(sym) => !looks_like_definition_of_symbol(&m.line, sym),
                    None => !looks_like_definition(&m.line),
                } {
                    file_has_non_def.insert(m.file.clone());
                    *file_non_definition_counts
                        .entry(m.file.clone())
                        .or_insert(0) += 1;
                }
                if let Some(sym) = query {
                    if looks_like_definition_of_symbol(&m.line, sym) {
                        file_has_exact_def.insert(m.file.clone());
                    }
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
                if contains_save_term(&m.line) {
                    file_has_save.insert(m.file.clone());
                }
            }
            for path in &self.search_candidate_paths {
                if file_has_non_def.contains(path) {
                    self.has_non_definition_candidates = true;
                    if let Some(count) = file_non_definition_counts.get(path) {
                        self.non_definition_match_counts
                            .insert(path.clone(), *count);
                    }
                } else {
                    self.definition_only_candidates.insert(path.clone());
                }
                if file_has_exact_def.contains(path) {
                    self.definition_site_candidates.insert(path.clone());
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
                if file_has_save.contains(path) {
                    self.save_candidates.insert(path.clone());
                }
                if is_lockfile_path(path) {
                    self.lockfile_candidates.insert(path.clone());
                }
            }

            if self.broad_usage_lookup && self.substantive_usage_candidate_count() >= 2 {
                self.useful_candidate_reads_target = 2;
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
                ("save_files", self.save_candidates.len().to_string()),
                ("lockfiles", self.lockfile_candidates.len().to_string()),
                (
                    "useful_target",
                    self.useful_candidate_reads_target.to_string(),
                ),
                ("broad_usage_lookup", self.broad_usage_lookup.to_string()),
            ],
        );
        was_empty
    }

    pub(crate) fn record_read_result(
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
            let is_save_candidate = self
                .save_candidates
                .iter()
                .any(|c| normalize_evidence_path(c) == read_path);
            let is_lockfile_candidate = self
                .lockfile_candidates
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
            }
            // Gate 7 (SaveLookup): non-save reads are structurally insufficient when
            // save candidates exist. Fire once; fallback accepts if no save candidates.
            else if matches!(mode, InvestigationMode::SaveLookup)
                && !is_save_candidate
                && !self.save_candidates.is_empty()
            {
                if !self.save_correction_issued {
                    self.save_correction_issued = true;
                    let suggested_path = self.first_save_candidate().map(str::to_string);
                    trace_runtime_decision(
                        on_event,
                        "read_evidence",
                        &[
                            ("path", read_path.clone()),
                            ("accepted", "false".into()),
                            ("reason", "save_non_save_candidate".into()),
                            (
                                "recovery_path",
                                suggested_path.clone().unwrap_or_else(|| "none".into()),
                            ),
                        ],
                    );
                    return suggested_path.map(|p| (p, RecoveryKind::Save));
                }
                trace_runtime_decision(
                    on_event,
                    "read_evidence",
                    &[
                        ("path", read_path.clone()),
                        ("accepted", "false".into()),
                        ("reason", "save_recovery_already_issued".into()),
                    ],
                );
                // Correction already issued: fall through without accepting.
            }
            // Gate 8 (DefinitionLookup): all non-definition-site reads are rejected.
            // A file passes when it is in definition_only_candidates (all lines are defs)
            // OR definition_site_candidates (at least one line is an exact definition).
            // When a definition candidate exists, RuntimeDispatch redirects to it.
            // When no definition candidate was classified, first_definition_candidate()
            // returns None and the gate returns None — the read does not satisfy evidence.
            else if matches!(mode, InvestigationMode::DefinitionLookup)
                && !is_def_only
                && !self
                    .definition_site_candidates
                    .iter()
                    .any(|c| normalize_evidence_path(c) == read_path)
            {
                let suggested_path = self.first_definition_candidate().map(str::to_string);
                trace_runtime_decision(
                    on_event,
                    "read_evidence",
                    &[
                        ("path", read_path.clone()),
                        ("accepted", "false".into()),
                        ("reason", "definition_lookup_non_definition_site".into()),
                        (
                            "recovery_path",
                            suggested_path.clone().unwrap_or_else(|| "none".into()),
                        ),
                    ],
                );
                return suggested_path.map(|p| (p, RecoveryKind::NonDefinitionSite));
            } else {
                if is_lockfile_candidate {
                    let suggested_path = self.first_source_candidate().map(str::to_string);
                    if suggested_path.is_some() {
                        if !self.lockfile_correction_issued {
                            self.lockfile_correction_issued = true;
                            trace_runtime_decision(
                                on_event,
                                "read_evidence",
                                &[
                                    ("path", read_path.clone()),
                                    ("accepted", "false".into()),
                                    ("reason", "lockfile_candidate".into()),
                                    (
                                        "recovery_path",
                                        suggested_path.clone().unwrap_or_else(|| "none".into()),
                                    ),
                                ],
                            );
                            return suggested_path.map(|p| (p, RecoveryKind::Lockfile));
                        }
                        trace_runtime_decision(
                            on_event,
                            "read_evidence",
                            &[
                                ("path", read_path.clone()),
                                ("accepted", "false".into()),
                                ("reason", "lockfile_recovery_already_issued".into()),
                            ],
                        );
                        return None;
                    }
                }
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
                self.useful_accepted_candidate_reads += 1;
                self.useful_accepted_candidate_paths
                    .insert(read_path.clone());
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
                        (
                            "useful_candidate_reads",
                            self.useful_accepted_candidate_reads.to_string(),
                        ),
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
        } else if matches!(mode, InvestigationMode::SaveLookup) && self.save_candidates.is_empty() {
            "save_fallback_no_save_candidates".into()
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

    pub(crate) fn preferred_usage_candidate(&self) -> Option<&str> {
        self.preferred_usage_candidate_with_filters(&HashSet::new(), false)
    }

    pub(crate) fn next_usage_evidence_candidate(&self) -> Option<&str> {
        if self.useful_accepted_candidate_reads == 0
            || self.useful_accepted_candidate_reads >= self.useful_candidate_reads_target
        {
            return None;
        }

        self.preferred_usage_candidate_with_filters(&self.useful_accepted_candidate_paths, true)
    }

    fn preferred_usage_candidate_with_filters(
        &self,
        excluded: &HashSet<String>,
        substantive_only: bool,
    ) -> Option<&str> {
        let mut best_index = None;
        let mut best_key = None;

        for (index, path) in self.search_candidate_paths.iter().enumerate() {
            if excluded.contains(&normalize_evidence_path(path)) {
                continue;
            }
            if substantive_only && !self.is_substantive_usage_candidate(path) {
                continue;
            }
            let key = self.usage_candidate_quality_key(path, index);
            if best_key.as_ref().is_none_or(|current| key < *current) {
                best_key = Some(key);
                best_index = Some(index);
            }
        }

        best_index.map(|index| self.search_candidate_paths[index].as_str())
    }

    fn substantive_usage_candidate_count(&self) -> usize {
        self.search_candidate_paths
            .iter()
            .filter(|path| self.is_substantive_usage_candidate(path))
            .count()
    }

    fn is_substantive_usage_candidate(&self, path: &str) -> bool {
        !self.definition_only_candidates.contains(path)
            && !self.import_only_candidates.contains(path)
            && !self.lockfile_candidates.contains(path)
    }

    fn first_definition_candidate(&self) -> Option<&str> {
        self.search_candidate_paths
            .iter()
            .find(|path| {
                self.definition_only_candidates.contains(*path)
                    || self.definition_site_candidates.contains(*path)
            })
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

    fn first_save_candidate(&self) -> Option<&str> {
        self.search_candidate_paths
            .iter()
            .find(|path| self.save_candidates.contains(*path))
            .map(String::as_str)
    }

    fn first_source_candidate(&self) -> Option<&str> {
        self.search_candidate_paths
            .iter()
            .find(|path| {
                !self.lockfile_candidates.contains(*path) && is_source_candidate_path(path)
            })
            .map(String::as_str)
    }

    fn usage_candidate_quality_key(&self, path: &str, index: usize) -> (u8, u8, u8, usize, usize) {
        let is_definition_only = self.definition_only_candidates.contains(path);
        let is_import_only = self.import_only_candidates.contains(path);
        let is_normal_source = is_source_candidate_path(path)
            && !self.config_file_candidates.contains(path)
            && !self.initialization_candidates.contains(path)
            && !self.lockfile_candidates.contains(path);
        let non_definition_match_count = self
            .non_definition_match_counts
            .get(path)
            .copied()
            .unwrap_or(0);

        (
            u8::from(is_definition_only),
            u8::from(is_import_only),
            u8::from(!is_normal_source),
            usize::MAX - non_definition_match_count,
            index,
        )
    }

    /// Returns a mode-specific candidate preference hint after a search, or None when:
    /// - no search candidates are recorded yet, or
    /// - all candidates are already of the preferred type (no misdirection possible), or
    /// - the mode has no preferred candidate class (General, UsageLookup, DefinitionLookup).
    ///
    /// DefinitionLookup is intentionally excluded: the definition_site_file preamble in
    /// tool_codec already handles that case directly in the rendered search output.
    pub(crate) fn candidate_preference_hint(&self, mode: InvestigationMode) -> Option<String> {
        if self.search_candidate_paths.is_empty() {
            return None;
        }
        match mode {
            InvestigationMode::InitializationLookup
                if !self.initialization_candidates.is_empty()
                    && self.has_non_initialization_candidates =>
            {
                let path = self.first_initialization_candidate()?;
                Some(format!(
                    "[initialization match found in {path} — read this file first]"
                ))
            }
            InvestigationMode::ConfigLookup
                if !self.config_file_candidates.is_empty() && self.has_non_config_candidates =>
            {
                let path = self.first_config_candidate()?;
                Some(format!(
                    "[config file found in {path} — read this file first]"
                ))
            }
            InvestigationMode::CreateLookup
                if !self.create_candidates.is_empty() && self.has_non_create_candidates =>
            {
                let path = self.first_create_candidate()?;
                Some(format!(
                    "[create match found in {path} — read this file first]"
                ))
            }
            InvestigationMode::RegisterLookup
                if !self.register_candidates.is_empty() && self.has_non_register_candidates =>
            {
                let path = self.first_register_candidate()?;
                Some(format!(
                    "[register match found in {path} — read this file first]"
                ))
            }
            InvestigationMode::LoadLookup
                if !self.load_candidates.is_empty() && self.has_non_load_candidates =>
            {
                let path = self.first_load_candidate()?;
                Some(format!(
                    "[load match found in {path} — read this file first]"
                ))
            }
            InvestigationMode::SaveLookup if !self.save_candidates.is_empty() => {
                let has_non_save = self
                    .search_candidate_paths
                    .iter()
                    .any(|p| !self.save_candidates.contains(p));
                if has_non_save {
                    let path = self.first_save_candidate()?;
                    Some(format!(
                        "[save match found in {path} — read this file first]"
                    ))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(matches!(
            detect_investigation_mode("Where is the session created and defined?"),
            InvestigationMode::CreateLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_initialization_priority_over_create() {
        assert!(matches!(
            detect_investigation_mode("Find where the session is initialized and created"),
            InvestigationMode::InitializationLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_usage_priority_over_create() {
        assert!(matches!(
            detect_investigation_mode("Where is the session used and created?"),
            InvestigationMode::UsageLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_config_priority_over_create() {
        assert!(matches!(
            detect_investigation_mode("Where is the session configured and created?"),
            InvestigationMode::ConfigLookup
        ));
    }

    #[test]
    fn contains_create_term_matches_exact_allowed_substrings_only() {
        assert!(contains_create_term("db.create(session)"));
        assert!(contains_create_term("session was created here"));
        assert!(contains_create_term("handles session creation"));
        assert!(contains_create_term("Session.Create()"));
        assert!(contains_create_term("CREATED_AT timestamp"));
        assert!(contains_create_term("recreate the session"));
        assert!(contains_create_term("createTable migration"));
        assert!(!contains_create_term("def handle_session(s):"));
        assert!(!contains_create_term("return session_id"));
    }

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
        assert!(matches!(
            detect_investigation_mode("Where is the command created and registered?"),
            InvestigationMode::CreateLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_register_priority_over_definition() {
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
        assert!(contains_register_term("registry.register(command)"));
        assert!(contains_register_term("command was registered here"));
        assert!(contains_register_term("command registration lives here"));
        assert!(contains_register_term("Registry.Register(command)"));
        assert!(contains_register_term("REGISTERED_COMMANDS"));
        assert!(contains_register_term("reregister command handlers"));
        assert!(contains_register_term("registration_notes = []"));
        assert!(!contains_register_term("def handle_command(command):"));
        assert!(!contains_register_term("return command_id"));
    }

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
        assert!(matches!(
            detect_investigation_mode("Where is the command registered and loaded?"),
            InvestigationMode::RegisterLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_load_priority_over_definition() {
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
        assert!(contains_load_term("session = load_session(session_id)"));
        assert!(contains_load_term("session was loaded here"));
        assert!(contains_load_term("session loading happens here"));
        assert!(contains_load_term("Session.Load()"));
        assert!(contains_load_term("LOADED_SESSION"));
        assert!(contains_load_term("session loader"));
        assert!(contains_load_term("reload session"));
        assert!(contains_load_term("autoload session"));
        assert!(!contains_load_term("def handle_session(session):"));
        assert!(!contains_load_term("return session_id"));
    }

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
        assert!(matches!(
            detect_investigation_mode("Where is the session loaded and saved?"),
            InvestigationMode::LoadLookup
        ));
    }

    #[test]
    fn detect_investigation_mode_save_priority_over_definition() {
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
        assert!(contains_save_term("save_session(session)"));
        assert!(contains_save_term("session was saved here"));
        assert!(contains_save_term("session saving happens here"));
        assert!(contains_save_term("Session.Save()"));
        assert!(contains_save_term("SAVED_SESSION"));
        assert!(contains_save_term("autosave session"));
        assert!(contains_save_term("savepoint created"));
        assert!(contains_save_term("saved_at timestamp"));
        assert!(!contains_save_term("def handle_session(session):"));
        assert!(!contains_save_term("return session_id"));
    }

    // candidate_preference_hint tests

    fn make_search_output_for_hint(matches: Vec<(&str, &str)>) -> crate::tools::ToolOutput {
        use crate::tools::types::{SearchMatch, SearchResultsOutput};
        let matches: Vec<SearchMatch> = matches
            .into_iter()
            .enumerate()
            .map(|(i, (file, line))| SearchMatch {
                file: file.to_string(),
                line_number: i + 1,
                line: line.to_string(),
            })
            .collect();
        let total = matches.len();
        crate::tools::ToolOutput::SearchResults(SearchResultsOutput {
            query: "test".into(),
            matches,
            total_matches: total,
            truncated: false,
        })
    }

    #[test]
    fn candidate_preference_hint_returns_none_when_no_candidates() {
        let state = InvestigationState::new();
        assert!(state
            .candidate_preference_hint(InvestigationMode::InitializationLookup)
            .is_none());
    }

    #[test]
    fn candidate_preference_hint_initialization_fires_with_mixed_candidates() {
        let mut state = InvestigationState::new();
        // z_init.py has an initialization term; commands.py does not
        let output = make_search_output_for_hint(vec![
            ("sandbox/cli/commands.py", "import logging"),
            ("sandbox/init/z_init.py", "def initialize_logging(): pass"),
        ]);
        state.record_search_results(&output, None, &mut |_| {});
        let hint = state.candidate_preference_hint(InvestigationMode::InitializationLookup);
        assert!(
            hint.is_some(),
            "hint must fire when init candidate exists alongside non-init"
        );
        assert!(
            hint.unwrap().contains("sandbox/init/z_init.py"),
            "hint must name the initialization candidate"
        );
    }

    #[test]
    fn candidate_preference_hint_initialization_suppressed_when_all_init() {
        let mut state = InvestigationState::new();
        // Both files have initialization terms — no non-init candidates exist
        let output = make_search_output_for_hint(vec![
            ("sandbox/init/a.py", "logging.initialize()"),
            ("sandbox/init/b.py", "def initialization_setup(): pass"),
        ]);
        state.record_search_results(&output, None, &mut |_| {});
        let hint = state.candidate_preference_hint(InvestigationMode::InitializationLookup);
        assert!(
            hint.is_none(),
            "hint must not fire when all candidates are initialization files"
        );
    }

    #[test]
    fn candidate_preference_hint_config_fires_with_mixed_candidates() {
        let mut state = InvestigationState::new();
        let output = make_search_output_for_hint(vec![
            (
                "services/database.py",
                "DATABASE_URL = os.getenv(\"DATABASE_URL\")",
            ),
            (
                "config/database.yaml",
                "database:\n  url: postgres://localhost/mydb",
            ),
        ]);
        state.record_search_results(&output, None, &mut |_| {});
        let hint = state.candidate_preference_hint(InvestigationMode::ConfigLookup);
        assert!(
            hint.is_some(),
            "hint must fire when config candidate exists alongside source"
        );
        assert!(
            hint.unwrap().contains("config/database.yaml"),
            "hint must name the config file candidate"
        );
    }

    #[test]
    fn candidate_preference_hint_config_suppressed_when_no_config_candidates() {
        let mut state = InvestigationState::new();
        let output = make_search_output_for_hint(vec![
            (
                "services/database.py",
                "DATABASE_URL = os.getenv(\"DATABASE_URL\")",
            ),
            ("services/user.py", "USER = UserService()"),
        ]);
        state.record_search_results(&output, None, &mut |_| {});
        let hint = state.candidate_preference_hint(InvestigationMode::ConfigLookup);
        assert!(
            hint.is_none(),
            "hint must not fire when no config-file candidates exist"
        );
    }

    #[test]
    fn candidate_preference_hint_general_mode_returns_none() {
        let mut state = InvestigationState::new();
        let output = make_search_output_for_hint(vec![
            ("sandbox/init/z_init.py", "logging.basicConfig()"),
            ("sandbox/cli/commands.py", "import logging"),
        ]);
        state.record_search_results(&output, None, &mut |_| {});
        assert!(
            state
                .candidate_preference_hint(InvestigationMode::General)
                .is_none(),
            "General mode must produce no candidate hint"
        );
    }

    #[test]
    fn candidate_preference_hint_definition_lookup_returns_none() {
        // DefinitionLookup is handled by definition_site_file in rendering — no hint here
        let mut state = InvestigationState::new();
        let output = make_search_output_for_hint(vec![
            ("models/enums.py", "class TaskStatus(str, Enum):"),
            ("cli/commands.py", "from models.enums import TaskStatus"),
        ]);
        state.record_search_results(&output, None, &mut |_| {});
        assert!(
            state
                .candidate_preference_hint(InvestigationMode::DefinitionLookup)
                .is_none(),
            "DefinitionLookup must not produce a candidate hint — handled by definition_site_file"
        );
    }

    #[test]
    fn candidate_preference_hint_names_first_init_candidate_in_search_order() {
        let mut state = InvestigationState::new();
        // Non-init first, then two init candidates — hint must name the first init candidate
        let output = make_search_output_for_hint(vec![
            ("sandbox/cli/commands.py", "import logging"),
            ("sandbox/init/a.py", "logging.initialize()"),
            ("sandbox/init/b.py", "def initialization_setup(): pass"),
        ]);
        state.record_search_results(&output, None, &mut |_| {});
        let hint = state.candidate_preference_hint(InvestigationMode::InitializationLookup);
        assert!(hint.is_some());
        let hint = hint.unwrap();
        assert!(
            hint.contains("sandbox/init/a.py"),
            "hint must name the first init candidate in search order, got: {hint}"
        );
        assert!(
            !hint.contains("sandbox/init/b.py"),
            "hint must not name second candidate when first already named"
        );
    }

    #[test]
    fn candidate_preference_hint_is_deterministic_for_same_inputs() {
        let mut state1 = InvestigationState::new();
        let mut state2 = InvestigationState::new();
        let matches = vec![
            ("sandbox/cli/commands.py", "import logging"),
            ("sandbox/init/z_init.py", "def initialize_logging(): pass"),
        ];
        let output1 = make_search_output_for_hint(matches.clone());
        let output2 = make_search_output_for_hint(matches);
        state1.record_search_results(&output1, None, &mut |_| {});
        state2.record_search_results(&output2, None, &mut |_| {});
        assert_eq!(
            state1.candidate_preference_hint(InvestigationMode::InitializationLookup),
            state2.candidate_preference_hint(InvestigationMode::InitializationLookup),
            "candidate_preference_hint must be deterministic for identical inputs"
        );
    }

    #[test]
    fn candidate_preference_hint_usage_lookup_returns_none() {
        let mut state = InvestigationState::new();
        let output = make_search_output_for_hint(vec![
            ("sandbox/init/z_init.py", "logging.basicConfig()"),
            ("sandbox/cli/commands.py", "logger.info(\"hello\")"),
        ]);
        state.record_search_results(&output, None, &mut |_| {});
        assert!(
            state
                .candidate_preference_hint(InvestigationMode::UsageLookup)
                .is_none(),
            "UsageLookup must produce no candidate hint"
        );
    }

    #[test]
    fn preferred_usage_candidate_prefers_substantive_source_over_import_only_and_definition() {
        let mut state = InvestigationState::new();
        let output = make_search_output_for_hint(vec![
            ("models/enums.py", "class TaskStatus(str, Enum):"),
            ("cli/header.py", "from models.enums import TaskStatus"),
            (
                "services/runner.py",
                "if task.status == TaskStatus.PENDING:",
            ),
            ("services/runner.py", "audit_status(TaskStatus.PENDING)"),
        ]);
        state.record_search_results(&output, Some("TaskStatus"), &mut |_| {});

        assert_eq!(
            state.preferred_usage_candidate(),
            Some("services/runner.py"),
            "substantive source file should outrank definition-only and import-only candidates"
        );
    }

    #[test]
    fn preferred_usage_candidate_prefers_normal_source_over_initialization_candidate() {
        let mut state = InvestigationState::new();
        let output = make_search_output_for_hint(vec![
            ("models/enums.py", "class TaskStatus(str, Enum):"),
            (
                "sandbox/init/bootstrap.py",
                "initialize_task_status(TaskStatus.PENDING)",
            ),
            (
                "sandbox/init/bootstrap.py",
                "INITIALIZED_STATUS = TaskStatus.PENDING",
            ),
            (
                "sandbox/services/runner.py",
                "if task.status == TaskStatus.PENDING:",
            ),
        ]);
        state.record_search_results(&output, Some("TaskStatus"), &mut |_| {});

        assert_eq!(
            state.preferred_usage_candidate(),
            Some("sandbox/services/runner.py"),
            "normal source files should outrank initialization candidates for UsageLookup"
        );
    }

    #[test]
    fn best_candidate_for_mode_general_prefers_source_over_docs_and_benchmarks() {
        let mut state = InvestigationState::new();
        let output = make_search_output_for_hint(vec![
            ("sandbox/README.md", "Completed tasks are documented here."),
            (
                "docs/benchmarks/runs/2026-04-29-phase16-baseline.md",
                "completed tasks benchmark notes",
            ),
            (
                "sandbox/services/task_service.py",
                "if task.completed:\n    filtered.append(task)",
            ),
        ]);
        state.record_search_results(&output, Some("completed"), &mut |_| {});

        assert_eq!(
            state.best_candidate_for_mode(InvestigationMode::General),
            Some("sandbox/services/task_service.py"),
            "General candidate preference should pick source over README/docs/benchmarks"
        );
    }

    #[test]
    fn preferred_usage_candidate_is_deterministic_for_same_inputs() {
        let matches = vec![
            ("models/enums.py", "class TaskStatus(str, Enum):"),
            ("cli/header.py", "from models.enums import TaskStatus"),
            (
                "services/runner.py",
                "if task.status == TaskStatus.PENDING:",
            ),
        ];
        let mut state1 = InvestigationState::new();
        let mut state2 = InvestigationState::new();
        let output1 = make_search_output_for_hint(matches.clone());
        let output2 = make_search_output_for_hint(matches);
        state1.record_search_results(&output1, Some("TaskStatus"), &mut |_| {});
        state2.record_search_results(&output2, Some("TaskStatus"), &mut |_| {});

        assert_eq!(
            state1.preferred_usage_candidate(),
            state2.preferred_usage_candidate(),
            "preferred usage candidate selection must be deterministic"
        );
    }

    #[test]
    fn definition_of_symbol_rejects_superstring_identifier() {
        assert!(!looks_like_definition_of_symbol(
            "class TaskStatus:",
            "Task"
        ));
        assert!(!looks_like_definition_of_symbol(
            "class TaskStatusEnum:",
            "Task"
        ));
        assert!(!looks_like_definition_of_symbol(
            "pub struct TaskRunner {",
            "Task"
        ));
        assert!(!looks_like_definition_of_symbol("fn create_task()", "task"));
    }

    #[test]
    fn definition_of_symbol_accepts_exact_identifier() {
        assert!(looks_like_definition_of_symbol("class Task:", "Task"));
        assert!(looks_like_definition_of_symbol("class Task(Base):", "Task"));
        assert!(looks_like_definition_of_symbol(
            "class Task(str, Enum):",
            "Task"
        ));
    }

    #[test]
    fn definition_of_symbol_accepts_exact_symbol_across_languages() {
        assert!(looks_like_definition_of_symbol(
            "class TaskStatus(str, Enum):",
            "TaskStatus"
        ));
        assert!(looks_like_definition_of_symbol(
            "pub struct TaskStatus {",
            "TaskStatus"
        ));
        assert!(looks_like_definition_of_symbol(
            "pub enum TaskStatus {",
            "TaskStatus"
        ));
        assert!(looks_like_definition_of_symbol(
            "def TaskStatus(self):",
            "TaskStatus"
        ));
        assert!(looks_like_definition_of_symbol(
            "func TaskStatus() error {",
            "TaskStatus"
        ));
        assert!(looks_like_definition_of_symbol(
            "function TaskStatus() {",
            "TaskStatus"
        ));
        assert!(looks_like_definition_of_symbol(
            "interface TaskStatus {",
            "TaskStatus"
        ));
    }

    #[test]
    fn definition_only_classification_uses_exact_symbol_when_query_given() {
        // query="Task": "class TaskStatus:" must NOT be definition-only —
        // the file has a non-definition match for the symbol Task.
        let mut state = InvestigationState::new();
        let output = make_search_output_for_hint(vec![(
            "models/task_status.py",
            "class TaskStatus(str, Enum):",
        )]);
        state.record_search_results(&output, Some("Task"), &mut |_| {});
        assert!(
            !state
                .definition_only_candidates
                .contains("models/task_status.py"),
            "class TaskStatus must not be definition-only for symbol 'Task'"
        );
        assert!(
            state.has_non_definition_candidates,
            "has_non_definition_candidates must be set when no exact-symbol definition exists"
        );
    }

    #[test]
    fn definition_only_classification_accepts_exact_symbol_match() {
        // query="Task": "class Task:" IS a definition-only line.
        let mut state = InvestigationState::new();
        let output = make_search_output_for_hint(vec![("models/task.py", "class Task(Base):")]);
        state.record_search_results(&output, Some("Task"), &mut |_| {});
        assert!(
            state.definition_only_candidates.contains("models/task.py"),
            "class Task must be definition-only for symbol 'Task'"
        );
        assert!(
            !state.has_non_definition_candidates,
            "has_non_definition_candidates must not be set when only exact definition exists"
        );
    }

    #[test]
    fn definition_only_classification_taskstatus_still_works() {
        // Regression: query="TaskStatus" — "class TaskStatus:" must still be definition-only.
        let mut state = InvestigationState::new();
        let output =
            make_search_output_for_hint(vec![("models/enums.py", "class TaskStatus(str, Enum):")]);
        state.record_search_results(&output, Some("TaskStatus"), &mut |_| {});
        assert!(
            state.definition_only_candidates.contains("models/enums.py"),
            "class TaskStatus must be definition-only for symbol 'TaskStatus'"
        );
    }
}
