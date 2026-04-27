use std::fs;
use std::path::Path;

use super::context::ToolContext;
use super::types::{
    ExecutionKind, SearchMatch, SearchResultsOutput, ToolError, ToolInput, ToolOutput,
    ToolRunResult, ToolSpec,
};
use super::Tool;

/// Internal upper bound on how many matches the walk collects.
/// Kept at 50 so total_matches is accurate up to that count without full-walk cost.
const MAX_COLLECT: usize = 50;

/// Maximum number of matches injected into the conversation as a single search result.
/// Each match adds one `file:line: content` line. 15 gives the model enough signal to
/// identify which file to read without the 50-match bulk that was causing long prefill.
const MAX_RESULTS_SHOWN: usize = 15;

/// Maximum number of matching lines collected from a single file.
/// Caps any one file's contribution to the collection budget so the walk visits more
/// distinct files before hitting MAX_COLLECT. Definition-site files that are
/// alphabetically late in the walk are then reached and promoted by the sort step.
const MAX_LINES_COLLECTED_PER_FILE: usize = 3;

/// Directory names that are always skipped during the recursive walk.
const SKIP_DIRS: &[&str] = &["target", "node_modules", ".git", ".hg", "dist", "build"];

/// File extensions treated as text. Everything else is skipped as likely binary.
const TEXT_EXTENSIONS: &[&str] = &[
    "rs",
    "toml",
    "md",
    "txt",
    "json",
    "yaml",
    "yml",
    "ts",
    "tsx",
    "js",
    "jsx",
    "py",
    "go",
    "c",
    "cpp",
    "h",
    "hpp",
    "sh",
    "bash",
    "zsh",
    "fish",
    "html",
    "css",
    "scss",
    "xml",
    "sql",
    "env",
    "gitignore",
    "lock",
];

pub struct SearchCodeTool {
    context: ToolContext,
}

impl SearchCodeTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

impl Tool for SearchCodeTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "search_code",
            description: "Search for a literal substring across source files in the project.",
            input_hint: "query_string [path/to/scope]",
            execution_kind: ExecutionKind::Immediate,
            default_risk: None,
        }
    }

    fn run(&self, input: &ToolInput) -> Result<ToolRunResult, ToolError> {
        let ToolInput::SearchCode { query, path } = input else {
            return Err(ToolError::InvalidInput(
                "search_code received wrong input variant".into(),
            ));
        };

        if query.is_empty() {
            return Err(ToolError::InvalidInput(
                "search query cannot be empty".into(),
            ));
        }

        let root = match path.as_deref() {
            Some(p) => self.context.resolve(p),
            None => self.context.root.clone(),
        };
        let root = root.as_path();

        let mut matches = Vec::new();
        walk_and_search(root, query, &mut matches)?;
        let mut matches = sort_by_file_group_priority(matches, query);

        let total_matches = matches.len();
        let truncated = total_matches > MAX_RESULTS_SHOWN;
        matches.truncate(MAX_RESULTS_SHOWN);

        Ok(ToolRunResult::Immediate(ToolOutput::SearchResults(
            SearchResultsOutput {
                query: query.clone(),
                matches,
                total_matches,
                truncated,
            },
        )))
    }
}

fn walk_and_search(
    dir: &Path,
    query: &str,
    matches: &mut Vec<SearchMatch>,
) -> Result<(), ToolError> {
    if matches.len() >= MAX_COLLECT {
        return Ok(());
    }

    let read = match fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return Ok(()), // skip unreadable dirs silently
    };

    let mut entries: Vec<_> = read.filter_map(|e| e.ok()).collect();
    // Sort for deterministic ordering across platforms.
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        if matches.len() >= MAX_COLLECT {
            break;
        }

        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if path.is_dir() {
            if !name_str.starts_with('.') && !SKIP_DIRS.contains(&name_str.as_ref()) {
                walk_and_search(&path, query, matches)?;
            }
        } else if is_text_file(&path) {
            search_in_file(&path, query, matches);
        }
    }

    Ok(())
}

fn search_in_file(path: &Path, query: &str, matches: &mut Vec<SearchMatch>) {
    let Ok(contents) = fs::read_to_string(path) else {
        return; // skip binary or unreadable files silently
    };

    let mut from_this_file = 0;
    for (idx, line) in contents.lines().enumerate() {
        if matches.len() >= MAX_COLLECT || from_this_file >= MAX_LINES_COLLECTED_PER_FILE {
            break;
        }
        if line.contains(query) {
            matches.push(SearchMatch {
                file: path.to_string_lossy().into_owned(),
                line_number: idx + 1,
                line: line.to_string(),
            });
            from_this_file += 1;
        }
    }
}

fn is_text_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| TEXT_EXTENSIONS.contains(&ext))
        .unwrap_or(false)
}

/// Groups matches by file and stable-sorts the groups so definition-containing source files
/// appear before usage-only files within the same class tier.
/// Within each priority bucket the original walk order (alphabetical DFS) is preserved.
///
/// Three-level key within each class tier:
///   (class, !has_exact_def, !has_all_def)
/// where has_exact_def = the query is defined exactly on at least one match line,
/// and   has_all_def   = every match line looks like a definition.
fn sort_by_file_group_priority(matches: Vec<SearchMatch>, query: &str) -> Vec<SearchMatch> {
    let mut groups: Vec<(String, Vec<SearchMatch>)> = Vec::new();
    for m in matches {
        if let Some(group) = groups.iter_mut().find(|(f, _)| *f == m.file) {
            group.1.push(m);
        } else {
            let file = m.file.clone();
            groups.push((file, vec![m]));
        }
    }
    groups.sort_by_key(|(file, file_matches)| {
        let class = file_class_priority(file);
        let has_exact_def = file_matches
            .iter()
            .any(|m| is_exact_symbol_definition(&m.line, query));
        let has_all_def = file_matches
            .iter()
            .all(|m| looks_like_definition(&m.line));
        (class, !has_exact_def, !has_all_def)
    });
    groups.into_iter().flat_map(|(_, ms)| ms).collect()
}

/// Returns true if the line defines exactly `query` as its top-level symbol.
/// Trims leading whitespace, strips the definition keyword prefix (e.g., `class `, `def `, `fn `),
/// extracts the first identifier after that prefix, and compares it exactly to `query`.
/// Does NOT match type annotations in function parameters (e.g., `def foo(task: Task)`).
/// Mirrors the heuristic in `runtime::tool_codec::is_exact_symbol_definition`.
fn is_exact_symbol_definition(line: &str, query: &str) -> bool {
    let line = line.trim_start();
    let def_prefixes = [
        "pub struct ", "pub const ", "pub static ", "pub enum ", "pub fn ",
        "pub type ", "pub trait ", "function ", "interface ", "struct ",
        "enum ", "class ", "impl ", "const ", "trait ", "def ", "func ",
        "type ", "fn ",
    ];
    for prefix in def_prefixes {
        if let Some(after_prefix) = line.strip_prefix(prefix) {
            let first_ident = after_prefix.split(|c: char| !c.is_alphanumeric() && c != '_').next();
            if let Some(ident) = first_ident {
                return ident == query;
            }
            return false;
        }
    }
    false
}

/// Returns true if the line looks like a top-level definition in any supported language.
/// Mirrors the heuristic in `runtime::tool_codec::looks_like_definition`.
fn looks_like_definition(line: &str) -> bool {
    let t = line.trim_start();
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
        || t.starts_with("class ")
        || t.starts_with("def ")
        || t.starts_with("func ")
        || t.starts_with("function ")
        || t.starts_with("interface ")
}

/// Returns a sort key that places source code files before config/data files,
/// and both before documentation/text files.  Within each class the existing
/// alphabetical walk order is preserved (sort_by_key is stable).
///
/// 0 — source:  rs go ts tsx js jsx py c cpp h hpp sh bash zsh fish html css scss sql xml
/// 1 — config:  toml json yaml yml env
/// 2 — docs:    md txt gitignore lock (and anything unrecognised)
fn file_class_priority(path: &str) -> u8 {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    match ext {
        "rs" | "go" | "ts" | "tsx" | "js" | "jsx" | "py" | "c" | "cpp" | "h" | "hpp" | "sh"
        | "bash" | "zsh" | "fish" | "html" | "css" | "scss" | "sql" | "xml" => 0,
        "toml" | "json" | "yaml" | "yml" | "env" => 1,
        _ => 2,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn search(query: &str, path: &str) -> Result<ToolRunResult, ToolError> {
        SearchCodeTool::new(ToolContext::new(PathBuf::from("."))).run(&ToolInput::SearchCode {
            query: query.to_string(),
            path: Some(path.to_string()),
        })
    }

    #[test]
    fn finds_matching_lines() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("lib.rs"), "fn foo() {}\nfn bar() {}\n").unwrap();

        let out = search("fn foo", tmp.path().to_str().unwrap()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::SearchResults(sr)) = out else {
            panic!("expected Immediate(SearchResults)")
        };

        assert_eq!(sr.matches.len(), 1);
        assert_eq!(sr.matches[0].line_number, 1);
        assert!(sr.matches[0].line.contains("fn foo"));
    }

    #[test]
    fn skips_target_directory() {
        let tmp = TempDir::new().unwrap();
        let target = tmp.path().join("target");
        fs::create_dir(&target).unwrap();
        fs::write(target.join("output.rs"), "needle in target").unwrap();
        fs::write(tmp.path().join("main.rs"), "no match here").unwrap();

        let out = search("needle", tmp.path().to_str().unwrap()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::SearchResults(sr)) = out else {
            panic!("expected Immediate(SearchResults)")
        };
        assert!(sr.matches.is_empty());
    }

    #[test]
    fn returns_error_on_empty_query() {
        let err = SearchCodeTool::new(ToolContext::new(PathBuf::from(".")))
            .run(&ToolInput::SearchCode {
                query: "".into(),
                path: None,
            })
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn truncates_to_match_cap_and_preserves_total_count() {
        let tmp = TempDir::new().unwrap();
        // 6 files × 3 lines each = 18 collected matches, exceeding MAX_RESULTS_SHOWN (15).
        // Each file is capped at MAX_LINES_COLLECTED_PER_FILE, so truncation must come from
        // spread across files rather than a single high-match file.
        for i in 0..6u8 {
            let content: String = (0..3).map(|j| format!("needle line {i}-{j}\n")).collect();
            fs::write(tmp.path().join(format!("file_{i}.rs")), content).unwrap();
        }

        let out = search("needle", tmp.path().to_str().unwrap()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::SearchResults(sr)) = out else {
            panic!("expected Immediate(SearchResults)")
        };

        assert_eq!(
            sr.matches.len(),
            MAX_RESULTS_SHOWN,
            "matches must be capped at MAX_RESULTS_SHOWN"
        );
        assert!(
            sr.total_matches > MAX_RESULTS_SHOWN,
            "total_matches must reflect all collected before truncation"
        );
        assert!(
            sr.truncated,
            "truncated must be true when results exceed the cap"
        );
    }

    #[test]
    fn searches_recursively() {
        let tmp = TempDir::new().unwrap();
        let sub = tmp.path().join("src");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("mod.rs"), "pub fn deep_fn() {}").unwrap();

        let out = search("deep_fn", tmp.path().to_str().unwrap()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::SearchResults(sr)) = out else {
            panic!("expected Immediate(SearchResults)")
        };
        assert_eq!(sr.matches.len(), 1);
    }

    #[test]
    fn source_files_ranked_before_docs() {
        // README.md and lib.rs both match — source file must appear first.
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("README.md"), "needle in docs").unwrap();
        fs::write(tmp.path().join("lib.rs"), "fn needle() {}").unwrap();

        let out = search("needle", tmp.path().to_str().unwrap()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::SearchResults(sr)) = out else {
            panic!("expected Immediate(SearchResults)")
        };

        assert_eq!(sr.matches.len(), 2);
        assert!(
            sr.matches[0].file.ends_with("lib.rs"),
            "source file must appear before doc file; got: {:?}",
            sr.matches.iter().map(|m| &m.file).collect::<Vec<_>>()
        );
        assert!(
            sr.matches[1].file.ends_with("README.md"),
            "doc file must appear after source file"
        );
    }

    #[test]
    fn config_files_ranked_between_source_and_docs() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("README.md"), "needle note").unwrap();
        fs::write(tmp.path().join("Cargo.toml"), "needle = true").unwrap();
        fs::write(tmp.path().join("lib.rs"), "fn needle() {}").unwrap();

        let out = search("needle", tmp.path().to_str().unwrap()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::SearchResults(sr)) = out else {
            panic!("expected Immediate(SearchResults)")
        };

        assert_eq!(sr.matches.len(), 3);
        let files: Vec<&str> = sr.matches.iter().map(|m| m.file.as_str()).collect();
        let rs_pos = files.iter().position(|f| f.ends_with("lib.rs")).unwrap();
        let toml_pos = files
            .iter()
            .position(|f| f.ends_with("Cargo.toml"))
            .unwrap();
        let md_pos = files.iter().position(|f| f.ends_with("README.md")).unwrap();

        assert!(rs_pos < toml_pos, "source must come before config");
        assert!(toml_pos < md_pos, "config must come before docs");
    }

    #[test]
    fn within_class_order_is_stable() {
        // Two source files: alphabetically a.rs < b.rs — within-class order must be preserved.
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("a.rs"), "fn needle() {}").unwrap();
        fs::write(tmp.path().join("b.rs"), "fn needle() {}").unwrap();

        let out = search("needle", tmp.path().to_str().unwrap()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::SearchResults(sr)) = out else {
            panic!("expected Immediate(SearchResults)")
        };

        assert_eq!(sr.matches.len(), 2);
        assert!(
            sr.matches[0].file.ends_with("a.rs"),
            "alphabetical order must be preserved within source class"
        );
        assert!(sr.matches[1].file.ends_with("b.rs"));
    }

    #[test]
    fn docs_only_results_are_unaffected() {
        // When only doc files match, they must still be returned (no filtering).
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("README.md"), "needle in readme").unwrap();
        fs::write(tmp.path().join("NOTES.md"), "needle in notes").unwrap();

        let out = search("needle", tmp.path().to_str().unwrap()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::SearchResults(sr)) = out else {
            panic!("expected Immediate(SearchResults)")
        };

        assert_eq!(sr.matches.len(), 2, "doc-only results must not be filtered");
    }

    #[test]
    fn file_class_priority_assigns_correct_tiers() {
        assert_eq!(file_class_priority("src/lib.rs"), 0);
        assert_eq!(file_class_priority("main.go"), 0);
        assert_eq!(file_class_priority("app.ts"), 0);
        assert_eq!(file_class_priority("Cargo.toml"), 1);
        assert_eq!(file_class_priority("config.json"), 1);
        assert_eq!(file_class_priority("settings.yaml"), 1);
        assert_eq!(file_class_priority("README.md"), 2);
        assert_eq!(file_class_priority("notes.txt"), 2);
        assert_eq!(file_class_priority("Cargo.lock"), 2);
        assert_eq!(file_class_priority("no_extension"), 2);
    }

    #[test]
    fn is_exact_symbol_definition_matches_exact_symbol() {
        assert!(is_exact_symbol_definition("class Task:", "Task"));
        assert!(is_exact_symbol_definition("class Task(Base):", "Task"));
        assert!(is_exact_symbol_definition("def Task(self):", "Task"));
        assert!(is_exact_symbol_definition("struct Task {", "Task"));
        assert!(is_exact_symbol_definition("pub struct Task {", "Task"));
        assert!(is_exact_symbol_definition("fn Task(", "Task"));
        assert!(is_exact_symbol_definition("pub fn Task(", "Task"));
        assert!(is_exact_symbol_definition("interface Task {", "Task"));
        assert!(is_exact_symbol_definition("func Task(", "Task"));
        assert!(is_exact_symbol_definition("type Task =", "Task"));
        assert!(is_exact_symbol_definition("enum Task {", "Task"));
    }

    #[test]
    fn is_exact_symbol_definition_rejects_prefix_symbols() {
        // "Task" should not match lines that define "TaskStatus", "TaskRunner", etc.
        assert!(!is_exact_symbol_definition("class TaskStatus:", "Task"));
        assert!(!is_exact_symbol_definition("class TaskRunner(", "Task"));
        assert!(!is_exact_symbol_definition("def TaskFactory(self):", "Task"));
        assert!(!is_exact_symbol_definition("struct TaskManager {", "Task"));
        assert!(!is_exact_symbol_definition("pub enum TaskState {", "Task"));
    }

    #[test]
    fn is_exact_symbol_definition_rejects_type_annotations() {
        // Type annotations in function signatures should not match as definitions.
        assert!(!is_exact_symbol_definition("def _format_task(task: Task) -> str:", "Task"));
        assert!(!is_exact_symbol_definition("fn process_data(data: Task) -> Result", "Task"));
        assert!(!is_exact_symbol_definition("def create_instance(task: Task) -> None:", "Task"));
    }

    #[test]
    fn is_exact_symbol_definition_rejects_non_definition_lines() {
        assert!(!is_exact_symbol_definition("x = Task()", "Task"));
        assert!(!is_exact_symbol_definition("let t = Task::new();", "Task"));
        assert!(!is_exact_symbol_definition("return Task.from_dict(data)", "Task"));
        assert!(!is_exact_symbol_definition("from models import Task", "Task"));
    }

    #[test]
    fn exact_def_file_promoted_over_prefix_def_file() {
        // omega.py defines "class Task:" (exact), alpha.py defines "class TaskStatus:" (prefix).
        // With query "Task", omega.py must appear first despite being alphabetically later.
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("alpha.py"), "class TaskStatus:\n    pass\n").unwrap();
        fs::write(tmp.path().join("omega.py"), "class Task:\n    pass\n").unwrap();

        let out = search("Task", tmp.path().to_str().unwrap()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::SearchResults(sr)) = out else {
            panic!("expected Immediate(SearchResults)")
        };

        assert_eq!(sr.matches.len(), 2);
        assert!(
            sr.matches[0].file.ends_with("omega.py"),
            "exact definition file must appear before prefix-definition file; first: {}",
            sr.matches[0].file
        );
    }

    #[test]
    fn definition_file_promoted_over_usage_file_within_source_tier() {
        let tmp = TempDir::new().unwrap();
        // alpha.py alphabetically first; only usage lines
        fs::write(tmp.path().join("alpha.py"), "x = Task()\ntask = Task.run()\n").unwrap();
        // omega.py alphabetically later; has a definition line
        fs::write(tmp.path().join("omega.py"), "class Task:\n    pass\n").unwrap();

        let out = search("Task", tmp.path().to_str().unwrap()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::SearchResults(sr)) = out else {
            panic!("expected Immediate(SearchResults)")
        };

        assert_eq!(sr.matches.len(), 3);
        assert!(
            sr.matches[0].file.ends_with("omega.py"),
            "definition file must appear before usage-only file; first: {}",
            sr.matches[0].file
        );
    }

    #[test]
    fn non_definition_files_within_class_preserve_walk_order() {
        let tmp = TempDir::new().unwrap();
        // Both usage-only — walk order (alpha < beta) must be preserved.
        fs::write(tmp.path().join("alpha.py"), "x = Task()\n").unwrap();
        fs::write(tmp.path().join("beta.py"), "y = Task.run()\n").unwrap();

        let out = search("Task", tmp.path().to_str().unwrap()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::SearchResults(sr)) = out else {
            panic!("expected Immediate(SearchResults)")
        };

        assert_eq!(sr.matches.len(), 2);
        assert!(
            sr.matches[0].file.ends_with("alpha.py"),
            "walk order must be preserved for equal-priority files"
        );
        assert!(sr.matches[1].file.ends_with("beta.py"));
    }

    #[test]
    fn class_priority_dominates_over_definition_signal() {
        let tmp = TempDir::new().unwrap();
        // source tier, usage-only
        fs::write(tmp.path().join("alpha.py"), "needle()\n").unwrap();
        // config tier, happens to contain a definition-keyword line ("fn = ...")
        fs::write(tmp.path().join("beta.toml"), "fn = \"needle\"\n").unwrap();

        let out = search("needle", tmp.path().to_str().unwrap()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::SearchResults(sr)) = out else {
            panic!("expected Immediate(SearchResults)")
        };

        let alpha_pos = sr.matches.iter().position(|m| m.file.ends_with("alpha.py")).unwrap();
        let beta_pos = sr.matches.iter().position(|m| m.file.ends_with("beta.toml")).unwrap();
        assert!(
            alpha_pos < beta_pos,
            "source tier must dominate over definition signal in config tier"
        );
    }

    #[test]
    fn definition_file_survives_cap_over_early_usage_file() {
        // Regression: many alphabetically-early usage files filling the collection budget must
        // not cut off an alphabetically-later definition file.
        // 6 usage files × 3 lines each = 18 usage matches + 1 definition match = 19 total,
        // which exceeds MAX_RESULTS_SHOWN (15) and triggers truncation.
        let tmp = TempDir::new().unwrap();
        for i in 0..6u8 {
            let content: String = (0..3).map(|_| "x = Task()\n").collect();
            fs::write(tmp.path().join(format!("aaa_{i}.py")), content).unwrap();
        }
        // zzz.py: the definition — alphabetically last, must survive the cap via sort promotion
        fs::write(tmp.path().join("zzz.py"), "class Task:\n    pass\n").unwrap();

        let out = search("Task", tmp.path().to_str().unwrap()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::SearchResults(sr)) = out else {
            panic!("expected Immediate(SearchResults)")
        };

        assert!(sr.truncated, "must be truncated (19 matches > cap of 15)");
        assert!(
            sr.matches[0].file.ends_with("zzz.py"),
            "definition file must be promoted to first; first: {}",
            sr.matches[0].file
        );
        let zzz_count = sr.matches.iter().filter(|m| m.file.ends_with("zzz.py")).count();
        assert_eq!(zzz_count, 1, "definition match must be within the shown cap");
    }
}
