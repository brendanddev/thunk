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
        matches.sort_by_key(|m| file_class_priority(&m.file));

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

    for (idx, line) in contents.lines().enumerate() {
        if matches.len() >= MAX_COLLECT {
            break;
        }
        if line.contains(query) {
            matches.push(SearchMatch {
                file: path.to_string_lossy().into_owned(),
                line_number: idx + 1,
                line: line.to_string(),
            });
        }
    }
}

fn is_text_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| TEXT_EXTENSIONS.contains(&ext))
        .unwrap_or(false)
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
        // 20 matching lines — exceeds MAX_RESULTS_SHOWN (15) but under MAX_COLLECT (50)
        let content: String = (0..20).map(|i| format!("needle line {i}\n")).collect();
        fs::write(tmp.path().join("matches.rs"), content).unwrap();

        let out = search("needle", tmp.path().to_str().unwrap()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::SearchResults(sr)) = out else {
            panic!("expected Immediate(SearchResults)")
        };

        assert_eq!(
            sr.matches.len(),
            MAX_RESULTS_SHOWN,
            "matches must be capped at MAX_RESULTS_SHOWN"
        );
        assert_eq!(
            sr.total_matches, 20,
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
}
