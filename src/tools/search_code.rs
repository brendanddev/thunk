use std::fs;
use std::path::Path;

use super::context::ToolContext;
use super::types::{ExecutionKind, SearchMatch, SearchResultsOutput, ToolError, ToolInput, ToolOutput, ToolRunResult, ToolSpec};
use super::Tool;

/// Maximum number of matches returned in a single search. Prevents context overload.
const MAX_MATCHES: usize = 50;

/// Directory names that are always skipped during the recursive walk.
const SKIP_DIRS: &[&str] = &["target", "node_modules", ".git", ".hg", "dist", "build"];

/// File extensions treated as text. Everything else is skipped as likely binary.
const TEXT_EXTENSIONS: &[&str] = &[
    "rs", "toml", "md", "txt", "json", "yaml", "yml", "ts", "tsx", "js", "jsx",
    "py", "go", "c", "cpp", "h", "hpp", "sh", "bash", "zsh", "fish",
    "html", "css", "scss", "xml", "sql", "env", "gitignore", "lock",
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
            return Err(ToolError::InvalidInput("search query cannot be empty".into()));
        }

        let root = match path.as_deref() {
            Some(p) => self.context.resolve(p),
            None => self.context.root.clone(),
        };
        let root = root.as_path();

        let mut matches = Vec::new();
        walk_and_search(root, query, &mut matches)?;

        let truncated = matches.len() >= MAX_MATCHES;

        Ok(ToolRunResult::Immediate(ToolOutput::SearchResults(
            SearchResultsOutput {
                query: query.clone(),
                matches,
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
    if matches.len() >= MAX_MATCHES {
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
        if matches.len() >= MAX_MATCHES {
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
        if matches.len() >= MAX_MATCHES {
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
        let ToolRunResult::Immediate(ToolOutput::SearchResults(sr)) = out else { panic!("expected Immediate(SearchResults)") };

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
        let ToolRunResult::Immediate(ToolOutput::SearchResults(sr)) = out else { panic!("expected Immediate(SearchResults)") };
        assert!(sr.matches.is_empty());
    }

    #[test]
    fn returns_error_on_empty_query() {
        let err = SearchCodeTool::new(ToolContext::new(PathBuf::from(".")))
            .run(&ToolInput::SearchCode { query: "".into(), path: None })
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn searches_recursively() {
        let tmp = TempDir::new().unwrap();
        let sub = tmp.path().join("src");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("mod.rs"), "pub fn deep_fn() {}").unwrap();

        let out = search("deep_fn", tmp.path().to_str().unwrap()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::SearchResults(sr)) = out else { panic!("expected Immediate(SearchResults)") };
        assert_eq!(sr.matches.len(), 1);
    }
}
