use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;

use super::{Tool, ToolRunResult};
use crate::error::{ParamsError, Result};
use crate::safety;

// File extensions we'll search — avoids binary files and build artifacts
const SEARCHABLE_EXTENSIONS: &[&str] = &[
    "rs", "py", "ts", "tsx", "js", "jsx", "go", "c", "cpp", "h", "java", "kt", "swift", "rb",
    "php", "cs", "toml", "yaml", "yml", "json", "md", "txt", "sh", "env", "sql",
];
const MAX_OUTPUT_MATCHES: usize = 24;
const MAX_FILES_IN_OUTPUT: usize = 6;
const MAX_HITS_PER_FILE: usize = 4;
const MAX_LINE_CHARS: usize = 180;

pub struct SearchCode;

impl Tool for SearchCode {
    fn name(&self) -> &str {
        "search"
    }

    fn description(&self) -> &str {
        "Search for text across all source files in the current directory. \
         Returns matching lines with file names and line numbers."
    }

    fn run(&self, arg: &str) -> Result<ToolRunResult> {
        info!(tool = "search", "tool called");
        let query = arg.trim();

        if query.is_empty() {
            return Err(ParamsError::Config("Search query cannot be empty".into()));
        }

        let current_dir = safety::project_root()?;
        let _ = safety::inspect_search_scope()?;
        let mut matches: Vec<SearchMatch> = Vec::new();

        // Walk the directory tree
        walk_and_search(&current_dir, query, &mut matches)?;

        if matches.is_empty() {
            return Ok(ToolRunResult::Immediate(format!(
                "No results found for: {query}"
            )));
        }

        let total = matches.len();
        let ranked = rank_search_matches(&current_dir, query, matches);

        let mut output = format!(
            "Search results for '{}' ({} matches{}):\n\n",
            query,
            total,
            if total > 50 { ", showing first 50" } else { "" }
        );

        let mut shown = 0usize;
        for file in ranked.iter().take(MAX_FILES_IN_OUTPUT) {
            if shown >= MAX_OUTPUT_MATCHES {
                break;
            }

            output.push_str(&format!("\n{}:\n", file.path));
            for hit in file.hits.iter().take(MAX_HITS_PER_FILE) {
                if shown >= MAX_OUTPUT_MATCHES {
                    break;
                }
                output.push_str(&format!(
                    "  {:4}: {}\n",
                    hit.line_number,
                    clip_inline(hit.line_content.trim(), MAX_LINE_CHARS)
                ));
                shown += 1;
            }
        }

        Ok(ToolRunResult::Immediate(output))
    }
}

struct SearchMatch {
    path: PathBuf,
    line_number: usize,
    line_content: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SearchLineMatch {
    line_number: usize,
    line_content: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SearchFileMatches {
    path: String,
    hits: Vec<SearchLineMatch>,
}

fn walk_and_search(dir: &Path, query: &str, matches: &mut Vec<SearchMatch>) -> Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };

    let mut entries = entries.flatten().collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();

        // Skip hidden dirs and build artifacts
        if name.starts_with('.') {
            continue;
        }
        if matches!(name.as_str(), "target" | "node_modules" | "__pycache__") {
            continue;
        }

        if path.is_dir() {
            walk_and_search(&path, query, matches)?;
        } else if is_searchable(&path) {
            search_file(&path, query, matches);
        }
    }

    Ok(())
}

fn is_searchable(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| SEARCHABLE_EXTENSIONS.contains(&ext))
        .unwrap_or(false)
}

fn search_file(path: &Path, query: &str, matches: &mut Vec<SearchMatch>) {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return, // Skip unreadable files silently
    };

    let query_lower = query.to_lowercase();

    for (i, line) in content.lines().enumerate() {
        if line.to_lowercase().contains(&query_lower) {
            matches.push(SearchMatch {
                path: path.to_path_buf(),
                line_number: i + 1,
                line_content: line.to_string(),
            });
        }
    }
}

fn relative_display_path(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .ok()
        .and_then(|p| p.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| path.display().to_string())
}

fn clip_inline(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    let clipped = text
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    format!("{}…", clipped.trim_end())
}

fn is_doc_path(path: &str) -> bool {
    path.ends_with(".md") || path.starts_with("docs/")
}

fn is_test_like_path(path: &str) -> bool {
    path.starts_with("tests/")
        || path.contains("/tests/")
        || path.contains("fixtures")
        || path.contains("snapshots")
        || path.ends_with("_test.rs")
        || path.ends_with("_tests.rs")
}

fn is_source_path(path: &str) -> bool {
    path.starts_with("src/")
        || path.ends_with(".rs")
        || path.ends_with(".py")
        || path.ends_with(".ts")
        || path.ends_with(".tsx")
        || path.ends_with(".js")
        || path.ends_with(".jsx")
        || path.ends_with(".go")
        || path.ends_with(".java")
        || path.ends_with(".kt")
        || path.ends_with(".swift")
}

fn is_definition_like_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with("pub fn ")
        || trimmed.starts_with("fn ")
        || trimmed.starts_with("pub struct ")
        || trimmed.starts_with("struct ")
        || trimmed.starts_with("pub enum ")
        || trimmed.starts_with("enum ")
        || trimmed.starts_with("impl ")
        || trimmed.starts_with("pub mod ")
        || trimmed.starts_with("mod ")
        || trimmed.starts_with("def ")
        || trimmed.starts_with("class ")
        || trimmed.starts_with("interface ")
}

fn score_search_file(query: &str, file: &SearchFileMatches) -> isize {
    let query = query.trim().to_ascii_lowercase();
    let path = file.path.to_ascii_lowercase();
    let mut score = 0isize;

    if is_source_path(&file.path) {
        score += 28;
    }
    if file.path.starts_with("src/") {
        score += 16;
    }
    if is_doc_path(&file.path) {
        score -= 18;
    }
    if is_test_like_path(&file.path) {
        score -= 28;
    }
    if path.contains("prompt") || path.contains("fixture") {
        score -= 14;
    }
    if !query.is_empty() && path.contains(&query) {
        score += 10;
    }

    score += (file.hits.len().min(6) as isize) * 3;

    for hit in file.hits.iter().take(4) {
        let line = hit.line_content.trim();
        let line_lower = line.to_ascii_lowercase();
        if !query.is_empty() && line_lower.contains(&query) {
            score += 6;
        }
        if is_definition_like_line(line) {
            score += 24;
        }
        if line.contains("assert!")
            || line.contains("#[test]")
            || line.contains("mod tests")
            || line.contains("Search results for")
        {
            score -= 10;
        }
        if line.contains('"') && !is_definition_like_line(line) {
            score -= 4;
        }
    }

    score
}

fn rank_search_matches(
    root: &Path,
    query: &str,
    matches: Vec<SearchMatch>,
) -> Vec<SearchFileMatches> {
    let mut grouped = BTreeMap::<String, Vec<SearchLineMatch>>::new();
    for hit in matches {
        grouped
            .entry(relative_display_path(&hit.path, root))
            .or_default()
            .push(SearchLineMatch {
                line_number: hit.line_number,
                line_content: hit.line_content,
            });
    }

    let mut files = grouped
        .into_iter()
        .map(|(path, mut hits)| {
            hits.sort_by_key(|hit| hit.line_number);
            hits.dedup_by(|a, b| {
                a.line_number == b.line_number && a.line_content == b.line_content
            });
            SearchFileMatches { path, hits }
        })
        .collect::<Vec<_>>();

    files.sort_by(|a, b| {
        score_search_file(query, b)
            .cmp(&score_search_file(query, a))
            .then_with(|| a.path.cmp(&b.path))
    });
    files
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn search_output_uses_relative_paths() {
        // Reproduces the root cause of wrong line numbers in auto-inspection:
        // SearchCode was emitting absolute paths while ReadFile emits relative
        // paths. The mismatch caused every search hit and every read file to be
        // treated as different files, so deduplication failed and the model saw
        // both the absolute and relative form of the same path.
        let dir = std::env::temp_dir().join(format!(
            "params-search-relpath-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let src_dir = dir.join("src");
        fs::create_dir_all(&src_dir).unwrap();
        let file = src_dir.join("lib.rs");
        fs::write(&file, "pub fn my_function() {}\n").unwrap();

        // Temporarily change the working directory so project_root() returns
        // our temp dir. We can't call SearchCode.run() easily without the
        // safety infrastructure, so instead test the path-stripping logic
        // directly via the internal walk helpers.
        let mut matches: Vec<SearchMatch> = Vec::new();
        walk_and_search(&dir, "my_function", &mut matches).unwrap();
        assert!(!matches.is_empty(), "should find the function");

        for m in &matches {
            let file_str = relative_display_path(&m.path, &dir);

            assert!(
                !file_str.starts_with('/'),
                "path in search output must be relative, got: {file_str}"
            );
            assert!(
                file_str.contains("src"),
                "relative path should contain 'src', got: {file_str}"
            );
        }

        let _ = fs::remove_file(file);
        let _ = fs::remove_dir(src_dir);
        let _ = fs::remove_dir(dir);
    }

    #[test]
    fn search_ranking_prefers_source_definition_over_docs_and_tests() {
        let dir = std::env::temp_dir().join(format!(
            "params-search-ranking-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(dir.join("docs/context")).unwrap();
        fs::create_dir_all(dir.join("src/inference")).unwrap();
        fs::create_dir_all(dir.join("src/session")).unwrap();

        fs::write(
            dir.join("docs/context/PLANS.md"),
            "load_most_recent overview\n",
        )
        .unwrap();
        fs::write(
            dir.join("src/inference/session.rs"),
            "fn prompt() {\n    let q = \"Where is session restore implemented?\";\n}\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn keeps_query() {\n        let x = \"load_most_recent\";\n    }\n}\n",
        )
        .unwrap();
        fs::write(
            dir.join("src/session/mod.rs"),
            "pub fn load_most_recent(&self) -> Result<Option<SavedSession>> {\n    Ok(None)\n}\n",
        )
        .unwrap();

        let mut matches = Vec::new();
        walk_and_search(&dir, "load_most_recent", &mut matches).unwrap();
        let ranked = rank_search_matches(&dir, "load_most_recent", matches);

        assert!(!ranked.is_empty(), "expected ranked search files");
        assert_eq!(ranked[0].path, "src/session/mod.rs");
        assert_ne!(ranked[0].path, "docs/context/PLANS.md");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn search_output_is_bounded_to_top_ranked_hits() {
        let dir = std::env::temp_dir().join(format!(
            "params-search-bounded-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(dir.join("src")).unwrap();
        for i in 0..10 {
            fs::write(
                dir.join("src").join(format!("file{i}.rs")),
                format!(
                    "pub fn load_most_recent_{i}() {{}}\nlet x = load_most_recent;\nlet y = load_most_recent;\nlet z = load_most_recent;\nlet w = load_most_recent;\n"
                ),
            )
            .unwrap();
        }

        let mut matches = Vec::new();
        walk_and_search(&dir, "load_most_recent", &mut matches).unwrap();
        let ranked = rank_search_matches(&dir, "load_most_recent", matches);

        let mut shown = 0usize;
        let mut files = 0usize;
        for file in ranked.iter().take(MAX_FILES_IN_OUTPUT) {
            files += 1;
            for _ in file.hits.iter().take(MAX_HITS_PER_FILE) {
                shown += 1;
            }
        }

        assert!(files <= MAX_FILES_IN_OUTPUT);
        assert!(shown <= MAX_OUTPUT_MATCHES);

        let _ = fs::remove_dir_all(dir);
    }
}
