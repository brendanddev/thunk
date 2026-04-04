// src/memory/index.rs
//
// Level 2: project file index.
//
// Stores a 2-3 sentence summary for each source file in a SQLite database at
// .local/memory/{project_hash}.db. The hash is derived from the current
// working directory path so each project gets its own database.
//
// Summary generation calls the active backend, so indexing runs on the model
// thread (or in the `params index .` command path) — never on the UI thread.
//
// find_relevant() does simple keyword scoring for now. Semantic search via
// embeddings can be layered on top later without changing the schema.

use std::collections::{HashSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use rusqlite::{params, Connection};
use tracing::{debug, info, warn};

use crate::config;
use crate::error::Result;
use crate::inference::{InferenceBackend, Message};
use super::run_prompt_sync;

pub struct ProjectIndex {
    conn: Connection,
}

pub struct IndexDelta {
    pub to_index: Vec<PathBuf>,
    pub removed: usize,
    pub unchanged: usize,
    pub skipped_large: usize,
}

const INDEXABLE_EXTENSIONS: &[&str] = &[
    "rs", "py", "ts", "tsx", "js", "jsx", "go", "c", "cpp", "h", "java", "kt", "swift",
    "rb", "php", "cs", "toml", "yaml", "yml", "json", "md", "txt", "sh", "sql",
];
const MAX_INDEXABLE_FILE_BYTES: u64 = 100_000;

impl ProjectIndex {
    /// Open or create the index database for the current working directory.
    pub fn open() -> Result<Self> {
        let cwd = std::env::current_dir()?;
        Self::open_for(&cwd)
    }

    /// Open or create the index database for a specific project root.
    pub fn open_for(project_root: &Path) -> Result<Self> {
        let db_path = db_path_for(project_root)?;

        let conn = Connection::open(&db_path)?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS files (
                path          TEXT    PRIMARY KEY,
                summary       TEXT    NOT NULL,
                embedding_json TEXT,
                last_modified INTEGER NOT NULL
            );",
        )?;

        info!(db = %db_path.display(), "project index opened");

        Ok(Self { conn })
    }

    /// Summarize `content` via the backend and store it against `path`.
    ///
    /// Replaces any existing entry for this path.
    pub fn index_file(
        &self,
        path: &Path,
        content: &str,
        backend: &dyn InferenceBackend,
    ) -> Result<()> {
        let path_str = path.to_string_lossy().to_string();
        let mtime = file_mtime(path).unwrap_or(0) as i64;

        let prompt = vec![
            Message::system(
                "You are a helpful assistant that writes concise code file summaries.",
            ),
            Message::user(&format!(
                "Summarize this file in 2-3 sentences. \
                 State what it does and name its key types or functions.\n\
                 File: {path_str}\n\n```\n{content}\n```"
            )),
        ];

        let summary = run_prompt_sync(backend, &prompt)?;
        let summary = summary.trim().to_string();

        if summary.is_empty() {
            warn!(path = %path_str, "summary was empty, skipping");
            return Ok(());
        }

        self.conn.execute(
            "INSERT OR REPLACE INTO files \
             (path, summary, embedding_json, last_modified) \
             VALUES (?1, ?2, NULL, ?3)",
            params![path_str, summary, mtime],
        )?;

        debug!(path = %path_str, "file indexed");

        Ok(())
    }

    /// Return up to `limit` (path, summary) pairs whose summaries match the
    /// query by keyword overlap. Unmatched entries are excluded; ties are
    /// broken by match count descending.
    pub fn find_relevant(&self, query: &str, limit: usize) -> Result<Vec<(String, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, summary FROM files")?;

        let query_lower = query.to_lowercase();
        let keywords: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored: Vec<(usize, String, String)> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .flatten()
            .filter_map(|(path, summary)| {
                let summary_lower = summary.to_lowercase();
                let score = keywords
                    .iter()
                    .filter(|kw| summary_lower.contains(*kw))
                    .count();
                if score > 0 {
                    Some((score, path, summary))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored.truncate(limit);

        Ok(scored.into_iter().map(|(_, p, s)| (p, s)).collect())
    }

    /// Returns true if `path` is not yet indexed or its mtime has changed.
    pub fn needs_reindex(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy().to_string();

        let current_mtime = match file_mtime(path) {
            Some(m) => m as i64,
            None => return false, // Can't stat — skip safely
        };

        let stored: rusqlite::Result<i64> = self.conn.query_row(
            "SELECT last_modified FROM files WHERE path = ?1",
            params![path_str],
            |row| row.get(0),
        );

        match stored {
            Ok(stored_mtime) => stored_mtime != current_mtime,
            Err(_) => true, // Not in the DB yet
        }
    }

    /// Scan the project tree for changed/new/deleted indexable files.
    ///
    /// This is intentionally cheap: it uses file metadata and the existing DB
    /// records to build a delta without reading file contents.
    pub fn collect_delta(&self, root: &Path) -> Result<IndexDelta> {
        let mut files = Vec::new();
        collect_indexable_files(root, &mut files)?;

        let mut current_paths = HashSet::with_capacity(files.len());
        let mut to_index = Vec::new();
        let mut unchanged = 0usize;
        let mut skipped_large = 0usize;

        for file in files {
            current_paths.insert(file.to_string_lossy().to_string());

            let size = std::fs::metadata(&file).map(|m| m.len()).unwrap_or(0);
            if size > MAX_INDEXABLE_FILE_BYTES {
                skipped_large += 1;
                continue;
            }

            if self.needs_reindex(&file) {
                to_index.push(file);
            } else {
                unchanged += 1;
            }
        }

        let indexed_paths = self.indexed_paths()?;
        let mut removed = 0usize;
        for path in indexed_paths {
            if !current_paths.contains(&path) {
                removed += self.remove_path_str(&path)?;
            }
        }

        Ok(IndexDelta {
            to_index,
            removed,
            unchanged,
            skipped_large,
        })
    }

    fn indexed_paths(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT path FROM files")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        Ok(rows.flatten().collect())
    }

    fn remove_path_str(&self, path: &str) -> Result<usize> {
        Ok(self
            .conn
            .execute("DELETE FROM files WHERE path = ?1", params![path])?)
    }
}

/// Recursively collect indexable files from a project root.
pub fn collect_indexable_files(root: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();

        if name.starts_with('.') {
            continue;
        }
        if matches!(name.as_ref(), "target" | "node_modules" | "__pycache__") {
            continue;
        }

        if path.is_dir() {
            collect_indexable_files(&path, files)?;
        } else if is_indexable_file(&path) {
            files.push(path);
        }
    }

    Ok(())
}

pub fn is_indexable_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| INDEXABLE_EXTENSIONS.contains(&ext))
        .unwrap_or(false)
}

/// Build the path to the project index database.
/// Creates `.local/memory/` if it doesn't exist.
fn db_path_for(project_root: &Path) -> Result<PathBuf> {
    let memory_dir = config::memory_dir()?;
    let hash = path_hash(&project_root.to_string_lossy());
    Ok(memory_dir.join(format!("{hash}.db")))
}

fn path_hash(s: &str) -> String {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn file_mtime(path: &Path) -> Option<u64> {
    std::fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_project_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("params-index-test-{label}-{nonce}"));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn test_index(root: &Path) -> ProjectIndex {
        ProjectIndex::open_for(root).expect("open project index")
    }

    #[test]
    fn collect_delta_detects_new_and_unchanged_files() {
        let root = temp_project_dir("delta");
        let file = root.join("main.rs");
        fs::write(&file, "fn main() {}\n").expect("write source file");

        let index = test_index(&root);
        let first = index.collect_delta(&root).expect("collect first delta");
        assert_eq!(first.to_index, vec![file.clone()]);
        assert_eq!(first.unchanged, 0);

        index.conn.execute(
            "INSERT INTO files (path, summary, embedding_json, last_modified) VALUES (?1, ?2, NULL, ?3)",
            params![
                file.to_string_lossy().to_string(),
                "summary",
                file_mtime(&file).unwrap() as i64
            ],
        ).expect("seed index row");

        let second = index.collect_delta(&root).expect("collect second delta");
        assert!(second.to_index.is_empty());
        assert_eq!(second.unchanged, 1);

        let _ = fs::remove_file(file);
        let _ = fs::remove_dir(root);
    }

    #[test]
    fn collect_delta_removes_deleted_rows() {
        let root = temp_project_dir("removed");
        let file = root.join("lib.rs");
        fs::write(&file, "pub fn value() {}\n").expect("write source file");

        let index = test_index(&root);
        index.conn.execute(
            "INSERT INTO files (path, summary, embedding_json, last_modified) VALUES (?1, ?2, NULL, ?3)",
            params![
                file.to_string_lossy().to_string(),
                "summary",
                file_mtime(&file).unwrap() as i64
            ],
        ).expect("seed index row");

        fs::remove_file(&file).expect("remove source file");
        let delta = index.collect_delta(&root).expect("collect delta");
        assert_eq!(delta.removed, 1);

        let count: i64 = index.conn.query_row(
            "SELECT COUNT(*) FROM files",
            [],
            |row| row.get(0),
        ).expect("count files");
        assert_eq!(count, 0);

        let _ = fs::remove_dir(root);
    }

    #[test]
    fn collect_delta_skips_large_files() {
        let root = temp_project_dir("large");
        let file = root.join("big.rs");
        fs::write(&file, "a".repeat((MAX_INDEXABLE_FILE_BYTES as usize) + 1))
            .expect("write large file");

        let index = test_index(&root);
        let delta = index.collect_delta(&root).expect("collect delta");
        assert!(delta.to_index.is_empty());
        assert_eq!(delta.skipped_large, 1);

        let _ = fs::remove_file(file);
        let _ = fs::remove_dir(root);
    }
}
