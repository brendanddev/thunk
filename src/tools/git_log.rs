use std::io::{self, Read};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;

use crate::runtime::ResolvedToolInput;

use super::context::ToolContext;
use super::types::{
    ExecutionKind, GitLogEntry, GitLogOutput, ToolError, ToolInput, ToolOutput, ToolRunResult,
    ToolSpec,
};
use super::Tool;

const MAX_LOG_ENTRIES: usize = 20;
const MAX_LOG_AUTHOR_CHARS: usize = 120;
const MAX_LOG_SUBJECT_CHARS: usize = 240;
const MAX_GIT_LOG_STDOUT_BYTES: usize = 64 * 1024;
const MAX_GIT_LOG_STDERR_BYTES: usize = 8 * 1024;
const GIT_LOG_FORMAT: &str = "%H%x1f%h%x1f%ad%x1f%an%x1f%s%x1e";

pub struct GitLogTool {
    context: ToolContext,
}

impl GitLogTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }

    fn run_legacy(&self, input: &ToolInput) -> Result<ToolRunResult, ToolError> {
        let ToolInput::GitLog = input else {
            return Err(ToolError::InvalidInput(
                "git_log received wrong input variant".into(),
            ));
        };

        let output = run_bounded_git_log(&self.context.root)?;

        if !output.status.success() {
            if is_empty_repo_log_error(&output.stderr.bytes) {
                return Ok(ToolRunResult::Immediate(ToolOutput::GitLog(
                    empty_git_log_output(),
                )));
            }
            return Err(git_log_error(&output.stderr.bytes));
        }

        let stdout = String::from_utf8_lossy(&output.stdout.bytes);
        Ok(ToolRunResult::Immediate(ToolOutput::GitLog(
            parse_git_log_output(&stdout, output.stdout.truncated),
        )))
    }
}

impl Tool for GitLogTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "git_log",
            description: "Show read-only recent git commit history for the project.",
            input_hint: "",
            execution_kind: ExecutionKind::Immediate,
            default_risk: None,
        }
    }

    fn run(&self, input: &ResolvedToolInput) -> Result<ToolRunResult, ToolError> {
        // Temporary Slice 15.3.3 shim: keep legacy git_log behavior unchanged
        // until the resolved-input-native migration lands in 15.3.5.
        let legacy = match input {
            ResolvedToolInput::GitLog => ToolInput::GitLog,
            _ => {
                return Err(ToolError::InvalidInput(
                    "git_log received wrong input variant".into(),
                ))
            }
        };

        self.run_legacy(&legacy)
    }
}

struct BoundedGitOutput {
    status: ExitStatus,
    stdout: BoundedCapture,
    stderr: BoundedCapture,
}

struct BoundedCapture {
    bytes: Vec<u8>,
    truncated: bool,
}

fn run_bounded_git_log(root: &std::path::Path) -> Result<BoundedGitOutput, ToolError> {
    let pretty = format!("--pretty=format:{GIT_LOG_FORMAT}");
    let mut child = Command::new("git")
        .args([
            "--no-pager",
            "log",
            "--max-count=20",
            "--no-show-signature",
            "--date=short",
            &pretty,
        ])
        .current_dir(root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(git_command_error)?;

    let stdout = child.stdout.take().ok_or_else(output_capture_error)?;
    let stderr = child.stderr.take().ok_or_else(output_capture_error)?;

    let stdout_reader =
        thread::spawn(move || read_bounded_stream(stdout, MAX_GIT_LOG_STDOUT_BYTES));
    let stderr_reader =
        thread::spawn(move || read_bounded_stream(stderr, MAX_GIT_LOG_STDERR_BYTES));

    let status = child.wait()?;
    let stdout = join_capture(stdout_reader)?;
    let stderr = join_capture(stderr_reader)?;

    Ok(BoundedGitOutput {
        status,
        stdout,
        stderr,
    })
}

fn read_bounded_stream<R: Read>(mut reader: R, limit: usize) -> io::Result<BoundedCapture> {
    let mut bytes = Vec::new();
    let mut truncated = false;
    let mut buf = [0u8; 8192];

    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }

        let remaining = limit.saturating_sub(bytes.len());
        if remaining > 0 {
            let keep = remaining.min(n);
            bytes.extend_from_slice(&buf[..keep]);
        }

        if n > remaining {
            truncated = true;
            break;
        }
    }

    if truncated {
        io::copy(&mut reader, &mut io::sink())?;
    }

    Ok(BoundedCapture { bytes, truncated })
}

fn join_capture(
    handle: thread::JoinHandle<io::Result<BoundedCapture>>,
) -> Result<BoundedCapture, ToolError> {
    handle
        .join()
        .map_err(|_| output_capture_error())?
        .map_err(ToolError::Io)
}

fn output_capture_error() -> ToolError {
    ToolError::InvalidInput("git_log failed: output capture failed".into())
}

fn git_command_error(error: io::Error) -> ToolError {
    if error.kind() == io::ErrorKind::NotFound {
        ToolError::InvalidInput("git_log failed: git executable unavailable".into())
    } else {
        ToolError::Io(error)
    }
}

fn git_log_error(stderr: &[u8]) -> ToolError {
    let stderr = String::from_utf8_lossy(stderr);
    if stderr.to_ascii_lowercase().contains("not a git repository") {
        ToolError::InvalidInput("git_log failed: not a Git repository".into())
    } else {
        ToolError::InvalidInput("git_log failed".into())
    }
}

fn is_empty_repo_log_error(stderr: &[u8]) -> bool {
    String::from_utf8_lossy(stderr)
        .to_ascii_lowercase()
        .contains("does not have any commits yet")
}

fn empty_git_log_output() -> GitLogOutput {
    GitLogOutput {
        entries: Vec::new(),
        truncated: false,
    }
}

fn parse_git_log_output(stdout: &str, capture_truncated: bool) -> GitLogOutput {
    let mut entries = Vec::new();
    let mut truncated = capture_truncated;

    for record in stdout.split('\x1e') {
        if entries.len() >= MAX_LOG_ENTRIES {
            truncated = true;
            break;
        }

        let record = record.trim_matches('\n');
        if record.is_empty() {
            continue;
        }

        let mut fields = record.split('\x1f');
        let Some(hash) = fields.next() else {
            continue;
        };
        let Some(short_hash) = fields.next() else {
            continue;
        };
        let Some(date) = fields.next() else {
            continue;
        };
        let Some(author) = fields.next() else {
            continue;
        };
        let Some(subject) = fields.next() else {
            continue;
        };

        let (author, author_truncated) = truncate_chars(author, MAX_LOG_AUTHOR_CHARS);
        let (subject, subject_truncated) = truncate_chars(subject, MAX_LOG_SUBJECT_CHARS);
        truncated |= author_truncated || subject_truncated;

        entries.push(GitLogEntry {
            hash: hash.to_string(),
            short_hash: short_hash.to_string(),
            date: date.to_string(),
            author,
            subject,
        });
    }

    GitLogOutput { entries, truncated }
}

fn truncate_chars(value: &str, max_chars: usize) -> (String, bool) {
    let char_count = value.chars().count();
    if char_count <= max_chars {
        return (value.to_string(), false);
    }

    let mut truncated = value
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>();
    truncated.push_str("...");
    (truncated, true)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Cursor;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};

    use tempfile::TempDir;

    use super::*;

    fn init_git_repo(path: &Path) {
        let status = Command::new("git")
            .args(["init"])
            .current_dir(path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "git init must succeed");
    }

    fn git(path: &Path, args: &[&str]) {
        let status = Command::new("git")
            .args(args)
            .current_dir(path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "git command must succeed: {args:?}");
    }

    fn commit_file_with(path: &Path, file: &str, contents: &str, author: &str, subject: &str) {
        fs::write(path.join(file), contents).unwrap();
        git(path, &["add", file]);
        git(
            path,
            &[
                "-c",
                &format!("user.name={author}"),
                "-c",
                "user.email=thunk@example.invalid",
                "commit",
                "-m",
                subject,
            ],
        );
    }

    fn commit_file(path: &Path, file: &str, contents: &str, subject: &str) {
        commit_file_with(path, file, contents, "thunk", subject);
    }

    fn run_log(path: &Path) -> Result<ToolRunResult, ToolError> {
        GitLogTool::new(ToolContext::new(PathBuf::from(path))).run_legacy(&ToolInput::GitLog)
    }

    #[test]
    fn spec_is_immediate() {
        let tool = GitLogTool::new(ToolContext::new(PathBuf::from(".")));
        let spec = tool.spec();
        assert_eq!(spec.name, "git_log");
        assert_eq!(spec.execution_kind, ExecutionKind::Immediate);
        assert!(spec.default_risk.is_none());
    }

    #[test]
    fn default_registry_dispatches_git_log() {
        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());
        let registry = crate::tools::default_registry(tmp.path().to_path_buf());

        let out = registry
            .dispatch(crate::runtime::ResolvedToolInput::GitLog)
            .unwrap();
        assert!(matches!(
            out,
            ToolRunResult::Immediate(ToolOutput::GitLog(_))
        ));
    }

    #[test]
    fn non_git_directory_returns_deterministic_error() {
        let tmp = TempDir::new().unwrap();
        let err = run_log(tmp.path()).unwrap_err();
        assert!(matches!(
            err,
            ToolError::InvalidInput(message)
                if message == "git_log failed: not a Git repository"
        ));
    }

    #[test]
    fn empty_repo_returns_deterministic_empty_log() {
        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());

        let out = run_log(tmp.path()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::GitLog(log)) = out else {
            panic!("expected Immediate(GitLog)");
        };
        assert!(log.entries.is_empty());
        assert!(!log.truncated);
    }

    #[test]
    fn repo_with_commits_returns_ordered_entries() {
        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());
        commit_file(tmp.path(), "first.txt", "first\n", "first commit");
        commit_file(tmp.path(), "second.txt", "second\n", "second commit");

        let out = run_log(tmp.path()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::GitLog(log)) = out else {
            panic!("expected Immediate(GitLog)");
        };
        assert_eq!(log.entries.len(), 2);
        assert_eq!(log.entries[0].subject, "second commit");
        assert_eq!(log.entries[1].subject, "first commit");
        assert_eq!(log.entries[0].author, "thunk");
        assert_eq!(log.entries[0].date.len(), 10);
        assert!(!log.entries[0].hash.is_empty());
        assert!(!log.entries[0].short_hash.is_empty());
        assert!(!log.truncated);
    }

    #[test]
    fn long_author_and_subject_are_truncated() {
        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());
        let author = "A".repeat(MAX_LOG_AUTHOR_CHARS + 10);
        let subject = "S".repeat(MAX_LOG_SUBJECT_CHARS + 10);
        commit_file_with(tmp.path(), "long.txt", "long\n", &author, &subject);

        let out = run_log(tmp.path()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::GitLog(log)) = out else {
            panic!("expected Immediate(GitLog)");
        };
        assert_eq!(log.entries.len(), 1);
        assert_eq!(log.entries[0].author.chars().count(), MAX_LOG_AUTHOR_CHARS);
        assert_eq!(
            log.entries[0].subject.chars().count(),
            MAX_LOG_SUBJECT_CHARS
        );
        assert!(log.entries[0].author.ends_with("..."));
        assert!(log.entries[0].subject.ends_with("..."));
        assert!(log.truncated);
    }

    #[test]
    fn parse_git_log_marks_truncated_when_capture_was_capped() {
        let stdout = "0123456789012345678901234567890123456789\x1f0123456\x1f2026-04-22\x1fthunk\x1fsubject\x1e";
        let log = parse_git_log_output(stdout, true);

        assert_eq!(log.entries.len(), 1);
        assert!(log.truncated);
    }

    #[test]
    fn bounded_capture_marks_truncated_without_retaining_extra_bytes() {
        let input = vec![b'x'; MAX_GIT_LOG_STDERR_BYTES + 10];
        let capture = read_bounded_stream(Cursor::new(input), MAX_GIT_LOG_STDERR_BYTES).unwrap();

        assert_eq!(capture.bytes.len(), MAX_GIT_LOG_STDERR_BYTES);
        assert!(capture.truncated);
    }
}
