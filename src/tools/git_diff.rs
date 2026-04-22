use std::io::{self, Read};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;

use super::context::ToolContext;
use super::types::{
    ExecutionKind, GitDiffOutput, ToolError, ToolInput, ToolOutput, ToolRunResult, ToolSpec,
};
use super::Tool;

const MAX_GIT_DIFF_STDOUT_BYTES: usize = 128 * 1024;
const MAX_GIT_DIFF_STDERR_BYTES: usize = 8 * 1024;

pub struct GitDiffTool {
    context: ToolContext,
}

impl GitDiffTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

impl Tool for GitDiffTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "git_diff",
            description: "Show read-only unstaged git working tree diff for the project.",
            input_hint: "",
            execution_kind: ExecutionKind::Immediate,
            default_risk: None,
        }
    }

    fn run(&self, input: &ToolInput) -> Result<ToolRunResult, ToolError> {
        let ToolInput::GitDiff = input else {
            return Err(ToolError::InvalidInput(
                "git_diff received wrong input variant".into(),
            ));
        };

        let output = run_bounded_git_diff(&self.context.root)?;

        if !output.status.success() {
            return Err(git_diff_error(&output.stderr.bytes));
        }

        Ok(ToolRunResult::Immediate(ToolOutput::GitDiff(
            git_diff_output(output.stdout),
        )))
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

fn run_bounded_git_diff(root: &std::path::Path) -> Result<BoundedGitOutput, ToolError> {
    let mut child = Command::new("git")
        .args(["diff", "--no-ext-diff", "--no-textconv", "--no-color", "--"])
        .current_dir(root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(git_command_error)?;

    let stdout = child.stdout.take().ok_or_else(output_capture_error)?;
    let stderr = child.stderr.take().ok_or_else(output_capture_error)?;

    let stdout_reader =
        thread::spawn(move || read_bounded_stream(stdout, MAX_GIT_DIFF_STDOUT_BYTES));
    let stderr_reader =
        thread::spawn(move || read_bounded_stream(stderr, MAX_GIT_DIFF_STDERR_BYTES));

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
    ToolError::InvalidInput("git_diff failed: output capture failed".into())
}

fn git_command_error(error: io::Error) -> ToolError {
    if error.kind() == io::ErrorKind::NotFound {
        ToolError::InvalidInput("git_diff failed: git executable unavailable".into())
    } else {
        ToolError::Io(error)
    }
}

fn git_diff_error(stderr: &[u8]) -> ToolError {
    let stderr = String::from_utf8_lossy(stderr);
    if stderr.to_ascii_lowercase().contains("not a git repository") {
        ToolError::InvalidInput("git_diff failed: not a Git repository".into())
    } else {
        ToolError::InvalidInput("git_diff failed".into())
    }
}

fn git_diff_output(capture: BoundedCapture) -> GitDiffOutput {
    let patch = String::from_utf8_lossy(&capture.bytes).to_string();
    GitDiffOutput {
        bytes_shown: capture.bytes.len(),
        patch,
        truncated: capture.truncated,
    }
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

    fn commit_file(path: &Path, file: &str, contents: &str) {
        fs::write(path.join(file), contents).unwrap();
        git(path, &["add", file]);
        git(
            path,
            &[
                "-c",
                "user.name=params",
                "-c",
                "user.email=params@example.invalid",
                "commit",
                "-m",
                "initial",
            ],
        );
    }

    fn run_diff(path: &Path) -> Result<ToolRunResult, ToolError> {
        GitDiffTool::new(ToolContext::new(PathBuf::from(path))).run(&ToolInput::GitDiff)
    }

    #[test]
    fn spec_is_immediate() {
        let tool = GitDiffTool::new(ToolContext::new(PathBuf::from(".")));
        let spec = tool.spec();
        assert_eq!(spec.name, "git_diff");
        assert_eq!(spec.execution_kind, ExecutionKind::Immediate);
        assert!(spec.default_risk.is_none());
    }

    #[test]
    fn default_registry_dispatches_git_diff() {
        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());
        let registry = crate::tools::default_registry(tmp.path().to_path_buf());

        let out = registry.dispatch(ToolInput::GitDiff).unwrap();
        assert!(matches!(
            out,
            ToolRunResult::Immediate(ToolOutput::GitDiff(_))
        ));
    }

    #[test]
    fn non_git_directory_returns_deterministic_error() {
        let tmp = TempDir::new().unwrap();
        let err = run_diff(tmp.path()).unwrap_err();
        assert!(matches!(
            err,
            ToolError::InvalidInput(message)
                if message == "git_diff failed: not a Git repository"
        ));
    }

    #[test]
    fn empty_repo_diff_returns_empty_patch() {
        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());

        let out = run_diff(tmp.path()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::GitDiff(diff)) = out else {
            panic!("expected Immediate(GitDiff)");
        };
        assert_eq!(diff.patch, "");
        assert_eq!(diff.bytes_shown, 0);
        assert!(!diff.truncated);
    }

    #[test]
    fn modified_file_returns_expected_diff_text() {
        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());
        commit_file(tmp.path(), "changed.txt", "old\n");
        fs::write(tmp.path().join("changed.txt"), "new\n").unwrap();

        let out = run_diff(tmp.path()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::GitDiff(diff)) = out else {
            panic!("expected Immediate(GitDiff)");
        };
        assert!(diff
            .patch
            .contains("diff --git a/changed.txt b/changed.txt"));
        assert!(diff.patch.contains("-old"));
        assert!(diff.patch.contains("+new"));
        assert_eq!(diff.bytes_shown, diff.patch.as_bytes().len());
        assert!(!diff.truncated);
    }

    #[test]
    fn configured_textconv_filter_is_not_invoked() {
        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());
        fs::write(
            tmp.path().join(".gitattributes"),
            "*.txt diff=params_textconv\n",
        )
        .unwrap();
        fs::write(tmp.path().join("converted.txt"), "old\n").unwrap();
        git(tmp.path(), &["add", ".gitattributes", "converted.txt"]);
        git(
            tmp.path(),
            &[
                "-c",
                "user.name=params",
                "-c",
                "user.email=params@example.invalid",
                "commit",
                "-m",
                "initial",
            ],
        );
        git(
            tmp.path(),
            &["config", "diff.params_textconv.textconv", "false"],
        );
        fs::write(tmp.path().join("converted.txt"), "new\n").unwrap();

        let out = run_diff(tmp.path()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::GitDiff(diff)) = out else {
            panic!("expected Immediate(GitDiff)");
        };
        assert!(diff
            .patch
            .contains("diff --git a/converted.txt b/converted.txt"));
        assert!(diff.patch.contains("-old"));
        assert!(diff.patch.contains("+new"));
        assert!(!diff.truncated);
    }

    #[test]
    fn large_diff_truncates_and_sets_truncated() {
        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());
        commit_file(tmp.path(), "large.txt", "old\n");
        let large = format!("{}\n", "new line\n".repeat(30_000));
        fs::write(tmp.path().join("large.txt"), large).unwrap();

        let out = run_diff(tmp.path()).unwrap();
        let ToolRunResult::Immediate(ToolOutput::GitDiff(diff)) = out else {
            panic!("expected Immediate(GitDiff)");
        };
        assert!(diff.bytes_shown <= MAX_GIT_DIFF_STDOUT_BYTES);
        assert_eq!(diff.bytes_shown, diff.patch.as_bytes().len());
        assert!(diff.truncated);
    }

    #[test]
    fn bounded_capture_marks_truncated_without_retaining_extra_bytes() {
        let input = vec![b'x'; MAX_GIT_DIFF_STDERR_BYTES + 10];
        let capture = read_bounded_stream(Cursor::new(input), MAX_GIT_DIFF_STDERR_BYTES).unwrap();

        assert_eq!(capture.bytes.len(), MAX_GIT_DIFF_STDERR_BYTES);
        assert!(capture.truncated);
    }
}
