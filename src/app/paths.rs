use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use super::Result;

pub const CONFIG_FILE_NAME: &str = "config.toml";

/// Struct to hold all relevant paths for the application
#[derive(Debug, Clone)]
pub struct AppPaths {
    /// Config/storage root: where config.toml lives, or cwd if absent.
    /// Storage (data/, logs/, session db) anchors here.
    pub root_dir: PathBuf,
    /// Runtime project root: nearest .git ancestor, or cwd as fallback.
    /// This is what ProjectRoot and all runtime tools operate within.
    pub project_root: PathBuf,
    pub config_file: PathBuf,
    pub data_dir: PathBuf,
    pub logs_dir: PathBuf,
    pub session_db: PathBuf,
}

/// Discovers the necessary paths for the application based on the current working directory
impl AppPaths {
    pub fn discover() -> Result<Self> {
        let start_dir = env::current_dir()?.canonicalize()?;

        // Config/storage root: where config.toml lives, or cwd when absent.
        let root_dir = find_config_root(&start_dir).unwrap_or_else(|| start_dir.clone());

        // Runtime project root: nearest .git ancestor, or cwd as fallback.
        let project_root = find_git_root(&start_dir).unwrap_or_else(|| start_dir.clone());

        Ok(Self {
            config_file: root_dir.join(CONFIG_FILE_NAME),
            data_dir: root_dir.join("data"),
            logs_dir: root_dir.join("logs"),
            session_db: root_dir.join("data").join("sessions.db"),
            root_dir,
            project_root,
        })
    }

    pub fn ensure_runtime_dirs(&self) -> Result<()> {
        fs::create_dir_all(&self.data_dir)?;
        fs::create_dir_all(&self.logs_dir)?;
        Ok(())
    }
}

/// Walks upward to find a directory containing config.toml.
fn find_config_root(start_dir: &Path) -> Option<PathBuf> {
    for candidate in start_dir.ancestors() {
        if candidate.join(CONFIG_FILE_NAME).is_file() {
            return Some(candidate.to_path_buf());
        }
    }
    None
}

/// Walks upward to find a directory containing a .git entry (file or directory).
fn find_git_root(start_dir: &Path) -> Option<PathBuf> {
    for candidate in start_dir.ancestors() {
        if candidate.join(".git").exists() {
            return Some(candidate.to_path_buf());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    // Builds an AppPaths as-if launched from `launch_dir`, using the same
    // discovery logic as AppPaths::discover() but without touching cwd.
    fn discover_from(launch_dir: &Path) -> AppPaths {
        let start_dir = launch_dir.canonicalize().unwrap();
        let root_dir = find_config_root(&start_dir).unwrap_or_else(|| start_dir.clone());
        let project_root = find_git_root(&start_dir).unwrap_or_else(|| start_dir.clone());
        AppPaths {
            config_file: root_dir.join(CONFIG_FILE_NAME),
            data_dir: root_dir.join("data"),
            logs_dir: root_dir.join("logs"),
            session_db: root_dir.join("data").join("sessions.db"),
            root_dir,
            project_root,
        }
    }

    #[test]
    fn launch_from_repo_with_config_toml() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("config.toml"), "").unwrap();
        fs::create_dir(dir.path().join(".git")).unwrap();

        let paths = discover_from(dir.path());

        // Config root and storage anchor at the dir containing config.toml.
        assert_eq!(paths.root_dir, dir.path().canonicalize().unwrap());
        // Runtime project root is the .git ancestor (same dir here).
        assert_eq!(paths.project_root, dir.path().canonicalize().unwrap());
        assert!(paths.config_file.ends_with("config.toml"));
    }

    #[test]
    fn launch_from_repo_without_config_toml() {
        let dir = tempdir().unwrap();
        fs::create_dir(dir.path().join(".git")).unwrap();

        let paths = discover_from(dir.path());

        // No config.toml: storage root falls back to cwd (launch dir).
        assert_eq!(paths.root_dir, dir.path().canonicalize().unwrap());
        // Runtime project root is the .git ancestor.
        assert_eq!(paths.project_root, dir.path().canonicalize().unwrap());
    }

    #[test]
    fn launch_from_nested_directory_inside_repo() {
        let dir = tempdir().unwrap();
        let git_root = dir.path();
        let sub = git_root.join("src").join("nested");
        fs::create_dir_all(&sub).unwrap();
        fs::create_dir(git_root.join(".git")).unwrap();

        let paths = discover_from(&sub);

        // No config: storage root is the nested launch dir.
        assert_eq!(paths.root_dir, sub.canonicalize().unwrap());
        // Runtime project root walks up to the .git ancestor.
        assert_eq!(paths.project_root, git_root.canonicalize().unwrap());
    }

    #[test]
    fn launch_from_plain_directory_no_git() {
        let dir = tempdir().unwrap();

        let paths = discover_from(dir.path());

        // No config, no .git: both roots fall back to cwd.
        assert_eq!(paths.root_dir, dir.path().canonicalize().unwrap());
        assert_eq!(paths.project_root, dir.path().canonicalize().unwrap());
    }

    #[test]
    fn config_root_and_project_root_can_differ() {
        // Config lives at the git root; we launch from a subdirectory.
        // project_root should reach the .git ancestor;
        // root_dir (config root) should also reach that ancestor via config.toml.
        let dir = tempdir().unwrap();
        let git_root = dir.path();
        fs::write(git_root.join("config.toml"), "").unwrap();
        fs::create_dir(git_root.join(".git")).unwrap();
        let sub = git_root.join("inner");
        fs::create_dir_all(&sub).unwrap();

        let paths = discover_from(&sub);

        let canonical_root = git_root.canonicalize().unwrap();
        // Config discovery walks up from sub and finds config.toml at git_root.
        assert_eq!(paths.root_dir, canonical_root);
        // Git root discovery also walks up to git_root.
        assert_eq!(paths.project_root, canonical_root);
    }

    #[test]
    fn project_root_does_not_escape_to_config_ancestor_above_git() {
        // Config exists two levels up from the .git root — project_root must
        // not escape past the .git boundary just because config is higher.
        // (find_git_root is independent of find_config_root.)
        let dir = tempdir().unwrap();
        let top = dir.path();
        let git_root = top.join("repo");
        fs::create_dir_all(&git_root).unwrap();
        fs::create_dir(git_root.join(".git")).unwrap();
        // Config lives above the git root — unusual but valid to test independence.
        fs::write(top.join("config.toml"), "").unwrap();
        let launch = git_root.join("src");
        fs::create_dir_all(&launch).unwrap();

        let paths = discover_from(&launch);

        // project_root should be the .git ancestor (git_root), not top.
        assert_eq!(paths.project_root, git_root.canonicalize().unwrap());
        // root_dir walks up to find config.toml at top.
        assert_eq!(paths.root_dir, top.canonicalize().unwrap());
    }
}
