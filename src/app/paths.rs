use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use super::{AppError, Result};

pub const CONFIG_FILE_NAME: &str = "config.toml";

/// Struct to hold all relevant paths for the application
#[derive(Debug, Clone)]
pub struct AppPaths {
    pub root_dir: PathBuf,
    pub config_file: PathBuf,
    pub data_dir: PathBuf,
    pub logs_dir: PathBuf,
}

/// Discovers the necessary paths for the application based on the current working directory
impl AppPaths {
    pub fn discover() -> Result<Self> {
        let start_dir = env::current_dir()?.canonicalize()?;
        let root_dir = find_project_root(&start_dir).ok_or_else(|| {
            AppError::Config(format!(
                "Could not find {CONFIG_FILE_NAME} starting from {}",
                start_dir.display()
            ))
        })?;

        Ok(Self {
            config_file: root_dir.join(CONFIG_FILE_NAME),
            data_dir: root_dir.join("data"),
            logs_dir: root_dir.join("logs"),
            root_dir,
        })
    }

    pub fn ensure_runtime_dirs(&self) -> Result<()> {
        fs::create_dir_all(&self.data_dir)?;
        fs::create_dir_all(&self.logs_dir)?;
        Ok(())
    }
}

fn find_project_root(start_dir: &Path) -> Option<PathBuf> {
    for candidate in start_dir.ancestors() {
        let config_file = candidate.join(CONFIG_FILE_NAME);
        if config_file.is_file() {
            return Some(candidate.to_path_buf());
        }
    }

    None
}
