pub mod config;
pub mod error;
pub mod paths;

pub use error::{AppError, Result};

pub fn run() -> Result<()> {
    let paths = paths::AppPaths::discover()?;
    paths.ensure_runtime_dirs()?;

    let config = config::load(&paths.config_file)?.resolve_paths(&paths.root_dir);
    let backend = crate::llm::providers::build_backend(&config)?;
    let mut runtime = crate::runtime::Runtime::new(&config, backend);
    crate::tui::run(&config, &paths, &mut runtime)
}
