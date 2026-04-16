pub mod config;
pub mod error;
pub mod paths;

pub use error::{AppError, Result};

pub fn run() -> Result<()> {
    let paths = paths::AppPaths::discover()?;
    paths.ensure_runtime_dirs()?;

    let config = config::load(&paths.config_file)?;
    let mut runtime = crate::runtime::Runtime::new(&config, &paths);
    crate::tui::run(&config, &paths, &mut runtime)
}
