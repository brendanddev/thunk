pub mod config;
pub mod context;
pub mod error;
pub mod paths;
pub mod session;

pub use context::AppContext;
pub use error::{AppError, Result};

pub fn run() -> Result<()> {
    let paths = paths::AppPaths::discover()?;
    paths.ensure_runtime_dirs()?;

    let config = config::load(&paths.config_file)?.resolve_paths(&paths.root_dir);
    let backend = crate::llm::providers::build_backend(&config)?;
    let registry = crate::tools::default_registry(paths.root_dir.clone());

    let (active_session, history) = session::ActiveSession::open_or_restore(&paths.session_db)?;
    let app = AppContext::build(&config, &paths.root_dir, backend, registry, active_session, history)?;

    crate::tui::run(&config, &paths, app)
}
