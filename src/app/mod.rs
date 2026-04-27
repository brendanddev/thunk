pub mod cli;
pub mod config;
pub mod context;
pub mod error;
pub mod paths;
pub mod session;

pub use context::AppContext;
pub use error::{AppError, Result};

use crate::llm::providers::build_backend;
use crate::tools::default_registry;
use crate::tui;

// Bootstraps the application: prepares paths and config, builds the backend and tools, restores session state,
// attaches logging, and starts the TUI.
pub fn run(cli: cli::Cli) -> Result<()> {
    let paths = paths::AppPaths::discover()?;
    paths.ensure_runtime_dirs()?;

    let mut config = config::load(&paths.config_file)?.resolve_paths(&paths.root_dir);
    if let Some(model) = cli.model {
        config.llm.provider = model;
    }
    let backend = build_backend(&config)?;
    let registry = default_registry(paths.root_dir.clone());
    let log = crate::logging::SessionLog::open(&paths.logs_dir);

    let (active_session, history) = session::ActiveSession::open_or_restore(&paths.session_db)?;
    let app = AppContext::build(
        &config,
        &paths.root_dir,
        backend,
        registry,
        active_session,
        history,
        log,
    )?;

    tui::run(&config, &paths, app)
}
