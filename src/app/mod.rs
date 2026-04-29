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
    let project_root = crate::runtime::ProjectRoot::new(paths.root_dir.clone())
        .map_err(|e| AppError::Config(e.to_string()))?;
    let registry = default_registry().with_project_root(project_root.as_path_buf());
    let log = crate::logging::SessionLog::open(&paths.logs_dir);

    let (active_session, history) =
        session::ActiveSession::open_or_restore(&paths.session_db, &project_root)?;
    let app = AppContext::build(
        &config,
        project_root,
        backend,
        registry,
        active_session,
        history,
        log,
    )?;

    tui::run(&config, &paths, app)
}
