use thunk::app;

fn main() -> app::Result<()> {
    let cli = app::cli::Cli::parse();
    app::run(cli)
}
