fmt:
    cargo fmt --all

check:
    cargo check --all-targets

test:
    cargo test

clippy:
    cargo clippy --all-targets

verify:
    cargo fmt --all --check
    cargo check --all-targets
    cargo clippy --all-targets
    cargo test

run:
    cargo run --release

trace:
    THUNK_TRACE_RUNTIME=1 cargo run --release

fresh:
    rm -f data/sessions.db

trace-fresh:
    just fresh
    just trace