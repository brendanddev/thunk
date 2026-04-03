// src/cache/mod.rs
//
// Exact response cache for repeated generations.

mod exact;

pub use exact::{build_cache_scope, ExactCache};
