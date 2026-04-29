mod project_path;
mod project_root;
mod project_snapshot;
mod resolved_input;
mod resolver;

pub(crate) use project_path::relative_display;
pub use project_path::{ProjectPath, ProjectScope};
pub use project_root::{ProjectRoot, ProjectRootError};
pub(crate) use project_snapshot::{
    ProjectStructureEntry, ProjectStructureEntryKind, ProjectStructureSnapshot,
    ProjectStructureSnapshotCache, MAX_SNAPSHOT_DEPTH, MAX_SNAPSHOT_NODES,
};
pub use resolved_input::ResolvedToolInput;
#[allow(unused_imports)]
pub use resolver::{resolve, PathResolutionError};
