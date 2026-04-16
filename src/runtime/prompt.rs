use std::path::Path;

use crate::tools::ToolSpec;

use super::tool_codec;

pub fn build_system_prompt(app_name: &str, project_root: &Path, specs: &[ToolSpec]) -> String {
    let mut prompt = format!(
        "You are {app_name}, a local AI coding assistant. \
You are working in the project at {}. \
Be concise, grounded, and practical. \
Prefer directly useful answers over long theory. \
If you are unsure, say so plainly. \
When you show code, keep it focused on the user's request.",
        project_root.display()
    );

    if !specs.is_empty() {
        prompt.push_str("\n\nYou have access to the following tools:\n\n");
        for spec in specs {
            prompt.push_str(&format!("  {}: {}\n", spec.name, spec.description));
        }
        prompt.push('\n');
        prompt.push_str(tool_codec::format_instructions());
    }

    prompt
}
