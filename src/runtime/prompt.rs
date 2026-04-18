use std::path::Path;

use crate::tools::ToolSpec;

use super::tool_codec;

pub fn build_system_prompt(app_name: &str, project_root: &Path, specs: &[ToolSpec]) -> String {
    let mut prompt = format!(
        "You are {app_name}, a local AI coding assistant.\n\
Project: {}\n\n\
Be concise, grounded, and practical. \
When the user asks about this project's code, investigate using the tools before responding — \
do not guess or ask the user for information the tools can find. \
When you show code, keep it focused on the user's request.",
        project_root.display()
    );

    if !specs.is_empty() {
        let instructions = tool_codec::format_instructions();

        // Guard: every registered tool must appear in the protocol instructions.
        // A missing entry means the model is told a tool exists but not how to call it.
        for spec in specs {
            debug_assert!(
                instructions.contains(spec.name),
                "tool '{}' is registered but its call syntax is missing from format_instructions()",
                spec.name
            );
        }

        prompt.push_str("\n\nYou have access to the following tools:\n\n");
        for spec in specs {
            prompt.push_str(&format!("  {}: {}\n", spec.name, spec.description));
        }
        prompt.push('\n');
        prompt.push_str(instructions);
    }

    prompt
}
