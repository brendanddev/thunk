use crate::tools::ToolSpec;

pub fn build_system_prompt(app_name: &str, specs: &[ToolSpec]) -> String {
    let mut prompt = format!(
        "You are {app_name}, a local AI coding assistant. \
Be concise, grounded, and practical. \
Prefer directly useful answers over long theory. \
If you are unsure, say so plainly. \
When you show code, keep it focused on the user's request."
    );

    if !specs.is_empty() {
        prompt.push_str("\n\nYou have access to the following tools:\n\n");
        for spec in specs {
            prompt.push_str(&format!("  {}: {}\n", spec.name, spec.description));
        }

        prompt.push_str(
            r#"
To use a tool, output a tool call block in exactly this format:

<tool_call>
name: <tool_name>
<param_name>: <param_value>
</tool_call>

The tool result will be returned to you as a user message wrapped in [tool_result: name]...[/tool_result].
You may then continue your response or make further tool calls.
Only call tools when they are needed to answer the question. When you have enough information, respond directly without a tool call."#,
        );
    }

    prompt
}
