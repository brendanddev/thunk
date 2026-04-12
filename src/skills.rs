use std::fs;
use std::path::Path;

const REPO_NAVIGATION_SKILL: &str = "repo-navigation";
const BENCHMARK_EVALUATOR_SKILL: &str = "benchmark-evaluator";

const REPO_NAVIGATION_SECTIONS: &[&str] = &[
    "Purpose",
    "Priorities",
    "Relevant Files To Inspect First",
    "Intent Rules",
    "Workflow",
    "Output Style",
    "Avoid",
];

const BENCHMARK_EVALUATOR_SECTIONS: &[&str] = &[
    "Purpose",
    "Source Of Truth",
    "Core Benchmark Set",
    "What To Record",
    "Scoring",
    "Workflow",
    "Evaluation Rules",
    "Avoid",
];

pub(crate) fn append_repo_navigation_skill_guidance(project_root: &Path, prompt: &mut String) {
    if let Some(body) = load_skill_excerpt(
        project_root,
        REPO_NAVIGATION_SKILL,
        REPO_NAVIGATION_SECTIONS,
    ) {
        append_skill_block(prompt, REPO_NAVIGATION_SKILL, &body);
    }
}

pub(crate) fn append_chat_skill_guidance(
    project_root: &Path,
    user_prompt: &str,
    prompt: &mut String,
) {
    if !looks_like_benchmark_prompt(user_prompt) {
        return;
    }

    if let Some(body) = load_skill_excerpt(
        project_root,
        BENCHMARK_EVALUATOR_SKILL,
        BENCHMARK_EVALUATOR_SECTIONS,
    ) {
        append_skill_block(prompt, BENCHMARK_EVALUATOR_SKILL, &body);
    }
}

fn append_skill_block(prompt: &mut String, skill_name: &str, body: &str) {
    let body = body.trim();
    if body.is_empty() {
        return;
    }

    prompt.push_str("\n\nBuilt-in skill active: `");
    prompt.push_str(skill_name);
    prompt.push_str("`\n");
    prompt.push_str(body);
}

fn load_skill_excerpt(project_root: &Path, skill_name: &str, sections: &[&str]) -> Option<String> {
    let skill_path = project_root
        .join("skills")
        .join(skill_name)
        .join("SKILL.md");
    let body = fs::read_to_string(skill_path).ok()?;
    let excerpt = extract_sections(&body, sections);
    if excerpt.trim().is_empty() {
        Some(body.trim().to_string())
    } else {
        Some(excerpt)
    }
}

fn extract_sections(markdown: &str, sections: &[&str]) -> String {
    let mut out = Vec::new();
    let mut current_heading: Option<String> = None;
    let mut current_lines = Vec::new();

    let flush_section =
        |heading: &Option<String>, lines: &mut Vec<String>, out: &mut Vec<String>| {
            let Some(heading) = heading.as_ref() else {
                lines.clear();
                return;
            };
            if sections.iter().any(|wanted| heading == wanted) && !lines.is_empty() {
                out.push(format!("## {heading}"));
                out.extend(lines.drain(..));
            } else {
                lines.clear();
            }
        };

    for line in markdown.lines() {
        if let Some(heading) = line.strip_prefix("## ") {
            flush_section(&current_heading, &mut current_lines, &mut out);
            current_heading = Some(heading.trim().to_string());
            continue;
        }

        if line.starts_with("# ") {
            continue;
        }

        if current_heading.is_some() {
            current_lines.push(line.to_string());
        }
    }

    flush_section(&current_heading, &mut current_lines, &mut out);

    out.join("\n").trim().to_string()
}

fn looks_like_benchmark_prompt(prompt: &str) -> bool {
    let normalized = normalize_prompt(prompt);
    let tokens = normalized.split_whitespace().collect::<Vec<_>>();
    let has_any = |needles: &[&str]| needles.iter().any(|needle| tokens.contains(needle));
    let contains_any = |needles: &[&str]| needles.iter().any(|needle| normalized.contains(needle));

    contains_any(&["benchmark", "benchmarks", "baseline", "regression"])
        || (has_any(&["evaluate", "grade", "score", "assess", "compare"])
            && contains_any(&["answer", "answers", "quality", "latency", "streaming"]))
        || (has_any(&["fix", "improved", "better", "regressed"])
            && contains_any(&["benchmark", "baseline"]))
}

fn normalize_prompt(prompt: &str) -> String {
    prompt
        .to_ascii_lowercase()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '/' {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::{
        append_chat_skill_guidance, append_repo_navigation_skill_guidance, extract_sections,
    };
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempProject {
        root: PathBuf,
    }

    impl TempProject {
        fn new() -> Self {
            let mut root = std::env::temp_dir();
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos();
            root.push(format!("params-skills-test-{unique}"));
            fs::create_dir_all(&root).expect("create temp root");
            Self { root }
        }

        fn write_skill(&self, name: &str, body: &str) {
            let path = self.root.join("skills").join(name);
            fs::create_dir_all(&path).expect("create skill dir");
            fs::write(path.join("SKILL.md"), body).expect("write skill");
        }

        fn path(&self) -> &Path {
            &self.root
        }
    }

    impl Drop for TempProject {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn extract_sections_keeps_only_requested_blocks() {
        let markdown = "# Skill\n\n## Purpose\none\n\n## Workflow\ntwo\n\n## Avoid\nthree\n";
        let excerpt = extract_sections(markdown, &["Purpose", "Avoid"]);

        assert!(excerpt.contains("## Purpose"));
        assert!(excerpt.contains("one"));
        assert!(excerpt.contains("## Avoid"));
        assert!(excerpt.contains("three"));
        assert!(!excerpt.contains("## Workflow"));
    }

    #[test]
    fn repo_navigation_skill_is_appended_when_present() {
        let project = TempProject::new();
        project.write_skill(
            "repo-navigation",
            "# Repo Navigation\n\n## Purpose\nUse source files.\n\n## Workflow\nRead then answer.\n",
        );

        let mut prompt = "Base prompt".to_string();
        append_repo_navigation_skill_guidance(project.path(), &mut prompt);

        assert!(prompt.contains("Built-in skill active: `repo-navigation`"));
        assert!(prompt.contains("## Purpose"));
        assert!(prompt.contains("Use source files."));
    }

    #[test]
    fn benchmark_skill_is_only_appended_for_benchmark_like_prompts() {
        let project = TempProject::new();
        project.write_skill(
            "benchmark-evaluator",
            "# Benchmark Evaluator\n\n## Purpose\nGrade benchmark runs.\n\n## Workflow\nRecord scores.\n",
        );

        let mut benchmark_prompt = "Base prompt".to_string();
        append_chat_skill_guidance(
            project.path(),
            "Can you evaluate these benchmark results?",
            &mut benchmark_prompt,
        );
        assert!(benchmark_prompt.contains("benchmark-evaluator"));
        assert!(benchmark_prompt.contains("Grade benchmark runs."));

        let mut plain_prompt = "Base prompt".to_string();
        append_chat_skill_guidance(
            project.path(),
            "How does session restore work?",
            &mut plain_prompt,
        );
        assert_eq!(plain_prompt, "Base prompt");
    }
}
