use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use serde::Deserialize;
use tracing::{info, warn};

use crate::config;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandOrigin {
    Local,
}

impl CommandOrigin {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
        }
    }
}

impl fmt::Display for CommandOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinKind {
    Context,
    Mutating,
    Session,
    Discovery,
}

#[derive(Debug, Clone, Copy)]
pub struct BuiltinCommandSpec {
    pub canonical: &'static str,
    pub aliases: &'static [&'static str],
    pub usage: &'static str,
    pub description: &'static str,
    pub kind: BuiltinKind,
}

pub const BUILTIN_COMMANDS: &[BuiltinCommandSpec] = &[
    BuiltinCommandSpec {
        canonical: "/read",
        aliases: &["/r"],
        usage: "/read <path>",
        description: "load a file into context",
        kind: BuiltinKind::Context,
    },
    BuiltinCommandSpec {
        canonical: "/ls",
        aliases: &["/list"],
        usage: "/ls [path]",
        description: "list a directory inside the project",
        kind: BuiltinKind::Context,
    },
    BuiltinCommandSpec {
        canonical: "/search",
        aliases: &["/s", "/grep"],
        usage: "/search <query>",
        description: "search source files inside the project",
        kind: BuiltinKind::Context,
    },
    BuiltinCommandSpec {
        canonical: "/git",
        aliases: &[],
        usage: "/git [status|diff|log]",
        description: "load read-only git context",
        kind: BuiltinKind::Context,
    },
    BuiltinCommandSpec {
        canonical: "/diag",
        aliases: &["/lsp"],
        usage: "/diag <file>",
        description: "Rust LSP diagnostics for a file",
        kind: BuiltinKind::Context,
    },
    BuiltinCommandSpec {
        canonical: "/hover",
        aliases: &[],
        usage: "/hover <file>:<line>:<col>",
        description: "Rust LSP hover at a file position",
        kind: BuiltinKind::Context,
    },
    BuiltinCommandSpec {
        canonical: "/def",
        aliases: &["/definition"],
        usage: "/def <file>:<line>:<col>",
        description: "Rust LSP definition at a file position",
        kind: BuiltinKind::Context,
    },
    BuiltinCommandSpec {
        canonical: "/lcheck",
        aliases: &["/lsp-check"],
        usage: "/lcheck",
        description: "check local rust-analyzer setup",
        kind: BuiltinKind::Context,
    },
    BuiltinCommandSpec {
        canonical: "/fetch",
        aliases: &["/web"],
        usage: "/fetch <url>",
        description: "fetch a public webpage into context",
        kind: BuiltinKind::Context,
    },
    BuiltinCommandSpec {
        canonical: "/run",
        aliases: &["/bash"],
        usage: "/run <command>",
        description: "propose a shell command for approval",
        kind: BuiltinKind::Mutating,
    },
    BuiltinCommandSpec {
        canonical: "/write",
        aliases: &[],
        usage: "/write <path> <content>",
        description: "propose a whole-file write for approval",
        kind: BuiltinKind::Mutating,
    },
    BuiltinCommandSpec {
        canonical: "/edit",
        aliases: &[],
        usage: "/edit <path>\\n```params-edit ... ```",
        description: "propose a targeted file edit for approval",
        kind: BuiltinKind::Mutating,
    },
    BuiltinCommandSpec {
        canonical: "/reflect",
        aliases: &[],
        usage: "/reflect <on|off|status>",
        description: "toggle reflection mode",
        kind: BuiltinKind::Session,
    },
    BuiltinCommandSpec {
        canonical: "/eco",
        aliases: &[],
        usage: "/eco <on|off|status>",
        description: "toggle eco mode",
        kind: BuiltinKind::Session,
    },
    BuiltinCommandSpec {
        canonical: "/debug-log",
        aliases: &[],
        usage: "/debug-log <on|off|status>",
        description: "toggle separate content debug logging",
        kind: BuiltinKind::Session,
    },
    BuiltinCommandSpec {
        canonical: "/approve",
        aliases: &[],
        usage: "/approve",
        description: "approve the pending action",
        kind: BuiltinKind::Session,
    },
    BuiltinCommandSpec {
        canonical: "/reject",
        aliases: &[],
        usage: "/reject",
        description: "reject the pending action",
        kind: BuiltinKind::Session,
    },
    BuiltinCommandSpec {
        canonical: "/clear",
        aliases: &["/c"],
        usage: "/clear",
        description: "clear the current conversation and active saved session",
        kind: BuiltinKind::Session,
    },
    BuiltinCommandSpec {
        canonical: "/clear-cache",
        aliases: &["/cache-clear"],
        usage: "/clear-cache",
        description: "clear the exact response cache",
        kind: BuiltinKind::Session,
    },
    BuiltinCommandSpec {
        canonical: "/clear-debug-log",
        aliases: &[],
        usage: "/clear-debug-log",
        description: "clear the separate content debug log",
        kind: BuiltinKind::Session,
    },
    BuiltinCommandSpec {
        canonical: "/help",
        aliases: &["/h", "/?"],
        usage: "/help",
        description: "show built-in command help",
        kind: BuiltinKind::Discovery,
    },
    BuiltinCommandSpec {
        canonical: "/commands",
        aliases: &[],
        usage: "/commands [list|reload]",
        description: "list or reload custom slash commands",
        kind: BuiltinKind::Discovery,
    },
    BuiltinCommandSpec {
        canonical: "/sessions",
        aliases: &[],
        usage: "/sessions <list|new|rename|resume|export>",
        description: "manage saved sessions for this project",
        kind: BuiltinKind::Session,
    },
    BuiltinCommandSpec {
        canonical: "/memory",
        aliases: &[],
        usage: "/memory [status|facts|last]",
        description: "inspect loaded memory and the latest memory update",
        kind: BuiltinKind::Session,
    },
    BuiltinCommandSpec {
        canonical: "/transcript",
        aliases: &[],
        usage: "/transcript [status|collapse|expand|toggle]",
        description: "control collapsible transcript context blocks",
        kind: BuiltinKind::Session,
    },
];

#[derive(Debug, Clone)]
pub struct CustomCommand {
    pub name: String,
    pub description: String,
    pub usage: Option<String>,
    pub origin: CommandOrigin,
    pub body: CustomCommandBody,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandSuggestion {
    pub name: String,
    pub usage: String,
    pub description: String,
    pub source: &'static str,
    pub group: &'static str,
}

#[derive(Debug, Clone)]
pub enum CustomCommandBody {
    Prompt(String),
    Workflow(Vec<CustomCommandStep>),
}

#[derive(Debug, Clone)]
pub enum CustomCommandStep {
    Slash(String),
    Prompt(String),
}

pub struct CommandRegistry {
    commands: BTreeMap<String, CustomCommand>,
}

impl CommandRegistry {
    pub fn load() -> Self {
        Self::load_report().registry
    }

    pub fn load_report() -> CommandLoadReport {
        let local_path = config::local_dir()
            .ok()
            .map(|dir| dir.join("commands.toml"));
        let report = Self::load_from_paths(local_path.as_deref());
        info!(
            loaded = report.loaded,
            invalid = report.invalid,
            sources = report.sources_loaded,
            "custom command registry loaded"
        );
        report
    }

    pub fn load_from_paths(local_path: Option<&Path>) -> CommandLoadReport {
        let mut commands = BTreeMap::new();
        let mut loaded = 0usize;
        let mut invalid = 0usize;
        let mut sources_loaded = 0usize;

        if let Some(path) = local_path {
            let layer = load_command_layer(path, CommandOrigin::Local);
            if layer.seen_file {
                sources_loaded += 1;
            }
            loaded += layer.loaded;
            invalid += layer.invalid;
            for command in layer.commands {
                commands.insert(command.name.clone(), command);
            }
        }

        CommandLoadReport {
            registry: Self { commands },
            loaded,
            invalid,
            sources_loaded,
        }
    }

    pub fn resolve(&self, name: &str) -> Option<&CustomCommand> {
        self.commands.get(&normalize_command_name(name).ok()?)
    }

    pub fn list(&self) -> Vec<&CustomCommand> {
        self.commands.values().collect()
    }

    pub fn autocomplete_names(&self) -> Vec<String> {
        let mut names = builtin_autocomplete_names();
        names.extend(self.commands.keys().cloned());
        names.sort();
        names.dedup();
        names
    }

    pub fn suggestions(&self) -> Vec<CommandSuggestion> {
        let mut suggestions = builtin_command_specs()
            .iter()
            .map(|spec| CommandSuggestion {
                name: spec.canonical.to_string(),
                usage: spec.usage.to_string(),
                description: spec.description.to_string(),
                source: "builtin",
                group: builtin_group_label(spec.kind),
            })
            .collect::<Vec<_>>();
        suggestions.extend(self.commands.values().map(|command| {
            CommandSuggestion {
                name: command.name.clone(),
                usage: command
                    .usage
                    .clone()
                    .unwrap_or_else(|| command.name.clone()),
                description: command.description.clone(),
                source: command.origin.as_str(),
                group: "custom",
            }
        }));
        suggestions.sort_by(|a, b| a.name.cmp(&b.name));
        suggestions
    }
}

fn builtin_group_label(kind: BuiltinKind) -> &'static str {
    match kind {
        BuiltinKind::Context => "context",
        BuiltinKind::Mutating => "action",
        BuiltinKind::Session => "session",
        BuiltinKind::Discovery => "help",
    }
}

pub struct CommandLoadReport {
    pub registry: CommandRegistry,
    pub loaded: usize,
    pub invalid: usize,
    pub sources_loaded: usize,
}

#[derive(Debug, Deserialize)]
struct CommandFile {
    #[serde(default)]
    commands: BTreeMap<String, RawCommandDefinition>,
}

#[derive(Debug, Deserialize)]
struct RawCommandDefinition {
    description: Option<String>,
    usage: Option<String>,
    prompt: Option<String>,
    steps: Option<Vec<RawCommandStep>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawCommandStep {
    Slash { slash: String },
    Prompt { prompt: String },
}

struct CommandLayer {
    commands: Vec<CustomCommand>,
    loaded: usize,
    invalid: usize,
    seen_file: bool,
}

fn load_command_layer(path: &Path, origin: CommandOrigin) -> CommandLayer {
    if !path.exists() {
        return CommandLayer {
            commands: Vec::new(),
            loaded: 0,
            invalid: 0,
            seen_file: false,
        };
    }

    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) => {
            warn!(path = %path.display(), origin = origin.as_str(), error = %error, "custom command file unreadable");
            return CommandLayer {
                commands: Vec::new(),
                loaded: 0,
                invalid: 1,
                seen_file: true,
            };
        }
    };

    let parsed: CommandFile = match toml::from_str(&contents) {
        Ok(parsed) => parsed,
        Err(error) => {
            warn!(path = %path.display(), origin = origin.as_str(), error = %error, "custom command file invalid");
            return CommandLayer {
                commands: Vec::new(),
                loaded: 0,
                invalid: 1,
                seen_file: true,
            };
        }
    };

    let mut commands = Vec::new();
    let mut loaded = 0usize;
    let mut invalid = 0usize;

    for (raw_name, raw) in parsed.commands {
        match validate_command(raw_name, raw, origin) {
            Ok(command) => {
                loaded += 1;
                commands.push(command);
            }
            Err(error) => {
                invalid += 1;
                warn!(
                    path = %path.display(),
                    origin = origin.as_str(),
                    error = error.as_str(),
                    "custom command skipped"
                );
            }
        }
    }

    CommandLayer {
        commands,
        loaded,
        invalid,
        seen_file: true,
    }
}

fn validate_command(
    raw_name: String,
    raw: RawCommandDefinition,
    origin: CommandOrigin,
) -> std::result::Result<CustomCommand, String> {
    let name = normalize_command_name(&raw_name)?;
    if is_reserved_builtin_name(&name) {
        return Err(format!("{name} is reserved by a built-in slash command"));
    }

    let description = raw
        .description
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{name} is missing a non-empty description"))?;

    match (raw.prompt, raw.steps) {
        (Some(prompt), None) => Ok(CustomCommand {
            name,
            description,
            usage: raw.usage,
            origin,
            body: CustomCommandBody::Prompt(prompt),
        }),
        (None, Some(steps)) => {
            if steps.is_empty() {
                return Err(format!("{name} must define at least one workflow step"));
            }

            let mut validated = Vec::new();
            let mut mutating_seen = false;
            let mut prompt_seen = false;
            let total_steps = steps.len();

            for (idx, step) in steps.into_iter().enumerate() {
                let is_last = idx == total_steps - 1;
                match step {
                    RawCommandStep::Slash { slash } => {
                        let slash = slash.trim().to_string();
                        if slash.is_empty() {
                            return Err(format!("{name} contains an empty slash step"));
                        }
                        let step_cmd = slash.split_whitespace().next().unwrap_or("");
                        let Some(spec) = resolve_builtin_command(step_cmd) else {
                            return Err(format!(
                                "{name} step `{step_cmd}` is not a supported built-in slash command"
                            ));
                        };
                        match spec.kind {
                            BuiltinKind::Context => {}
                            BuiltinKind::Mutating => {
                                if mutating_seen {
                                    return Err(format!(
                                        "{name} can contain at most one mutating built-in step"
                                    ));
                                }
                                if !is_last {
                                    return Err(format!(
                                        "{name} mutating step `{}` must be final",
                                        spec.canonical
                                    ));
                                }
                                mutating_seen = true;
                            }
                            BuiltinKind::Session | BuiltinKind::Discovery => {
                                return Err(format!(
                                    "{name} step `{}` is not supported inside custom workflows",
                                    spec.canonical
                                ));
                            }
                        }
                        validated.push(CustomCommandStep::Slash(slash));
                    }
                    RawCommandStep::Prompt { prompt } => {
                        if prompt_seen {
                            return Err(format!("{name} can contain at most one prompt step"));
                        }
                        if !is_last {
                            return Err(format!("{name} prompt step must be final"));
                        }
                        prompt_seen = true;
                        validated.push(CustomCommandStep::Prompt(prompt));
                    }
                }
            }

            Ok(CustomCommand {
                name,
                description,
                usage: raw.usage,
                origin,
                body: CustomCommandBody::Workflow(validated),
            })
        }
        (Some(_), Some(_)) => Err(format!(
            "{name} must define either `prompt` or `steps`, not both"
        )),
        (None, None) => Err(format!("{name} must define either `prompt` or `steps`")),
    }
}

fn normalize_command_name(name: &str) -> std::result::Result<String, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("custom command name cannot be empty".to_string());
    }

    let normalized = trimmed.strip_prefix('/').unwrap_or(trimmed);
    if normalized.is_empty() {
        return Err("custom command name cannot be just `/`".to_string());
    }

    if !normalized
        .chars()
        .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-' || ch == '_')
    {
        return Err(format!(
            "invalid custom command name `{trimmed}`; use lowercase letters, digits, `-`, or `_`"
        ));
    }

    Ok(format!("/{normalized}"))
}

pub fn builtin_autocomplete_names() -> Vec<String> {
    let mut set = BTreeSet::new();
    for spec in BUILTIN_COMMANDS {
        set.insert(spec.canonical.to_string());
        for alias in spec.aliases {
            set.insert((*alias).to_string());
        }
    }
    set.into_iter().collect()
}

pub fn builtin_command_specs() -> &'static [BuiltinCommandSpec] {
    BUILTIN_COMMANDS
}

pub fn resolve_builtin_command(name: &str) -> Option<&'static BuiltinCommandSpec> {
    BUILTIN_COMMANDS.iter().find(|spec| {
        spec.canonical.eq_ignore_ascii_case(name)
            || spec
                .aliases
                .iter()
                .any(|alias| alias.eq_ignore_ascii_case(name))
    })
}

fn is_reserved_builtin_name(name: &str) -> bool {
    resolve_builtin_command(name).is_some()
}

pub fn expand_positional_args(template: &str, args: &[&str]) -> String {
    let joined = args.join(" ");
    let chars: Vec<char> = template.chars().collect();
    let mut out = String::with_capacity(template.len());
    let mut i = 0usize;

    while i < chars.len() {
        if chars[i] != '$' {
            out.push(chars[i]);
            i += 1;
            continue;
        }

        if i + 1 >= chars.len() {
            out.push('$');
            i += 1;
            continue;
        }

        let next = chars[i + 1];
        if next == '@' {
            out.push_str(&joined);
            i += 2;
            continue;
        }

        if next.is_ascii_digit() {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            let index_text: String = chars[i + 1..j].iter().collect();
            if let Ok(index) = index_text.parse::<usize>() {
                if index > 0 {
                    out.push_str(args.get(index - 1).copied().unwrap_or(""));
                    i = j;
                    continue;
                }
            }
        }

        out.push('$');
        i += 1;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        let unique = format!(
            "params-cli-commands-{}-{}",
            name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        std::env::temp_dir().join(unique)
    }

    #[test]
    fn positional_expansion_handles_numbers_and_joined_tail() {
        let expanded = expand_positional_args(
            "Review $1 and $2 with tail: $@",
            &["src/main.rs", "carefully", "please"],
        );
        assert_eq!(
            expanded,
            "Review src/main.rs and carefully with tail: src/main.rs carefully please"
        );
    }

    #[test]
    fn load_reads_local_commands_file() {
        let root = temp_path("local");
        std::fs::create_dir_all(root.join(".local")).unwrap();
        std::fs::write(
            root.join(".local/commands.toml"),
            r#"
[commands.review]
description = "Local review"
prompt = "Local prompt: $@"

[commands.auth_context]
description = "Load auth files"
steps = [{ slash = "/read src/auth.rs" }]
"#,
        )
        .unwrap();

        let report = CommandRegistry::load_from_paths(Some(&root.join(".local/commands.toml")));

        let review = report.registry.resolve("/review").unwrap();
        assert_eq!(review.description, "Local review");
        assert_eq!(review.origin, CommandOrigin::Local);
        assert!(report.registry.resolve("/auth_context").is_some());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn built_in_names_are_reserved() {
        let raw = RawCommandDefinition {
            description: Some("Bad".to_string()),
            usage: None,
            prompt: Some("Hi".to_string()),
            steps: None,
        };
        let error = validate_command("read".to_string(), raw, CommandOrigin::Local).unwrap_err();
        assert!(error.contains("reserved"));
    }

    #[test]
    fn invalid_workflow_steps_are_skipped_on_load() {
        let root = temp_path("invalid");
        std::fs::create_dir_all(root.join(".local")).unwrap();
        std::fs::write(
            root.join(".local/commands.toml"),
            r#"
[commands.good]
description = "Good"
steps = [{ slash = "/read src/main.rs" }, { prompt = "Explain $@" }]

[commands.bad]
description = "Bad"
steps = [{ slash = "/clear" }]
"#,
        )
        .unwrap();

        let report = CommandRegistry::load_from_paths(Some(&root.join(".local/commands.toml")));

        assert!(report.registry.resolve("/good").is_some());
        assert!(report.registry.resolve("/bad").is_none());
        assert_eq!(report.loaded, 1);
        assert_eq!(report.invalid, 1);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn workflow_rejects_mutating_step_before_the_end() {
        let raw = RawCommandDefinition {
            description: Some("Bad workflow".to_string()),
            usage: None,
            prompt: None,
            steps: Some(vec![
                RawCommandStep::Slash {
                    slash: "/run cargo check".to_string(),
                },
                RawCommandStep::Prompt {
                    prompt: "Explain $@".to_string(),
                },
            ]),
        };

        let error =
            validate_command("bad_flow".to_string(), raw, CommandOrigin::Local).unwrap_err();
        assert!(error.contains("must be final"));
    }

    #[test]
    fn autocomplete_names_include_builtins_and_custom_commands() {
        let root = temp_path("autocomplete");
        std::fs::create_dir_all(root.join(".local")).unwrap();
        std::fs::write(
            root.join(".local/commands.toml"),
            r#"
[commands.review_auth]
description = "Review auth code"
prompt = "Review $@"
"#,
        )
        .unwrap();

        let report = CommandRegistry::load_from_paths(Some(&root.join(".local/commands.toml")));
        let names = report.registry.autocomplete_names();

        assert!(names.iter().any(|name| name == "/read"));
        assert!(names.iter().any(|name| name == "/commands"));
        assert!(names.iter().any(|name| name == "/review_auth"));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn workflow_allows_edit_as_final_mutating_step() {
        let raw = RawCommandDefinition {
            description: Some("Edit focused code".to_string()),
            usage: Some("/edit_focus".to_string()),
            prompt: None,
            steps: Some(vec![RawCommandStep::Slash {
                slash: "/edit src/main.rs\n```params-edit\n<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n```".to_string(),
            }]),
        };

        let command =
            validate_command("edit_focus".to_string(), raw, CommandOrigin::Local).expect("valid");
        match command.body {
            CustomCommandBody::Workflow(steps) => assert_eq!(steps.len(), 1),
            CustomCommandBody::Prompt(_) => panic!("expected workflow"),
        }
    }
}
