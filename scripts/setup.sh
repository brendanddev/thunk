#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DIR="$ROOT_DIR/.local"
MODELS_DIR="$LOCAL_DIR/models"
MEMORY_DIR="$LOCAL_DIR/memory"
EXPORTS_DIR="$LOCAL_DIR/exports"
LOGS_DIR="$LOCAL_DIR/logs"
CONFIG_PATH="$LOCAL_DIR/config.toml"
KEYS_PATH="$LOCAL_DIR/keys.env"

say() {
  printf '%s\n' "$*"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

write_default_config() {
  cat >"$CONFIG_PATH" <<'EOF'
# params-cli local config
# Backend options: "llama_cpp", "ollama", "openai_compat"
backend = "llama_cpp"

[llama_cpp]
# model_path = ".local/models/your-model.gguf"

[ollama]
url = "http://localhost:11434"
model = "qwen2.5-coder:7b"

[openai_compat]
url = "https://api.openai.com/v1"
# api_key = ""
model = "gpt-4o-mini"

[generation]
max_tokens = 512
temperature = 0.8

[cache]
ttl_seconds = 21600

[reflection]
enabled = false

[eco]
enabled = false

[memory]
fact_ttl_days = 90
max_facts_per_project = 150

[safety]
enabled = true
read_scope = "project_only"
block_private_network = true
inspect_network = true
shell_mode = "approve_inspect"
block_destructive_shell = true
shell_allowlist = []
shell_denylist = []
network_allowlist = []
inspect_cloud_requests = true
EOF
}

write_keys_template() {
  cat >"$KEYS_PATH" <<'EOF'
# Optional API keys for openai_compat backends.
# Uncomment only the provider(s) you use.

# export OPENAI_API_KEY="..."
# export GROQ_API_KEY="..."
# export OPENROUTER_API_KEY="..."
# export XAI_API_KEY="..."
EOF
}

say "Preparing params-cli workspace in:"
say "  $ROOT_DIR"

mkdir -p "$MODELS_DIR" "$MEMORY_DIR" "$EXPORTS_DIR" "$LOGS_DIR"
say "Created local directories under .local/"

if [[ ! -f "$CONFIG_PATH" ]]; then
  write_default_config
  say "Created starter config at .local/config.toml"
else
  say "Kept existing config at .local/config.toml"
fi

if [[ ! -f "$KEYS_PATH" ]]; then
  write_keys_template
  say "Created optional keys template at .local/keys.env"
else
  say "Kept existing keys file at .local/keys.env"
fi

say ""
say "Environment checks:"

if have_cmd cargo; then
  say "  [ok] cargo: $(cargo --version)"
else
  say "  [missing] cargo (install Rust from https://rustup.rs)"
fi

if have_cmd cmake; then
  say "  [ok] cmake: $(cmake --version | head -n 1)"
else
  say "  [missing] cmake (needed for llama.cpp builds)"
fi

if have_cmd rust-analyzer; then
  say "  [ok] rust-analyzer: $(rust-analyzer --version 2>/dev/null || printf 'found')"
else
  say "  [optional] rust-analyzer not found"
fi

say ""
say "Next steps:"
say "  1. Review .local/config.toml and choose a backend."
say "  2. For llama.cpp, place a .gguf model in .local/models/."
say "  3. For openai_compat, add keys to .local/keys.env or config."
say "  4. Run ./scripts/check.sh to verify the environment."
say "  5. Start the app with: cargo run --release"

