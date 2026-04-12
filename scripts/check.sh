#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/.local/config.toml"
KEYS_PATH="$ROOT_DIR/.local/keys.env"
MODELS_DIR="$ROOT_DIR/.local/models"
PROJECT_PROFILE_PATH="$ROOT_DIR/.params.toml"

status=0

pass() {
  printf '[ok] %s\n' "$*"
}

warn() {
  printf '[warn] %s\n' "$*"
}

fail() {
  printf '[missing] %s\n' "$*"
  status=1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

config_value() {
  local key="$1"
  if [[ ! -f "$CONFIG_PATH" ]]; then
    return 1
  fi
  awk -F'"' -v key="$key" '
    $0 ~ "^[[:space:]]*" key "[[:space:]]*=" { print $2; exit }
  ' "$CONFIG_PATH"
}

printf 'Checking params-cli environment in %s\n\n' "$ROOT_DIR"

if have_cmd cargo; then
  pass "cargo available: $(cargo --version)"
else
  fail "cargo not found"
fi

if have_cmd cmake; then
  pass "cmake available: $(cmake --version | head -n 1)"
else
  warn "cmake not found (required for llama.cpp builds)"
fi

if [[ -f "$CONFIG_PATH" ]]; then
  pass "config present: .local/config.toml"
else
  fail "config missing: run ./scripts/setup.sh"
fi

mkdir -p "$MODELS_DIR"
if [[ -d "$MODELS_DIR" ]]; then
  pass "models directory present: .local/models"
fi

if [[ -f "$KEYS_PATH" ]]; then
  pass "keys file present: .local/keys.env"
else
  warn "keys file missing: .local/keys.env"
fi

if [[ -f "$PROJECT_PROFILE_PATH" ]]; then
  pass "project profile present: .params.toml"
else
  warn "project profile not set"
fi

backend="$(config_value backend || true)"
if [[ -z "$backend" ]]; then
  warn "backend not detected in .local/config.toml"
else
  pass "backend configured: $backend"
fi

case "$backend" in
  llama_cpp)
    if find "$MODELS_DIR" -maxdepth 1 -type f \( -name '*.gguf' -o -name '*.GGUF' \) | grep -q .; then
      pass "llama.cpp model found in .local/models"
    else
      warn "no .gguf model found in .local/models"
    fi
    ;;
  ollama)
    if have_cmd ollama; then
      pass "ollama command available"
    else
      warn "ollama command not found"
    fi
    ;;
  openai_compat)
    if [[ -f "$KEYS_PATH" ]]; then
      pass "openai_compat can use keys from .local/keys.env"
    else
      warn "add API keys via .local/keys.env or .local/config.toml"
    fi
    ;;
  "")
    ;;
  *)
    warn "unknown backend value: $backend"
    ;;
esac

if have_cmd rust-analyzer; then
  pass "rust-analyzer available"
else
  warn "rust-analyzer not found (LSP commands may fail)"
fi

printf '\n'
if [[ "$status" -eq 0 ]]; then
  printf 'Environment looks usable. Start with: cargo run --release\n'
else
  printf 'Environment is missing required pieces. Run ./scripts/setup.sh first.\n'
fi

exit "$status"
