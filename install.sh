#!/usr/bin/env bash
# Fully bootstrap Parakeet on a new machine: virtualenv, model assets, and systemd unit.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
SERVICE_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_TEMPLATE="$ROOT_DIR/systemd/parakeet-ptt.service"
SERVICE_PATH="$SERVICE_DIR/parakeet-ptt.service"
MODEL_ID="nvidia/parakeet-tdt-1.1b"
CORPUS_ROOT="$HOME/Documents/parakeet_corpus"
SYSTEM_PACKAGES=(ffmpeg xdotool libsndfile1 default-jre)
COMMAND_CHECKS=(ffmpeg xdotool java)
DEFAULT_HOTWORDS_SRC="$ROOT_DIR/vocab.d/hotwords.tsv"
DEFAULT_LEXICON_SRC="$ROOT_DIR/vocab.d/lexicon.tsv"
GRAMMAR_LANGUAGE="en-US"

step() {
  printf '\n==> %s\n' "$1"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' is required but not found in PATH." >&2
    exit 1
  fi
}

warn_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Warning: '$1' not found; install it for full functionality." >&2
    return 1
  fi
  return 0
}

install_system_packages() {
  if command -v apt-get >/dev/null 2>&1; then
    step "Installing system packages (${SYSTEM_PACKAGES[*]})"
    sudo apt-get update -y
    sudo apt-get install -y "${SYSTEM_PACKAGES[@]}"
  else
    echo "Warning: no supported package manager detected; install ${SYSTEM_PACKAGES[*]} manually." >&2
  fi
}

ensure_corpus_root() {
  if [ ! -d "$CORPUS_ROOT" ]; then
    step "Creating corpus directory at $CORPUS_ROOT"
    mkdir -p "$CORPUS_ROOT"
  fi
}

has_corpus_files() {
  find "$1" -type f -print -quit | grep -q .
}

seed_default_vocab() {
  if [ -f "$DEFAULT_HOTWORDS_SRC" ] && [ ! -f "$CORPUS_ROOT/hotwords.tsv" ]; then
    step "Seeding default hotwords into $CORPUS_ROOT/hotwords.tsv"
    install -m 644 "$DEFAULT_HOTWORDS_SRC" "$CORPUS_ROOT/hotwords.tsv"
  fi
  if [ -f "$DEFAULT_LEXICON_SRC" ] && [ ! -f "$CORPUS_ROOT/lexicon.tsv" ]; then
    step "Seeding default lexicon into $CORPUS_ROOT/lexicon.tsv"
    install -m 644 "$DEFAULT_LEXICON_SRC" "$CORPUS_ROOT/lexicon.tsv"
  fi
}

step "Checking prerequisites"
require_cmd python3
require_cmd systemctl
require_cmd bash
install_system_packages

# soundfile requires libsndfile at runtime; probe after venv setup
for pkg in "${COMMAND_CHECKS[@]}"; do
  warn_cmd "$pkg" || true
done

step "Bootstrapping Python environment"
"$ROOT_DIR/scripts/bootstrap.sh"
source "$VENV_DIR/bin/activate"

step "Prefetching Parakeet model ($MODEL_ID)"
python "$ROOT_DIR/scripts/download_model.py" --model "$MODEL_ID"

if ! python -c "import soundfile" >/dev/null 2>&1; then
  echo "Warning: python package 'soundfile' failed to import. Ensure libsndfile1 is installed (e.g., 'sudo apt install libsndfile1')." >&2
fi

step "Prefetching LanguageTool resources ($GRAMMAR_LANGUAGE)"
if ! python -m language_tool_python.download_lt >/dev/null 2>&1; then
  echo "Warning: Failed to prefetch LanguageTool resources; grammar cleanup may attempt to download at runtime." >&2
fi

ensure_corpus_root
seed_default_vocab

HOTWORD_SOURCES=("$ROOT_DIR")
if has_corpus_files "$CORPUS_ROOT"; then
  HOTWORD_SOURCES+=("$CORPUS_ROOT")
fi

step "Refreshing hotword TSV from: ${HOTWORD_SOURCES[*]}"
if ! python "$ROOT_DIR/scripts/refresh_hotwords.py" "${HOTWORD_SOURCES[@]}"; then
  echo "Warning: hotword refresh failed; inspect output above." >&2
else
  if ! python "$ROOT_DIR/scripts/curate_hotwords.py"; then
    echo "Warning: hotword curation failed; inspect output above." >&2
  fi
fi

KENLM_SOURCE="$CORPUS_ROOT"
if ! has_corpus_files "$KENLM_SOURCE"; then
  echo "Warning: no corpus files detected in $KENLM_SOURCE; falling back to repository sources." >&2
  KENLM_SOURCE="$ROOT_DIR"
fi

if warn_cmd lmplz && warn_cmd build_binary; then
  step "Building KenLM binary from $KENLM_SOURCE"
  if ! python "$ROOT_DIR/scripts/build_kenlm.py" --source "$KENLM_SOURCE"; then
    echo "Warning: KenLM build failed; install KenLM binaries and retry." >&2
  fi
else
  echo "Warning: KenLM binaries not found; skipping language model build." >&2
fi

deactivate

if [ ! -d "$SERVICE_DIR" ]; then
  step "Creating systemd user directory at $SERVICE_DIR"
  mkdir -p "$SERVICE_DIR"
fi

step "Deploying user-level systemd service"
TMP_SERVICE="$(mktemp)"
sed \
  -e "s|WorkingDirectory=%h/wa_parakeet|WorkingDirectory=$ROOT_DIR|" \
  -e "s|ExecStart=%h/wa_parakeet/scripts/parakeet-ptt-start|ExecStart=$ROOT_DIR/scripts/parakeet-ptt-start|" \
  -e "s|ExecStop=%h/wa_parakeet/scripts/parakeet-ptt-stop|ExecStop=$ROOT_DIR/scripts/parakeet-ptt-stop|" \
  "$SERVICE_TEMPLATE" > "$TMP_SERVICE"
install -m 644 "$TMP_SERVICE" "$SERVICE_PATH"
rm "$TMP_SERVICE"

if [ -n "${DISPLAY:-}" ] && [ -n "${XAUTHORITY:-}" ]; then
  step "Importing DISPLAY and XAUTHORITY into systemd user session"
  systemctl --user import-environment DISPLAY XAUTHORITY || true
else
  echo "Warning: DISPLAY/XAUTHORITY not set; run 'systemctl --user import-environment DISPLAY XAUTHORITY' after logging into a graphical session." >&2
fi

step "Reloading and enabling parakeet-ptt.service"
systemctl --user daemon-reload
systemctl --user enable --now parakeet-ptt.service

step "Parakeet install completed"
echo "Logs live at $HOME/.cache/Parakeet/service.log"
echo "Use 'systemctl --user status parakeet-ptt.service' to check service state."
