#!/usr/bin/env bash
# Fully bootstrap Parakeet on a new machine: virtualenv, model assets, and systemd unit.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
SERVICE_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_TEMPLATE="$ROOT_DIR/systemd/parakeet-ptt.service"
SERVICE_PATH="$SERVICE_DIR/parakeet-ptt.service"
MODEL_ID="nvidia/parakeet-tdt-1.1b"
HOTWORD_SOURCE_DEFAULT="$HOME/Documents/wise_apple"
HOTWORD_SOURCE="${PARAKEET_HOTWORD_SOURCE:-$HOTWORD_SOURCE_DEFAULT}"
SKIP_HOTWORDS="${PARAKEET_SKIP_HOTWORDS:-0}"

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

step "Checking prerequisites"
require_cmd python3
require_cmd systemctl
require_cmd bash
warn_cmd ffmpeg || true
warn_cmd xdotool || true

# soundfile requires libsndfile at runtime; probe after venv setup

step "Bootstrapping Python environment"
"$ROOT_DIR/scripts/bootstrap.sh"
source "$VENV_DIR/bin/activate"

step "Prefetching Parakeet model ($MODEL_ID)"
python "$ROOT_DIR/scripts/download_model.py" --model "$MODEL_ID"

if ! python -c "import soundfile" >/dev/null 2>&1; then
  echo "Warning: python package 'soundfile' failed to import. Ensure libsndfile1 is installed (e.g., 'sudo apt install libsndfile1')." >&2
fi

if [ "$SKIP_HOTWORDS" != "1" ]; then
  if [ -d "$HOTWORD_SOURCE" ]; then
    step "Refreshing hotword TSV from $HOTWORD_SOURCE"
    if ! python "$ROOT_DIR/scripts/refresh_hotwords.py" "$HOTWORD_SOURCE"; then
      echo "Warning: hotword refresh failed; inspect output above." >&2
    else
      if ! python "$ROOT_DIR/scripts/curate_hotwords.py"; then
        echo "Warning: hotword curation failed; inspect output above." >&2
      fi
    fi

    if warn_cmd lmplz && warn_cmd build_binary; then
      step "Building KenLM binary from $HOTWORD_SOURCE"
      if ! python "$ROOT_DIR/scripts/build_kenlm.py" --source "$HOTWORD_SOURCE"; then
        echo "Warning: KenLM build failed; install KenLM binaries and retry." >&2
      fi
    else
      echo "Warning: KenLM binaries not found; skipping language model build." >&2
    fi
  else
    echo "Warning: hotword source directory '$HOTWORD_SOURCE' not found; set PARAKEET_HOTWORD_SOURCE or export PARAKEET_SKIP_HOTWORDS=1 to silence." >&2
  fi
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
