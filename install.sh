#!/usr/bin/env bash
# Fully bootstrap Parakeet on a new machine: virtualenv, model assets, and systemd unit.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
SERVICE_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_TEMPLATE="$ROOT_DIR/systemd/parakeet-ptt.service"
SERVICE_PATH="$SERVICE_DIR/parakeet-ptt.service"
MODEL_ID="nvidia/parakeet-tdt-1.1b"

step() {
  printf '\n==> %s\n' "$1"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' is required but not found in PATH." >&2
    exit 1
  fi
}

step "Checking prerequisites"
require_cmd python3
require_cmd systemctl
require_cmd bash

step "Bootstrapping Python environment"
"$ROOT_DIR/scripts/bootstrap.sh"
source "$VENV_DIR/bin/activate"

step "Prefetching Parakeet model ($MODEL_ID)"
python "$ROOT_DIR/scripts/download_model.py" --model "$MODEL_ID"

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
