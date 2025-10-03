# wa_parakeet

`wa_parakeet` packages the Parakeet push-to-talk dictation workflow into a portable, git-tracked project. It bundles the Python listener, helper scripts, and optional systemd unit so you can install the stack on any Linux desktop and have it auto-start after login.

This is my personal project, built while I migrated from Windows to Pop!_OS and needed a reliable voice-to-text tool. I made it available because I could not find an easy plug-and-play option.

## Features
- Right-Alt push-to-talk capture with silence timeout and preamp controls
- NVIDIA NeMo Parakeet transcription (`nvidia/parakeet-tdt-0.6b-v2` by default)
- Text injection into the focused window via `xdotool`
- Automatic muting of desktop audio while recording (disable with `--no-auto-mute`)
- Helper scripts for local runs, service management, and model prefetching
- Optional user-level systemd unit with restart-on-failure semantics

## Requirements
- Linux desktop with X11 (tested on Pop!_OS / GNOME)
- Python 3.10+ and `python3-venv`
- System packages: `ffmpeg`, `xdotool`, `libsndfile1`
- GPU optional: CUDA accelerates inference but the script runs on CPU as well

## Quick Start
```bash
# Clone or copy the repository, then bootstrap dependencies
cd ~/wa_parakeet
./scripts/bootstrap.sh

# (Optional) Prefetch the model so the first run is faster
./scripts/download_model.py

# Launch manually
./scripts/parakeet-ptt --append-space --allow-esc
```

The listener logs to `~/.cache/Parakeet/push_to_talk.log` and stores a PID cookie at `/tmp/parakeet-ptt.pid` to prevent duplicate launches.

## Selecting an Input Device
By default the script uses the system default capture device. Override selection via:

- `--device-index <portaudio-index>` for an explicit device ID
- `--device-keyword "USB Microphone"` to search by substring (case-insensitive)

List available devices with `python -m sounddevice` after activating the virtual environment.

## Systemd Integration
1. Copy `systemd/parakeet-ptt.service` to `~/.config/systemd/user/` and adjust `WorkingDirectory` if the clone lives elsewhere.
2. Ensure the unit has access to your X session. Either set static values inside the service (e.g., `Environment=DISPLAY=:1` and `Environment=XAUTHORITY=%h/.Xauthority`) or import the live environment after login:
   ```bash
   systemctl --user import-environment DISPLAY XAUTHORITY
   ```
3. Reload and enable the service:
   ```bash
   systemctl --user daemon-reload
   systemctl --user enable --now parakeet-ptt.service
   ```

The unit restarts on failure and inherits `DISPLAY`/`XAUTHORITY` via systemd specifiers.

## Repository Layout
```
wa_parakeet/
├── README.md                # Project overview and setup steps
├── TODO.md                  # Backlog for future enhancements
├── requirements.txt         # Python dependencies
├── src/                     # Dictation listener source
├── scripts/                 # Helper shell/python scripts (bootstrap, run, stop)
└── systemd/                 # Example user service definition
```

## Safety & Privacy
- No API keys or credentials are required; ensure you do not commit personal model caches or recordings.
- Review `TODO.md` for planned work such as configurable hotkeys and audio cues.
- When sharing logs, redact window titles and dictated content if it may contain sensitive information.

## License
MIT License (see `LICENSE`).
