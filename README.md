# wa_parakeet

`wa_parakeet` packages the Parakeet push-to-talk dictation workflow into a portable, git-tracked project. It bundles the Python listener, helper scripts, and optional systemd unit so you can install the stack on any Linux desktop and have it auto-start after login.

This is my personal project, built while I migrated from Windows to Pop!_OS and needed a reliable voice-to-text tool. I made it available because I could not find an easy plug-and-play option.

## Features
- Right-Alt push-to-talk capture with silence timeout and preamp controls
- NVIDIA NeMo Parakeet-TDT-1.1B transcription with optional Flashlight beam search, KenLM fusion, and hotword boosting
- DTLN voice isolation pre-processing for cleaner audio
- Optional emotion detection (SpeechBrain wav2vec2 SER) to inject emphasis or emotion tags
- Number word → digit conversion with ordinal handling, slash cleanup, and heuristics to skip pronoun phrases ("that one")
- LanguageTool-powered grammar and punctuation cleanup with opt-out flag
- Acronym formatting for Codex CLI, API/APIs, and domain terms
- Text injection into the focused window via `xdotool`
- Automatic muting of desktop audio while recording (disable with `--no-auto-mute`)
- Helper scripts for local runs, service management, and model/decoder management
- Optional user-level systemd unit with restart-on-failure semantics
- DTLN voice isolation pre-processing for cleaner audio
- Number word → digit conversion with ordinal handling, slash cleanup, and heuristics to skip pronoun phrases ("that one")
- Acronym formatting for Codex CLI, API/APIs, and domain terms
- Right-Alt push-to-talk capture with silence timeout and preamp controls
- NVIDIA NeMo Parakeet-TDT-1.1B transcription with optional Flashlight beam search, KenLM fusion, and hotword boosting
- Optional emotion detection (SpeechBrain wav2vec2 SER) to inject emphasis or emotion tags
- Text injection into the focused window via `xdotool`
- Automatic muting of desktop audio while recording (disable with `--no-auto-mute`)
- Helper scripts for local runs, service management, and model/decoder management
- Optional user-level systemd unit with restart-on-failure semantics
- LanguageTool-powered grammar and punctuation cleanup with opt-out flag

### Silence Guard Tuning
- `--speech-min-duration-ms` *(default 250)* – minimum cumulative speech energy before transcription.
- `--speech-rms-delta-db` *(default 8)* – minimum dB gap between peak speech energy and the silence floor.
- `--speech-min-total-ms` *(default 0, disabled)* – optional minimum total clip duration.
- `--speech-min-ratio` *(default 0.35)* – minimum proportion of the clip classified as speech.
- `--speech-min-streak-ms` *(default 150)* – minimum contiguous speech streak required.

## Requirements
- Linux desktop with X11 (tested on Pop!_OS / GNOME)
- Python 3.10+ and `python3-venv`
- System packages: `ffmpeg`, `xdotool`, `libsndfile1`, `default-jre` (LanguageTool Java runtime)
- GPU optional: CUDA accelerates inference but the script runs on CPU as well
- Flashlight text decoder and KenLM binaries (build from source) for advanced beam-search support
- NeMo NLP extras for punctuation/capitalization post-processing (`nemo-toolkit[asr,nlp]==2.4.1`)
- SpeechBrain for emotion recognition (`speechbrain==0.5.15`)

### Flashlight & KenLM Setup

1. Install KenLM (build from source) so `lmplz` and `build_binary` are on your `PATH`.
2. Build the Flashlight text decoder bindings following the [Flashlight instructions](https://github.com/flashlight/text). Ensure the Python wheel installs into this virtualenv (`pip install ./build/dist/*.whl`).
3. Verify imports work inside the virtualenv:
   ```bash
   python -c "from flashlight.lib.text.decoder import KenLM"
   ```
4. Use the helper scripts in this repo to generate resources:
   ```bash
   ./scripts/refresh_hotwords.py ~/wa_parakeet ~/Documents/parakeet_corpus
   ./scripts/curate_hotwords.py --limit 500
   ./scripts/build_kenlm.py --source ~/Documents/parakeet_corpus --output lm/programming_5gram.binary
   ```
   Adjust source paths or pass `--input` if you maintain a curated corpus. If the corpus folder is empty the build step can instead point at `~/wa_parakeet`.

## Turnkey Install
Run the bundled installer to fetch dependencies, download the latest Parakeet model, and install the systemd unit under your user:

```bash
cd ~/wa_parakeet
./install.sh
```

The script will:

- Create/refresh the project virtual environment (`scripts/bootstrap.sh`).
- Prefetch `nvidia/parakeet-tdt-1.1b` so first-run latency stays low.
- Copy `systemd/parakeet-ptt.service` to `~/.config/systemd/user/`, rewriting paths so it points at your clone.
- Import `DISPLAY`/`XAUTHORITY` (when available), run `systemctl --user daemon-reload`, and enable+start the service.
- Install OS dependencies (`ffmpeg`, `xdotool`, `libsndfile1`) via `apt-get` when available.
- Create `~/Documents/parakeet_corpus` (drop extra source text there), seed it with the curated `hotwords.tsv` and `lexicon.tsv`, and refresh `vocab.d/hotwords*.tsv` from both that folder and this repository.
- Enable DTLN voice isolation automatically (uses `pretrained_models/dtln/model_*.onnx`) so noisy audio is cleaned before transcription.
- Prefetch LanguageTool (en-US) grammar resources so cleanups run offline by default.
- Rebuild `lm/programming_5gram.binary` when KenLM binaries (`lmplz`, `build_binary`) are present, falling back to repo sources if the corpus folder is empty.

Log output lands in `~/.cache/Parakeet/service.log`. Check service health with `systemctl --user status parakeet-ptt.service`.

## Manual Quick Start
Use these commands if you prefer to wire things up yourself or need to customise individual steps:

```bash
# Clone or copy the repository, then bootstrap dependencies
cd ~/wa_parakeet
./scripts/bootstrap.sh

# Prefetch the default model so the first run is faster
./scripts/download_model.py --model nvidia/parakeet-tdt-1.1b

# (Recommended) Build vocab + LM assets (adjust source path to your repos)
cp vocab.d/hotwords.tsv ~/Documents/parakeet_corpus/hotwords.tsv
cp vocab.d/lexicon.tsv ~/Documents/parakeet_corpus/lexicon.tsv
./scripts/refresh_hotwords.py ~/wa_parakeet ~/Documents/parakeet_corpus
./scripts/curate_hotwords.py --limit 500
./scripts/build_kenlm.py --source ~/Documents/parakeet_corpus --output lm/programming_5gram.binary

# Launch manually
./scripts/parakeet-ptt --append-space --allow-esc
# Disable voice isolation or point to a custom ONNX directory if needed
# ./scripts/parakeet-ptt --disable-voice-isolation
# ./scripts/parakeet-ptt --voice-isolation-model-dir /path/to/dtln
```

The listener logs to `~/.cache/Parakeet/push_to_talk.log` and stores a PID cookie at `/tmp/parakeet-ptt.pid` to prevent duplicate launches.

## Selecting an Input Device
By default the script uses the system default capture device. Override selection via:

- `--device-index <portaudio-index>` for an explicit device ID
- `--device-keyword "USB Microphone"` to search by substring (case-insensitive)

List available devices with `python -m sounddevice` after activating the virtual environment.

## Systemd Integration
1. Copy `systemd/parakeet-ptt.service` to `~/.config/systemd/user/` and adjust `WorkingDirectory` if the clone lives elsewhere. (The `install.sh` script performs this rewrite automatically.)
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

## Model Upgrades
Run `./scripts/upgrade_parakeet_model` any time NVIDIA publishes a new Parakeet-TDT release (0.6B or 1.1B). The helper script:

- Detects the latest release from Hugging Face and compares it to the configured model.
- Stops the running `parakeet-ptt` service, backs up the cached weights, and downloads the newer model.
- Updates `DEFAULT_MODEL` in `src/parakeet_push_to_talk.py`, restarts the service, and tails the log to confirm the upgrade succeeded.

Use `--dry-run` to check the latest version without making changes.

## Flashlight Decoder & Vocabulary

The 1.1B Parakeet stack can be combined with the Flashlight beam-search decoder for maximum accuracy:

- Decoder presets live in `config/decoder_presets.yaml`. Each preset specifies beam search parameters plus resource paths.
- Runtime resources sit under `vocab.d/` (hotword boosts + pronunciation lexicon) and `lm/` (KenLM binary). Populate these before launching `parakeet-ptt`.
- Use `./scripts/refresh_hotwords.py` to regenerate the hotword TSV and `./scripts/build_kenlm.py` to rebuild the KenLM binary whenever your projects change.
- ⚠️ NeMo currently exposes only the built-in greedy/beam/MAES decoders for Parakeet-TDT models; if Flashlight integration is unavailable the listener logs a warning and keeps the model default decoder.
- Punctuation and capitalization post-processing defaults to the NeMo `punctuation_en_bert` model; disable with `--disable-punctuation` or swap via `--punctuation-model`.
- Emotion detection defaults to `speechbrain/emotion-recognition-wav2vec2-IEMOCAP`; disable with `--disable-emotion`, adjust confidence with `--emotion-threshold`, and add tags using `--emotion-tag`.
- Cached Hugging Face models live under `~/.cache/huggingface/hub/`; remove folders for retired releases (for example, `models--nvidia--parakeet-tdt-0.6b-*`) if you need to reclaim disk space. The service boots with `nvidia/parakeet-tdt-1.1b` by default.
- SpeechBrain uses a shimmed `speechbrain_modules/custom_interface.py` so the upstream wav2vec2 encoder resolves cleanly without editing installed packages. Keep it in sync with upstream releases when upgrading models.

## Emotion-Aware Punctuation

- Emotion inference is applied to the raw transcript before punctuation so exclamations or tags reflect the detected tone once the punctuation model runs.
- Supported toggles:
  - `--emotion-model` – switch to another SpeechBrain-compatible classifier.
  - `--emotion-threshold` – confidence (0–1) required before adding emphasis (default `0.35`).
  - `--emotion-tag` – prefix output with `[EMOTION]` when above threshold.
  - `--disable-emotion` – bypass the SER stage entirely.
- By default we append `!` when the detector is confident the speaker sounds excited/angry/surprised.
- Canonical label mapping normalizes SpeechBrain short codes (`ang`, `hap`, `neu`, `sad`, etc.) before applying emphasis so thresholds behave consistently across classifiers.
- Runtime logs live at `~/.cache/Parakeet/push_to_talk.log`; look for `Emotion detection: <label> (<score>)` entries when tuning thresholds or verifying the pipeline. Helper demos write to `/tmp/ptt_demo.log`.
- `./scripts/parakeet-ptt` accepts overrides:
  - `--decoder-config PATH` – YAML with preset definitions.
  - `--decoder-preset NAME` – preset to load (defaults to `live_fast`).
  - `--lm-path`, `--lexicon-path`, `--hotword-path` – per-run overrides.
  - `--disable-decoder-tuning` – fall back to the model’s default greedy decoder.

Presets target two workflows out of the box:

- `offline_best` – large beam, highest accuracy for batch/offline dictation.
- `live_fast` – smaller beam tuned for interactive push-to-talk.

Update the vocab files as projects change; the listener reloads them on the next start.

## Grammar Cleanup
- LanguageTool runs after the NeMo punctuation model to tidy sentence punctuation and basic grammar while respecting numeric normalization.
- Ordinal words (e.g., 'twenty first') become digits with suffixes and spoken 'forward slash' collapses to '/'.
- Trailing stray '/' at the end of a question normalizes to '?' when the sentence reads like a question.
- Requires a local Java runtime (installed via `default-jre`) and the `language-tool-python` package; the installer prefetches the core resources for offline use.
- Toggle with `--disable-grammar-cleanup` or switch locales via `--grammar-language en-GB`.
- Corrections are logged to `push_to_talk.log` so you can audit changes if a sentence feels off.

## Acronym Normalization
- `config/acronyms.yaml` defines case-safe replacements (Codex CLI, API/APIs, GPU/GPUs, SQL, etc.).
- Update the YAML to add new entries; the listener reloads them on restart.
- Use `--acronym-config /path/to/file.yaml` to point at a custom set for your environment.
  - Quick smoke test: "Codex CLI handles deployments", "install APIs today", "query SQL reports".

## Repository Layout
```
wa_parakeet/
├── README.md                # Project overview and setup steps
├── TODO.md                  # Backlog for future enhancements
├── requirements.txt         # Python dependencies
├── src/                     # Dictation listener source
├── scripts/                 # Helper shell/python scripts (bootstrap, run, stop)
├── config/                  # Decoder preset definitions
├── vocab.d/                 # Hotword boosts and pronunciation lexicon
├── lm/                      # KenLM binaries and helper docs
└── systemd/                 # Example user service definition
```

## Safety & Privacy
- No API keys or credentials are required; ensure you do not commit personal model caches or recordings.
- Review `TODO.md` for planned work such as configurable hotkeys and audio cues.
- When sharing logs, redact window titles and dictated content if it may contain sensitive information.

## License
MIT License (see `LICENSE`).
