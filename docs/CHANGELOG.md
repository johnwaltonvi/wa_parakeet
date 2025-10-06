# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- Added configurable acronym normalization (config/acronyms.yaml) with Codex CLI, API/APIs, GPU, and SQL defaults.

- Added optional LanguageTool grammar cleanup stage with CLI toggles and installer support.

## [2025-10-03] Auto-mute system audio during push-to-talk
- Added an `AudioMuteController` that mutes desktop audio via `wpctl`/`pactl` while the recorder is active and restores the previous state afterward.
- Introduced a `--no-auto-mute` flag for advanced users who prefer to manage audio manually.
- Logged mute/unmute actions to `push_to_talk.log` for easier troubleshooting.
- Documented the systemd environment requirements and created unit tests covering both wpctl and pactl paths.
