# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [2025-10-03] Auto-mute system audio during push-to-talk
- Added an `AudioMuteController` that mutes desktop audio via `wpctl`/`pactl` while the recorder is active and restores the previous state afterward.
- Introduced a `--no-auto-mute` flag for advanced users who prefer to manage audio manually.
- Logged mute/unmute actions to `push_to_talk.log` for easier troubleshooting.
- Documented the systemd environment requirements and created unit tests covering both wpctl and pactl paths.
