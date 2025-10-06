# TODO

- [ ] Add audio toggle cues (beep when recording starts, alternate when it stops)
- [ ] Move hotkey configuration into a `config.toml` file with support for multiple bindings
- [ ] Review polish and portability items (packaging, release checklist, optional GUI)
- [ ] Provide unit tests or smoke checks for audio device discovery logic
- [ ] Document alternative speech models and instructions for offline installation
- [x] Add post-processing pass to normalize spoken numbers into digits (e.g., "redis eight" -> "redis 8") with minimal configuration
- [ ] Implement acronym detection to enforce uppercase formatting and optional dotted forms (e.g., "s. q. l." variants)
- [ ] Explore optional syntax cleanup that collapses repeated punctuation (",," "+!" etc.) without altering intended wording; evaluate grammar toggles separately
- [ ] Prototype DTLN-based voice isolation pre-processing to feed cleaner audio into Parakeet (benchmark latency/quality)
