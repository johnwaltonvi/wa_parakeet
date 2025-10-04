# KenLM Artifacts

Place the compiled 4-gram or 5-gram KenLM binary built from your programming corpora here.

Recommended workflow:

1. Aggregate text sources (README.md, docs/, code comments, transcripts).
2. Normalize text (lowercase, strip timestamps) and save as `corpus.txt`.
3. Build LM using KenLM:
   ```bash
   lmplz -o 5 --memory 8G --text corpus.txt --arpa programming_5gram.arpa
   build_binary programming_5gram.arpa programming_5gram.binary
   ```
4. Move `programming_5gram.binary` into this directory.

The decoder preset file (`config/decoder_presets.yaml`) expects the binary at `lm/programming_5gram.binary`.
