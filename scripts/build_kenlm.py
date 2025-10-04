#!/usr/bin/env python3
"""Assemble a programming-focused corpus and invoke KenLM to build an n-gram binary."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

SUPPORTED_EXTENSIONS = {
    ".rs",
    ".py",
    ".toml",
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".ts",
    ".js",
    ".dart",
    ".txt",
}


def gather_corpus(root: Path, destination: Path, include_hidden: bool = False) -> None:
    with destination.open("w", encoding="utf-8") as out_f:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if not include_hidden and any(part.startswith(".") for part in path.parts):
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            out_f.write(text)
            if not text.endswith("\n"):
                out_f.write("\n")


def run(cmd: Iterable[str]) -> None:
    proc = subprocess.run(list(cmd), text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed with exit code {proc.returncode}")


def ensure_binary(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        raise FileNotFoundError(
            f"Required KenLM binary '{name}' not found in PATH. Install KenLM or provide --{name}-path."
        )
    return binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a KenLM binary from local programming text")
    parser.add_argument("--source", type=Path, required=False, help="Directory to crawl for corpus text")
    parser.add_argument("--input", type=Path, required=False, help="Existing text file to use as corpus")
    parser.add_argument("--output", type=Path, default=Path("lm/programming_5gram.binary"), help="Destination binary path")
    parser.add_argument("--order", type=int, default=5, help="n-gram order (default: 5)")
    parser.add_argument("--lmplz-path", type=Path, help="Explicit path to lmplz binary")
    parser.add_argument("--build-binary-path", type=Path, help="Explicit path to build_binary binary")
    parser.add_argument("--keep-temps", action="store_true", help="Keep intermediate corpus/arpa files")
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden files when crawling source")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source and not args.input:
        print("Either --source or --input must be provided", file=sys.stderr)
        return 1

    lmplz = str(args.lmplz_path) if args.lmplz_path else ensure_binary("lmplz")
    build_binary = str(args.build_binary_path) if args.build_binary_path else ensure_binary("build_binary")

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix="kenlm_build_"))
    corpus_path = args.input.expanduser() if args.input else temp_dir / "corpus.txt"

    try:
        if args.source:
            source = args.source.expanduser()
            if not source.exists():
                raise FileNotFoundError(f"Source directory not found: {source}")
            if not args.input:
                gather_corpus(source, corpus_path, include_hidden=args.include_hidden)

        arpa_path = temp_dir / "model.arpa"

        run([lmplz, "-o", str(args.order), "--text", str(corpus_path), "--arpa", str(arpa_path)])
        run([build_binary, str(arpa_path), str(output_path)])
        print(f"KenLM binary written to {output_path}")

    finally:
        if not args.keep_temps:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
