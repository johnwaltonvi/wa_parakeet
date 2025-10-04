#!/usr/bin/env python3
"""Generate a hotword TSV by scanning project sources for domain terms."""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Counter

DEFAULT_EXTENSIONS = {
    ".rs",
    ".py",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".dart",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_:+.\-]{3,}")

BOOST_DEFAULT = 60
CRITICAL_TOKENS = {
    "CUDA": 80,
    "C++": 100,
    "Cargo.toml": 80,
    "VecDeque": 70,
    "Tokio": 60,
}


def should_consider_token(token: str) -> bool:
    if len(token) <= 2:
        return False
    if token.islower():
        return False
    if token.isdigit():
        return False
    return True


def scan_file(path: Path) -> Iterable[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    for match in TOKEN_RE.finditer(text):
        token = match.group(0)
        if should_consider_token(token):
            yield token


def collect_tokens(root: Path, extensions: set[str], include_hidden: bool) -> Counter[str]:
    counter: Counter[str] = collections.Counter()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if not include_hidden and any(part.startswith(".") for part in path.parts):
            continue
        if path.suffix.lower() not in extensions:
            continue
        counter.update(scan_file(path))
    return counter


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh hotwords.tsv by scanning source repositories")
    parser.add_argument("repos", nargs="+", type=Path, help="Directories to scan for vocabulary")
    parser.add_argument("--output", type=Path, default=Path("vocab.d/hotwords_raw.tsv"), help="Output TSV path")
    parser.add_argument("--max", type=int, default=120, help="Maximum boost value")
    parser.add_argument("--min-count", type=int, default=3, help="Minimum occurrences before a token is considered")
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden files/directories")
    parser.add_argument(
        "--extensions",
        type=str,
        help="Comma-separated list of file extensions to include (overrides defaults)",
    )
    parser.add_argument("--print-json", action="store_true", help="Dump the resulting mapping as JSON to stdout")
    args = parser.parse_args()

    extensions = DEFAULT_EXTENSIONS
    if args.extensions:
        extensions = {ext if ext.startswith('.') else f'.{ext}' for ext in args.extensions.split(',') if ext}
    token_counts: Counter[str] = collections.Counter()
    for repo in args.repos:
        repo = repo.expanduser()
        if not repo.exists():
            print(f"warning: {repo} does not exist", file=sys.stderr)
            continue
        token_counts.update(collect_tokens(repo, extensions, include_hidden=args.include_hidden))

    output_map: dict[str, int] = {}
    for token, boost in CRITICAL_TOKENS.items():
        output_map[token] = min(boost, args.max)

    for token, count in token_counts.most_common():
        if count < args.min_count:
            continue
        boost = min(BOOST_DEFAULT + (count // 5) * 5, args.max)
        output_map.setdefault(token, boost)

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("# token\tboost\tdisplay\ttags\n")
        for token, boost in sorted(output_map.items()):
            display = token
            tags = "auto"
            fh.write(f"{token}\t{boost}\t{display}\t{tags}\n")

    if args.print_json:
        print(json.dumps(output_map, indent=2))

    print(f"Wrote {len(output_map)} hotwords to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
