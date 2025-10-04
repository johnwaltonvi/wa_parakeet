#!/usr/bin/env python3
"""Curate a compact hotword TSV tuned for programming, Rust, trading/crypto, ML, and marketing domains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

STOPWORDS = {
    'the', 'and', 'for', 'with', 'from', 'into', 'this', 'that', 'will', 'your', 'have',
    'been', 'shall', 'would', 'should', 'could', 'there', 'their', 'about', 'which',
    'each', 'other', 'such', 'more', 'less', 'also', 'than', 'some', 'many', 'when',
    'where', 'while', 'because', 'however', 'within', 'without', 'through', 'before',
    'after', 'upon', 'here', 'must', 'were', 'are', 'was', 'can', 'may', 'not', 'only',
    'over', 'per', 'its', 'any', 'all', 'our', 'these', 'those', 'once', 'every', 'very',
    'much', 'like', 'just', 'being', 'still', 'they', 'them', 'you', 'yours', 'his',
    'her', 'who', 'what', 'whom', 'why', 'how', 'under', 'between', 'across', 'until',
    'again', 'during', 'even', 'both', 'off', 'own', 'same'
}

MANUAL_TERMS: Dict[str, Tuple[str, int]] = {
    # Programming / Rust concepts & crates
    'Borrow Checker': ('rust', 95),
    'Ownership': ('rust', 95),
    'Lifetimes': ('rust', 90),
    'Trait Object': ('rust', 85),
    'VecDeque': ('rust', 85),
    'HashMap': ('rust', 85),
    'HashSet': ('rust', 85),
    'BTreeMap': ('rust', 85),
    'BTreeSet': ('rust', 85),
    'Arc': ('rust', 80),
    'Rc': ('rust', 80),
    'Send': ('rust', 80),
    'Sync': ('rust', 80),
    'Pin': ('rust', 80),
    'UnsafeCell': ('rust', 80),
    'macro_rules!': ('rust', 85),
    'async/await': ('rust', 90),
    'no_std': ('rust', 85),
    'Cargo.toml': ('rust', 95),
    'Cargo.lock': ('rust', 90),
    'WasmBindgen': ('rust', 85),
    'WebAssembly': ('programming', 85),
    'Serde': ('rust', 85),
    'Tokio': ('rust', 85),
    'Reqwest': ('rust', 80),
    'Hyper': ('rust', 80),
    'Axum': ('rust', 80),
    'Actix': ('rust', 80),
    'Tonic': ('rust', 80),
    'SeaORM': ('rust', 80),
    'Polars': ('rust', 85),
    'Rust Analyzer': ('rust', 85),
    'Rustfmt': ('rust', 80),
    'Clippy': ('rust', 80),
    'Nightly Rust': ('rust', 80),
    'Zero Copy': ('rust', 80),
    'Crate Graph': ('rust', 80),
    'BorrowMut': ('rust', 80),
    'MutexGuard': ('rust', 80),
    'RwLock': ('rust', 80),

    # Machine learning & AI
    'TensorFlow': ('ml', 95),
    'PyTorch': ('ml', 95),
    'JAX': ('ml', 90),
    'XGBoost': ('ml', 90),
    'LightGBM': ('ml', 90),
    'CatBoost': ('ml', 90),
    'scikit-learn': ('ml', 90),
    'NumPy': ('ml', 90),
    'Pandas': ('ml', 90),
    'Transformer': ('ml', 95),
    'Self-Attention': ('ml', 90),
    'Cross-Entropy': ('ml', 90),
    'Learning Rate': ('ml', 85),
    'Gradient Descent': ('ml', 90),
    'Backpropagation': ('ml', 95),
    'Fine-Tuning': ('ml', 90),
    'Zero-Shot': ('ml', 90),
    'Few-Shot': ('ml', 90),
    'Prompt Engineering': ('ml', 90),
    'BLEU Score': ('ml', 85),
    'Perplexity Score': ('ml', 85),
    'Autoencoder': ('ml', 85),
    'Contrastive Learning': ('ml', 90),
    'Reinforcement Learning': ('ml', 90),
    'Diffusion Model': ('ml', 90),
    'Latent Space': ('ml', 85),
    'Embeddings': ('ml', 85),
    'Tokenization': ('ml', 85),

    # Trading & cryptocurrency jargon
    'OHLC': ('trading', 95),
    'VWAP': ('trading', 95),
    'EMA': ('trading', 95),
    'SMA': ('trading', 95),
    'RSI': ('trading', 95),
    'MACD': ('trading', 95),
    'ATR': ('trading', 95),
    'ADX': ('trading', 95),
    'Bollinger Bands': ('trading', 95),
    'Fibonacci Retracement': ('trading', 90),
    'Candlestick Pattern': ('trading', 90),
    'Candlestick': ('trading', 90),
    'Order Book': ('trading', 90),
    'Depth Chart': ('trading', 90),
    'Bid Ask Spread': ('trading', 90),
    'Stop Loss': ('trading', 90),
    'Take Profit': ('trading', 90),
    'Dollar Cost Averaging': ('trading', 90),
    'Funding Rate': ('crypto', 90),
    'Mark Price': ('crypto', 90),
    'Perpetual Swap': ('crypto', 90),
    'Liquidity Pool': ('crypto', 90),
    'Automated Market Maker': ('crypto', 90),
    'Layer 2': ('crypto', 85),
    'Stablecoin': ('crypto', 90),
    'Altcoin': ('crypto', 90),
    'Bitcoin': ('crypto', 90),
    'Ethereum': ('crypto', 90),
    'Satoshi': ('crypto', 90),
    'HODL': ('crypto', 90),
    'Whale': ('crypto', 90),
    'Bear Trap': ('trading', 90),
    'Bull Trap': ('trading', 90),
    'Gas Fees': ('crypto', 85),
    'Proof of Stake': ('crypto', 90),
    'Proof of Work': ('crypto', 90),
    'Staking': ('crypto', 85),
    'Staking Rewards': ('crypto', 85),
    'DeFi Yield': ('crypto', 85),

    # Digital marketing & advertising
    'CPC': ('marketing', 95),
    'CPM': ('marketing', 95),
    'CTR': ('marketing', 95),
    'ROAS': ('marketing', 95),
    'ROI': ('marketing', 95),
    'CAC': ('marketing', 95),
    'LTV': ('marketing', 95),
    'Conversion Rate': ('marketing', 90),
    'Retargeting': ('marketing', 90),
    'Marketing Funnel': ('marketing', 85),
    'Customer Journey': ('marketing', 85),
    'Lookalike Audience': ('marketing', 90),
    'Attribution Model': ('marketing', 90),
    'Programmatic Advertising': ('marketing', 90),
    'UTM Parameter': ('marketing', 85),
    'Click Through Rate': ('marketing', 90),
    'Impression Share': ('marketing', 85),
    'Cost Per Lead': ('marketing', 90),
    'Landing Page': ('marketing', 85),
    'A/B Test': ('marketing', 85),
    'Session Recording': ('marketing', 80),
    'Customer Lifetime Value': ('marketing', 90),
    'Demand Generation': ('marketing', 85),
}

MANUAL_LOWERCASE = {
    'serde': ('rust', 85),
    'tokio': ('rust', 85),
    'wasmtime': ('rust', 80),
    'neon': ('rust', 75),
    'xgboost': ('ml', 90),
    'lightgbm': ('ml', 90),
    'catboost': ('ml', 90),
    'tensorflow': ('ml', 95),
    'pytorch': ('ml', 95),
    'llm': ('ml', 95),
    'embedding': ('ml', 85),
    'tokenization': ('ml', 85),
    'vwap': ('trading', 95),
    'ema': ('trading', 95),
    'sma': ('trading', 95),
    'adx': ('trading', 95),
    'atr': ('trading', 95),
    'defi': ('crypto', 90),
    'dydx': ('crypto', 90),
    'amm': ('crypto', 90),
    'hodl': ('crypto', 90),
    'fomo': ('crypto', 85),
    'cpc': ('marketing', 95),
    'cpm': ('marketing', 95),
    'ctr': ('marketing', 95),
    'roas': ('marketing', 95),
    'kpi': ('marketing', 90),
    'mql': ('marketing', 90),
    'sql': ('marketing', 90),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate hotwords for programming/trading/marketing dictation")
    parser.add_argument("--input", type=Path, default=Path("vocab.d/hotwords_raw.tsv"), help="Raw hotword TSV")
    parser.add_argument("--output", type=Path, default=Path("vocab.d/hotwords.tsv"), help="Curated TSV output")
    parser.add_argument("--max", type=int, default=120, help="Maximum boost value")
    parser.add_argument("--min", type=int, default=60, help="Minimum boost value")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of auto-selected tokens (0 = manual only)")
    parser.add_argument("--dump-json", action="store_true", help="Print resulting tokens to stdout as JSON")
    return parser.parse_args()


def load_raw(path: Path) -> Dict[str, Tuple[int, str, str]]:
    entries: Dict[str, Tuple[int, str, str]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Raw hotword list not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        token = parts[0]
        boost = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 60
        display = parts[2] if len(parts) > 2 and parts[2] else token
        tags = parts[3] if len(parts) > 3 and parts[3] else 'auto'
        entries[token] = (boost, display, tags)
    return entries


def is_interesting(token: str) -> bool:
    lower = token.lower()
    if lower in STOPWORDS:
        return False
    if token in MANUAL_TERMS or lower in MANUAL_LOWERCASE:
        return True
    if len(token) <= 2:
        return False
    if token.isdigit():
        return False
    simple = token.replace('.', '').replace('-', '')
    if simple.isdigit():
        return False
    if set(token) <= set('.-0123456789'):
        return False
    if token in {'---', '...'}:
        return False
    if token.endswith(('L', 'l')) and token[:-1].isdigit():
        return False
    if token.upper() == token and len(token) > 4:
        return False
    if token[0].isupper() and token[1:].islower():
        return False
    return True


def main() -> int:
    args = parse_args()
    raw = load_raw(args.input)
    curated: Dict[str, Dict[str, str | int]] = {}

    for token, (tag, default_boost) in MANUAL_TERMS.items():
        curated[token] = {
            'boost': min(default_boost, args.max),
            'display': token,
            'tags': tag,
        }
    for token, (tag, default_boost) in MANUAL_LOWERCASE.items():
        curated[token] = {
            'boost': min(default_boost, args.max),
            'display': token,
            'tags': tag,
        }

    if args.limit > 0:
        auto_candidates: Dict[str, Dict[str, str | int]] = {}
        for token, (boost, display, tags) in raw.items():
            if token in curated or token.lower() in curated:
                continue
            if not is_interesting(token):
                continue
            adjusted = max(min(boost, args.max), args.min)
            auto_candidates[token] = {
                'boost': adjusted,
                'display': display,
                'tags': 'auto',
            }
        if len(auto_candidates) > args.limit:
            ranked = sorted(auto_candidates.items(), key=lambda item: (item[1]['boost'], item[0]), reverse=True)
            auto_candidates = dict(ranked[: args.limit])
        curated.update(auto_candidates)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as fh:
        fh.write('# token\tboost\tdisplay\ttags\n')
        for token in sorted(curated.keys(), key=lambda s: (curated[s]['tags'], s.lower())):
            entry = curated[token]
            fh.write(f"{token}\t{entry['boost']}\t{entry['display']}\t{entry['tags']}\n")

    if args.dump_json:
        print(json.dumps(curated, indent=2))

    print(f"Curated {len(curated)} hotwords -> {output_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
