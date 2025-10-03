#!/usr/bin/env python3
"""Prefetch the Parakeet ASR model so the first run is faster."""
import argparse
from pathlib import Path

import torch
from nemo.collections.asr.models import ASRModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a NeMo ASR model into the local cache")
    parser.add_argument(
        "--model",
        default="nvidia/parakeet-tdt-0.6b-v2",
        help="Model identifier to download",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASRModel.from_pretrained(model_name=args.model, map_location=device)
    cache_dir = Path(model._cfg.trainer.get("default_save_path", ""))  # type: ignore[attr-defined]
    print(f"Model {args.model} is ready. Local cache lives under {cache_dir} (if set by NeMo).")


if __name__ == "__main__":
    main()
