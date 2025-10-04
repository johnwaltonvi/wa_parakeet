#!/usr/bin/env python3
"""Automate upgrades of the Parakeet ASR model."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_FILE = REPO_ROOT / "src" / "parakeet_push_to_talk.py"
MODEL_REGEX = re.compile(r'^DEFAULT_MODEL\s*=\s*"(?P<model>[^"]+)"', re.MULTILINE)
HF_SEARCH_URL = "https://huggingface.co/api/models?author=nvidia&search=parakeet-tdt-"
CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"
UPGRADE_LOG = Path.home() / ".cache" / "Parakeet" / "model-upgrades.json"
SERVICE_NAME = "parakeet-ptt.service"
SERVICE_LOG = Path.home() / ".cache" / "Parakeet" / "push_to_talk.log"


@dataclass
class ModelRelease:
    model_id: str
    size: float
    variant: int
    created_at: Optional[str]
    last_modified: Optional[str]

    @property
    def cache_dir(self) -> Path:
        return CACHE_ROOT / f"models--{self.model_id.replace('/', '--')}"

    @property
    def descriptor(self) -> str:
        variant_label = f"v{self.variant}" if self.variant else "base"
        return f"{self.size:.1f}B/{variant_label}"


class UpgradeError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[upgrade] {msg}")


def run_cmd(cmd: List[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise UpgradeError(
            f"Command {' '.join(cmd)} failed with code {result.returncode}:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
    if result.stdout:
        log(result.stdout.strip())
    if result.stderr:
        log(result.stderr.strip())
    return result


def load_current_model() -> str:
    contents = DEFAULT_MODEL_FILE.read_text(encoding="utf-8")
    match = MODEL_REGEX.search(contents)
    if not match:
        raise UpgradeError(f"Could not locate DEFAULT_MODEL in {DEFAULT_MODEL_FILE}")
    return match.group("model")


def write_current_model(model_id: str) -> None:
    contents = DEFAULT_MODEL_FILE.read_text(encoding="utf-8")
    new_contents = MODEL_REGEX.sub(f'DEFAULT_MODEL = "{model_id}"', contents)
    DEFAULT_MODEL_FILE.write_text(new_contents, encoding="utf-8")


def _parse_release_descriptor(model_id: str) -> Optional[tuple[float, int]]:
    name = model_id.split("/")[-1]
    if not name.startswith("parakeet-tdt-"):
        return None
    descriptor = name[len("parakeet-tdt-") :]
    parts = descriptor.split("-")
    size_part = parts[0].rstrip("bB")
    try:
        size = float(size_part)
    except ValueError:
        return None
    variant = 0
    for part in parts[1:]:
        if part.startswith("v") and part[1:].isdigit():
            variant = int(part[1:])
            break
    return size, variant


def fetch_available_models() -> List[ModelRelease]:
    log("Fetching model list from Hugging Face…")
    with urllib.request.urlopen(HF_SEARCH_URL, timeout=30) as response:
        data = json.loads(response.read())
    releases: List[ModelRelease] = []
    for item in data:
        model_id = item.get("modelId")
        if not isinstance(model_id, str) or not model_id.startswith("nvidia/parakeet-tdt-"):
            continue
        parsed = _parse_release_descriptor(model_id)
        if not parsed:
            continue
        releases.append(
            ModelRelease(
                model_id=model_id,
                size=parsed[0],
                variant=parsed[1],
                created_at=item.get("createdAt"),
                last_modified=item.get("lastModified"),
            )
        )
    if not releases:
        raise UpgradeError("Could not find any Parakeet TDT releases on Hugging Face")
    releases.sort(key=lambda r: (r.size, r.variant), reverse=True)
    return releases


def ensure_service_stopped() -> None:
    status = run_cmd(["systemctl", "--user", "is-active", SERVICE_NAME], check=False)
    if status.returncode == 0:
        log("Stopping running parakeet-ptt service…")
        run_cmd(["systemctl", "--user", "stop", SERVICE_NAME])
    else:
        log("Service already stopped or inactive.")


def restart_service() -> None:
    log("Starting parakeet-ptt service…")
    run_cmd(["systemctl", "--user", "start", SERVICE_NAME])


def wait_for_service_log(model_id: str, timeout: int = 120) -> None:
    log("Waiting for service to confirm model load…")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if SERVICE_LOG.exists():
            tail = SERVICE_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()[-50:]
            if any(model_id in line and "Loading model" in line for line in tail):
                log("Service log confirms new model is active.")
                return
        time.sleep(3)
    raise UpgradeError(f"Timed out waiting for service log to mention {model_id}")


def backup_cache(dirs: Iterable[Path]) -> Optional[Path]:
    dirs = list(dirs)
    if not dirs:
        return None
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_root = Path.home() / ".cache" / "Parakeet" / f"backup-{timestamp}"
    backup_root.mkdir(parents=True, exist_ok=True)
    for directory in dirs:
        if directory.exists():
            shutil.move(str(directory), backup_root / directory.name)
            log(f"Backed up {directory} to {backup_root / directory.name}")
    return backup_root


def install_model(model_id: str) -> None:
    log(f"Downloading {model_id} via NeMo…")
    try:
        import torch
        from nemo.collections.asr.models import ASRModel
    except ImportError as exc:
        raise UpgradeError("torch and nemo.collections must be installed in the wa_parakeet virtualenv") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASRModel.from_pretrained(model_name=model_id, map_location=device)
    model.eval()
    log(f"Model {model_id} loaded on {device} and cached successfully.")


def collect_cache_dirs(pattern: str) -> List[Path]:
    if not CACHE_ROOT.exists():
        return []
    return [path for path in CACHE_ROOT.glob(pattern) if path.is_dir()]


def update_history(old_model: str, new_model: str, status: str, extra: Optional[dict] = None) -> None:
    UPGRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
    history: List[dict]
    if UPGRADE_LOG.exists():
        history = json.loads(UPGRADE_LOG.read_text(encoding="utf-8"))
    else:
        history = []
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "from": old_model,
        "to": new_model,
        "status": status,
    }
    if extra:
        entry.update(extra)
    history.append(entry)
    UPGRADE_LOG.write_text(json.dumps(history, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upgrade the Parakeet ASR model to the latest release")
    parser.add_argument("--dry-run", action="store_true", help="Inspect latest version without making changes")
    parser.add_argument("--force", action="store_true", help="Reinstall the latest model even if already configured")
    args = parser.parse_args()

    current_model = load_current_model()
    log(f"Current configured model: {current_model}")

    releases = fetch_available_models()
    latest = releases[0]
    log(f"Latest available release: {latest.model_id} ({latest.descriptor})")

    if latest.model_id == current_model and not args.force:
        log("Already using the latest model. Nothing to do.")
        return
    if args.force and latest.model_id == current_model:
        log("Force flag set – reinstalling the currently configured model.")

    if args.dry_run:
        log("Dry run requested; exiting before making changes.")
        return

    old_model = current_model
    backup_root: Optional[Path] = None
    try:
        ensure_service_stopped()
        cache_dirs = collect_cache_dirs("models--nvidia--parakeet-tdt-0.6b-*")
        backup_root = backup_cache(cache_dirs)
        install_model(latest.model_id)
        write_current_model(latest.model_id)
        restart_service()
        wait_for_service_log(latest.model_id)
        update_history(
            old_model,
            latest.model_id,
            "success",
            {
                "createdAt": latest.created_at,
                "lastModified": latest.last_modified,
                "descriptor": latest.descriptor,
                "backup": str(backup_root) if backup_root else None,
            },
        )
        log("Upgrade completed successfully.")
    except Exception as exc:  # noqa: BLE001
        update_history(
            old_model,
            latest.model_id,
            "failed",
            {"error": str(exc), "descriptor": latest.descriptor},
        )
        if backup_root and backup_root.exists():
            log("Restoring previous cache from backup…")
            for item in backup_root.iterdir():
                shutil.move(str(item), CACHE_ROOT / item.name)
        write_current_model(old_model)
        run_cmd(["systemctl", "--user", "start", SERVICE_NAME], check=False)
        raise


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except UpgradeError as err:
        log(f"FAILED: {err}")
        sys.exit(1)
    except Exception as err:  # noqa: BLE001
        log(f"UNEXPECTED ERROR: {err}")
        sys.exit(1)
