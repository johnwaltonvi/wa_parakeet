#!/usr/bin/env python3
"""Parakeet push-to-talk dictation using a configurable hotkey."""
import argparse
import atexit
import difflib
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from nemo.collections.asr.models import ASRModel
try:
    from nemo.collections.nlp.models import PunctuationCapitalizationModel
except ImportError:  # pragma: no cover - optional dependency
    PunctuationCapitalizationModel = None  # type: ignore[assignment]
try:
    from speechbrain.pretrained.interfaces import foreign_class
except ImportError:  # pragma: no cover - optional dependency
    foreign_class = None  # type: ignore[assignment]
from pynput import keyboard
from word2number import w2n
from dataclasses import dataclass

from voice_isolation import DtlnVoiceIsolation

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - handled at runtime
    OmegaConf = None

try:
    import language_tool_python
except ImportError:  # pragma: no cover - handled at runtime
    language_tool_python = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = Path.home() / ".cache" / "Parakeet" / "push_to_talk.log"
DEFAULT_COOKIE = Path("/tmp/parakeet-ptt.pid")
DEFAULT_MODEL = "nvidia/parakeet-tdt-1.1b"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_DECODER_CONFIG = REPO_ROOT / "config" / "decoder_presets.yaml"
DEFAULT_DECODER_PRESET = "live_fast"
DEFAULT_PUNCTUATION_MODEL = "punctuation_en_bert"
DEFAULT_EMOTION_MODEL = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
DEFAULT_EMOTION_THRESHOLD = 0.35
DEFAULT_VOICE_ISOLATION_MODEL_DIR = REPO_ROOT / "pretrained_models" / "dtln"
DEFAULT_GRAMMAR_LANGUAGE = "en-US"
DEFAULT_ACRONYM_CONFIG = REPO_ROOT / "config" / "acronyms.yaml"
EMOTION_EXCLAIM_LABELS = {"angry", "happy", "excited", "surprised", "frustrated", "fearful"}
EMOTION_CANONICAL_MAP = {
    "ang": "angry",
    "anger": "angry",
    "angry": "angry",
    "hap": "happy",
    "happy": "happy",
    "joy": "happy",
    "excited": "excited",
    "surprise": "surprised",
    "surprised": "surprised",
    "fru": "frustrated",
    "frustrated": "frustrated",
    "fea": "fearful",
    "fear": "fearful",
    "fearful": "fearful",
    "sad": "sad",
    "sorrow": "sad",
    "neu": "neutral",
    "neutral": "neutral",
}


_PUNCTUATION_MODELS: dict[str, PunctuationCapitalizationModel] = {}
_EMOTION_MODELS: dict[str, Any] = {}

_NUMBER_PRIMARY_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
    "trillion",
}
_NUMBER_CONNECTOR_WORDS = {"and"}
_NUMBER_DECIMAL_WORDS = {"point"}
_NUMBER_ALLOWED_WORDS = _NUMBER_PRIMARY_WORDS | _NUMBER_CONNECTOR_WORDS | _NUMBER_DECIMAL_WORDS
_NUMBER_DIGIT_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
}
_NUMBER_WORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(sorted(_NUMBER_ALLOWED_WORDS, key=len, reverse=True)) + r")(?:[\s-]+(?:"
    + "|".join(sorted(_NUMBER_ALLOWED_WORDS, key=len, reverse=True))
    + r"))*\b",
    re.IGNORECASE,
)

_NUMBER_SINGLE_SKIP_PRECEDERS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "another",
    "each",
    "every",
    "some",
    "any",
    "no",
    "my",
    "your",
    "our",
    "their",
    "his",
    "her",
    "its",
    "whose",
    "which",
}

_NUMBER_SINGLE_SKIP_FOLLOWERS = {
    "just",
    "like",
    "also",
    "too",
    "either",
    "neither",
}

_LANGUAGE_TOOLS: Dict[str, Any] = {}

_ORDINAL_BASE_MAP = {
    "first": "one",
    "second": "two",
    "third": "three",
    "fourth": "four",
    "fifth": "five",
    "sixth": "six",
    "seventh": "seven",
    "eighth": "eight",
    "ninth": "nine",
    "tenth": "ten",
    "eleventh": "eleven",
    "twelfth": "twelve",
    "thirteenth": "thirteen",
    "fourteenth": "fourteen",
    "fifteenth": "fifteen",
    "sixteenth": "sixteen",
    "seventeenth": "seventeen",
    "eighteenth": "eighteen",
    "nineteenth": "nineteen",
    "twentieth": "twenty",
    "thirtieth": "thirty",
    "fortieth": "forty",
    "fiftieth": "fifty",
    "sixtieth": "sixty",
    "seventieth": "seventy",
    "eightieth": "eighty",
    "ninetieth": "ninety",
    "hundredth": "hundred",
    "thousandth": "thousand",
    "millionth": "million",
    "billionth": "billion",
    "trillionth": "trillion",
    "zeroth": "zero",
}

_ORDINAL_ALLOWED_WORDS = _NUMBER_ALLOWED_WORDS | set(_ORDINAL_BASE_MAP.keys())
_ORDINAL_WORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(sorted(_ORDINAL_ALLOWED_WORDS, key=len, reverse=True)) + r")(?:[\s-]+(?:"
    + "|".join(sorted(_ORDINAL_ALLOWED_WORDS, key=len, reverse=True))
    + r"))*\b",
    re.IGNORECASE,
)

_FORWARD_SLASH_PATTERN = re.compile(r"\bforward[\s,-]*(?:slash|slashes)\b", re.IGNORECASE)

_FUZZY_SPAN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9'\-\s]{1,40}")


@dataclass
class AcronymRule:
    canonical: str
    patterns: list[re.Pattern[str]]
    fuzzy_stems: tuple[str, ...] = ()
    fuzzy_threshold: float = 0.82


_DEFAULT_ACRONYM_ENTRIES: list[dict[str, Any]] = [
    {
        "canonical": "Codex CLI",
        "aliases": [
            "codex cli",
            "codexcli",
            "codexcly",
            "codecly",
            "codecscly",
            "codex kli",
            "codex*cli",
        ],
        "fuzzy_stems": ["codexcli", "codexcly", "codecly", "codexcl"],
        "fuzzy_threshold": 0.8,
    },
    {
        "canonical": "Codex CLI's",
        "aliases": [
            "codex cli's",
            "codexcli's",
            "codexcly's",
            "codex*cli's",
        ],
        "fuzzy_stems": ["codexclis", "codexclis"],
        "fuzzy_threshold": 0.78,
    },
    {"canonical": "API", "aliases": ["api"]},
    {"canonical": "APIs", "aliases": ["apis"]},
    {"canonical": "CLI", "aliases": ["cli"]},
    {"canonical": "CLIs", "aliases": ["clis"]},
    {"canonical": "GPU", "aliases": ["gpu"]},
    {"canonical": "GPUs", "aliases": ["gpus"]},
    {"canonical": "SQL", "aliases": ["sql"]},
    {"canonical": "TDD", "aliases": ["tdd"]},
    {"canonical": "MCP", "aliases": ["mcp"]},
]


def canonicalize_emotion_label(label: str) -> str:
    key = label.strip().lower()
    return EMOTION_CANONICAL_MAP.get(key, key)


def _normalize_number_words(text: str) -> str:
    if not text:
        return text

    original_text = text

    def replace(match: re.Match[str]) -> str:
        phrase = match.group(0)
        raw_tokens = [tok for tok in re.split(r"[\s-]+", phrase.lower()) if tok]
        tokens = [re.sub(r"[^a-z]", "", tok) for tok in raw_tokens]
        tokens = [tok for tok in tokens if tok]
        if not tokens:
            return phrase
        if not any(tok in _NUMBER_PRIMARY_WORDS for tok in tokens):
            return phrase

        start, end = match.span()
        if len(tokens) == 1 and tokens[0] in _NUMBER_DIGIT_WORDS:
            prev_tokens = re.findall(r"[A-Za-z']+", original_text[:start])
            next_match = re.search(r"[A-Za-z']+", original_text[end:])
            prev_word = prev_tokens[-1].lower() if prev_tokens else ""
            next_word = next_match.group(0).lower() if next_match else ""
            if prev_word in _NUMBER_SINGLE_SKIP_PRECEDERS or next_word in _NUMBER_SINGLE_SKIP_FOLLOWERS:
                return phrase

        normalized_phrase = " ".join(tokens)
        if all(tok in _NUMBER_DIGIT_WORDS for tok in tokens):
            converted_tokens = []
            for tok in tokens:
                try:
                    converted_tokens.append(str(w2n.word_to_num(tok)))
                except ValueError:
                    return phrase
            return " ".join(converted_tokens)
        had_point = "point" in tokens
        try:
            value = w2n.word_to_num(normalized_phrase)
        except ValueError:
            return phrase
        if isinstance(value, float) and value.is_integer() and not had_point:
            value = int(value)
        if had_point and isinstance(value, int):
            return f"{value}.0"
        return str(value)

    return _NUMBER_WORD_PATTERN.sub(replace, text)




def _load_language_tool(language: str, log_path: Path):
    if language_tool_python is None:
        write_log("language_tool_python not installed; grammar cleanup disabled.", log_path)
        return None
    cached = _LANGUAGE_TOOLS.get(language)
    if cached is not None:
        return cached
    try:
        tool = language_tool_python.LanguageTool(language)
    except Exception as exc:  # noqa: BLE001
        write_log(f"Failed to initialize LanguageTool for {language}: {exc}", log_path)
        return None
    _LANGUAGE_TOOLS[language] = tool
    write_log(f"Loaded LanguageTool resources for {language}", log_path)
    return tool


def _ordinal_suffix(value: int) -> str:
    value_abs = abs(value)
    if 10 <= (value_abs % 100) <= 20:
        return "th"
    last_digit = value_abs % 10
    if last_digit == 1:
        return "st"
    if last_digit == 2:
        return "nd"
    if last_digit == 3:
        return "rd"
    return "th"


def _normalize_ordinal_words(text: str) -> str:
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        phrase = match.group(0)
        tokens = [tok for tok in re.split(r"[\s-]+", phrase.lower()) if tok]
        if not tokens:
            return phrase
        if not any(tok in _ORDINAL_BASE_MAP for tok in tokens):
            return phrase
        converted = [
            re.sub(r"[^a-z]", "", _ORDINAL_BASE_MAP.get(tok, tok))
            for tok in tokens
        ]
        converted = [tok for tok in converted if tok]
        if not converted:
            return phrase
        normalized_phrase = " ".join(converted)
        try:
            value = w2n.word_to_num(normalized_phrase)
        except ValueError:
            return phrase
        if isinstance(value, float):
            if not value.is_integer():
                return phrase
            value = int(value)
        suffix = _ordinal_suffix(int(value))
        return f"{int(value)}{suffix}"

    return _ORDINAL_WORD_PATTERN.sub(replace, text)


def _normalize_slash_phrases(text: str) -> str:
    if not text:
        return text

    replaced = _FORWARD_SLASH_PATTERN.sub("/", text)
    replaced = re.sub(r"\s*/\s*", " / ", replaced)
    return re.sub(r"\s{2,}", " ", replaced)




def _fix_trailing_artifacts(text: str, force_terminal: Optional[str]) -> str:
    if not text:
        return text

    stripped = text.rstrip()
    if stripped.endswith('/') and not force_terminal:
        base = stripped[:-1].rstrip()
        if base and '?' not in base:
            tokens = re.findall(r"[A-Za-z']+", base.lower())
            if tokens:
                question_leads = {
                    "who",
                    "what",
                    "when",
                    "where",
                    "why",
                    "how",
                    "do",
                    "does",
                    "did",
                    "can",
                    "could",
                    "should",
                    "would",
                    "will",
                    "is",
                    "are",
                    "am",
                    "have",
                    "has",
                    "may",
                    "might",
                }
                if tokens[0] in question_leads or tokens[-1] in {"who", "what", "where", "why", "how"}:
                    suffix = text[len(stripped):]
                    text = base + '?' + suffix
    return text




def _sanitize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


def _compile_acronym_pattern(alias: str) -> re.Pattern[str]:
    alias = alias.strip()
    if not alias:
        raise ValueError("Acronym alias may not be empty")
    escaped = re.escape(alias)
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\*", r"[A-Za-z0-9]*")
    pattern = rf"(?i)(?<!\w){escaped}(?!\w)"
    return re.compile(pattern)


def _load_acronym_rules(config_path: Path, log_path: Path) -> list[AcronymRule]:
    entries: list[dict[str, Any]] = []
    if yaml is not None and config_path.exists():
        try:
            data = yaml.safe_load(config_path.read_text())
            if isinstance(data, dict):
                entries = data.get("acronyms", []) or []
            else:
                write_log(f"Acronym config {config_path} must define a top-level mapping", log_path)
        except Exception as exc:  # noqa: BLE001
            write_log(f"Failed to parse acronym config {config_path}: {exc}", log_path)
    if not entries:
        entries = _DEFAULT_ACRONYM_ENTRIES
        write_log(
            f"Using built-in acronym defaults (config missing or empty at {config_path})",
            log_path,
        )

    rules: list[AcronymRule] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        canonical = str(entry.get("canonical", "")).strip()
        if not canonical:
            continue
        aliases = entry.get("aliases", []) or []
        patterns: list[re.Pattern[str]] = []
        for alias in aliases:
            try:
                patterns.append(_compile_acronym_pattern(str(alias)))
            except ValueError:
                write_log(f"Ignoring empty alias for acronym {canonical}", log_path)
        fuzzy_sources = entry.get("fuzzy_stems", []) or []
        fuzzy_stems = tuple(_sanitize_token(str(src)) for src in fuzzy_sources if str(src).strip())
        threshold = float(entry.get("fuzzy_threshold", 0.82))
        if not patterns and not fuzzy_stems:
            continue
        rules.append(
            AcronymRule(
                canonical=canonical,
                patterns=patterns,
                fuzzy_stems=fuzzy_stems,
                fuzzy_threshold=threshold,
            )
        )
    return rules

def ensure_speechbrain_wav2vec_shim() -> None:
    target_pkg = "speechbrain.lobes.models.huggingface_transformers"
    target_module = f"{target_pkg}.wav2vec2"
    if target_module in sys.modules:
        return
    try:
        from speechbrain.lobes.models import huggingface_wav2vec  # type: ignore[attr-defined]
        import speechbrain.lobes.models as sb_models  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - optional dependency
        return

    wav2vec_cls = getattr(huggingface_wav2vec, "HuggingFaceWav2Vec2", None)
    if wav2vec_cls is None:
        return

    pkg_module = sys.modules.get(target_pkg)
    if pkg_module is None:
        pkg_module = types.ModuleType(target_pkg)
        sys.modules[target_pkg] = pkg_module
    if not hasattr(pkg_module, "__path__"):
        pkg_module.__path__ = []  # type: ignore[attr-defined]

    shim_module = types.ModuleType(target_module)
    setattr(shim_module, "Wav2Vec2", wav2vec_cls)
    sys.modules[target_module] = shim_module

    setattr(pkg_module, "wav2vec2", shim_module)
    if not hasattr(sb_models, "huggingface_transformers"):
        setattr(sb_models, "huggingface_transformers", pkg_module)
    if hasattr(sb_models, "__all__") and "huggingface_transformers" not in sb_models.__all__:
        sb_models.__all__.append("huggingface_transformers")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parakeet push-to-talk dictation")
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Explicit PortAudio input device index to use",
    )
    parser.add_argument(
        "--device-keyword",
        default=None,
        help="Substring to match desired input device name (default: use system default)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Recording sample rate",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Seconds of silence after key release to continue recording before stopping",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="NeMo model identifier",
    )
    parser.add_argument(
        "--append-space",
        action="store_true",
        help="Append a trailing space to emitted text",
    )
    parser.add_argument(
        "--rms-threshold",
        type=float,
        default=1.5e-5,
        help="Minimum RMS level to treat as speech (lower = more sensitive)",
    )
    parser.add_argument(
        "--preamp",
        type=float,
        default=1.8,
        help="Multiplier applied to audio samples before transcription",
    )
    parser.add_argument(
        "--allow-esc",
        action="store_true",
        help="Permit exiting the listener when ESC is pressed (disabled by default)",
    )
    parser.add_argument(
        "--log",
        default=str(DEFAULT_LOG),
        help="Log file path",
    )
    parser.add_argument(
        "--cookie",
        default=str(DEFAULT_COOKIE),
        help="PID file used to avoid duplicate instances",
    )
    parser.add_argument(
        "--no-auto-mute",
        action="store_false",
        dest="auto_mute",
        help="Disable automatic muting of system audio while recording",
    )
    parser.set_defaults(auto_mute=True)
    parser.add_argument(
        "--decoder-config",
        default=str(DEFAULT_DECODER_CONFIG),
        help="Path to YAML file containing decoder presets",
    )
    parser.add_argument(
        "--decoder-preset",
        default=DEFAULT_DECODER_PRESET,
        help="Name of the decoder preset to apply (see decoder config)",
    )
    parser.add_argument(
        "--lm-path",
        default=None,
        help="Override KenLM binary path for decoding",
    )
    parser.add_argument(
        "--lexicon-path",
        default=None,
        help="Override pronunciation lexicon path",
    )
    parser.add_argument(
        "--hotword-path",
        default=None,
        help="Override hotword TSV for runtime boosts",
    )
    parser.add_argument(
        "--disable-decoder-tuning",
        action="store_true",
        help="Skip Flashlight/LM configuration and rely on the model default decoder",
    )
    parser.add_argument(
        "--punctuation-model",
        default=DEFAULT_PUNCTUATION_MODEL,
        help="Pretrained NeMo punctuation+capitalization model to apply",
    )
    parser.add_argument(
        "--disable-punctuation",
        action="store_true",
        help="Disable punctuation and capitalization post-processing",
    )
    parser.add_argument(
        "--emotion-model",
        default=DEFAULT_EMOTION_MODEL,
        help="Speech emotion recognition model to load (SpeechBrain foreign class string)",
    )
    parser.add_argument(
        "--disable-emotion",
        action="store_true",
        help="Disable emotion detection and emphasis adjustments",
    )
    parser.add_argument(
        "--emotion-threshold",
        type=float,
        default=DEFAULT_EMOTION_THRESHOLD,
        help="Confidence threshold [0-1] before applying emotion-driven punctuation",
    )
    parser.add_argument(
        "--emotion-tag",
        action="store_true",
        help="Prefix transcripts with the dominant emotion label when above threshold",
    )
    parser.add_argument(
        "--disable-number-normalization",
        action="store_true",
        help="Skip converting spoken number words (e.g., 'eight point zero') into digits",
    )
    parser.add_argument(
        "--disable-voice-isolation",
        action="store_true",
        help="Disable the DTLN voice isolation preprocessor",
    )
    parser.add_argument(
        "--voice-isolation-model-dir",
        default=str(DEFAULT_VOICE_ISOLATION_MODEL_DIR),
        help="Directory containing DTLN ONNX models (model_1.onnx and model_2.onnx)",
    )
    parser.add_argument(
        "--disable-grammar-cleanup",
        action="store_true",
        help="Skip LanguageTool grammar and punctuation cleanup",
    )
    parser.add_argument(
        "--grammar-language",
        default=DEFAULT_GRAMMAR_LANGUAGE,
        help="LanguageTool locale to use for grammar cleanup (default: en-US)",
    )
    parser.add_argument(
        "--acronym-config",
        default=str(DEFAULT_ACRONYM_CONFIG),
        help="YAML file defining custom acronym normalization rules",
    )
    parser.add_argument(
        "--speech-min-duration-ms",
        type=float,
        default=250.0,
        help="Minimum cumulative speech duration (ms) required before transcription",
    )
    parser.add_argument(
        "--speech-rms-delta-db",
        type=float,
        default=8.0,
        help="Minimum peak vs silence energy delta (dB) required before transcription",
    )
    parser.add_argument(
        "--speech-min-total-ms",
        type=float,
        default=0.0,
        help="Optional minimum total clip duration (ms); 0 disables this check",
    )
    parser.add_argument(
        "--speech-min-ratio",
        type=float,
        default=0.35,
        help="Minimum fraction of the clip that must contain speech energy (0-1)",
    )
    parser.add_argument(
        "--speech-min-streak-ms",
        type=float,
        default=150.0,
        help="Minimum contiguous speech streak (ms) required before transcription",
    )
    return parser.parse_args()


def write_log(msg: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{timestamp}] {msg}\n")


def deliver_to_runelite(message: str, log_path: Path) -> bool:
    message = message.strip()
    if not message:
        write_log("RuneLite delivery skipped because message was empty after trimming", log_path)
        return False
    try:
        search_output = subprocess.check_output(["xdotool", "search", "--name", "RuneLite"], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        write_log("RuneLite window not found via xdotool search", log_path)
        return False
    window_ids = [line.strip() for line in search_output.decode().splitlines() if line.strip()]
    if not window_ids:
        write_log("RuneLite search returned no window IDs", log_path)
        return False
    target_window = window_ids[-1]
    try:
        previous_focus = subprocess.check_output(["xdotool", "getwindowfocus"], stderr=subprocess.DEVNULL).strip().decode()
    except subprocess.CalledProcessError:
        previous_focus = ""

    subprocess.run(["xdotool", "windowactivate", "--sync", target_window], check=False)
    time.sleep(0.12)

    ydotool_bin = shutil.which("ydotool")
    if ydotool_bin:
        yd_success = True
        for cmd in ([ydotool_bin, "type", message],):
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                yd_success = False
                break
            time.sleep(0.05)
        if yd_success:
            if previous_focus and previous_focus != target_window:
                subprocess.run(["xdotool", "windowactivate", "--sync", previous_focus], check=False)
            write_log(f"RuneLite message typed via ydotool (manual confirmation required): {message!r}", log_path)
            return True
        write_log("ydotool delivery failed; falling back to xdotool", log_path)

    subprocess.run(
        [
            "xdotool",
            "type",
            "--clearmodifiers",
            "--delay",
            "15",
            "--window",
            target_window,
            "--",
            message,
        ],
        check=False,
    )
    time.sleep(0.08)
    if previous_focus and previous_focus != target_window:
        subprocess.run(["xdotool", "windowactivate", "--sync", previous_focus], check=False)
    write_log(f"RuneLite message typed via xdotool (manual confirmation required): {message!r}", log_path)
    return True


def ensure_single_instance(cookie_path: Path) -> None:
    if cookie_path.exists():
        try:
            pid = int(cookie_path.read_text().strip())
        except (ValueError, OSError):
            pid = None
        else:
            if pid and Path(f"/proc/{pid}").exists():
                print("Another instance is already running", file=sys.stderr)
                sys.exit(1)
    cookie_path.write_text(str(os.getpid()))
    atexit.register(lambda: cookie_path.unlink(missing_ok=True))


def _load_decoder_presets(config_path: Path) -> Dict[str, Dict[str, Any]]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to parse decoder preset files. Install pyyaml to continue.")
    if not config_path.exists():
        raise FileNotFoundError(f"Decoder preset file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "presets" not in data:
        raise ValueError(f"Decoder preset file {config_path} is missing a top-level 'presets' mapping")
    presets = data["presets"]
    if not isinstance(presets, dict):
        raise ValueError("Decoder presets must be a mapping of preset names to configuration blocks")
    return presets


def _resolve_resource(path_hint: str | None) -> Path | None:
    if not path_hint:
        return None
    path = Path(path_hint).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _load_punctuation_model(name: str, device: torch.device, log_path: Path) -> PunctuationCapitalizationModel | None:
    if PunctuationCapitalizationModel is None:
        write_log("Punctuation model requested but NeMo NLP extras are unavailable.", log_path)
        return None
    cached = _PUNCTUATION_MODELS.get(name)
    if cached is not None:
        return cached
    try:
        model = PunctuationCapitalizationModel.from_pretrained(name)
    except Exception as exc:  # noqa: BLE001
        write_log(f"Failed to load punctuation model {name}: {exc}", log_path)
        return None
    model = model.to(device)
    model.eval()
    _PUNCTUATION_MODELS[name] = model
    write_log(f"Loaded punctuation model {name} on {device}", log_path)
    return model


def _load_emotion_model(name: str, device: torch.device, log_path: Path) -> Optional[Any]:
    if foreign_class is None:
        write_log("SpeechBrain is not installed; skipping emotion detection.", log_path)
        return None
    cached = _EMOTION_MODELS.get(name)
    if cached is not None:
        return cached
    device_str = str(device)
    ensure_speechbrain_wav2vec_shim()
    try:
        classifier = foreign_class(
            source=name,
            pymodule_file=str((REPO_ROOT / "speechbrain_modules" / "custom_interface.py").resolve()),
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": device_str},
        )
    except Exception as exc:  # noqa: BLE001
        write_log(f"Failed to load emotion model {name}: {exc}", log_path)
        return None
    _EMOTION_MODELS[name] = classifier
    write_log(f"Loaded emotion model {name} on {device_str}", log_path)
    return classifier


def _configure_decoder_from_preset(
    model: ASRModel,
    preset: Dict[str, Any],
    log_path: Path,
    lm_override: str | None = None,
    lexicon_override: str | None = None,
    hotword_override: str | None = None,
) -> None:
    if OmegaConf is None:
        raise RuntimeError("omegaconf is required to configure the decoder. Please install omegaconf.")

    beam_size = int(preset.get("beam_size", 32))
    beam_size_token = int(preset.get("beam_size_token", beam_size))
    beam_threshold = float(preset.get("beam_threshold", 25.0))
    lm_weight = float(preset.get("lm_weight", 2.0))
    word_bonus = float(preset.get("word_insertion_bonus", 0.0))
    silence_weight = float(preset.get("silence_weight", 0.0))
    unknown_weight = float(preset.get("unk_weight", -10.0))

    lm_path = _resolve_resource(lm_override or preset.get("lm_binary"))
    lexicon_path = _resolve_resource(lexicon_override or preset.get("lexicon"))
    hotword_path = _resolve_resource(hotword_override or preset.get("hotwords"))

    if lm_path is None or not lm_path.exists():
        raise FileNotFoundError(
            "KenLM binary not found. Provide one via decoder preset or --lm-path override."
        )
    if lexicon_path is None or not lexicon_path.exists():
        raise FileNotFoundError(
            "Lexicon TSV not found. Provide one via decoder preset or --lexicon-path override."
        )
    if hotword_path is not None and not hotword_path.exists():
        write_log(f"Hotword file {hotword_path} does not exist; continuing without boosts", log_path)
        hotword_path = None

    decoding_cfg = {
        "strategy": "flashlight",
        "beam": {
            "search_type": "flashlight",
            "beam_size": beam_size,
            "beam_size_token": beam_size_token,
            "beam_threshold": beam_threshold,
            "kenlm_path": str(lm_path),
            "beam_alpha": lm_weight,
            "beam_beta": word_bonus,
            "flashlight_cfg": {
                "lexicon_path": str(lexicon_path),
                "beam_size_token": beam_size_token,
                "beam_threshold": beam_threshold,
                "lm_weight": lm_weight,
                "word_score": word_bonus,
                "sil_weight": silence_weight,
                "unk_weight": unknown_weight,
            },
        },
    }

    if hotword_path is not None:
        decoding_cfg["beam"]["flashlight_cfg"]["boost_path"] = str(hotword_path)

    # Allow presets to toggle timestamps from the decoder side
    if "use_timestamps" in preset:
        decoding_cfg["use_timestamps"] = bool(preset["use_timestamps"])

    cfg = OmegaConf.create(decoding_cfg)
    model.change_decoding_strategy(cfg)
    write_log(
        "Applied Flashlight decoder preset with KenLM={}, lexicon={}, hotwords={}".format(
            lm_path, lexicon_path, hotword_path or "<none>"
        ),
        log_path,
    )


def resolve_input_device(keyword: str | None, explicit_index: int | None) -> int:
    if explicit_index is not None:
        return explicit_index

    devices = sd.query_devices()
    if keyword:
        for idx, dev in enumerate(devices):
            name = dev.get("name", "")
            if keyword.lower() in name.lower() and dev.get("max_input_channels", 0) > 0:
                return idx

    default_input, _ = sd.default.device
    if default_input is not None:
        return int(default_input)

    available = [d.get("name", "<unknown>") for d in devices if d.get("max_input_channels", 0) > 0]
    msg = "No matching input device found. Available inputs: " + ", ".join(available)
    raise RuntimeError(msg)


class Recorder:
    def __init__(self, sample_rate: int, device_index: int, log_path: Path, rms_threshold: float, preamp: float):
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.log_path = log_path
        self.rms_threshold = rms_threshold
        self.preamp = preamp
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._writer: sf.SoundFile | None = None
        self._file: Path | None = None
        self._running = False
        self._worker: threading.Thread | None = None
        self._last_audio_time = 0.0
        self._speech_duration_ms = 0.0
        self._total_duration_ms = 0.0
        self._max_rms = 0.0
        self._silence_rms_sum = 0.0
        self._silence_block_count = 0
        self._speech_block_count = 0
        self._total_block_count = 0
        self._max_speech_streak_ms = 0.0
        self._current_speech_streak_ms = 0.0
        self._last_capture_stats: Dict[str, float | int | None] | None = None

    def start(self) -> Path:
        if self._running:
            return self._file  # type: ignore

        tmp_dir = Path(tempfile.gettempdir()) / "parakeet_ptt"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self._file = Path(tempfile.mkstemp(suffix=".wav", dir=tmp_dir)[1])
        self._writer = sf.SoundFile(
            str(self._file), mode="w", samplerate=self.sample_rate, channels=1, subtype="PCM_16"
        )

        def audio_callback(indata, frames, time_info, status):
            if status:
                write_log(f"Audio status: {status}", self.log_path)
            self._queue.put(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            device=self.device_index,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        )
        self._stream.start()
        self._running = True
        self._last_audio_time = time.time()
        self._speech_duration_ms = 0.0
        self._total_duration_ms = 0.0
        self._max_rms = 0.0
        self._silence_rms_sum = 0.0
        self._silence_block_count = 0
        self._speech_block_count = 0
        self._total_block_count = 0
        self._max_speech_streak_ms = 0.0
        self._current_speech_streak_ms = 0.0
        self._last_capture_stats = None

        def worker():
            while self._running or not self._queue.empty():
                try:
                    block = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self.preamp != 1.0:
                    block = np.clip(block * self.preamp, -1.0, 1.0)
                self._writer.write(block)
                rms = float(np.sqrt(np.mean(np.square(block), dtype=np.float64)))
                frames = int(block.shape[0]) if block.ndim > 0 else 0
                block_ms = (frames / self.sample_rate) * 1000.0 if frames > 0 else 0.0
                self._total_duration_ms += block_ms
                self._total_block_count += 1
                self._max_rms = max(self._max_rms, rms)
                if rms > self.rms_threshold:
                    self._last_audio_time = time.time()
                    self._speech_duration_ms += block_ms
                    self._speech_block_count += 1
                    self._current_speech_streak_ms += block_ms
                    if self._current_speech_streak_ms > self._max_speech_streak_ms:
                        self._max_speech_streak_ms = self._current_speech_streak_ms
                else:
                    if rms > 0.0:
                        self._silence_rms_sum += rms
                        self._silence_block_count += 1
                    self._current_speech_streak_ms = 0.0

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()
        write_log("Recorder started", self.log_path)
        return self._file

    def stop(self, timeout: float) -> Path | None:
        if not self._running:
            return None
        deadline = self._last_audio_time + timeout
        while time.time() < deadline:
            time.sleep(0.05)
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._worker is not None:
            self._worker.join()
            self._worker = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        avg_silence_rms = None
        if self._silence_block_count > 0:
            avg_silence_rms = self._silence_rms_sum / self._silence_block_count
        speech_max_db = 20 * math.log10(self._max_rms) if self._max_rms > 0 else None
        silence_avg_db = (
            20 * math.log10(avg_silence_rms) if avg_silence_rms and avg_silence_rms > 0 else None
        )
        delta_db = (
            speech_max_db - silence_avg_db if speech_max_db is not None and silence_avg_db is not None else None
        )
        speech_ratio = (self._speech_duration_ms / self._total_duration_ms) if self._total_duration_ms > 0 else None
        self._last_capture_stats = {
            "speech_ms": self._speech_duration_ms,
            "total_ms": self._total_duration_ms,
            "speech_blocks": self._speech_block_count,
            "total_blocks": self._total_block_count,
            "speech_max_db": speech_max_db,
            "silence_avg_db": silence_avg_db,
            "speech_delta_db": delta_db,
            "speech_ratio": speech_ratio,
            "max_speech_streak_ms": self._max_speech_streak_ms,
        }
        write_log("Recorder stopped", self.log_path)
        return self._file

    def last_capture_stats(self) -> Dict[str, float | int | None] | None:
        return self._last_capture_stats


class AudioMuteError(Exception):
    """Raised when system audio mute operations fail."""


class AudioMuteController:
    """Wrap wpctl/pactl to mute desktop audio during capture."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._strategy: "_BaseMuteStrategy | None" = self._detect_strategy()

    def _detect_strategy(self) -> "_BaseMuteStrategy | None":
        if shutil.which("wpctl"):
            return _WpctlStrategy(self.log_path)
        if shutil.which("pactl"):
            return _PactlStrategy(self.log_path)
        write_log("Auto-mute unavailable: wpctl/pactl not found", self.log_path)
        return None

    def mute(self) -> None:
        if not self._strategy:
            return
        try:
            self._strategy.mute()
        except AudioMuteError as exc:
            write_log(f"Failed to mute system audio: {exc}", self.log_path)
            self._strategy = None

    def restore(self) -> None:
        if not self._strategy:
            return
        try:
            self._strategy.restore()
        except AudioMuteError as exc:
            write_log(f"Failed to restore system audio: {exc}", self.log_path)
            self._strategy = None


class _BaseMuteStrategy:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._active = False
        self._previously_muted: bool | None = None

    def mute(self) -> None:
        raise NotImplementedError

    def restore(self) -> None:
        raise NotImplementedError


class _WpctlStrategy(_BaseMuteStrategy):
    _TARGET = "@DEFAULT_AUDIO_SINK@"

    def mute(self) -> None:
        if self._active:
            return
        self._previously_muted = self._read_muted()
        try:
            subprocess.run(["wpctl", "set-mute", self._TARGET, "1"], check=True)
        except subprocess.CalledProcessError as exc:
            raise AudioMuteError(f"wpctl set-mute failed: {exc}") from exc
        self._active = True
        write_log("Muted system audio via wpctl", self.log_path)

    def restore(self) -> None:
        if not self._active:
            return
        try:
            if self._previously_muted is False:
                subprocess.run(["wpctl", "set-mute", self._TARGET, "0"], check=True)
                write_log("Restored system audio via wpctl", self.log_path)
            else:
                write_log("System audio was muted before recording; leaving muted", self.log_path)
        except subprocess.CalledProcessError as exc:
            raise AudioMuteError(f"wpctl set-mute restore failed: {exc}") from exc
        finally:
            self._active = False
            self._previously_muted = None

    def _read_muted(self) -> bool | None:
        try:
            output = subprocess.check_output(["wpctl", "get-volume", self._TARGET], text=True)
        except subprocess.CalledProcessError as exc:
            raise AudioMuteError(f"wpctl get-volume failed: {exc}") from exc
        normalized = output.strip().lower()
        if "muted:" in normalized:
            return "muted: yes" in normalized
        if "[muted]" in normalized:
            return True
        return False


class _PactlStrategy(_BaseMuteStrategy):
    def __init__(self, log_path: Path):
        super().__init__(log_path)
        self._sink = self._detect_sink()

    def mute(self) -> None:
        if self._active:
            return
        self._previously_muted = self._read_muted()
        try:
            subprocess.run(["pactl", "set-sink-mute", self._sink, "1"], check=True)
        except subprocess.CalledProcessError as exc:
            raise AudioMuteError(f"pactl set-sink-mute failed: {exc}") from exc
        self._active = True
        write_log(f"Muted system audio via pactl (sink {self._sink})", self.log_path)

    def restore(self) -> None:
        if not self._active:
            return
        try:
            if self._previously_muted is False:
                subprocess.run(["pactl", "set-sink-mute", self._sink, "0"], check=True)
                write_log(f"Restored system audio via pactl (sink {self._sink})", self.log_path)
            else:
                write_log("System audio was muted before recording; leaving muted", self.log_path)
        except subprocess.CalledProcessError as exc:
            raise AudioMuteError(f"pactl set-sink-mute restore failed: {exc}") from exc
        finally:
            self._active = False
            self._previously_muted = None

    def _read_muted(self) -> bool | None:
        try:
            output = subprocess.check_output(["pactl", "get-sink-mute", self._sink], text=True)
        except subprocess.CalledProcessError as exc:
            raise AudioMuteError(f"pactl get-sink-mute failed: {exc}") from exc
        normalized = output.strip().lower()
        if "yes" in normalized:
            return True
        if "no" in normalized:
            return False
        return False

    def _detect_sink(self) -> str:
        try:
            sink = subprocess.check_output(["pactl", "get-default-sink"], text=True).strip()
            if sink:
                return sink
        except subprocess.CalledProcessError:
            pass
        try:
            output = subprocess.check_output(["pactl", "list", "short", "sinks"], text=True)
        except subprocess.CalledProcessError as exc:
            raise AudioMuteError(f"pactl list short sinks failed: {exc}") from exc
        for line in output.splitlines():
            parts = line.split("\t")
            if parts:
                return parts[0]
        raise AudioMuteError("No PulseAudio sinks found")


class ParakeetPTT:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.log_path = Path(args.log)
        try:
            self.device_index = resolve_input_device(args.device_keyword, args.device_index)
        except Exception as exc:
            raise SystemExit(f"Failed to select input device: {exc}")
        info = sd.query_devices(self.device_index)
        write_log(
            (
                f"Using input device index {self.device_index}: {info.get('name')} "
                f"(inputs={info.get('max_input_channels')}, outputs={info.get('max_output_channels')})"
            ),
            self.log_path,
        )
        self.model = self._load_model(args.model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.disable_punctuation:
            write_log("Punctuation post-processing disabled via flag", self.log_path)
            self.punctuation_model = None
        else:
            self.punctuation_model = _load_punctuation_model(args.punctuation_model, device, self.log_path)
            if self.punctuation_model is None:
                write_log("Continuing without punctuation post-processing", self.log_path)
        if args.disable_emotion:
            write_log("Emotion detection disabled via flag", self.log_path)
            self.emotion_model = None
        else:
            self.emotion_model = _load_emotion_model(args.emotion_model, device, self.log_path)
        self.emotion_threshold = args.emotion_threshold
        self.emotion_tag = args.emotion_tag
        self.normalize_numbers = not args.disable_number_normalization
        if args.disable_number_normalization:
            write_log("Number normalization disabled via flag", self.log_path)
        if args.disable_voice_isolation:
            self.voice_isolator = None
            write_log("Voice isolation disabled via flag", self.log_path)
        else:
            try:
                model_dir = Path(args.voice_isolation_model_dir)
                self.voice_isolator = DtlnVoiceIsolation(
                    model_dir / "model_1.onnx",
                    model_dir / "model_2.onnx",
                )
                write_log(
                    f"Voice isolation enabled with models from {args.voice_isolation_model_dir}",
                    self.log_path,
                )
            except Exception as exc:  # noqa: BLE001
                self.voice_isolator = None
                write_log(f"Failed to initialize voice isolation: {exc}", self.log_path)
        self.grammar_language = args.grammar_language
        self.grammar_tool = None
        if args.disable_grammar_cleanup:
            write_log("Grammar cleanup disabled via flag", self.log_path)
        else:
            tool = _load_language_tool(self.grammar_language, self.log_path)
            if tool is None:
                write_log("Continuing without grammar cleanup", self.log_path)
            else:
                self.grammar_tool = tool
        self.last_emotion: Optional[Dict[str, Any]] = None
        self.recorder = Recorder(args.sample_rate, self.device_index, self.log_path, args.rms_threshold, args.preamp)
        self.recording = False
        self.lock = threading.Lock()
        self.allow_esc = args.allow_esc
        self.audio_muter = AudioMuteController(self.log_path) if args.auto_mute else None
        if not args.auto_mute:
            write_log("Auto-mute disabled via CLI flag", self.log_path)

        acronym_config = Path(args.acronym_config).expanduser()
        self.acronym_rules = _load_acronym_rules(acronym_config, self.log_path)
        self.speech_min_duration_ms = max(0.0, args.speech_min_duration_ms)
        self.speech_rms_delta_db = max(0.0, args.speech_rms_delta_db)
        self.speech_min_total_ms = max(0.0, args.speech_min_total_ms)
        self.speech_min_ratio = min(max(args.speech_min_ratio, 0.0), 1.0)
        self.speech_min_streak_ms = max(0.0, args.speech_min_streak_ms)

    def _load_model(self, model_name: str) -> ASRModel:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        write_log(f"Loading model {model_name} on {device}", self.log_path)
        model = ASRModel.from_pretrained(model_name=model_name, map_location=device)
        model.eval()
        if self.args.disable_decoder_tuning:
            write_log("Decoder tuning disabled; using model default decoding strategy", self.log_path)
            return model

        try:
            presets = _load_decoder_presets(Path(self.args.decoder_config).expanduser())
        except Exception as exc:  # noqa: BLE001
            write_log(f"Unable to load decoder presets ({exc}); using default decoder", self.log_path)
            return model

        preset = presets.get(self.args.decoder_preset)
        if preset is None:
            write_log(
                f"Decoder preset '{self.args.decoder_preset}' not found in {self.args.decoder_config}; using default decoder",
                self.log_path,
            )
            return model

        try:
            _configure_decoder_from_preset(
                model,
                preset,
                self.log_path,
                lm_override=self.args.lm_path,
                lexicon_override=self.args.lexicon_path,
                hotword_override=self.args.hotword_path,
            )
        except Exception as exc:  # noqa: BLE001
            write_log(f"Failed to apply Flashlight decoder preset: {exc}", self.log_path)
        return model

    def _evaluate_emotion(self, audio_path: Path) -> Optional[Dict[str, Any]]:
        if self.emotion_model is None:
            return None
        try:
            raw = self.emotion_model.classify_file(str(audio_path))
        except Exception as exc:  # noqa: BLE001
            write_log(f"Emotion model inference failed: {exc}", self.log_path)
            return None

        labels: Optional[list[str]] = None
        raw_mapping: Dict[str, float] = {}

        if isinstance(raw, dict):
            labels = raw.get("labels") or raw.get("classes")
            scores = raw.get("scores")
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().tolist()
            if isinstance(scores, (list, tuple)) and scores and isinstance(scores[0], (list, tuple)):
                scores = scores[0]
            if labels and isinstance(scores, (list, tuple)):
                raw_mapping = {str(label): float(score) for label, score in zip(labels, scores)}
        elif isinstance(raw, (tuple, list)) and len(raw) >= 4:
            out_prob, score_tensor, _, text_lab = raw
            if isinstance(out_prob, torch.Tensor):
                probs_tensor = torch.softmax(out_prob, dim=-1) if out_prob.dim() > 1 else torch.softmax(out_prob.unsqueeze(0), dim=-1)
                probs = probs_tensor.squeeze(0).detach().cpu().tolist()
            else:
                probs = out_prob
            label_encoder = getattr(getattr(self.emotion_model, "hparams", None), "label_encoder", None)
            decoded_labels: list[str] | None = None
            if label_encoder is not None:
                try:
                    decoded = label_encoder.decode_ndim(list(range(len(probs))))
                    decoded_labels = [str(item) for item in decoded]
                except Exception:  # pragma: no cover - defensive fallback
                    decoded_labels = None
            if decoded_labels is None:
                decoded_labels = [str(idx) for idx in range(len(probs))]
            raw_mapping = {label: float(prob) for label, prob in zip(decoded_labels, probs)}

        if not raw_mapping:
            write_log("Emotion model returned unsupported format", self.log_path)
            return None

        canonical_scores: Dict[str, float] = {}
        for raw_label, value in raw_mapping.items():
            canonical = canonicalize_emotion_label(raw_label)
            current = canonical_scores.get(canonical)
            if current is None or value > current:
                canonical_scores[canonical] = value

        if not canonical_scores:
            write_log("Emotion model produced labels but none could be canonicalized", self.log_path)
            return None

        label, score = max(canonical_scores.items(), key=lambda item: item[1])
        raw_label = next((rl for rl, val in raw_mapping.items() if canonicalize_emotion_label(rl) == label and val == score), None)
        if raw_label is None:
            raw_label = next(iter(raw_mapping))

        info = {
            "label": str(label),
            "score": float(score),
            "scores": canonical_scores,
            "raw_label": str(raw_label),
            "raw_scores": raw_mapping,
        }
        self.last_emotion = info
        if raw_label and raw_label != label:
            write_log(
                f"Emotion detection: {label} ({score:.2f}) [raw={raw_label}]",
                self.log_path,
            )
        else:
            write_log(f"Emotion detection: {label} ({score:.2f})", self.log_path)
        if score < self.emotion_threshold:
            ordered = sorted(canonical_scores.items(), key=lambda item: item[1], reverse=True)
            summary = ", ".join(f"{lbl}:{val:.2f}" for lbl, val in ordered[:4])
            write_log(
                f"Emotion scores below threshold ({self.emotion_threshold:.2f}): {summary}",
                self.log_path,
            )
        return info

    def _apply_emotion_rules(self, text: str, info: Dict[str, Any]) -> tuple[str, Optional[str]]:
        label = canonicalize_emotion_label(str(info.get("label", ""))).lower()
        score = float(info.get("score", 0.0))
        force_terminal: Optional[str] = None

        if self.emotion_tag and label:
            tag = label.upper()
            text = f"[{tag}] {text}" if text else f"[{tag}]"

        if score >= self.emotion_threshold:
            if label in EMOTION_EXCLAIM_LABELS and text:
                force_terminal = "!"
                if text[-1] not in {"!", "?"}:
                    text = text.rstrip()
                    text = text.rstrip(". ")
                    text += "!"
        return text, force_terminal


    def _apply_grammar_cleanup(self, text: str) -> str:
        if self.grammar_tool is None or language_tool_python is None:
            return text
        try:
            matches = self.grammar_tool.check(text)
        except Exception as exc:  # noqa: BLE001
            write_log(f"Grammar cleanup failed (check): {exc}", self.log_path)
            return text
        if not matches:
            return text
        try:
            corrected = language_tool_python.utils.correct(text, matches)
        except Exception as exc:  # noqa: BLE001
            write_log(f"Grammar cleanup failed (apply): {exc}", self.log_path)
            return text
        corrected = corrected.strip()
        if corrected != text:
            write_log(f"Grammar cleanup applied: {text!r} -> {corrected!r}", self.log_path)
        return corrected


    def _apply_acronym_formatting(self, text: str) -> str:
        if not getattr(self, "acronym_rules", None):
            return text

        updated = text

        for rule in self.acronym_rules:
            for pattern in rule.patterns:
                updated = pattern.sub(rule.canonical, updated)

            if rule.fuzzy_stems:
                def fuzzy_replace(match: re.Match[str]) -> str:
                    phrase = match.group(0)
                    sanitized = _sanitize_token(phrase)
                    canonical_norm = _sanitize_token(rule.canonical)
                    if not sanitized or sanitized == canonical_norm:
                        return phrase
                    for stem in rule.fuzzy_stems:
                        if not stem:
                            continue
                        ratio = difflib.SequenceMatcher(None, sanitized, stem).ratio()
                        if ratio >= rule.fuzzy_threshold:
                            return rule.canonical
                    return phrase

                updated = _FUZZY_SPAN_PATTERN.sub(fuzzy_replace, updated)

        if updated != text:
            old_preview = text if len(text) <= 120 else text[:117] + '...'
            new_preview = updated if len(updated) <= 120 else updated[:117] + '...'
            write_log(f"Acronym formatting applied: {old_preview!r} -> {new_preview!r}", self.log_path)
        return updated

    def _should_transcribe(self, stats: Dict[str, float | int | None] | None) -> bool:
        if stats is None:
            write_log("No capture statistics collected; defaulting to transcription", self.log_path)
            return True
        total_ms = float(stats.get("total_ms") or 0.0)
        speech_ms = float(stats.get("speech_ms") or 0.0)
        speech_ratio = stats.get("speech_ratio")
        speech_ratio = float(speech_ratio) if speech_ratio is not None else None
        delta_db = stats.get("speech_delta_db")
        delta_db = float(delta_db) if isinstance(delta_db, (int, float)) else None
        max_streak_ms = float(stats.get("max_speech_streak_ms") or 0.0)

        if total_ms <= 0.0:
            write_log("Skipping transcription: capture contained zero duration", self.log_path)
            return False
        if self.speech_min_total_ms > 0.0 and total_ms < self.speech_min_total_ms:
            write_log(
                (
                    "Skipping transcription: total duration "
                    f"{total_ms:.1f} ms below configured floor {self.speech_min_total_ms:.1f} ms"
                ),
                self.log_path,
            )
            return False
        if speech_ms < self.speech_min_duration_ms:
            write_log(
                (
                    "Skipping transcription: accumulated speech energy "
                    f"{speech_ms:.1f} ms below threshold {self.speech_min_duration_ms:.1f} ms"
                ),
                self.log_path,
            )
            return False
        if speech_ratio is not None and speech_ratio < self.speech_min_ratio:
            write_log(
                (
                    "Skipping transcription: speech ratio "
                    f"{speech_ratio:.2f} below threshold {self.speech_min_ratio:.2f}"
                ),
                self.log_path,
            )
            return False
        if max_streak_ms < self.speech_min_streak_ms:
            write_log(
                (
                    "Skipping transcription: max speech streak "
                    f"{max_streak_ms:.1f} ms below threshold {self.speech_min_streak_ms:.1f} ms"
                ),
                self.log_path,
            )
            return False
        if delta_db is not None and delta_db < self.speech_rms_delta_db:
            write_log(
                (
                    "Skipping transcription: speech energy delta "
                    f"{delta_db:.1f} dB below threshold {self.speech_rms_delta_db:.1f} dB"
                ),
                self.log_path,
            )
            return False
        return True

    def start_recording(self):
        with self.lock:
            if self.recording:
                return
            if self.audio_muter:
                self.audio_muter.mute()
            try:
                self.recorder.start()
            except Exception:
                if self.audio_muter:
                    self.audio_muter.restore()
                raise
            self.recording = True

    def stop_recording_and_transcribe(self):
        with self.lock:
            if not self.recording:
                return
            try:
                audio_path = self.recorder.stop(self.args.timeout)
            finally:
                if self.audio_muter:
                    self.audio_muter.restore()
                self.recording = False

        if not audio_path or audio_path.stat().st_size == 0:
            write_log("No audio captured", self.log_path)
            return
        stats = getattr(self.recorder, "last_capture_stats", None)
        stats = stats() if callable(stats) else None
        if not self._should_transcribe(stats):
            if stats:
                speech_ms = float(stats.get("speech_ms") or 0.0)
                total_ms = float(stats.get("total_ms") or 0.0)
                ratio = stats.get("speech_ratio")
                delta_db = stats.get("speech_delta_db")
                streak_ms = float(stats.get("max_speech_streak_ms") or 0.0)
                write_log(
                    (
                        "Capture summary: speech_ms={speech_ms:.1f}, total_ms={total_ms:.1f}, "
                        "ratio={ratio}, delta_db={delta}, max_streak_ms={streak:.1f}"
                    ).format(
                        speech_ms=speech_ms,
                        total_ms=total_ms,
                        ratio=(
                            f"{float(ratio):.2f}" if isinstance(ratio, (int, float)) else "n/a"
                        ),
                        delta=(
                            f"{float(delta_db):.1f}"
                            if isinstance(delta_db, (int, float))
                            else "n/a"
                        ),
                        streak=streak_ms,
                    ),
                    self.log_path,
                )
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass
            return

        write_log(f"Transcribing {audio_path}", self.log_path)
        time.sleep(0.15)

        denoised_path: Optional[Path] = None
        if self.voice_isolator is not None:
            try:
                denoised_path = self.voice_isolator.process_file(audio_path)
                audio_path = denoised_path
                write_log("Applied DTLN voice isolation", self.log_path)
            except Exception as exc:  # noqa: BLE001
                write_log(f"Voice isolation failed: {exc}", self.log_path)

        with torch.inference_mode():
            hypotheses = self.model.transcribe(audio=[str(audio_path)], batch_size=1, return_hypotheses=True)
        text = ""
        if isinstance(hypotheses, list) and hypotheses:
            hyp = hypotheses[0]
            text = hyp.text if hasattr(hyp, "text") else str(hyp)
        else:
            text = str(hypotheses)
        text = text.strip()
        force_terminal: Optional[str] = None
        emotion_info = self._evaluate_emotion(audio_path)
        if emotion_info:
            text, force_terminal = self._apply_emotion_rules(text, emotion_info)
        if text:
            text = _normalize_ordinal_words(text)
            text = _normalize_slash_phrases(text)
        if self.normalize_numbers and text:
            text = _normalize_number_words(text)
        if text:
            text = self._apply_acronym_formatting(text)
        if text and self.punctuation_model is not None:
            try:
                punctuated = self.punctuation_model.add_punctuation_capitalization([text])[0]
                if punctuated:
                    text = punctuated.strip()
            except Exception as exc:  # noqa: BLE001
                write_log(f"Failed to apply punctuation model: {exc}", self.log_path)
        if text and self.grammar_tool is not None:
            text = self._apply_grammar_cleanup(text)
        if text:
            text = _normalize_ordinal_words(text)
            text = _normalize_slash_phrases(text)
            text = _fix_trailing_artifacts(text, force_terminal)
        if force_terminal and text:
            stripped = text.rstrip()
            if not stripped.endswith(force_terminal):
                stripped = stripped.rstrip(". ?!")
                stripped += force_terminal
                text = stripped
        if self.normalize_numbers and text:
            text = _normalize_number_words(text)
        if text:
            text = self._apply_acronym_formatting(text)
        if self.args.append_space and text:
            text += " "
        write_log(f"Recognized text: {text!r}", self.log_path)
        if text:
            try:
                window_id = subprocess.check_output(["xdotool", "getwindowfocus"]).strip().decode()
            except subprocess.CalledProcessError:
                window_id = ""
            window_name = ""
            if window_id:
                try:
                    window_name = (
                        subprocess.check_output(["xdotool", "getwindowname", window_id]).strip().decode()
                    )
                except subprocess.CalledProcessError:
                    window_name = ""
            if window_name and "runelite" in window_name.lower():
                if deliver_to_runelite(text, self.log_path):
                    return
            cmd = [
                "xdotool",
                "type",
                "--clearmodifiers",
                "--delay",
                "0",
            ]
            if window_id:
                cmd.extend(["--window", window_id])
            cmd.extend(["--", text])
            subprocess.run(cmd, check=False)
        if denoised_path is not None:
            try:
                denoised_path.unlink(missing_ok=True)
            except OSError:
                pass

    def run(self):
        def on_press(key):
            if key in (keyboard.Key.alt_gr, keyboard.Key.alt_r):
                self.start_recording()

        def on_release(key):
            if key in (keyboard.Key.alt_gr, keyboard.Key.alt_r):
                threading.Thread(target=self.stop_recording_and_transcribe, daemon=True).start()

            if key == keyboard.Key.esc:
                if self.allow_esc:
                    write_log("ESC pressed – exiting listener", self.log_path)
                    return False
                write_log("ESC pressed – ignored (allow_esc disabled)", self.log_path)

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


def main() -> int:
    args = parse_args()
    cookie_path = Path(args.cookie)
    ensure_single_instance(cookie_path)
    ptt = ParakeetPTT(args)

    def handle_signal(signum, frame):
        write_log(f"Received signal {signum}", ptt.log_path)
        if ptt.recording:
            ptt.stop_recording_and_transcribe()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_signal)

    write_log("Parakeet push-to-talk ready (Right Alt)", ptt.log_path)
    ptt.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
