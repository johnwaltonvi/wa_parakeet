#!/usr/bin/env python3
"""Parakeet push-to-talk dictation using a configurable hotkey."""
import argparse
import atexit
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from nemo.collections.asr.models import ASRModel
from pynput import keyboard


DEFAULT_LOG = Path.home() / ".cache" / "Parakeet" / "push_to_talk.log"
DEFAULT_COOKIE = Path("/tmp/parakeet-ptt.pid")
DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
DEFAULT_SAMPLE_RATE = 16_000


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
    return parser.parse_args()


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


def write_log(msg: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{timestamp}] {msg}\n")


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
                if rms > self.rms_threshold:
                    self._last_audio_time = time.time()

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
        write_log("Recorder stopped", self.log_path)
        return self._file


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
        self.recorder = Recorder(args.sample_rate, self.device_index, self.log_path, args.rms_threshold, args.preamp)
        self.recording = False
        self.lock = threading.Lock()
        self.allow_esc = args.allow_esc

    def _load_model(self, model_name: str) -> ASRModel:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        write_log(f"Loading model {model_name} on {device}", self.log_path)
        model = ASRModel.from_pretrained(model_name=model_name, map_location=device)
        model.eval()
        return model

    def start_recording(self):
        with self.lock:
            if self.recording:
                return
            self.recorder.start()
            self.recording = True

    def stop_recording_and_transcribe(self):
        with self.lock:
            if not self.recording:
                return
            audio_path = self.recorder.stop(self.args.timeout)
            self.recording = False

        if not audio_path or audio_path.stat().st_size == 0:
            write_log("No audio captured", self.log_path)
            return

        write_log(f"Transcribing {audio_path}", self.log_path)
        time.sleep(0.15)

        with torch.inference_mode():
            hypotheses = self.model.transcribe(audio=[str(audio_path)], batch_size=1, return_hypotheses=True)
        text = ""
        if isinstance(hypotheses, list) and hypotheses:
            hyp = hypotheses[0]
            text = hyp.text if hasattr(hyp, "text") else str(hyp)
        else:
            text = str(hypotheses)
        text = text.strip()
        if self.args.append_space and text:
            text += " "
        write_log(f"Recognized text: {text!r}", self.log_path)
        if text:
            try:
                window_id = subprocess.check_output(["xdotool", "getwindowfocus"]).strip().decode()
            except subprocess.CalledProcessError:
                window_id = ""
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
