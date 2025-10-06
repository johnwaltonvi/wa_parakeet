from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf


@dataclass
class DtlnVoiceIsolation:
    """Applies the DTLN speech enhancement model via ONNX Runtime."""

    model_path: Path
    sample_rate: int = 16000
    _session: Optional[ort.InferenceSession] = None
    _input_name: Optional[str] = None
    _output_name: Optional[str] = None

    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"DTLN model not found at {self.model_path}")
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self._session = ort.InferenceSession(str(self.model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def process_file(self, input_path: Path) -> Path:
        self._ensure_session()
        assert self._session is not None
        assert self._input_name is not None and self._output_name is not None

        audio, fs = sf.read(str(input_path), always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if fs != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=fs, target_sr=self.sample_rate)
        audio = audio.astype(np.float32)
        len_orig = len(audio)
        pad = np.zeros(384, dtype=np.float32)
        padded = np.concatenate((pad, audio, pad), axis=0)
        inputs = {self._input_name: np.expand_dims(padded, axis=0)}
        denoised = self._session.run([self._output_name], inputs)[0]
        denoised = np.squeeze(denoised)[384 : 384 + len_orig]
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(tmp_path, denoised, self.sample_rate)
        return Path(tmp_path)
