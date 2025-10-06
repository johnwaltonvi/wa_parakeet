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

_BLOCK_LEN = 512
_BLOCK_SHIFT = 128


def _prepare_inputs(session: ort.InferenceSession) -> tuple[list[str], dict[str, np.ndarray]]:
    names = [inp.name for inp in session.get_inputs()]
    tensors = {
        inp.name: np.zeros(
            [dim if isinstance(dim, int) else 1 for dim in inp.shape], dtype=np.float32
        )
        for inp in session.get_inputs()
    }
    return names, tensors


@dataclass
class DtlnVoiceIsolation:
    """Applies the DTLN speech enhancement model (ONNX) to denoise audio."""

    model_1_path: Path
    model_2_path: Path
    sample_rate: int = 16000
    _session_1: Optional[ort.InferenceSession] = None
    _session_2: Optional[ort.InferenceSession] = None
    _inputs_1: Optional[tuple[list[str], dict[str, np.ndarray]]] = None
    _inputs_2: Optional[tuple[list[str], dict[str, np.ndarray]]] = None

    def _ensure_sessions(self) -> None:
        if self._session_1 is not None and self._session_2 is not None:
            return
        if not self.model_1_path.exists() or not self.model_2_path.exists():
            raise FileNotFoundError("DTLN ONNX model files are missing")
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self._session_1 = ort.InferenceSession(str(self.model_1_path), providers=providers)
        self._session_2 = ort.InferenceSession(str(self.model_2_path), providers=providers)
        self._inputs_1 = _prepare_inputs(self._session_1)
        self._inputs_2 = _prepare_inputs(self._session_2)

    def process_file(self, input_path: Path) -> Path:
        self._ensure_sessions()
        assert self._session_1 is not None and self._session_2 is not None
        assert self._inputs_1 is not None and self._inputs_2 is not None

        audio, fs = sf.read(str(input_path), always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if fs != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=fs, target_sr=self.sample_rate)
        audio = audio.astype(np.float32)

        out_audio = np.zeros_like(audio)
        in_buffer = np.zeros((_BLOCK_LEN), dtype=np.float32)
        out_buffer = np.zeros((_BLOCK_LEN), dtype=np.float32)
        num_blocks = max(0, (audio.shape[0] - (_BLOCK_LEN - _BLOCK_SHIFT)) // _BLOCK_SHIFT)

        names_1, inputs_1_template = self._inputs_1
        names_2, inputs_2_template = self._inputs_2
        inputs_1 = {k: v.copy() for k, v in inputs_1_template.items()}
        inputs_2 = {k: v.copy() for k, v in inputs_2_template.items()}

        for idx in range(num_blocks):
            in_buffer[:-_BLOCK_SHIFT] = in_buffer[_BLOCK_SHIFT:]
            in_buffer[-_BLOCK_SHIFT:] = audio[idx * _BLOCK_SHIFT : (idx + 1) * _BLOCK_SHIFT]
            in_block_fft = np.fft.rfft(in_buffer)
            in_mag = np.abs(in_block_fft).reshape(1, 1, -1).astype(np.float32)
            in_phase = np.angle(in_block_fft)
            inputs_1[names_1[0]] = in_mag
            outputs_1 = self._session_1.run(None, inputs_1)
            for name, output in zip(names_1[1:], outputs_1[1:]):
                inputs_1[name] = output
            estimated_complex = in_mag * outputs_1[0] * np.exp(1j * in_phase)
            estimated_block = np.fft.irfft(estimated_complex)
            estimated_block = estimated_block.reshape(1, 1, -1).astype(np.float32)
            inputs_2[names_2[0]] = estimated_block
            outputs_2 = self._session_2.run(None, inputs_2)
            for name, output in zip(names_2[1:], outputs_2[1:]):
                inputs_2[name] = output
            out_block = np.squeeze(outputs_2[0])
            out_buffer[:-_BLOCK_SHIFT] = out_buffer[_BLOCK_SHIFT:]
            out_buffer[-_BLOCK_SHIFT:] = 0.0
            out_buffer += out_block
            out_audio[idx * _BLOCK_SHIFT : (idx + 1) * _BLOCK_SHIFT] = out_buffer[:_BLOCK_SHIFT]

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(tmp_path, out_audio, self.sample_rate)
        return Path(tmp_path)
