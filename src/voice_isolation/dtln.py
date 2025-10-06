from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

try:  # Lazy import so environments without TensorFlow fail gracefully.
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.layers import Layer  # type: ignore
    from tensorflow.keras.models import load_model  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "TensorFlow is required for DTLN voice isolation. Install the 'tensorflow' package."
    ) from exc


class InstantLayerNormalization(Layer):
    """Instant layer normalization from the original DTLN implementation."""

    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self.epsilon = 1e-7
        self.gamma = None
        self.beta = None

    def build(self, input_shape):  # type: ignore[no-untyped-def]
        shape = input_shape[-1:]
        self.gamma = self.add_weight(
            shape=shape, initializer="ones", trainable=True, name="gamma"
        )
        self.beta = self.add_weight(
            shape=shape, initializer="zeros", trainable=True, name="beta"
        )

    def call(self, inputs):  # type: ignore[no-untyped-def]
        mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)
        variance = tf.math.reduce_mean(tf.math.square(inputs - mean), axis=[-1], keepdims=True)
        std = tf.math.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        outputs = outputs * self.gamma
        outputs = outputs + self.beta
        return outputs


@dataclass
class DtlnVoiceIsolation:
    """Wraps the DTLN speech enhancement model for audio pre-processing."""

    model_path: Path
    sample_rate: int = 16000
    _model: Optional[tf.keras.Model] = None  # type: ignore[name-defined]

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"DTLN model not found at {self.model_path}")
        custom_objects = {"InstantLayerNormalization": InstantLayerNormalization}
        self._model = load_model(
            str(self.model_path), custom_objects=custom_objects, compile=False
        )

    def process_file(self, input_path: Path) -> Path:
        self._ensure_model()
        assert self._model is not None
        audio, fs = sf.read(str(input_path), always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if fs != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=fs, target_sr=self.sample_rate)
        audio = audio.astype(np.float32)
        len_orig = len(audio)
        pad = np.zeros(384, dtype=np.float32)
        padded = np.concatenate((pad, audio, pad), axis=0)
        predicted = self._model.predict_on_batch(
            np.expand_dims(padded, axis=0).astype(np.float32)
        )
        denoised = np.squeeze(predicted)[384 : 384 + len_orig]
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(tmp_path, denoised, self.sample_rate)
        return Path(tmp_path)
