import sys
import types
from pathlib import Path
from unittest import TestCase
from unittest.mock import call, patch

# Stub heavy dependencies so the module imports without GPU/audio libraries present.
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.device = lambda name: name

    class _InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    torch_stub.inference_mode = lambda: _InferenceMode()
    sys.modules["torch"] = torch_stub

if "nemo" not in sys.modules:
    nemo_stub = types.ModuleType("nemo")
    collections_stub = types.ModuleType("nemo.collections")
    asr_stub = types.ModuleType("nemo.collections.asr")
    models_stub = types.ModuleType("nemo.collections.asr.models")

    class _DummyModel:
        @classmethod
        def from_pretrained(cls, model_name, map_location):
            return cls()

        def eval(self):
            return None

        def transcribe(self, audio, batch_size, return_hypotheses):
            return []

    models_stub.ASRModel = _DummyModel
    asr_stub.models = models_stub
    collections_stub.asr = asr_stub
    nemo_stub.collections = collections_stub

    sys.modules["nemo"] = nemo_stub
    sys.modules["nemo.collections"] = collections_stub
    sys.modules["nemo.collections.asr"] = asr_stub
    sys.modules["nemo.collections.asr.models"] = models_stub

if "sounddevice" not in sys.modules:
    sounddevice_stub = types.ModuleType("sounddevice")
    sounddevice_stub.default = types.SimpleNamespace(device=(None, None))

    class _InputStream:  # pragma: no cover - simple placeholder
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

    sounddevice_stub.InputStream = _InputStream
    sounddevice_stub.query_devices = lambda: []
    sys.modules["sounddevice"] = sounddevice_stub

if "soundfile" not in sys.modules:
    soundfile_stub = types.ModuleType("soundfile")

    class _SoundFile:  # pragma: no cover - placeholder
        def __init__(self, *args, **kwargs):
            pass

        def write(self, *_args, **_kwargs):
            return None

        def close(self):
            return None

    soundfile_stub.SoundFile = _SoundFile
    sys.modules["soundfile"] = soundfile_stub

if "pynput" not in sys.modules:
    pynput_stub = types.ModuleType("pynput")
    keyboard_stub = types.ModuleType("pynput.keyboard")
    keyboard_stub.Key = types.SimpleNamespace(alt_gr="alt_gr", alt_r="alt_r", esc="esc")

    class _Listener:  # pragma: no cover - placeholder
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def join(self):
            return None

    keyboard_stub.Listener = _Listener
    pynput_stub.keyboard = keyboard_stub
    sys.modules["pynput"] = pynput_stub
    sys.modules["pynput.keyboard"] = keyboard_stub

if "numpy" not in sys.modules:
    import math

    numpy_stub = types.ModuleType("numpy")

    def _clip(values, min_value, max_value):  # pragma: no cover - test stub
        return values

    def _square(values, dtype=None):  # pragma: no cover - test stub
        return [v * v for v in values]

    def _mean(values, dtype=None):  # pragma: no cover - test stub
        return sum(values) / len(values) if values else 0.0

    def _sqrt(value, dtype=None):  # pragma: no cover - test stub
        return math.sqrt(value)

    numpy_stub.clip = _clip
    numpy_stub.square = _square
    numpy_stub.mean = _mean
    numpy_stub.sqrt = _sqrt
    numpy_stub.ndarray = list
    sys.modules["numpy"] = numpy_stub


from src.parakeet_push_to_talk import AudioMuteController


class AudioMuteControllerTests(TestCase):
    def setUp(self):
        self.log_path = Path("/tmp/parakeet_test.log")
        if self.log_path.exists():
            self.log_path.unlink()

    def test_wpctl_mute_and_restore_unmutes_when_needed(self):
        with patch("shutil.which", side_effect=lambda name: "/usr/bin/wpctl" if name == "wpctl" else None), \
            patch("subprocess.check_output", return_value="Volume: 0.50\nMuted: no\n") as mock_co, \
            patch("subprocess.run") as mock_run:
            controller = AudioMuteController(self.log_path)
            controller.mute()
            controller.restore()

        mock_co.assert_called_once_with(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"], text=True)
        mock_run.assert_has_calls(
            [
                call(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "1"], check=True),
                call(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "0"], check=True),
            ]
        )

    def test_wpctl_restore_keeps_muted_state(self):
        with patch("shutil.which", side_effect=lambda name: "/usr/bin/wpctl" if name == "wpctl" else None), \
            patch("subprocess.check_output", return_value="Volume: 0.50\nMuted: yes\n"), \
            patch("subprocess.run") as mock_run:
            controller = AudioMuteController(self.log_path)
            controller.mute()
            controller.restore()

        mock_run.assert_called_once_with(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "1"], check=True)

    def test_pactl_fallback_uses_sink_detection(self):
        def which(name):
            if name == "wpctl":
                return None
            if name == "pactl":
                return "/usr/bin/pactl"
            return None

        with patch("shutil.which", side_effect=which), \
            patch("subprocess.check_output") as mock_co, \
            patch("subprocess.run") as mock_run:
            mock_co.side_effect = ["alsa_output.0\n", "Mute: no\n"]
            controller = AudioMuteController(self.log_path)
            controller.mute()
            controller.restore()

        expected_calls = [
            call(["pactl", "get-default-sink"], text=True),
            call(["pactl", "get-sink-mute", "alsa_output.0"], text=True),
        ]
        mock_co.assert_has_calls(expected_calls)
        mock_run.assert_has_calls(
            [
                call(["pactl", "set-sink-mute", "alsa_output.0", "1"], check=True),
                call(["pactl", "set-sink-mute", "alsa_output.0", "0"], check=True),
            ]
        )
