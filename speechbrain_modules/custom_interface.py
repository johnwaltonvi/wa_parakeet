import sys
import types

import torch
try:  # pragma: no cover - compatibility bridge
    from speechbrain.inference.interfaces import Pretrained  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - legacy speechbrain versions
    from speechbrain.pretrained.interfaces import Pretrained

try:
    from speechbrain.lobes.models import huggingface_wav2vec as _hf_wav2vec
except ImportError:  # pragma: no cover - optional dependency
    _hf_wav2vec = None
else:
    parent_name = "speechbrain.lobes.models"
    shim_pkg_name = f"{parent_name}.huggingface_transformers"
    shim_mod_name = f"{shim_pkg_name}.wav2vec2"

    parent_module = sys.modules.get(parent_name)
    if parent_module is None:
        import speechbrain.lobes.models as parent_module  # type: ignore[attr-defined]

    if shim_pkg_name not in sys.modules:
        shim_pkg = types.ModuleType(shim_pkg_name)
        sys.modules[shim_pkg_name] = shim_pkg
    else:
        shim_pkg = sys.modules[shim_pkg_name]

    if shim_mod_name not in sys.modules:
        shim_module = types.ModuleType(shim_mod_name)
        sys.modules[shim_mod_name] = shim_module
    else:
        shim_module = sys.modules[shim_mod_name]

    wav2vec_cls = getattr(_hf_wav2vec, "HuggingFaceWav2Vec2", None)
    if wav2vec_cls is not None and not hasattr(shim_module, "Wav2Vec2"):
        setattr(shim_module, "Wav2Vec2", wav2vec_cls)

    if parent_module is not None and not hasattr(parent_module, "huggingface_transformers"):
        setattr(parent_module, "huggingface_transformers", shim_pkg)


class CustomEncoderWav2vec2Classifier(Pretrained):
    """A ready-to-use class for utterance-level classification (e.g, speaker-id,
    language-id, emotion recognition, keyword spotting, etc).

    The class assumes that an self-supervised encoder like wav2vec2/hubert and a classifier model
    are defined in the yaml file. If you want to
    convert the predicted index into a corresponding text label, please
    provide the path of the label_encoder in a variable called 'lab_encoder_file'
    within the yaml.

    The class can be used either to run only the encoder (encode_batch()) to
    extract embeddings or to run a classification step (classify_batch()).
    ```

    Example
    -------
    >>> import torchaudio
    >>> from speechbrain.pretrained import EncoderClassifier
    >>> # Model is downloaded from the speechbrain HuggingFace repo
    >>> tmpdir = getfixture("tmpdir")
    >>> classifier = EncoderClassifier.from_hparams(
    ...     source="speechbrain/spkrec-ecapa-voxceleb",
    ...     savedir=tmpdir,
    ... )

    >>> # Compute embeddings
    >>> signal, fs = torchaudio.load("samples/audio_samples/example1.wav")
    >>> embeddings =  classifier.encode_batch(signal)

    >>> # Classification
    >>> prediction =  classifier .classify_batch(signal)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        """Encodes the input audio into a single vector embedding.

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = <this>.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.
        normalize : bool
            If True, it normalizes the embeddings with the statistics
            contained in mean_var_norm_emb.

        Returns
        -------
        torch.tensor
            The encoded batch
        """
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        # Computing features and embeddings
        outputs = self.mods.wav2vec2(wavs)

        # last dim will be used for AdaptativeAVG pool
        outputs = self.mods.avg_pool(outputs, wav_lens)
        outputs = outputs.view(outputs.shape[0], -1)
        return outputs

    def classify_batch(self, wavs, wav_lens=None):
        """Performs classification on the top of the encoded features.

        It returns the posterior probabilities, the index and, if the label
        encoder is specified it also the text label.

        Arguments
        ---------
        wavs : torch.tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        out_prob
            The log posterior probabilities of each class ([batch, N_class])
        score:
            It is the value of the log-posterior for the best class ([batch,])
        index
            The indexes of the best class ([batch,])
        text_lab:
            List with the text labels corresponding to the indexes.
            (label encoder should be provided).
        """
        outputs = self.encode_batch(wavs, wav_lens)
        outputs = self.mods.output_mlp(outputs)
        out_prob = self.hparams.softmax(outputs)
        score, index = torch.max(out_prob, dim=-1)
        text_lab = self.hparams.label_encoder.decode_torch(index)
        return out_prob, score, index, text_lab

    def classify_file(self, path):
        """Classifies the given audiofile into the given set of labels.

        Arguments
        ---------
        path : str
            Path to audio file to classify.

        Returns
        -------
        out_prob
            The log posterior probabilities of each class ([batch, N_class])
        score:
            It is the value of the log-posterior for the best class ([batch,])
        index
            The indexes of the best class ([batch,])
        text_lab:
            List with the text labels corresponding to the indexes.
            (label encoder should be provided).
        """
        waveform = self.load_audio(path)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        outputs = self.encode_batch(batch, rel_length)
        outputs = self.mods.output_mlp(outputs).squeeze(1)
        out_prob = self.hparams.softmax(outputs)
        score, index = torch.max(out_prob, dim=-1)
        text_lab = self.hparams.label_encoder.decode_torch(index)
        return out_prob, score, index, text_lab

    def forward(self, wavs, wav_lens=None, normalize=False):
        return self.encode_batch(
            wavs=wavs, wav_lens=wav_lens, normalize=normalize
        )
