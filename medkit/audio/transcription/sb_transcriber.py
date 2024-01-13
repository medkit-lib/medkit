"""This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[sb-transcriber]`.
"""

__all__ = ["SBTranscriber"]

from pathlib import Path
from typing import List, Optional, Union

import speechbrain as sb

import medkit.core.utils
from medkit.core import Attribute, Operation
from medkit.core.audio import AudioBuffer, Segment


class SBTranscriber(Operation):
    """Transcriber operation based on a SpeechBrain model.

    For each segment given as input, a transcription attribute will be created
    with the transcribed text as value. If needed, a text document can later be
    created from all the transcriptions of a audio document using
    :func:`~medkit.audio.transcription.TranscribedTextDocument.from_audio_doc
    <TranscribedTextDocument.from_audio_doc>`
    """

    def __init__(
        self,
        model: Union[str, Path],
        needs_decoder: bool,
        output_label: str = "transcribed_text",
        add_trailing_dot: bool = True,
        capitalize: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        device: int = -1,
        batch_size: int = 1,
        uid: Optional[str] = None,
    ):
        """Parameters
        ----------
        model:
            Name of the model on the Hugging Face models hub, or local path.
        output_label:
            Label of the attribute containing the transcribed text that will be
            attached to the input segments
        needs_decoder:
            Whether the model should be used with the speechbrain
            `EncoderDecoderASR` class or the `EncoderASR` class. If unsure,
            check the code snippets on the model card on the hub.
        add_trailing_dot:
            If `True`, a dot will be added at the end of each transcription text.
        capitalize:
            It `True`, the first letter of each transcription text will be
            uppercased and the rest lowercased.
        cache_dir:
            Directory where to store the downloaded model. If `None`,
            speechbrain will use "pretrained_models/" and "model_checkpoints/"
            directories in the current working directory.
        device:
            Device to use for pytorch models. Follows the Hugging Face convention
            (`-1` for cpu and device number for gpu, for instance `0` for "cuda:0")
        batch_size:
            Number of segments in batches processed by the model.
        uid:
            Identifier of the transcriber.
        """
        if cache_dir is not None:
            cache_dir = Path(cache_dir)

        super().__init__(
            model=model,
            needs_decoder=needs_decoder,
            output_label=output_label,
            add_trailing_dot=add_trailing_dot,
            capitalize=capitalize,
            cache_dir=cache_dir,
            device=device,
            batch_size=batch_size,
            uid=uid,
        )

        self.model_name = model
        self.output_label = output_label
        self.add_trailing_dot = add_trailing_dot
        self.capitalize = capitalize
        self.cache_dir = cache_dir
        self.device = device
        self.batch_size = batch_size
        self._torch_device = "cpu" if self.device < 0 else f"cuda:{self.device}"

        asr_class = sb.pretrained.EncoderDecoderASR if needs_decoder else sb.pretrained.EncoderASR

        self._asr = asr_class.from_hparams(source=model, savedir=cache_dir, run_opts={"device": self._torch_device})

        self._sample_rate = self._asr.audio_normalizer.sample_rate

    def run(self, segments: List[Segment]):
        """Add a transcription attribute to each segment with a text value
        containing the transcribed text.

        Parameters
        ----------
        segments:
            List of segments to transcribe
        """
        audios = [s.audio for s in segments]
        texts = self._transcribe_audios(audios)

        for segment, text in zip(segments, texts):
            attr = Attribute(label=self.output_label, value=text)
            segment.attrs.add(attr)
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(attr, self.description, [segment])

    def _transcribe_audios(self, audios: List[AudioBuffer]) -> List[str]:
        if not all(a.sample_rate == self._sample_rate for a in audios):
            msg = (
                "SBTranscriber received audio buffers with incompatible sample"
                f" rates (model expected {self._sample_rate} Hz)"
            )
            raise ValueError(msg)
        if not all(a.nb_channels == 1 for a in audios):
            msg = "SBTranscriber only supports mono audio buffers"
            raise ValueError(msg)

        texts = []

        # group audios in batch of same length with padding
        for batched_audios in medkit.core.utils.batch_list(audios, self.batch_size):
            padded_batch = sb.dataio.batch.PaddedBatch([{"wav": a.read().reshape((-1,))} for a in batched_audios])
            padded_batch.to(self._torch_device)

            batch_texts, _ = self._asr.transcribe_batch(padded_batch.wav.data, padded_batch.wav.lengths)
            texts += batch_texts

        if self.capitalize:
            texts = [t.capitalize() for t in texts]
        if self.add_trailing_dot:
            texts = [t + "." for t in texts]

        return texts
