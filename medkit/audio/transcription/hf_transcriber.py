"""This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[hf-transcriber]`.
"""
from __future__ import annotations

__all__ = ["HFTranscriber"]


from typing import TYPE_CHECKING

import transformers
from transformers import AutomaticSpeechRecognitionPipeline

from medkit.core import Attribute, Operation

if TYPE_CHECKING:
    from pathlib import Path

    from medkit.core.audio import AudioBuffer, Segment


class HFTranscriber(Operation):
    """Transcriber operation based on a Hugging Face transformers model.

    For each segment given as input, a transcription attribute will be created
    with the transcribed text as value. If needed, a text document can later be
    created from all the transcriptions of a audio document using
    :func:`~medkit.audio.transcription.TranscribedTextDocument.from_audio_doc
    <TranscribedTextDocument.from_audio_doc>`
    """

    def __init__(
        self,
        model: str = "facebook/s2t-large-librispeech-asr",
        output_label: str = "transcribed_text",
        language: str | None = None,
        add_trailing_dot: bool = True,
        capitalize: bool = True,
        device: int = -1,
        batch_size: int = 1,
        hf_auth_token: str | None = None,
        cache_dir: str | Path | None = None,
        uid: str | None = None,
    ):
        """Parameters
        ----------
        model : str, default="facebook/s2t-large-librispeech-asr"
            Name of the ASR model on the Hugging Face models hub. Must be a
            model compatible with the `AutomaticSpeechRecognitionPipeline`
            transformers class.
        output_label : str, default="transcribed_text"
            Label of the attribute containing the transcribed text that will be
            attached to the input segments
        language : str, optional
            Optional output language to be forced on the model (useful for some
            multilingual models such as Whisper)
        add_trailing_dot : bool, default=True
            If `True`, a dot will be added at the end of each transcription text.
        capitalize : bool, default=True
            It `True`, the first letter of each transcription text will be
            uppercased and the rest lowercased.
        device : int, default=-1
            Device to use for pytorch models. Follows the Hugging Face convention
            (`-1` for cpu and device number for gpu, for instance `0` for "cuda:0")
        batch_size : int, default=1
            Size of batches processed by ASR pipeline.
        hf_auth_token : str, optional
            HuggingFace Authentication token (to access private models on the
            hub)
        cache_dir : str or Path, optional
            Directory where to store downloaded models. If not set, the default
            HuggingFace cache dir is used.
        uid : str, optional
            Identifier of the transcriber.
        """
        super().__init__(
            model=model,
            output_label=output_label,
            add_trailing_dot=add_trailing_dot,
            capitalize=capitalize,
            device=device,
            batch_size=batch_size,
            cache_dir=cache_dir,
            uid=uid,
        )

        self.model_name = model
        self.output_label = output_label
        self.add_trailing_dot = add_trailing_dot
        self.capitalize = capitalize
        self.device = device

        task = transformers.pipelines.get_task(self.model_name, token=hf_auth_token)
        if task != "automatic-speech-recognition":
            msg = (
                f"Model {self.model_name} is not associated to a speech"
                " recognition task and cannot be use with HFTranscriber"
            )
            raise ValueError(msg)

        self._pipeline = transformers.pipeline(
            task=task,
            model=self.model_name,
            feature_extractor=self.model_name,
            pipeline_class=AutomaticSpeechRecognitionPipeline,
            device=self.device,
            batch_size=batch_size,
            token=hf_auth_token,
            model_kwargs={"cache_dir": cache_dir},
        )

        if language is not None:
            self._pipeline.model.config.forced_decoder_ids = self._pipeline.tokenizer.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )

    def run(self, segments: list[Segment]):
        """Add a transcription attribute to each segment with a text value
        containing the transcribed text.

        Parameters
        ----------
        segments : list of Segment
            List of segments to transcribe
        """
        audios = [s.audio for s in segments]
        texts = self._transcribe_audios(audios)

        for segment, text in zip(segments, texts):
            attr = Attribute(label=self.output_label, value=text)
            segment.attrs.add(attr)
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(attr, self.description, [segment])

    def _transcribe_audios(self, audios: list[AudioBuffer]) -> list[str]:
        # generate iterator of all audio dicts to pass to the transformers
        # pipeline (which will handle the batching)
        audio_dicts_gen = (
            {
                "raw": audio.read().reshape((-1,)),
                "sampling_rate": audio.sample_rate,
            }
            for audio in audios
        )
        text_dicts = self._pipeline(audio_dicts_gen)
        texts_gen = (text_dict["text"] for text_dict in text_dicts)

        # post-process transcribed texts
        if self.capitalize and self.add_trailing_dot:
            texts = [t.capitalize() + "." for t in texts_gen]
        elif self.capitalize:
            texts = [t.capitalize() for t in texts_gen]
        elif self.add_trailing_dot:
            texts = [t + "." for t in texts_gen]
        else:
            texts = list(texts_gen)

        return texts
