"""This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[pa-speaker-detector]`.
"""
from __future__ import annotations

__all__ = ["PASpeakerDetector"]

from typing import TYPE_CHECKING, Iterator

# When pyannote and spacy are both installed, a conflict might occur between the
# ujson library used by pandas (a pyannote dependency) and the ujson library used
# by srsrly (a spacy dependency), especially in docker environments.
# srsly seems to end up using the ujson library from pandas, which is older and does not
# support the escape_forward_slashes parameters, instead of its own.
# The bug seems to only happen when pandas is imported from pyannote, not if
# we import pandas manually first.
# So as a workaround, we always import pandas before importing something from pyannote
import pandas as pd  # noqa: F401
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization

from medkit.core import Attribute
from medkit.core.audio import Segment, SegmentationOperation, Span

if TYPE_CHECKING:
    from pathlib import Path

# margin (in seconds) by which a turn segment
# may overrun the input segment due to imprecision
_DURATION_MARGIN = 0.1


class PASpeakerDetector(SegmentationOperation):
    """Speaker diarization operation relying on `pyannote.audio`

    Each input segment will be split into several sub-segments corresponding
    to speech turn, and an attribute will be attached to each of these sub-segments
    indicating the speaker of the turn.

    `PASpeakerDetector` uses the `SpeakerDiarization` pipeline from
    `pyannote.audio`, which performs the following steps:

    - perform multi-speaker VAD with a `PyanNet` segmentation model and extract \
    voiced segments ;

    - compute embeddings for each voiced segment with a \
    embeddings model (typically speechbrain ECAPA-TDNN) ;

    - group voice segments by speakers using a clustering algorithm such as
      agglomerative clustering, HMM, etc.

    """

    def __init__(
        self,
        model: str | Path,
        output_label: str,
        min_nb_speakers: int | None = None,
        max_nb_speakers: int | None = None,
        min_duration: float = 0.1,
        device: int = -1,
        segmentation_batch_size: int = 1,
        embedding_batch_size: int = 1,
        hf_auth_token: str | None = None,
        uid: str | None = None,
    ):
        """Parameters
        ----------
        model : str or Path
            Name (on the HuggingFace models hub) or path of a pretrained
            pipeline. When a path, should point to the .yaml file containing the
            pipeline configuration.
        output_label : str
            Label of generated turn segments.
        min_nb_speakers : int, optional
            Minimum number of speakers expected to be found.
        max_nb_speakers : int, optional
            Maximum number of speakers expected to be found.
        min_duration : float, default=0.1
            Minimum duration of speech segments, in seconds (short segments will
            be discarded).
        device : int, default=-1
            Device to use for pytorch models. Follows the Hugging Face
            convention (`-1` for cpu and device number for gpu, for instance `0`
            for "cuda:0").
        segmentation_batch_size : int, default=1
            Number of input segments in batches processed by segmentation model.
        embedding_batch_size : int, default=1
            Number of pre-segmented audios in batches processed by embedding model.
        hf_auth_token : str, optional
            HuggingFace Authentication token (to access private models on the
            hub)
        uid : str, optional
            Identifier of the detector.
        """
        # Pass all arguments to super (remove self and confidential hf_auth_token)
        init_args = locals()
        init_args.pop("self")
        init_args.pop("hf_auth_token")
        super().__init__(**init_args)

        self.output_label = output_label
        self.min_nb_speakers = min_nb_speakers
        self.max_nb_speakers = max_nb_speakers
        self.min_duration = min_duration

        torch_device = torch.device("cpu" if device < 0 else f"cuda:{device}")
        self._pipeline = Pipeline.from_pretrained(model, use_auth_token=hf_auth_token)
        if self._pipeline is None:
            msg = f"Could not instantiate pretrained pipeline with '{model}'"
            raise ValueError(msg)
        if not isinstance(self._pipeline, SpeakerDiarization):
            msg = (
                f"'{model}' does not correspond to a SpeakerDiarization pipeline. Got"
                f" object of type {type(self._pipeline)}"
            )
            raise TypeError(msg)
        self._pipeline.to(torch_device)
        self._pipeline.segmentation_batch_size = segmentation_batch_size
        self._pipeline.embedding_batch_size = embedding_batch_size

    def run(self, segments: list[Segment]) -> list[Segment]:
        """Return all turn segments detected for all input `segments`.

        Parameters
        ----------
        segments : list of Segment
            Audio segments on which to perform diarization.

        Returns
        -------
        list of Segment
            Segments detected as containing speech activity (with speaker
            attributes)
        """
        return [turn_seg for seg in segments for turn_seg in self._detect_turns_in_segment(seg)]

    def _detect_turns_in_segment(self, segment: Segment) -> Iterator[Segment]:
        audio = segment.audio
        file = {
            "waveform": torch.from_numpy(audio.read()),
            "sample_rate": audio.sample_rate,
        }

        diarization = self._pipeline.apply(
            file,
            min_speakers=self.min_nb_speakers,
            max_speakers=self.max_nb_speakers,
        )

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.duration < self.min_duration:
                continue

            # trim original audio to turn start/end points
            # (allow pyannote's turn to be slighty over the total input duration)
            if turn.end > (audio.duration + _DURATION_MARGIN):
                msg = f"Turn end {turn.end} exceeds audio duration {audio.duration}"
                raise ValueError(msg)

            turn_end = min(turn.end, audio.duration)
            turn_audio = audio.trim_duration(turn.start, turn_end)

            turn_span = Span(
                start=segment.span.start + turn.start,
                end=segment.span.start + turn_end,
            )
            speaker_attr = Attribute(label="speaker", value=speaker)
            turn_segment = Segment(
                label=self.output_label,
                span=turn_span,
                audio=turn_audio,
                attrs=[speaker_attr],
            )

            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(turn_segment, self.description, [segment])
                self._prov_tracer.add_prov(speaker_attr, self.description, [segment])

            yield turn_segment
