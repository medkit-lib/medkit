"""
This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[srt-io-convert]`.
"""

__all__ = ["SRTInputConverter", "SRTOutputConverter"]

import logging
from pathlib import Path
from typing import List, Optional, Union

import pysrt

from medkit.core import (
    generate_id,
    InputConverter,
    OutputConverter,
    OperationDescription,
    ProvTracer,
    Attribute,
)

from medkit.core.audio import AudioDocument, Segment, Span, FileAudioBuffer

logger = logging.getLogger(__name__)


class SRTInputConverter(InputConverter):
    """
    Convert .srt files containing transcription information into turn segments
    with transcription attributes.

    For each turn in a .srt file, a
    :class:`~medkit.core.audio.annotation.Segment` will be created, with an
    associated :class:`~medkit.core.Attribute` holding the transcribed text as
    value. The segments can be retrieved directly or as part of an
    :class:`~medkit.core.audio.document.AudioDocument` instance.

    If a :class:`~medkit.core.ProvTracer` is set, provenance information will be
    added for each segment and each attribute (referencing the input converter
    as the operation).
    """

    def __init__(
        self,
        turn_segment_label: str = "turn",
        transcription_attr_label: str = "transcribed_text",
        converter_id: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        turn_segment_label:
            Label to use for segments representing turns in the .srt file.
        transcription_attr_label:
            Label to use for segments attributes containing the transcribed text.
        converter_id:
            Identifier of the converter.
        """

        if converter_id is None:
            converter_id = generate_id()

        self.uid = converter_id
        self.turn_segment_label = turn_segment_label
        self.transcription_attr_label = transcription_attr_label

        self._prov_tracer: Optional[ProvTracer] = None

    @property
    def description(self) -> OperationDescription:
        """Contains all the input converter init parameters."""
        return OperationDescription(
            uid=self.uid,
            name=self.__class__.__name__,
            class_name=self.__class__.__name__,
            config={
                "turn_segment_label": self.turn_segment_label,
                "transcription_attr_label": self.transcription_attr_label,
            },
        )

    def set_prov_tracer(self, prov_tracer: ProvTracer):
        """Enable provenance tracing.

        Parameters
        ----------
        prov_tracer:
            The provenance tracer used to trace the provenance.
        """

        self._prov_tracer = prov_tracer

    def load(
        self,
        srt_dir: Union[str, Path],
        audio_dir: Optional[Union[str, Path]] = None,
        audio_ext: str = ".wav",
    ) -> List[AudioDocument]:
        """
        Load all .srt files in a directory into a list of
        :class:`~medkit.core.audio.document.AudioDocument` objects.

        For each .srt file, they must be a corresponding audio file with the
        same basename, either in the same directory or in an separated audio
        directory.

        Parameters
        ----------
        srt_dir:
            Directory containing the .srt files.
        audio_dir:
            Directory containing the audio files corresponding to the .srt files,
            if they are not in `srt_dir`.
        audio_ext:
            File extension to use for audio files.

        Returns
        -------
        List[AudioDocument]
            List of generated documents.
        """

        srt_dir = Path(srt_dir)
        audio_dir = Path(audio_dir) if audio_dir else None

        docs = []
        for srt_file in sorted(srt_dir.glob("*.srt")):
            # corresponding audio file must have same base name with audio extension,
            # either in the same directory or in audio_dir if provided
            if audio_dir:
                audio_file = (audio_dir / srt_file.stem).with_suffix(audio_ext)
            else:
                audio_file = srt_file.with_suffix(audio_ext)

            doc = self.load_doc(srt_file, audio_file)
            docs.append(doc)

        if len(docs) == 0:
            logger.warning(f"No .srt found in '{srt_dir}'")

        return docs

    def load_doc(
        self, srt_file: Union[str, Path], audio_file: Union[str, Path]
    ) -> AudioDocument:
        """Load a single .srt file into an
        :class:`~medkit.core.audio.document.AudioDocument` containing
        turn segments with transcription attributes.

        Parameters
        ----------
        srt_file:
            Path to the .srt file.
        audio_file:
            Path to the corresponding audio file.

        Returns
        -------
        AudioDocument:
            Generated document.
        """

        audio_file = Path(audio_file)

        srt_items = pysrt.open(str(srt_file))
        full_audio = FileAudioBuffer(path=audio_file)
        segments = [self._build_segment(srt_item, full_audio) for srt_item in srt_items]

        doc = AudioDocument(audio=full_audio)
        for segment in segments:
            doc.anns.add(segment)

        return doc

    def load_segments(
        self, srt_file: Union[str, Path], audio_file: Union[str, Path]
    ) -> List[Segment]:
        """Load a .srt file and return a list of
        :class:`~medkit.core.audio.annotation.Segment` objects corresponding to
        turns, with transcription attributes.

        Parameters
        ----------
        srt_file:
            Path to the .srt file.
        audio_file:
            Path to the corresponding audio file.

        Returns
        -------
        List[:class:`~medkit.core.audio.annotation.Segment`]:
            Turn segments as found in the .srt file, with transcription
            attributes attached.
        """

        audio_file = Path(audio_file)

        srt_items = pysrt.open(str(srt_file))
        full_audio = FileAudioBuffer(path=audio_file)

        segments = [self._build_segment(srt_item, full_audio) for srt_item in srt_items]
        return segments

    def _build_segment(
        self, srt_item: pysrt.SubRipItem, full_audio: FileAudioBuffer
    ) -> Segment:
        # milliseconds to seconds
        start = srt_item.start.ordinal / 1000
        end = srt_item.end.ordinal / 1000

        audio = full_audio.trim_duration(start, end)
        segment = Segment(
            label=self.turn_segment_label, span=Span(start, end), audio=audio
        )
        transcription_attr = Attribute(
            label=self.transcription_attr_label, value=srt_item.text
        )
        segment.attrs.add(transcription_attr)

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(segment, self.description, source_data_items=[])
            self._prov_tracer.add_prov(
                transcription_attr, self.description, source_data_items=[]
            )

        return segment


class SRTOutputConverter(OutputConverter):
    """
    Build .srt files containing transcription information from
    :class:`~medkit.core.audio.annotation.Segment` objects.

    There must be a segment for each turn, with an associated
    :class:`~medkit.core.Attribute` holding the transcribed text as
    value. The segments can be passed directly or as part of
    :class:`~medkit.core.audio.document.AudioDocument` instances.
    """

    def __init__(
        self,
        segment_turn_label: str = "turn",
        transcription_attr_label: str = "transcribed_text",
    ):
        """
        Parameters
        ----------
        segment_turn_label:
            Label of segments representing turns in the audio documents.
        transcription_attr_label:
            Label of segments attributes containing the transcribed text.
        """

        super().__init__()

        self.segment_turn_label = segment_turn_label
        self.transcription_attr_label = transcription_attr_label

    def save(
        self,
        docs: List[AudioDocument],
        srt_dir: Union[str, Path],
        doc_names: Optional[List[str]] = None,
    ):
        """Save :class:`~medkit.core.audio.document.AudioDocument` instances as
        .srt files in a directory.

        Parameters
        ----------
        docs:
            List of audio documents to save.
        str_dir:
            Directory into which the generated .str files will be stored.
        doc_names:
            Optional list of names to use as basenames for the generated .srt
            files.
        """

        srt_dir = Path(srt_dir)

        if doc_names is not None:
            if len(doc_names) != len(docs):
                raise ValueError(
                    "doc_names must have the same length as docs when provided"
                )
        else:
            doc_names = [doc.uid for doc in docs]

        srt_dir.mkdir(parents=True, exist_ok=True)

        for doc_name, doc in zip(doc_names, docs):
            srt_file = srt_dir / f"{doc_name}.srt"
            self.save_doc(doc, srt_file=srt_file)

    def save_doc(
        self,
        doc: AudioDocument,
        srt_file: Union[str, Path],
    ):
        """Save a single :class:`~medkit.core.audio.document.AudioDocument` as a
        .srt file.

        Parameters
        ----------
        doc:
            Audio document to save.
        srt_file:
            Path of the generated .srt file.
        """

        srt_file = Path(srt_file)

        segments = doc.anns.get(label=self.segment_turn_label)
        self.save_segments(segments, srt_file)

    def save_segments(self, segments: List[Segment], srt_file: Union[str, Path]):
        """Save :class:`~medkit.core.audio.annotation.Segment` objects representing
        turns into a .srt file.

        Parameters
        ----------
        segments:
            Turn segments to save.
        srt_file:
            Path of the generated .srt file.
        """
        srt_items = pysrt.SubRipFile(path=str(srt_file))

        for i, segment in enumerate(segments):
            transcription_attr = segment.attrs.get(label=self.transcription_attr_label)[
                0
            ]
            srt_item = pysrt.SubRipItem(
                index=i,
                start=pysrt.SubRipTime(seconds=segment.span.start),
                end=pysrt.SubRipTime(seconds=segment.span.end),
                text=transcription_attr.value,
            )
            srt_items.append(srt_item)

        srt_items.save()
