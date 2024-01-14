from __future__ import annotations

__all__ = ["RTTMInputConverter", "RTTMOutputConverter"]

import csv
import logging
from pathlib import Path
from typing import Any

from medkit.core import (
    Attribute,
    InputConverter,
    OperationDescription,
    OutputConverter,
    ProvTracer,
    generate_id,
)
from medkit.core.audio import AudioDocument, FileAudioBuffer, Segment, Span

logger = logging.getLogger(__name__)

# cf https://github.com/nryant/dscore#rttm
_RTTM_FIELDS = [
    "type",
    "file_id",
    "channel",
    "onset",
    "duration",
    "na_1",
    "na_2",
    "speaker_name",
    "na_3",
    "na_4",
]


class RTTMInputConverter(InputConverter):
    """Convert Rich Transcription Time Marked (.rttm) files containing diarization
    information into turn segments.

    For each turn in a .rttm file, a
    :class:`~medkit.core.audio.annotation.Segment` will be created, with an
    associated :class:`~medkit.core.Attribute` holding the name of the turn
    speaker as value. The segments can be retrieved directly or as part of an
    :class:`~medkit.core.audio.document.AudioDocument` instance.

    If a :class:`~medkit.core.ProvTracer` is set, provenance information will be
    added for each segment and each attribute (referencing the input converter
    as the operation).
    """

    def __init__(
        self,
        turn_label: str = "turn",
        speaker_label: str = "speaker",
        converter_id: str | None = None,
    ):
        """Parameters
        ----------
        turn_label : str, default="turn"
            Label of segments representing turns in the .rttm file.
        speaker_label : str, default="speaker"
            Label of speaker attributes to add to each segment.
        converter_id : str, optional
            Identifier of the converter.
        """
        if converter_id is None:
            converter_id = generate_id()

        self.uid = converter_id
        self.turn_label = turn_label
        self.speaker_label = speaker_label

        self._prov_tracer: ProvTracer | None = None

    @property
    def description(self) -> OperationDescription:
        """Contains all the input converter init parameters."""
        return OperationDescription(
            uid=self.uid,
            name=self.__class__.__name__,
            class_name=self.__class__.__name__,
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
        rttm_dir: str | Path,
        audio_dir: str | Path | None = None,
        audio_ext: str = ".wav",
    ) -> list[AudioDocument]:
        """Load all .rttm files in a directory into a list of
        :class:`~medkit.core.audio.document.AudioDocument` objects.

        For each .rttm file, they must be a corresponding audio file with the
        same basename, either in the same directory or in an separated audio
        directory.

        Parameters
        ----------
        rttm_dir : str or Path
            Directory containing the .rttm files.
        audio_dir : str or Path, optional
            Directory containing the audio files corresponding to the .rttm files,
            if they are not in `rttm_dir`.
        audio_ext : str, default=".wav"
            File extension to use for audio files.

        Returns
        -------
        list of AudioDocument
            List of generated documents.
        """
        rttm_dir = Path(rttm_dir)
        if audio_dir is not None:
            audio_dir = Path(audio_dir)

        docs = []
        for rttm_file in sorted(rttm_dir.glob("*.rttm")):
            # corresponding audio file must have same base name with audio extension,
            # either in the same directory or in audio_dir if provided
            audio_file = (
                (audio_dir / rttm_file.stem).with_suffix(audio_ext) if audio_dir else rttm_file.with_suffix(audio_ext)
            )

            doc = self.load_doc(rttm_file, audio_file)
            docs.append(doc)

        if len(docs) == 0:
            logger.warning("No .rttm found in '%s'", rttm_dir)

        return docs

    def load_doc(self, rttm_file: str | Path, audio_file: str | Path) -> AudioDocument:
        """Load a single .rttm file into an
        :class:`~medkit.core.audio.document.AudioDocument`.

        Parameters
        ----------
        rttm_file : str or Path
            Path to the .rttm file.
        audio_file : str or Path
            Path to the corresponding audio file.

        Returns
        -------
        AudioDocument
            Generated document.
        """
        rttm_file = Path(rttm_file)
        audio_file = Path(audio_file)

        rows = self._load_rows(rttm_file)
        full_audio = FileAudioBuffer(path=audio_file)
        turn_segments = [self._build_turn_segment(row, full_audio) for row in rows]

        doc = AudioDocument(audio=full_audio)
        for turn_segment in turn_segments:
            doc.anns.add(turn_segment)

        return doc

    def load_turns(self, rttm_file: str | Path, audio_file: str | Path) -> list[Segment]:
        """Load a .rttm file and return a list of
        :class:`~medkit.core.audio.annotation.Segment` objects.

        Parameters
        ----------
        rttm_file : str or Path
            Path to the .rttm file.
        audio_file : str or Path
            Path to the corresponding audio file.

        Returns
        -------
        list of Segment
            Turn segments as found in the .rttm file.
        """
        rttm_file = Path(rttm_file)
        audio_file = Path(audio_file)

        rows = self._load_rows(rttm_file)
        full_audio = FileAudioBuffer(path=audio_file)
        return [self._build_turn_segment(row, full_audio) for row in rows]

    @staticmethod
    def _load_rows(rttm_file: Path):
        with Path(rttm_file).open() as fp:
            csv_reader = csv.DictReader(fp, fieldnames=_RTTM_FIELDS, delimiter=" ")
            rows = list(csv_reader)

        file_id = rows[0]["file_id"]
        if not all(r["file_id"] == file_id for r in rows):
            msg = "Multi-file .rttm are not supported, all entries should have same file_id or <NA>"
            raise RuntimeError(msg)

        return rows

    def _build_turn_segment(self, row: dict[str, Any], full_audio: FileAudioBuffer) -> Segment:
        start = float(row["onset"])
        end = start + float(row["duration"])
        audio = full_audio.trim_duration(start, end)
        segment = Segment(label=self.turn_label, span=Span(start, end), audio=audio)
        speaker_attr = Attribute(label=self.speaker_label, value=row["speaker_name"])
        segment.attrs.add(speaker_attr)

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(segment, self.description, source_data_items=[])
            self._prov_tracer.add_prov(speaker_attr, self.description, source_data_items=[])

        return segment


class RTTMOutputConverter(OutputConverter):
    """Build Rich Transcription Time Marked (.rttm) files containing diarization
    information from :class:`~medkit.core.audio.annotation.Segment` objects.

    There must be a segment for each turn, with an associated
    :class:`~medkit.core.Attribute` holding the name of the turn speaker as
    value. The segments can be passed directly or as part of
    :class:`~medkit.core.audio.document.AudioDocument` instances.
    """

    def __init__(self, turn_label: str = "turn", speaker_label: str = "speaker"):
        """Parameters
        ----------
        turn_label : str, default="turn"
            Label of segments representing turns in the audio documents.
        speaker_label : str, default="speaker"
            Label of speaker attributes attached to each turn segment.
        """
        super().__init__()

        self.turn_label = turn_label
        self.speaker_label = speaker_label

    def save(
        self,
        docs: list[AudioDocument],
        rttm_dir: str | Path,
        doc_names: list[str] | None = None,
    ):
        """Save :class:`~medkit.core.audio.document.AudioDocument` instances as
        .rttm files in a directory.

        Parameters
        ----------
        docs : list of AudioDocument
            List of audio documents to save.
        rttm_dir : str or Path
            Directory into which the generated .rttm files will be stored.
        doc_names : list of str, optional
            Optional list of names to use as basenames and file ids for the
            generated .rttm files (2d column). If none provided, the document
            ids will be used.
        """
        rttm_dir = Path(rttm_dir)

        if doc_names is not None:
            if len(doc_names) != len(docs):
                msg = "doc_names must have the same length as docs when provided"
                raise ValueError(msg)
        else:
            doc_names = [doc.uid for doc in docs]

        rttm_dir.mkdir(parents=True, exist_ok=True)

        for doc_name, doc in zip(doc_names, docs):
            rttm_file = rttm_dir / f"{doc_name}.rttm"
            self.save_doc(doc, rttm_file=rttm_file, rttm_doc_id=doc_name)

    def save_doc(
        self,
        doc: AudioDocument,
        rttm_file: str | Path,
        rttm_doc_id: str | None = None,
    ):
        """Save a single :class:`~medkit.core.audio.document.AudioDocument` as a
        .rttm file.

        Parameters
        ----------
        doc : AudioDocument
            Audio document to save.
        rttm_file : str or Path
            Path of the generated .rttm file.
        rttm_doc_id : str, optional
            File uid to use for the generated .rttm file (2d column). If none
            provided, the document uid will be used.
        """
        rttm_file = Path(rttm_file)
        if rttm_doc_id is None:
            rttm_doc_id = doc.uid

        turns = doc.anns.get(label=self.turn_label)
        self.save_turn_segments(turns, rttm_file, rttm_doc_id)

    def save_turn_segments(
        self,
        turn_segments: list[Segment],
        rttm_file: str | Path,
        rttm_doc_id: str | None,
    ):
        """Save :class:`~medkit.core.audio.annotation.Segment` objects into a .rttm file.

        Parameters
        ----------
        turn_segments : list of Segment
            Turn segments to save.
        rttm_file : str or Path
            Path of the generated .rttm file.
        rttm_doc_id : str, optional
            File uid to use for the generated .rttm file (2d column).
        """
        rows = [self._build_rttm_row(s, rttm_doc_id) for s in turn_segments]
        rows.sort(key=lambda r: r["onset"])

        with Path(rttm_file).open(mode="w", encoding="utf-8") as fp:
            csv_writer = csv.DictWriter(fp, fieldnames=_RTTM_FIELDS, delimiter=" ")
            csv_writer.writerows(rows)

    def _build_rttm_row(self, turn_segment: Segment, rttm_doc_id: str | None) -> dict[str, Any]:
        speaker_attrs = turn_segment.attrs.get(label=self.speaker_label)
        if len(speaker_attrs) == 0:
            msg = f"Found no attribute with label '{self.speaker_label}' on turn segment"
            raise RuntimeError(msg)

        speaker_attr = speaker_attrs[0]
        span = turn_segment.span

        return {
            "type": "SPEAKER",
            "file_id": rttm_doc_id if rttm_doc_id is not None else "<NA>",
            "channel": "1",
            "onset": f"{span.start:.3f}",
            "duration": f"{span.length:.3f}",
            "na_1": "<NA>",
            "na_2": "<NA>",
            "speaker_name": speaker_attr.value,
            "na_3": "<NA>",
            "na_4": "<NA>",
        }
