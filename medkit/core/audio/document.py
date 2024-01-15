from __future__ import annotations

__all__ = ["AudioDocument"]

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

from typing_extensions import Self

from medkit.core import Attribute, AttributeContainer, dict_conv
from medkit.core.audio.annotation import Segment
from medkit.core.audio.annotation_container import AudioAnnotationContainer
from medkit.core.audio.audio_buffer import (
    AudioBuffer,
    FileAudioBuffer,
    MemoryAudioBuffer,
    PlaceholderAudioBuffer,
)
from medkit.core.audio.span import Span
from medkit.core.id import generate_deterministic_id, generate_id

if TYPE_CHECKING:
    import os


@dataclasses.dataclass(init=False)
class AudioDocument(dict_conv.SubclassMapping):
    """Document holding audio annotations.

    Attributes
    ----------
    uid: str
        Unique identifier of the document.
    audio: AudioBuffer
        Audio buffer containing the entire signal of the document.
    anns: :class:`~.audio.AudioAnnotationContainer`
        Annotations of the document. Stored in an
        :class:`~.audio.AudioAnnotationContainer` but can be passed as a list at init.
    attrs: :class:`~.core.AttributeContainer`
        Attributes of the document. Stored in an
        :class:`~.core.AttributeContainer` but can be passed as a list at init
    metadata: dict of str to Any
        Document metadata.
    raw_segment: :class:`~.audio.Segment`
        Auto-generated segment containing the full unprocessed document audio.
    """

    RAW_LABEL: ClassVar[str] = "RAW_AUDIO"
    """Label to be used for raw segment"""

    uid: str
    anns: AudioAnnotationContainer
    attrs: AttributeContainer
    metadata: dict[str, Any]
    raw_segment: Segment

    def __init__(
        self,
        audio: AudioBuffer,
        anns: Sequence[Segment] | None = None,
        attrs: Sequence[Attribute] | None = None,
        metadata: dict[str, Any] | None = None,
        uid: str | None = None,
    ):
        if anns is None:
            anns = []
        if attrs is None:
            attrs = []
        if metadata is None:
            metadata = {}
        if uid is None:
            uid = generate_id()

        self.uid = uid
        self.metadata = metadata

        # auto-generated raw segment to hold the audio buffer
        self.raw_segment = self._generate_raw_segment(audio, uid)

        self.anns = AudioAnnotationContainer(doc_id=self.uid, raw_segment=self.raw_segment)
        for ann in anns:
            self.anns.add(ann)

        self.attrs = AttributeContainer(owner_id=self.uid)
        for attr in attrs:
            self.attrs.add(attr)

    @classmethod
    def _generate_raw_segment(cls, audio: AudioBuffer, doc_id: str) -> Segment:
        uid = str(generate_deterministic_id(reference_id=doc_id))

        return Segment(
            label=cls.RAW_LABEL,
            span=Span(0.0, audio.duration),
            audio=audio,
            uid=uid,
        )

    @property
    def audio(self) -> AudioBuffer:
        return self.raw_segment.audio

    def __init_subclass__(cls):
        AudioDocument.register_subclass(cls)
        super().__init_subclass__()

    def to_dict(self, with_anns: bool = True) -> dict[str, Any]:
        # convert MemoryAudioBuffer to PlaceholderAudioBuffer
        # because we can't serialize the actual signal
        if isinstance(self.audio, MemoryAudioBuffer):
            placeholder = PlaceholderAudioBuffer.from_audio_buffer(self.audio)
            audio = placeholder.to_dict()
        else:
            audio = self.audio.to_dict()
        doc_dict: dict[str, Any] = {
            "uid": self.uid,
            "audio": audio,
            "metadata": self.metadata,
        }
        if with_anns:
            doc_dict["anns"] = [a.to_dict() for a in self.anns]
        if self.attrs:
            doc_dict["attrs"] = [a.to_dict() for a in self.attrs]

        dict_conv.add_class_name_to_data_dict(self, doc_dict)
        return doc_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        subclass = cls.get_subclass_for_data_dict(data)
        if subclass is not None:
            return subclass.from_dict(data)

        audio = AudioBuffer.from_dict(data["audio"])
        anns = [Segment.from_dict(a) for a in data.get("anns", [])]
        attrs = [Attribute.from_dict(a) for a in data.get("attrs", [])]
        return cls(
            uid=data["uid"],
            audio=audio,
            anns=anns,
            attrs=attrs,
            metadata=data["metadata"],
        )

    @classmethod
    def from_file(cls, path: os.PathLike) -> Self:
        """Create document from an audio file

        Parameters
        ----------
        path: path-like
            Path to the audio file. Supports all file formats handled by
            `libsndfile` (http://www.mega-nerd.com/libsndfile/#Features)

        Returns
        -------
        AudioDocument
            Audio document with signal of `path` as audio. The file path is
            included in the document metadata.
        """
        path = Path(path)
        audio = FileAudioBuffer(path)
        return cls(audio=audio, metadata={"path_to_audio": str(path.absolute())})

    @classmethod
    def from_dir(
        cls,
        path: os.PathLike,
        pattern: str = "*.wav",
    ) -> list[Self]:
        """Create documents from audio files in a directory

        Parameters
        ----------
        path: path-like
            Path of the directory containing audio files
        pattern: str, default="*.wav"
            Glob pattern to match audio files in `path`. Supports all file
            formats handled by `libsndfile`
            (http://www.mega-nerd.com/libsndfile/#Features)

        Returns
        -------
        List[AudioDocument]
            Audio documents with signal of each file as audio
        """
        path = Path(path)
        files = sorted(path.glob(pattern))
        return [cls.from_file(f) for f in files]
