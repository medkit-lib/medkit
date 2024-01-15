from __future__ import annotations

__all__ = ["Segment"]

import dataclasses
from typing import Any

from medkit.core import dict_conv
from medkit.core.attribute import Attribute
from medkit.core.attribute_container import AttributeContainer
from medkit.core.audio.audio_buffer import (
    AudioBuffer,
    MemoryAudioBuffer,
    PlaceholderAudioBuffer,
)
from medkit.core.audio.span import Span
from medkit.core.id import generate_id


@dataclasses.dataclass(init=False)
class Segment(dict_conv.SubclassMapping):
    """Audio segment referencing part of an :class:`~.core.audio.AudioDocument`.

    Attributes
    ----------
    uid: str
        Unique identifier of the segment.
    label: str
        Label of the segment.
    audio: AudioBuffer
        The audio signal of the segment. It must be consistent with the span,
        in the sense that it must correspond to the audio signal of the document
        at the span boundaries. But it can be a modified, processed version of this
        audio signal.
    span: Span
        Span (in seconds) indicating the part of the document's full signal that
        this segment references.
    attrs: AttributeContainer
        Attributes of the segment. Stored in a
        :class:{~medkit.core.AttributeContainer} but can be passed as a list at
        init.
    metadata: dict of str to Any
        Metadata of the segment.
    keys: set of str
        Pipeline output keys to which the annotation belongs to.
    """

    uid: str
    label: str
    audio: AudioBuffer
    span: Span
    attrs: AttributeContainer
    metadata: dict[str, Any]
    keys: set[str]

    def __init__(
        self,
        label: str,
        audio: AudioBuffer,
        span: Span,
        attrs: list[Attribute] | None = None,
        metadata: dict[str, Any] | None = None,
        uid: str | None = None,
    ):
        if attrs is None:
            attrs = []
        if metadata is None:
            metadata = {}
        if uid is None:
            uid = generate_id()

        self.label = label
        self.audio = audio
        self.span = span
        self.metadata = metadata
        self.keys = set()
        self.uid = uid

        self.attrs = AttributeContainer(owner_id=self.uid)
        for attr in attrs:
            self.attrs.add(attr)

    def __init_subclass__(cls):
        Segment.register_subclass(cls)
        super().__init_subclass__()

    def to_dict(self) -> dict[str, Any]:
        # convert MemoryAudioBuffer to PlaceholderAudioBuffer
        # because we can't serialize the actual signal
        if isinstance(self.audio, MemoryAudioBuffer):
            placeholder = PlaceholderAudioBuffer.from_audio_buffer(self.audio)
            audio = placeholder.to_dict()
        else:
            audio = self.audio.to_dict()

        span = self.span.to_dict()
        attrs = [a.to_dict() for a in self.attrs]
        segment_dict = {
            "uid": self.uid,
            "label": self.label,
            "audio": audio,
            "span": span,
            "attrs": attrs,
            "metadata": self.metadata,
        }
        dict_conv.add_class_name_to_data_dict(self, segment_dict)
        return segment_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Segment:
        subclass = cls.get_subclass_for_data_dict(data)
        if subclass is not None:
            return subclass.from_dict(data)

        audio = AudioBuffer.from_dict(data["audio"])
        span = Span.from_dict(data["span"])
        attrs = [Attribute.from_dict(a) for a in data["attrs"]]
        return cls(
            label=data["label"],
            audio=audio,
            span=span,
            attrs=attrs,
            uid=data["uid"],
            metadata=data["metadata"],
        )
