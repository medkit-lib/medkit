from __future__ import annotations

__all__ = ["TranscribedTextDocument"]

import dataclasses
from typing import Any, Sequence

from typing_extensions import Self

from medkit.core import Attribute, dict_conv
from medkit.core.audio import Span as AudioSpan
from medkit.core.text import AnySpan as AnyTextSpan
from medkit.core.text import Segment as TextSegment
from medkit.core.text import Span as TextSpan
from medkit.core.text import TextAnnotation, TextDocument
from medkit.core.text import span_utils as text_span_utils


@dataclasses.dataclass(init=False)
class TranscribedTextDocument(TextDocument):
    """Text document generated by audio transcription.

    Parameters
    ----------
    text: str
        The full transcribed text.
    text_spans_to_audio_spans: dict of TextSpan to AudioSpan
        Mapping between text characters spans in this document and
        corresponding audio spans in the original audio.
    audio_doc_id: str, optional
        Id of the original
        :class:`~medkit.core.audio.document.AudioDocument` that was
        transcribed, if known.
    anns: sequence of TextAnnotation, optional
        Annotations of the document.
    attrs: sequence of Attribute, optional
        Attributes of the document.
    metadata: dict of str to Any
        Document metadata.
    uid: str, optional
        Document identifier.

    Attributes
    ----------
    raw_segment: TextSegment
        Auto-generated segment containing the raw full transcribed text.
    """

    text_spans_to_audio_spans: dict[TextSpan, AudioSpan]
    audio_doc_id: str | None

    def __init__(
        self,
        text: str,
        text_spans_to_audio_spans: dict[TextSpan, AudioSpan],
        audio_doc_id: str | None,
        anns: Sequence[TextAnnotation] | None = None,
        attrs: Sequence[Attribute] | None = None,
        metadata: dict[str, Any] | None = None,
        uid: str | None = None,
    ):
        assert all(s.end <= len(text) for s in text_spans_to_audio_spans)

        super().__init__(text=text, anns=anns, attrs=attrs, metadata=metadata, uid=uid)

        self.audio_doc_id = audio_doc_id
        self.text_spans_to_audio_spans = text_spans_to_audio_spans

    def get_containing_audio_spans(self, text_ann_spans: list[AnyTextSpan]) -> list[AudioSpan]:
        """Return the audio spans used to transcribe the text referenced by a text annotation.

        For instance, if the audio ranging from 1.0 to 20.0 seconds is
        transcribed to some text ranging from character 10 to 56 in the
        transcribed document, and then a text annotation is created referencing
        the span 15 to 25, then the containing audio span will be the one ranging
        from 1.0 to 20.0 seconds.

        Note that some text annotations maybe be contained in more that one
        audio spans.

        Parameters
        ----------
        text_ann_spans: list of AnyTextSpan
            Text spans of a text annotation referencing some characters in the
            transcribed document.

        Returns
        -------
        list of AudioSpan
            Audio spans used to transcribe the text referenced by the spans of `text_ann`.
        """
        ann_text_spans = text_span_utils.normalize_spans(text_ann_spans)
        # TODO: use interval tree instead of nested iteration
        return [
            audio_span
            for ann_text_span in ann_text_spans
            for text_span, audio_span in self.text_spans_to_audio_spans.items()
            if text_span.overlaps(ann_text_span)
        ]

    def to_dict(self, with_anns: bool = True) -> dict[str, Any]:
        text_spans = [s.to_dict() for s in self.text_spans_to_audio_spans]
        audio_spans = [s.to_dict() for s in self.text_spans_to_audio_spans.values()]
        doc_dict = {
            "uid": self.uid,
            "text": self.text,
            "metadata": self.metadata,
            "text_spans": text_spans,
            "audio_spans": audio_spans,
            "audio_doc_id": self.audio_doc_id,
        }
        if with_anns:
            doc_dict["anns"] = [a.to_dict() for a in self.anns]

        if self.attrs:
            doc_dict["attrs"] = [a.to_dict() for a in self.attrs]
        dict_conv.add_class_name_to_data_dict(self, doc_dict)
        return doc_dict

    @classmethod
    def from_dict(cls, doc_dict: dict[str, Any]) -> Self:
        """Create a `TranscribedTextDocument` from a dict.

        Parameters
        ----------
        doc_dict: dict of str to Any
            A dictionary from a serialized `TranscribedTextDocument` as generated by to_dict()
        """
        text_spans = [TextSpan.from_dict(s) for s in doc_dict["text_spans"]]
        audio_spans = [AudioSpan.from_dict(s) for s in doc_dict["audio_spans"]]
        text_spans_to_audio_spans = dict(zip(text_spans, audio_spans))
        anns = [TextSegment.from_dict(a) for a in doc_dict["anns"]]
        attrs = [Attribute.from_dict(a) for a in doc_dict.get("attrs", [])]
        return cls(
            uid=doc_dict["uid"],
            text=doc_dict["text"],
            text_spans_to_audio_spans=text_spans_to_audio_spans,
            audio_doc_id=doc_dict["audio_doc_id"],
            anns=anns,
            attrs=attrs,
            metadata=doc_dict["metadata"],
        )
