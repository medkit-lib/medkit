from __future__ import annotations

__all__ = ["TextDocument"]

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

from typing_extensions import Self

from medkit.core import Attribute, AttributeContainer, dict_conv
from medkit.core.id import generate_deterministic_id, generate_id
from medkit.core.text import span_utils
from medkit.core.text.annotation import Segment, TextAnnotation
from medkit.core.text.annotation_container import TextAnnotationContainer
from medkit.core.text.span import Span

if TYPE_CHECKING:
    import os


@dataclasses.dataclass(init=False)
class TextDocument(dict_conv.SubclassMapping):
    """Document holding text annotations

    Annotations must be subclasses of `TextAnnotation`.

    Attributes
    ----------
    uid : str
        Unique identifier of the document.
    text : str
        Full document text.
    anns : TextAnnotationContainer
        Annotations of the document. Stored in an
        :class:`~.text.TextAnnotationContainer` but can be passed as a list at init.
    attrs : AttributeContainer
        Attributes of the document. Stored in an
        :class:`~.core.AttributeContainer` but can be passed as a list at init
    metadata : dict of str to Any
        Document metadata.
    raw_segment : Segment
        Auto-generated segment containing the full unprocessed document text. To
        get the raw text as an annotation to pass to processing operations:

    Examples
    --------
    >>> doc = TextDocument(text="hello")
    >>> raw_text = doc.anns.get(label=TextDocument.RAW_LABEL)[0]
    """

    RAW_LABEL: ClassVar[str] = "RAW_TEXT"

    uid: str
    anns: TextAnnotationContainer
    attrs: AttributeContainer
    metadata: dict[str, Any]
    raw_segment: Segment

    def __init__(
        self,
        text: str,
        anns: Sequence[TextAnnotation] | None = None,
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

        # auto-generated raw segment to hold the text
        self.raw_segment = self._generate_raw_segment(text, uid)

        self.anns = TextAnnotationContainer(doc_id=self.uid, raw_segment=self.raw_segment)
        for ann in anns:
            self.anns.add(ann)

        self.attrs = AttributeContainer(
            owner_id=self.uid,
        )

        for attr in attrs:
            self.attrs.add(attr)

    @classmethod
    def _generate_raw_segment(cls, text: str, doc_id: str) -> Segment:
        uid = str(generate_deterministic_id(reference_id=doc_id))

        return Segment(
            label=cls.RAW_LABEL,
            spans=[Span(0, len(text))],
            text=text,
            uid=uid,
        )

    @property
    def text(self) -> str:
        return self.raw_segment.text

    def __init_subclass__(cls):
        TextDocument.register_subclass(cls)
        super().__init_subclass__()

    def to_dict(self, with_anns: bool = True) -> dict[str, Any]:
        doc_dict = dict(
            uid=self.uid,
            text=self.text,
            metadata=self.metadata,
        )
        if with_anns:
            doc_dict["anns"] = [a.to_dict() for a in self.anns]

        if self.attrs:
            doc_dict["attrs"] = [a.to_dict() for a in self.attrs]

        dict_conv.add_class_name_to_data_dict(self, doc_dict)
        return doc_dict

    @classmethod
    def from_dict(cls, doc_dict: dict[str, Any]) -> Self:
        """Creates a TextDocument from a dict

        Parameters
        ----------
        doc_dict : dict of str to Any
            A dictionary from a serialized TextDocument as generated by to_dict()
        """
        # if class method is not the same as the TextDocument one
        # (e.g., when subclassing with an overriding method)
        subclass = cls.get_subclass_for_data_dict(doc_dict)
        if subclass is not None:
            return subclass.from_dict(doc_dict)

        anns = [TextAnnotation.from_dict(a) for a in doc_dict.get("anns", [])]
        attrs = [Attribute.from_dict(a) for a in doc_dict.get("attrs", [])]
        return cls(
            uid=doc_dict["uid"],
            text=doc_dict["text"],
            anns=anns,
            attrs=attrs,
            metadata=doc_dict["metadata"],
        )

    @classmethod
    def from_file(cls, path: os.PathLike, encoding: str = "utf-8") -> Self:
        """Create a document from a text file

        Parameters
        ----------
        path : Path
            Path of the text file
        encoding : str, default="utf-8"
            Text encoding to use

        Returns
        -------
        TextDocument
            Text document with contents of `path` as text. The file path is
            included in the document metadata.
        """
        path = Path(path)
        text = path.read_text(encoding=encoding)
        return cls(text=text, metadata={"path_to_text": str(path.absolute())})

    @classmethod
    def from_dir(
        cls,
        path: os.PathLike,
        pattern: str = "*.txt",
        encoding: str = "utf-8",
    ) -> list[Self]:
        """Create documents from text files in a directory

        Parameters
        ----------
        path : Path
            Path of the directory containing text files
        pattern : str
            Glob pattern to match text files in `path`
        encoding : str
            Text encoding to use

        Returns
        -------
        list of TextDocument
            Text documents with contents of each file as text
        """
        path = Path(path)
        files = sorted(path.glob(pattern))
        return [cls.from_file(f, encoding) for f in files]

    def get_snippet(self, segment: Segment, max_extend_length: int) -> str:
        """Return a portion of the original text containing the annotation

        Parameters
        ----------
        segment : Segment
            The annotation
        max_extend_length : int
            Maximum number of characters to use around the annotation

        Returns
        -------
        str
            A portion of the text around the annotation
        """
        spans_normalized = span_utils.normalize_spans(segment.spans)
        start = min(s.start for s in spans_normalized)
        end = max(s.end for s in spans_normalized)
        start_extended = max(start - max_extend_length // 2, 0)
        remaining_max_extend_length = max_extend_length - (start - start_extended)
        end_extended = min(end + remaining_max_extend_length, len(self.text))
        return self.text[start_extended:end_extended]
