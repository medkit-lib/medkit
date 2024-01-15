from __future__ import annotations

__all__ = ["DuplicateFinder", "DuplicationAttribute"]

import dataclasses
import re
from typing import Any, ClassVar

import duptextfinder  # type: ignore
from typing_extensions import Literal, Self

from medkit.core import Attribute, Collection, Operation, dict_conv
from medkit.core.text import AnySpan, Segment, TextDocument, span_utils


@dataclasses.dataclass
class DuplicationAttribute(Attribute):
    """Attribute indicating if some text is a duplicate of some other text in
    another document

    Attributes
    ----------
    uid : str
        Identifier of the attribute
    label : str
        The attribute label, always set to :attr:`DuplicationAttribute.LABEL`
    value : Any, optional
        `True` if the segment or entity to which the attribute belongs is a
        duplicate of the part of another document, `False` otherwise.
    source_doc_id : str, optional
        Identifier of the document from which the text was copied
    source_spans : list of AnySpan, optional
        Spans of the duplicated text in the source document
    source_doc_date : Any, optional
        Date of the source document, if known
    """

    source_doc_id: str | None
    source_spans: list[AnySpan] | None
    # TODO do we need to duplicate this info for convenience,
    # or should source_doc_id be enough?
    source_doc_date: Any | None

    LABEL: ClassVar[str] = "is_duplicate"
    """
    Label used for all TNM attributes
    """

    def __init__(
        self,
        value: bool,
        source_doc_id: str | None = None,
        source_spans: list[AnySpan] | None = None,
        source_doc_date: Any | None = None,
        metadata: dict[str, Any] | None = None,
        uid: str | None = None,
    ):
        super().__init__(label=self.LABEL, value=value, metadata=metadata, uid=uid)

        self.source_doc_id = source_doc_id
        self.source_spans = source_spans
        self.source_doc_date = source_doc_date

    def to_dict(self) -> dict[str, Any]:
        attr_dict = {
            "uid": self.uid,
            "value": self.value,
            "source_doc_id": self.source_doc_id,
            "source_spans": self.source_spans,
            "source_doc_date": self.source_doc_date,
            "metadata": self.metadata,
        }
        dict_conv.add_class_name_to_data_dict(self, attr_dict)
        return attr_dict

    @classmethod
    def from_dict(cls, attr_dict: dict[str, Any]) -> Self:
        return cls(
            uid=attr_dict["uid"],
            value=attr_dict["value"],
            source_doc_id=attr_dict["source_doc_id"],
            source_spans=attr_dict["source_spans"],
            source_doc_date=attr_dict["source_doc_date"],
            metadata=attr_dict["metadata"],
        )


class DuplicateFinder(Operation):
    """Detect duplicated chunks of text across a collection of text documents,
    relying on the `duptextfinder` library.

    When a duplicated chunk of text is found, a segment is created on the newest
    document covering the span that is duplicated. A
    :class:`~.DuplicationAttribute` having `"is_duplicate"` as label and `True`
    as value is attached to the segment. It can later be propagated to the
    entities created from the duplicate segments.

    The attribute also holds the id of the source document from which the text
    was copied, the spans of the text in the source document, and optionally the
    date of the source document if provided.

    Optionally, segments can also be created for non-duplicate zones to make it
    easier to process only those parts of the documents. For these segments, the
    attribute value is `False` and the source, spans and date fields are `None`.

    NB: better performance may be achieved by installing the `ncls` python
    package, which will then be used by `duptextfinder` library.
    """

    _NON_EMPTY_REGEXP = re.compile(r"\w")

    def __init__(
        self,
        output_label: str,
        segments_to_output: Literal["dup", "nondup", "both"] = "dup",
        min_duplicate_length: int = 5,
        fingerprint_type: Literal["char", "word"] = "word",
        fingerprint_length: int = 2,
        date_metadata_key: str | None = None,
        case_sensitive: bool = True,
        allow_multiline: bool = True,
        orf: int = 1,
    ):
        """Parameters
        ----------
        output_label : str
            Label of created segments
        segments_to_output : str, default="dup"
            Type of segments to create: only duplicate segments (`"dup"`), only
            non-duplicate segments (`"nondup"`), or both (`"both"`)
        min_duplicate_length : int, default=5
            Minimum length of duplicated segments, in characters (shorter
            segments will be discarded)
        fingerprint_type : str, default="word"
            Base unit to use for fingerprinting (either `"char"` or `"word"`)
        fingerprint_length : int, default=2
            Number of chars or words in each fingerprint. If `fingerprint_type`
            is set to `"char"`, this should be the same value as
            `min_duplicate_length`. If `fingerprint_type` is set to `"word"`,
            this should be around the average word size multiplied by
            `min_duplicate_length`
        date_metadata_key : str, optional
            Key to use to retrieve the date of each document from their metadata
            dicts. When provided, this is used to determine which document
            should be the source of a duplicate (the older) and which document
            should be the recipient (the newer). If None, the order of the
            documents in the collection will be used.
        case_sensitive : bool, default=True
            Whether duplication detection should be case-sensitive or not
        allow_multiline : bool, default=True
            Whether detected duplicates can span across multiline lines, or
            each line should be handled separately
        orf : int, default=1
            Step size when building fingerprints, cf the `duptextfinder`
            documentation
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        self.output_label = output_label
        self._output_duplicate = segments_to_output in ("dup", "both")
        self._output_nondup = segments_to_output in ("nondup", "both")
        self.date_metadata_key = date_metadata_key
        self.fingerprint_type = fingerprint_type
        self.fingerprint_length = fingerprint_length
        self.min_duplicate_length = min_duplicate_length
        self.orf = orf
        self.case_sensitive = case_sensitive
        self.allow_multiline = allow_multiline

    def run(self, collections: list[Collection]):
        """Find duplicates in each collection of documents

        For each duplicate found, a :class:`~.core.text.Segment` object with a
        :class:`~.DuplicationAttribute` will be created and attached to the document that
        is the recipient of the duplication (ie not the source document).
        """
        for collection in collections:
            self._find_duplicate_in_docs(collection.text_docs)

    def _find_duplicate_in_docs(self, docs: list[TextDocument]):
        """Find duplicates among a set of documents"""
        # configure new fingerprint builder and duplicate finder for set of
        # documents (can't reuse the same ones for all collections because they
        # remember previously seen documents)
        if self.fingerprint_type == "char":
            fingerprint_builder = duptextfinder.CharFingerprintBuilder(
                fingerprintLength=self.fingerprint_length,
                orf=self.orf,
                caseSensitive=self.case_sensitive,
                allowMultiline=self.allow_multiline,
            )
        else:
            assert self.fingerprint_type == "word"
            fingerprint_builder = duptextfinder.WordFingerprintBuilder(
                fingerprintLength=self.fingerprint_length,
                orf=self.orf,
                caseSensitive=self.case_sensitive,
                allowMultiline=self.allow_multiline,
            )
        duplicate_finder = duptextfinder.DuplicateFinder(
            fingerprint_builder, minDuplicateLength=self.min_duplicate_length
        )

        docs_by_id = {}
        # iterate over docs, from older to newer
        if self.date_metadata_key is not None:
            docs = sorted(docs, key=lambda d: d.metadata[self.date_metadata_key])
        for doc in docs:
            self._find_duplicates_in_doc(doc, duplicate_finder, docs_by_id)

    def _find_duplicates_in_doc(
        self,
        doc: TextDocument,
        duplicate_finder: duptextfinder.DuplicateFinder,
        docs_by_id: dict[str, TextDocument],
    ):
        """Find duplicates between a document and previously processed documents

        Parameters
        ----------
        doc : TextDocument
            Document in which to look for duplicates
        duplicate_finder : DuplicateFinder
            Duplicate finder to use, that has already processed previous documents if any
        docs_by_id : dict of str to TextDocument
            Previously processed documents, by id
        """
        docs_by_id[doc.uid] = doc
        target_segment = doc.raw_segment

        # get iterator to duplicate parts
        duplicates = duplicate_finder.findDuplicates(doc.uid, target_segment.text)

        # create segments for non-duplicate and duplicate ranges
        char_cursor = 0
        for duplicate in duplicates:
            if self._output_nondup and char_cursor < duplicate.targetSpan.start:
                nondup_seg = self._create_nondup_segment(
                    target_segment,
                    range_=(char_cursor, duplicate.targetSpan.start),
                )
                if nondup_seg is not None:
                    doc.anns.add(nondup_seg)

            if self._output_duplicate:
                dup_seg = self._create_duplicate_segment(
                    target_segment,
                    target_range=(duplicate.targetSpan.start, duplicate.targetSpan.end),
                    source_range=(duplicate.sourceSpan.start, duplicate.sourceSpan.end),
                    source_doc=docs_by_id[duplicate.sourceDocId],
                )
                if dup_seg is not None:
                    doc.anns.add(dup_seg)

            char_cursor = duplicate.targetSpan.end

        # handle tail non-duplicate segment
        if self._output_nondup and char_cursor < len(target_segment.text):
            nondup_seg = self._create_nondup_segment(
                target_segment,
                range_=(char_cursor, len(target_segment.text)),
            )
            if nondup_seg is not None:
                doc.anns.add(nondup_seg)

    def _create_nondup_segment(self, target_segment, range_):
        """Create a segment representing a non-duplicated zone"""
        # "rebase" the range taking into accounts spans of the target segment
        text, spans = span_utils.extract(target_segment.text, target_segment.spans, ranges=[range_])

        # skip if empty
        if not self._NON_EMPTY_REGEXP.search(text):
            return None

        attr = DuplicationAttribute(value=False)
        segment = Segment(label=self.output_label, text=text, spans=spans, attrs=[attr])

        # handle provenance
        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(
                segment,
                self.description,
                # should we put docs instead of segments as source data items?
                source_data_items=[target_segment],
            )
            self._prov_tracer.add_prov(
                attr,
                self.description,
                # should we put docs instead of segments as source data items?
                source_data_items=[target_segment],
            )

        return segment

    def _create_duplicate_segment(self, target_segment, target_range, source_doc, source_range):
        """Create a segment representing a duplicated zone"""
        # "rebase" the target range taking into accounts spans of the target segment
        text, spans = span_utils.extract(target_segment.text, target_segment.spans, ranges=[target_range])

        # skip if empty
        if not self._NON_EMPTY_REGEXP.search(text):
            return None

        # "rebase" the source range taking into accounts spans of the source
        # segment
        source_segment = source_doc.raw_segment
        _, source_spans = span_utils.extract(source_segment.text, source_segment.spans, ranges=[source_range])

        # store info about source in duplication attr
        source_doc_date = source_doc.metadata[self.date_metadata_key] if self.date_metadata_key is not None else None
        attr = DuplicationAttribute(
            value=True,
            source_doc_id=source_doc.uid,
            source_spans=source_spans,
            source_doc_date=source_doc_date,
        )
        segment = Segment(label=self.output_label, text=text, spans=spans, attrs=[attr])

        # handle provenance
        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(
                segment,
                self.description,
                # should we put docs instead of segments as source data items?
                source_data_items=[source_segment, target_segment],
            )
            self._prov_tracer.add_prov(
                attr,
                self.description,
                # should we put docs instead of segments as source data items?
                source_data_items=[source_segment, target_segment],
            )

        return segment
