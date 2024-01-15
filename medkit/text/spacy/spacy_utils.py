from __future__ import annotations

__all__ = [
    "extract_anns_and_attrs_from_spacy_doc",
    "build_spacy_doc_from_medkit_doc",
    "build_spacy_doc_from_medkit_segment",
]

import warnings
from typing import TYPE_CHECKING, Callable

from spacy.tokens import Doc
from spacy.tokens import Span as SpacySpan
from spacy.tokens.underscore import Underscore
from spacy.util import filter_spans

from medkit.core import Attribute
from medkit.core.text import AnySpan, Entity, Segment, Span, TextDocument, span_utils
from medkit.io._common import get_anns_by_type

if TYPE_CHECKING:
    from spacy import Language

_ATTR_MEDKIT_ID = "medkit_id"


def extract_anns_and_attrs_from_spacy_doc(
    spacy_doc: Doc,
    medkit_source_ann: Segment | None = None,
    entities: list[str] | None = None,
    span_groups: list[str] | None = None,
    attrs: list[str] | None = None,
    attribute_factories: dict[str, Callable[[SpacySpan, str], Attribute]] | None = None,
    rebuild_medkit_anns_and_attrs: bool = False,
) -> tuple[list[Segment], dict[str, list[Attribute]]]:
    """Given a spacy document, convert selected entities or spans into Segments.
    Extract attributes for each annotation in the document.

    Parameters
    ----------
    spacy_doc : Doc
         A Spacy Doc with spans to be converted
    medkit_source_ann : Segment, optional
        Segment used to rebuild spans referencing the original text
    entities : list of str, optional
        Labels of entities to be extracted
        If `None` (default) all new entities will be extracted as annotations
    span_groups : list of str, optional
        Name of span groups to be extracted
        If `None` (default) all new spans will be extracted as annotations
    attrs : list of str, optional
        Name of custom attributes to extract from the annotations that will be included.
        If `None` (default) all the custom attributes will be extracted
    attribute_factories : dict of str to Callable, optional
        Mapping of factories in charge of converting spacy attributes to medkit
        attributes. Factories will receive a spacy span and an attribute label
        when called. The key in the mapping is the attribute label.
    rebuild_medkit_anns_and_attrs : bool, default=False
        If True the annotations and attributes with medkit ids will become
        new annotations/attributes with new ids.
        If False (default) the annotations and attributes with medkit ids are not
        rebuilt, only new annotations and attributes are returned

    Returns
    -------
    annotations: list of Segment
        Segments extracted from the spacy Doc object
    attributes_by_ann: dict of str to list of Attribute
        Attributes extracted for each annotation, the key is a medkit uid

    Raises
    ------
    ValueError
        Raises when the given medkit source and the spacy doc do not have the same medkit uid
    """
    if attribute_factories is None:
        attribute_factories = {}

    # extensions to indicate the medkit origin
    _define_default_extensions()
    spacy_doc_medkit_id = spacy_doc._.get(_ATTR_MEDKIT_ID)
    if spacy_doc_medkit_id and medkit_source_ann and medkit_source_ann.uid != spacy_doc_medkit_id:
        msg = (
            "The medkit uid of the Doc object is"
            f" {spacy_doc_medkit_id}, the medkit source annotation"
            f" provided has a different uid: {medkit_source_ann.uid}."
        )
        raise ValueError(msg)

    # get annotations according to entities and name_spans_to_transfer
    spacy_entities = _get_ents_by_label(spacy_doc, entities)
    spacy_spans = _get_spans_by_label(spacy_doc, span_groups)
    spacy_attrs = _get_custom_attrs_by_label(rebuild_medkit_anns_and_attrs, attrs)

    annotations = []
    attributes_by_ann = {}

    # convert spacy entities
    for entity_spacy in spacy_entities:
        medkit_id = entity_spacy._.get(_ATTR_MEDKIT_ID)

        if medkit_id is None or rebuild_medkit_anns_and_attrs:
            # create a new entity annotation
            label = entity_spacy.label_
            text, spans = _get_text_and_spans_from_span_spacy(
                span_spacy=entity_spacy, medkit_source_ann=medkit_source_ann
            )

            entity = Entity(label=label, spans=spans, text=text)
            medkit_id = entity.uid
            annotations.append(entity)

        # for each spacy extension having a value other than None,
        # a medkit Attribute is created
        attributes = []
        for attr_label in spacy_attrs:
            value = entity_spacy._.get(attr_label)
            if value is None:
                continue
            factory = attribute_factories.get(attr_label)
            attribute = factory(entity_spacy, attr_label) if factory else Attribute(attr_label, value)
            attributes.append(attribute)

        if attributes:
            attributes_by_ann[medkit_id] = attributes

    # convert spacy span groups
    for label, spans in spacy_spans.items():
        for span_spacy in spans:
            # ignore spans that have a corresponding entity
            # (some matchers, for instance in EDS-NLP create both an entity and
            # a span for each match)
            if span_spacy in spacy_entities:
                continue

            medkit_id = span_spacy._.get(_ATTR_MEDKIT_ID)

            if medkit_id is None or rebuild_medkit_anns_and_attrs:
                # create new segment annotation
                text, new_spans = _get_text_and_spans_from_span_spacy(
                    span_spacy=span_spacy, medkit_source_ann=medkit_source_ann
                )
                segment = Segment(
                    label=label,
                    spans=new_spans,
                    text=text,
                    attrs=[],
                    metadata={"name": span_spacy.label_},
                )
                # 'label' represents 'span_key' from spacy
                # 'name' in metadata represents the original label of the span in spacy
                medkit_id = segment.uid
                annotations.append(segment)

            # for each spacy extension having a value other than None,
            # a medkit Attribute is created
            attributes = []
            for attr_label in spacy_attrs:
                value = span_spacy._.get(attr_label)
                if value is None:
                    continue
                factory = attribute_factories.get(attr_label)
                attribute = factory(span_spacy, attr_label) if factory else Attribute(attr_label, value)
                attributes.append(attribute)

            if attributes:
                attributes_by_ann[medkit_id] = attributes

    return annotations, attributes_by_ann


def build_spacy_doc_from_medkit_doc(
    nlp: Language,
    medkit_doc: TextDocument,
    labels_anns: list[str] | None = None,
    attrs: list[str] | None = None,
    include_medkit_info: bool = True,
) -> Doc:
    """Create a Spacy Doc from a TextDocument.

    Parameters
    ----------
    nlp:
        Language object with the loaded pipeline from Spacy
    medkit_doc:
        TextDocument to convert
    labels_anns:
        Labels of annotations to include in the spacy document.
        If `None` (default) all the annotations will be included.
    attrs:
        Labels of attributes to add in the annotations that will be included.
        If `None` (default) all the attributes will be added as `custom attributes`
        in each annotation included.
    include_medkit_info:
        If True, medkitID is included as an extension in the Doc object
        to identify the medkit source annotation.
        If False, no information about IDs is included

    Returns
    -------
    Doc:
        A Spacy Doc with the selected annotations included.
    """
    # extensions to indicate the medkit origin
    _define_default_extensions()

    # get the raw text segment to transfer
    raw_segment = medkit_doc.raw_segment
    annotations = get_anns_by_type(medkit_doc, anns_labels=labels_anns)

    # create a spacy doc
    return build_spacy_doc_from_medkit_segment(
        nlp=nlp,
        segment=raw_segment,
        annotations=annotations["segments"] + annotations["entities"],
        attrs=attrs,
        include_medkit_info=include_medkit_info,
    )


def build_spacy_doc_from_medkit_segment(
    nlp: Language,
    segment: Segment,
    annotations: list[Segment] | None = None,
    attrs: list[str] | None = None,
    include_medkit_info: bool = True,
) -> Doc:
    """Create a Spacy Doc from a Segment.

    Parameters
    ----------
    nlp:
        Language object with the loaded pipeline from Spacy
    segment:
        Segment to convert, this annotation contains the text to create the spacy doc
    annotations:
        List of annotations in `segment` to include
    attrs:
        Labels of attributes to add in the annotations that will be included.
        If `None` (default) all the attributes will be added as `custom attributes`
        in each annotation included.
    include_medkit_info:
        If True, medkitID is included as an extension in the Doc object
        to identify the medkit source annotation.
        If False, no information about IDs is included.

    Returns
    -------
    Doc:
        A Spacy Doc with the selected annotations included.
    """
    # extensions to indicate the medkit origin
    _define_default_extensions()

    # create spacy doc
    doc = nlp.make_doc(segment.text)
    if include_medkit_info:
        doc._.set(_ATTR_MEDKIT_ID, segment.uid)

    annotations = annotations or []
    if not annotations:
        return doc

    # include annotations in the Doc object
    # define custom attributes in spacy from selected annotations
    if attrs is None:
        # include all attributes
        attrs = {attr.label for ann in annotations for attr in ann.attrs}
    _define_attrs_extensions(attrs)

    entities = []
    segments = []
    for ann in annotations:
        if isinstance(ann, Entity):
            # intermediate list to check for overlaps
            entities.append(ann)
        elif isinstance(ann, Segment):
            segments.append(ann)

    _add_entities_in_spacy_doc(
        spacy_doc=doc,
        entities=entities,
        attrs=attrs,
        include_medkit_info=include_medkit_info,
    )

    _add_segments_in_spacy_doc(
        spacy_doc=doc,
        segments=segments,
        attrs=attrs,
        include_medkit_info=include_medkit_info,
    )
    return doc


def _add_entities_in_spacy_doc(spacy_doc: Doc, entities: list[Entity], attrs: list[str], include_medkit_info: bool):
    """Convert entities into spacy spans and modifies
    the entities in the Doc object (doc.ents)
    """
    # create an intermediate list to check for overlaps
    spacy_entities = []
    for medkit_ent in entities:
        spacy_span = _segment_to_spacy_span(
            spacy_doc_target=spacy_doc,
            medkit_segment=medkit_ent,
            attrs=attrs,
            include_medkit_info=include_medkit_info,
        )
        spacy_entities.append(spacy_span)
    # since Spacy does not allow overlaps in entities,
    # `filter_spans` suppresses duplicates or overlaps.
    ents_filtered = filter_spans(spacy_entities)
    # overwrite entities in the document, ensure the transfer
    # of the medkit entities
    spacy_doc.ents = ents_filtered

    discarded_str = "--".join([ent.text for ent in spacy_entities if ent not in ents_filtered])
    if discarded_str:
        warnings.warn(
            f"Spacy does not allow entity overlapping, these entities ({discarded_str})" "  were discarded",
            stacklevel=2,
        )


def _add_segments_in_spacy_doc(
    spacy_doc: Doc,
    segments: list[Segment],
    attrs: list[str],
    include_medkit_info: bool,
):
    """Convert segments into a spacy spans and modifies
    the spans in the Doc object (doc.spans)
    """
    for medkit_seg in segments:
        spacy_span = _segment_to_spacy_span(
            spacy_doc_target=spacy_doc,
            medkit_segment=medkit_seg,
            attrs=attrs,
            include_medkit_info=include_medkit_info,
        )
        # it is not necessary to check overlaps,
        # the spans are added directly into the Doc object
        if medkit_seg.label not in spacy_doc.spans:
            spacy_doc.spans[medkit_seg.label] = [spacy_span]
        else:
            spacy_doc.spans[medkit_seg.label].append(spacy_span)


def _get_defined_spacy_attrs(include_medkit_attrs: bool = False) -> list[str]:
    """Returns the name of the custom attributes configured in spacy spans.

    Parameters
    ----------
    include_medkit_attrs:
        If True, medkit attrs (attrs transferred from medkit) are included

    Returns
    -------
    List[str]:
        Name of spans extensions defined in Spacy
    """
    # `get_state` is a spacy function, it returns a tuple of dictionaries
    # with the information of the defined extensions (custom attributes)
    # where ([0]= token_extensions,[1]=span_extensions,[2]=doc_extensions)
    available_attrs = Underscore.get_state()[1].keys()
    # remove default medkit attributes
    attrs = [attr for attr in available_attrs if not attr.endswith(_ATTR_MEDKIT_ID)]
    if include_medkit_attrs:
        return attrs
    # does not include medkit-defined attributes
    # remove attrs that have a medkit ID
    return [attr for attr in attrs if f"{attr}_{_ATTR_MEDKIT_ID}" not in available_attrs]


def _define_spacy_span_extension(custom_attr: str):
    if not SpacySpan.has_extension(custom_attr):
        SpacySpan.set_extension(custom_attr, default=None)


def _define_spacy_doc_extension(custom_attr: str):
    if not Doc.has_extension(custom_attr):
        Doc.set_extension(custom_attr, default=None)


def _define_default_extensions():
    """Define default attributes to identify origin from medkit"""
    _define_spacy_doc_extension(_ATTR_MEDKIT_ID)
    _define_spacy_span_extension(_ATTR_MEDKIT_ID)


def _define_attrs_extensions(attrs_to_transfer: list[str]):
    """Define attributes as span extensions in the Spacy context."""
    for attr in attrs_to_transfer:
        # `attr_medkit_id` is the medkit ID of the original attribute
        _define_spacy_span_extension(f"{attr}_{_ATTR_MEDKIT_ID}")
        _define_spacy_span_extension(attr)


def _get_span_boundaries(spans: list[AnySpan]) -> tuple[int, int]:
    """Return boundaries (start,end) from a list of spans"""
    spans_norm: list[Span] = span_utils.normalize_spans(spans)
    start = spans_norm[0].start
    end = spans_norm[-1].end

    if len(spans_norm) > 1:
        # Spacy does not allow discontinuous spans
        # for compatibility, get a continuous span from the list
        warnings.warn(
            f"These spans {spans} are discontinuous, they were converted"
            f" into its expanded version, from {start} to {end}.",
            stacklevel=2,
        )

    return (start, end)


def _segment_to_spacy_span(
    spacy_doc_target: Doc,
    medkit_segment: Segment,
    attrs: list[str],
    include_medkit_info: bool,
) -> Span:
    """Create a spacy span given a medkit segment."""
    # create a spacy span from characters in the text instead of tokens
    start, end = _get_span_boundaries(medkit_segment.spans)
    label = medkit_segment.metadata.get("name", medkit_segment.label)
    span = spacy_doc_target.char_span(start, end, alignment_mode="expand", label=label)

    if include_medkit_info:
        span._.set(_ATTR_MEDKIT_ID, medkit_segment.uid)

    for label in attrs:
        for attr in medkit_segment.attrs.get(label=label):
            value = attr.to_spacy()
            if value is None:
                # in medkit having an attribute, indicates that the attribute exists
                # for the given annotation, we force True as value
                value = True
            # set attributes as extensions
            span._.set(attr.label, value)
            if include_medkit_info:
                span._.set(f"{attr.label}_{_ATTR_MEDKIT_ID}", attr.uid)

    return span


def _get_text_and_spans_from_span_spacy(
    span_spacy: SpacySpan, medkit_source_ann: Segment | None
) -> tuple[str, list[AnySpan]]:
    """Return text and spans depending on the origin of the spacy span"""
    if medkit_source_ann is None:
        text = span_spacy.text
        spans = [Span(span_spacy.start_char, span_spacy.end_char)]
    else:
        # the origin is a medkit annotation
        text, spans = span_utils.extract(
            medkit_source_ann.text,
            medkit_source_ann.spans,
            [(span_spacy.start_char, span_spacy.end_char)],
        )
    return text, spans


def _get_ents_by_label(spacy_doc: Doc, entities: list[str] | None = None) -> list[SpacySpan]:
    return [ent for ent in spacy_doc.ents if ent.label_ in entities] if entities else list(spacy_doc.ents)


def _get_spans_by_label(spacy_doc: Doc, span_groups: list[str] | None = None) -> dict[str, list[SpacySpan]]:
    if span_groups is None:
        spans = dict(spacy_doc.spans)
    else:
        spans = {label: sp for label, sp in spacy_doc.spans.items() if label in span_groups}
    return spans


def _get_custom_attrs_by_label(rebuild_medkit_anns_and_attrs: bool, attributes: list[str] | None = None) -> list[str]:
    spacy_attrs = _get_defined_spacy_attrs(rebuild_medkit_anns_and_attrs)
    if attributes is not None:
        # filter attributes by label
        spacy_attrs = [attr for attr in spacy_attrs if attr in attributes]

    return spacy_attrs
