from __future__ import annotations

__all__ = ["DocumentSplitter"]

from medkit.core import Attribute, Operation
from medkit.core.text import (
    Entity,
    ModifiedSpan,
    Relation,
    Segment,
    Span,
    TextAnnotation,
    TextDocument,
    span_utils,
)
from medkit.text.postprocessing.alignment_utils import compute_nested_segments


class DocumentSplitter(Operation):
    """Split text documents using its segments as a reference.

    The resulting 'mini-documents' contain the entities belonging to each
    segment along with their attributes.

    This operation can be used to create datasets from medkit text documents.
    """

    def __init__(
        self,
        segment_label: str,
        entity_labels: list[str] | None = None,
        attr_labels: list[str] | None = None,
        relation_labels: list[str] | None = None,
        name: str | None = None,
        uid: str | None = None,
    ):
        """Instantiate the document splitter

        Parameters
        ----------
        segment_label : str
            Label of the segments to use as references for the splitter
        entity_labels : list of str, optional
            Labels of entities to be included in the mini documents.
            If None, all entities from the document will be included.
        attr_labels : list of str, optional
            Labels of the attributes to be included into the new annotations.
            If None, all attributes will be included.
        relation_labels : list of str, optional
            Labels of relations to be included in the mini documents.
            If None, all relations will be included.
        name : str, optional
            Name describing the splitter (default to the class name).
        uid : str, Optional
            Identifier of the operation
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        self.segment_label = segment_label
        self.entity_labels = entity_labels
        self.attr_labels = attr_labels
        self.relation_labels = relation_labels

    def run(self, docs: list[TextDocument]) -> list[TextDocument]:
        """Split docs into mini documents

        Parameters
        ----------
        docs: list of TextDocument
            List of text documents to split

        Returns
        -------
        list of TextDocument
            List of documents created from the selected segments
        """
        segment_docs = []

        for doc in docs:
            segments = doc.anns.get_segments(label=self.segment_label)

            # filter entities
            entities = (
                doc.anns.get_entities()
                if self.entity_labels is None
                else [ent for label in self.entity_labels for ent in doc.anns.get_entities(label=label)]
            )

            # align segment and entities (fully contained)
            segment_and_entities = compute_nested_segments(segments, entities)

            # filter relations in the document
            relations = (
                doc.anns.get_relations()
                if self.relation_labels is None
                else [rel for label in self.relation_labels for rel in doc.anns.get_relations(label=label)]
            )

            # Iterate over all segments and corresponding nested entities
            for segment, nested_entities in segment_and_entities:
                # filter relations in nested entities
                entities_uid = {ent.uid for ent in nested_entities}
                nested_relations = [
                    relation
                    for relation in relations
                    if relation.source_id in entities_uid and relation.target_id in entities_uid
                ]
                # create new document from segment
                segment_doc = self._create_segment_doc(
                    segment=segment,
                    entities=nested_entities,
                    relations=nested_relations,
                    doc_source=doc,
                )
                segment_docs.append(segment_doc)

        return segment_docs

    def _create_segment_doc(
        self,
        segment: Segment,
        entities: list[Entity],
        relations: list[Relation],
        doc_source: TextDocument,
    ) -> TextDocument:
        """Create a TextDocument from a segment and its entities.
        The original zone of the segment becomes the text of the document.

        Parameters
        ----------
        segment : Segment
            Segment to use as reference for the new document
        entities : list of Entity
            Entities inside the segment
        relations : list of Relation
            Relations inside the segment
        doc_source : TextDocument
            Initial document from which annotations where extracted

        Returns
        -------
        TextDocument
            A new document with entities, the metadata includes the original span and metadata
        """
        normalized_spans = span_utils.normalize_spans(segment.spans)

        # create an empty mini-doc with the raw text of the segment
        offset, end_span = normalized_spans[0].start, normalized_spans[-1].end
        metadata = doc_source.metadata.copy()
        metadata.update(segment.metadata)

        segment_doc = TextDocument(text=doc_source.text[offset:end_span], metadata=metadata)

        # handle provenance
        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(segment_doc, self.description, source_data_items=[segment])

        # Copy segment attributes
        segment_attrs = self._filter_attrs_from_ann(segment)
        for attr in segment_attrs:
            new_doc_attr = attr.copy()
            segment_doc.attrs.add(new_doc_attr)
            # handle provenance
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(
                    new_doc_attr,
                    self.description,
                    source_data_items=[attr],
                )

        # Add selected entities
        uid_mapping = {}
        for ent in entities:
            spans = []
            for span in ent.spans:
                # relocate entity spans using segment offset
                if isinstance(span, Span):
                    spans.append(Span(span.start - offset, span.end - offset))
                else:
                    replaced_spans = [Span(sp.start - offset, sp.end - offset) for sp in span.replaced_spans]
                    spans.append(ModifiedSpan(length=span.length, replaced_spans=replaced_spans))
            # define the new entity
            relocated_ent = Entity(
                text=ent.text,
                label=ent.label,
                spans=spans,
                metadata=ent.metadata.copy(),
            )
            # add mapping for relations
            uid_mapping[ent.uid] = relocated_ent.uid

            # handle provenance
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(relocated_ent, self.description, source_data_items=[ent])

            # Copy entity attributes
            entity_attrs = self._filter_attrs_from_ann(ent)
            for attr in entity_attrs:
                new_ent_attr = attr.copy()
                relocated_ent.attrs.add(new_ent_attr)
                # handle provenance
                if self._prov_tracer is not None:
                    self._prov_tracer.add_prov(
                        new_ent_attr,
                        self.description,
                        source_data_items=[attr],
                    )

            # add entity to the new document
            segment_doc.anns.add(relocated_ent)

        for rel in relations:
            relation = Relation(
                label=rel.label,
                source_id=uid_mapping[rel.source_id],
                target_id=uid_mapping[rel.target_id],
                metadata=rel.metadata.copy(),
            )
            # handle provenance
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(relation, self.description, source_data_items=[rel])

            # Copy relation attributes
            relation_attrs = self._filter_attrs_from_ann(rel)
            for attr in relation_attrs:
                new_rel_attr = attr.copy()
                relation.attrs.add(new_rel_attr)
                # handle provenance
                if self._prov_tracer is not None:
                    self._prov_tracer.add_prov(
                        new_rel_attr,
                        self.description,
                        source_data_items=[attr],
                    )

            # add relation to the new document
            segment_doc.anns.add(relation)

        return segment_doc

    def _filter_attrs_from_ann(self, ann: TextAnnotation) -> list[Attribute]:
        """Filter attributes from an annotation using 'attr_labels'"""
        return (
            ann.attrs.get()
            if self.attr_labels is None
            else [attr for label in self.attr_labels for attr in ann.attrs.get(label=label)]
        )
