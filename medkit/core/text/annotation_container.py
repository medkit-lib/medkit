__all__ = ["TextAnnotationContainer"]

import typing
from typing import Dict, List, Optional

from medkit.core.annotation_container import AnnotationContainer
from medkit.core.text.annotation import Entity, Relation, Segment, TextAnnotation


class TextAnnotationContainer(AnnotationContainer[TextAnnotation]):
    """Manage a list of text annotations belonging to a text document.

    This behaves more or less like a list: calling `len()` and iterating are
    supported. Additional filtering is available through the `get()` method.

    Also provides retrieval of entities, segments, relations, and handling of
    raw segment.
    """

    def __init__(self, doc_id: str, raw_segment: Segment):
        super().__init__(doc_id=doc_id)

        # auto-generated raw segment
        # not stored with other annotations but injected in calls to get()
        # and get_by_id()
        self.raw_segment = raw_segment

        self._segment_ids: List[str] = []
        self._entity_ids: List[str] = []
        self._relation_ids: List[str] = []
        self._relation_ids_by_source_id: Dict[str, List[str]] = {}

    @property
    def segments(self) -> List[Segment]:
        """Return the list of segments"""
        return self.get_segments()

    @property
    def entities(self) -> List[Entity]:
        """Return the list of entities"""
        return self.get_entities()

    @property
    def relations(self) -> List[Relation]:
        """Return the list of relations"""
        return self.get_relations()

    def add(self, ann: TextAnnotation):
        if ann.label == self.raw_segment.label:
            msg = f"Cannot add annotation with reserved label {self.raw_segment.label}"
            raise RuntimeError(msg)

        super().add(ann)

        # update entity/segments/relations index
        if isinstance(ann, Entity):
            self._entity_ids.append(ann.uid)
        elif isinstance(ann, Segment):
            self._segment_ids.append(ann.uid)
        elif isinstance(ann, Relation):
            self._relation_ids.append(ann.uid)
            if ann.source_id not in self._relation_ids_by_source_id:
                self._relation_ids_by_source_id[ann.source_id] = []
            self._relation_ids_by_source_id[ann.source_id].append(ann.uid)

    def get(self, *, label: Optional[str] = None, key: Optional[str] = None) -> List[TextAnnotation]:
        # inject raw segment
        if label == self.raw_segment.label and key is None:
            return [self.raw_segment]
        return super().get(label=label, key=key)

    def get_by_id(self, uid) -> TextAnnotation:
        # inject raw segment
        if uid == self.raw_segment.uid:
            return self.raw_segment
        return super().get_by_id(uid)

    def get_segments(self, *, label: Optional[str] = None, key: Optional[str] = None) -> List[Segment]:
        """Return a list of the segments of the document (not including entities),
        optionally filtering by label or key.

        Parameters
        ----------
        label:
            Label to use to filter segments.
        key:
            Key to use to filter segments.
        """
        # get ids filtered by label/key
        uids = self.get_ids(label=label, key=key)
        # keep only segment ids
        uids = (uid for uid in uids if uid in self._segment_ids)

        segments = [self.get_by_id(uid) for uid in uids]
        return typing.cast(List[Segment], segments)

    def get_entities(self, *, label: Optional[str] = None, key: Optional[str] = None) -> List[Entity]:
        """Return a list of the entities of the document, optionally filtering
        by label or key.

        Parameters
        ----------
        label:
            Label to use to filter entities.
        key:
            Key to use to filter entities.
        """
        # get ids filtered by label/key
        uids = self.get_ids(label=label, key=key)
        # keep only entity ids
        uids = (uid for uid in uids if uid in self._entity_ids)

        entities = [self.get_by_id(uid) for uid in uids]
        return typing.cast(List[Entity], entities)

    def get_relations(
        self,
        *,
        label: Optional[str] = None,
        key: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> List[Relation]:
        """Return a list of the relations of the document, optionally filtering
        by label, key or source entity.

        Parameters
        ----------
        label:
            Label to use to filter relations.
        key:
            Key to use to filter relations.
        source_id:
            Identifier of the source entity to use to filter relations.
        """
        # get ids filtered by label/key
        uids = self.get_ids(label=label, key=key)
        # keep only relation ids
        # (either all relations or relations with specific source)
        if source_id is None:
            uids = (uid for uid in uids if uid in self._relation_ids)
        else:
            relation_ids = self._relation_ids_by_source_id.get(source_id, [])
            uids = (uid for uid in uids if uid in relation_ids)

        entities = [self.get_by_id(uid) for uid in uids]
        return typing.cast(List[Relation], entities)
