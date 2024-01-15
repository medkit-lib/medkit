from __future__ import annotations

__all__ = ["TextAnnotationContainer"]

from typing import List, cast

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

        self._segment_ids: list[str] = []
        self._entity_ids: list[str] = []
        self._relation_ids: list[str] = []
        self._relation_ids_by_source_id: dict[str, list[str]] = {}

    @property
    def segments(self) -> list[Segment]:
        """Return the list of segments"""
        return self.get_segments()

    @property
    def entities(self) -> list[Entity]:
        """Return the list of entities"""
        return self.get_entities()

    @property
    def relations(self) -> list[Relation]:
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

    def get(self, *, label: str | None = None, key: str | None = None) -> list[TextAnnotation]:
        # inject raw segment
        if label == self.raw_segment.label and key is None:
            return [self.raw_segment]
        return super().get(label=label, key=key)

    def get_by_id(self, uid) -> TextAnnotation:
        # inject raw segment
        if uid == self.raw_segment.uid:
            return self.raw_segment
        return super().get_by_id(uid)

    def get_segments(self, *, label: str | None = None, key: str | None = None) -> list[Segment]:
        """Return a list of the segments of the document (not including entities),
        optionally filtering by label or key.

        Parameters
        ----------
        label : str, optional
            Label to use to filter segments.
        key : str, optional
            Key to use to filter segments.
        """
        # get ids filtered by label/key
        uids = self.get_ids(label=label, key=key)
        # keep only segment ids
        uids = (uid for uid in uids if uid in self._segment_ids)

        segments = [self.get_by_id(uid) for uid in uids]
        return cast(List[Segment], segments)

    def get_entities(self, *, label: str | None = None, key: str | None = None) -> list[Entity]:
        """Return a list of the entities of the document, optionally filtering
        by label or key.

        Parameters
        ----------
        label : str, optional
            Label to use to filter entities.
        key : str, optional
            Key to use to filter entities.
        """
        # get ids filtered by label/key
        uids = self.get_ids(label=label, key=key)
        # keep only entity ids
        uids = (uid for uid in uids if uid in self._entity_ids)

        entities = [self.get_by_id(uid) for uid in uids]
        return cast(List[Entity], entities)

    def get_relations(
        self,
        *,
        label: str | None = None,
        key: str | None = None,
        source_id: str | None = None,
    ) -> list[Relation]:
        """Return a list of the relations of the document, optionally filtering
        by label, key or source entity.

        Parameters
        ----------
        label : str, optional
            Label to use to filter relations.
        key : str, optional
            Key to use to filter relations.
        source_id : str, optional
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
        return cast(List[Relation], entities)
