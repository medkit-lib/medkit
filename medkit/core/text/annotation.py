from __future__ import annotations

__all__ = ["TextAnnotation", "Segment", "Entity", "Relation"]

import abc
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from medkit.core.annotation import Annotation, Attribute
from medkit.core.text import span_utils
from medkit.core.text.normalization import EntityNormalization
from medkit.core.text.span import AnySpan, AnySpanType

if TYPE_CHECKING:
    from medkit.core.text.document import TextDocument


class TextAnnotation(Annotation):
    """Base abstract class for all text annotations"""

    @abc.abstractmethod
    def __init__(
        self,
        label: str,
        attrs: Optional[List[Attribute]] = None,
        uid: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        keys: Optional[Set[str]] = None,
    ):
        super().__init__(
            label=label, attrs=attrs, uid=uid, metadata=metadata, keys=keys
        )


class Segment(TextAnnotation):
    def __init__(
        self,
        label: str,
        spans: List[AnySpanType],
        text: str,
        attrs: Optional[List[Attribute]] = None,
        uid: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        keys: Optional[Set[str]] = None,
    ):
        """
        Initialize a medkit segment

        Parameters
        ----------
        label: str
            The label for this annotation (e.g., SENTENCE)
        spans:
            The annotation span
        text: str
            The annotation text
        attrs:
            The attributes of the segment
        uid: str, Optional
            The identifier of the annotation (if existing)
        metadata: dict[str, Any], Optional
            The metadata of the annotation
        keys:
            Pipeline output keys to which the segment belongs to.
        """
        super().__init__(
            uid=uid, label=label, attrs=attrs, metadata=metadata, keys=keys
        )
        self.spans: List[AnySpanType] = spans
        self.text: str = text

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(spans=[s.to_dict() for s in self.spans], text=self.text)
        data.update(class_name=self.__class__.__name__)
        return data

    @classmethod
    def from_dict(cls, segment_dict: Dict[str, Any]) -> Segment:
        """
        Creates a Segment from a dict

        Parameters
        ----------
        segment_dict: dict
            A dictionary from a serialized segment as generated by to_dict()
        """
        # TODO: there is an issue for duplicate attributes (TOFIX with attr storage)
        attrs = [Attribute.from_dict(a) for a in segment_dict["attrs"]]
        spans = [AnySpan.from_dict(span) for span in segment_dict["spans"]]
        segment = cls(
            uid=segment_dict["uid"],
            label=segment_dict["label"],
            attrs=attrs,
            spans=spans,
            text=segment_dict["text"],
            metadata=segment_dict["metadata"],
            keys=set(segment_dict["keys"]),
        )

        return segment

    def get_snippet(self, doc: TextDocument, max_extend_length: int) -> str:
        """Return a portion of the original text contaning the annotation

        Parameters
        ----------
        doc:
            The document to which the annotation is attached

        max_extend_length:
            Maximum number of characters to use around the annotation

        Returns
        -------
        str:
            A portion of the text around the annotation
        """
        spans_normalized = span_utils.normalize_spans(self.spans)
        start = min(s.start for s in spans_normalized)
        end = max(s.end for s in spans_normalized)
        start_extended = max(start - max_extend_length // 2, 0)
        remaining_max_extend_length = max_extend_length - (start - start_extended)
        end_extended = min(end + remaining_max_extend_length, len(doc.text))
        return doc.text[start_extended:end_extended]


class Entity(Segment):
    NORM_LABEL = "NORMALIZATION"
    """
    Label to use for normalization attributes
    """

    def __init__(
        self,
        label: str,
        spans: List[AnySpanType],
        text: str,
        attrs: Optional[List[Attribute]] = None,
        uid: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        keys: Optional[Set[str]] = None,
    ):
        """
        Initialize a medkit text entity

        Parameters
        ----------
        label: str
            The entity label
        spans:
            The entity span
        text: str
            The entity text
        attrs:
            The attributes of the entity
        uid: str, Optional
            The identifier of the entity (if existing)
        metadata: dict[str, Any], Optional
            The metadata of the entity
        keys:
            Pipeline output keys to which the entity belongs to.
        """
        super().__init__(
            label=label,
            spans=spans,
            text=text,
            attrs=attrs,
            uid=uid,
            metadata=metadata,
            keys=keys,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(class_name=self.__class__.__name__)
        return data

    @classmethod
    def from_dict(cls, entity_dict: Dict[str, Any]) -> Segment:
        """
        Creates an Entity from a dict

        Parameters
        ----------
        entity_dict: dict
            A dictionary from a serialized entity as generated by to_dict()
        """
        attrs = [Attribute.from_dict(a) for a in entity_dict["attrs"]]
        spans = [AnySpan.from_dict(span) for span in entity_dict["spans"]]

        entity = cls(
            uid=entity_dict["uid"],
            label=entity_dict["label"],
            attrs=attrs,
            spans=spans,
            text=entity_dict["text"],
            metadata=entity_dict["metadata"],
            keys=set(entity_dict["keys"]),
        )

        return entity

    def add_norm(self, normalization: EntityNormalization) -> Attribute:
        """
        Attach an :class:`~medkit.core.text.normalization.EntityNormalization`
        object to the entity.

        This helper will wrap `normalization` in an
        :class:`~medkit.core.annotation.Attribute` with
        :attr:`Entity.NORM_LABEL` as label and add it to the entity.

        Returns
        -------
        Attribute:
            The attribute that was created and added to the entity
        """

        attr = Attribute(label=self.NORM_LABEL, value=normalization)
        self.add_attr(attr)
        return attr

    def get_norms(self) -> List[EntityNormalization]:
        """
        Return all :class:`~medkit.core.text.normalization.EntityNormalization`
        objects attached to the entity.

        This helper will retrieve all the entity attributes with
        :attr:`Entity.NORM_LABEL` as label and return their
        :class:`~medkit.core.text.normalization.EntityNormalization` values.

        Returns
        -------
        List[EntityNormalization]:
            All normalizations attached to the entity.
        """
        return [a.value for a in self.get_attrs_by_label(label=self.NORM_LABEL)]


class Relation(TextAnnotation):
    def __init__(
        self,
        label: str,
        source_id: str,
        target_id: str,
        attrs: Optional[List[Attribute]] = None,
        uid: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        keys: Optional[Set[str]] = None,
    ):
        """
        Initialize the medkit relation

        Parameters
        ----------
        label: str
            The relation label
        source_id: str
            The identifier of the entity from which the relation is defined
        target_id: str
            The identifier of the entity to which the relation is defined
        attrs:
            The attributes of the relation
        uid: str, Optional
            The identifier of the relation (if existing)
        metadata: Dict[str, Any], Optional
            The metadata of the relation
        keys:
            Pipeline output keys to which the relation belongs to.
        """
        super().__init__(
            uid=uid, label=label, attrs=attrs, metadata=metadata, keys=keys
        )
        self.source_id: str = source_id
        self.target_id: str = target_id

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(
            source_id=self.source_id,
            target_id=self.target_id,
            class_name=self.__class__.__name__,
        )
        return data

    @classmethod
    def from_dict(cls, relation_dict):
        """
        Creates a Relation from a dict

        Parameters
        ----------
        relation_dict: dict
            A dictionary from a serialized relation as generated by to_dict()
        """

        attrs = [Attribute.from_dict(a) for a in relation_dict["attrs"]]

        relation = cls(
            uid=relation_dict["uid"],
            label=relation_dict["label"],
            attrs=attrs,
            source_id=relation_dict["source_id"],
            target_id=relation_dict["target_id"],
            metadata=relation_dict["metadata"],
            keys=set(relation_dict["keys"]),
        )

        return relation
