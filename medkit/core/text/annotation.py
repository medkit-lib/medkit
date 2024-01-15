from __future__ import annotations

__all__ = ["TextAnnotation", "Segment", "Entity", "Relation"]

import abc
import dataclasses
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from medkit.core import dict_conv
from medkit.core.attribute import Attribute
from medkit.core.attribute_container import AttributeContainer
from medkit.core.id import generate_id
from medkit.core.text.entity_attribute_container import EntityAttributeContainer
from medkit.core.text.span import AnySpan

if TYPE_CHECKING:
    from medkit.core.store import Store


@dataclasses.dataclass(init=False)
class TextAnnotation(abc.ABC, dict_conv.SubclassMapping):
    """Base abstract class for all text annotations

    Attributes
    ----------
    uid : str
        Unique identifier of the annotation.
    label : str
        The label for this annotation (e.g., SENTENCE)
    attrs : AttributeContainer
        Attributes of the annotation. Stored in a
        :class:{~medkit.core.AttributeContainer} but can be passed as a list at
        init.
    metadata : dict of str to Any
        The metadata of the annotation
    keys : set of str
        Pipeline output keys to which the annotation belongs to.
    """

    uid: str
    label: str
    attrs: AttributeContainer
    metadata: dict[str, Any]
    keys: set[str]

    @abc.abstractmethod
    def __init__(
        self,
        label: str,
        attrs: list[Attribute] | None = None,
        metadata: dict[str, Any] | None = None,
        uid: str | None = None,
        attr_container_class: type[AttributeContainer] = AttributeContainer,
    ):
        if attrs is None:
            attrs = []
        if metadata is None:
            metadata = {}
        if uid is None:
            uid = generate_id()

        self.uid = uid
        self.label = label
        self.metadata = metadata
        self.keys = set()

        self.attrs = attr_container_class(owner_id=self.uid)
        for attr in attrs:
            self.attrs.add(attr)

    def __init_subclass__(cls):
        TextAnnotation.register_subclass(cls)
        super().__init_subclass__()

    @classmethod
    def from_dict(cls, ann_dict: dict[str, Any]) -> Self:
        subclass = cls.get_subclass_for_data_dict(ann_dict)
        if subclass is None:
            msg = (
                "TextAnnotation is an abstract class. Its class method `from_dict` is"
                " only used for calling the correct subclass `from_dict`. Subclass is"
                f" {subclass}"
            )
            raise NotImplementedError(msg)
        return subclass.from_dict(ann_dict)

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclasses.dataclass(init=False)
class Segment(TextAnnotation):
    """Text segment referencing part of an :class:`~medkit.core.text.TextDocument`.

    Attributes
    ----------
    uid : str
        The segment identifier.
    label : str
        The label for this segment (e.g., SENTENCE)
    text : str
        Text of the segment.
    spans : list of AnySpan
        List of spans indicating which parts of the segment text correspond to
        which part of the document's full text.
    attrs : AttributeContainer
        Attributes of the segment. Stored in a
        :class:{~medkit.core.AttributeContainer} but can be passed as a list at
        init.
    metadata : dict of str to Any
        The metadata of the segment
    keys : set of str
        Pipeline output keys to which the segment belongs to.
    """

    spans: list[AnySpan]
    text: str

    def __init__(
        self,
        label: str,
        text: str,
        spans: list[AnySpan],
        attrs: list[Attribute] | None = None,
        metadata: dict[str, Any] | None = None,
        uid: str | None = None,
        store: Store | None = None,
        attr_container_class: type[AttributeContainer] = AttributeContainer,
    ):
        super().__init__(
            label=label,
            attrs=attrs,
            metadata=metadata,
            uid=uid,
            attr_container_class=attr_container_class,
        )

        self.text = text
        self.spans = spans

        # check if spans length is equal to text length
        length = sum(s.length for s in self.spans)
        assert len(self.text) == length, "Spans length does not match text length"

    def to_dict(self) -> dict[str, Any]:
        spans = [s.to_dict() for s in self.spans]
        attrs = [a.to_dict() for a in self.attrs]
        segment_dict = {
            "uid": self.uid,
            "label": self.label,
            "text": self.text,
            "spans": spans,
            "attrs": attrs,
            "metadata": self.metadata,
        }
        dict_conv.add_class_name_to_data_dict(self, segment_dict)
        return segment_dict

    @classmethod
    def from_dict(cls, segment_dict: dict[str, Any]) -> Self:
        """Creates a Segment from a dict

        Parameters
        ----------
        segment_dict : dict of str to Any
            A dictionary from a serialized segment as generated by to_dict()
        """
        spans = [AnySpan.from_dict(s) for s in segment_dict["spans"]]
        attrs = [Attribute.from_dict(a) for a in segment_dict["attrs"]]
        return cls(
            uid=segment_dict["uid"],
            label=segment_dict["label"],
            text=segment_dict["text"],
            spans=spans,
            attrs=attrs,
            metadata=segment_dict["metadata"],
        )


@dataclasses.dataclass(init=False)
class Entity(Segment):
    """Text entity referencing part of an :class:`~medkit.core.text.TextDocument`.

    Attributes
    ----------
    uid : str
        The entity identifier.
    label : str
        The label for this entity (e.g., DISEASE)
    text : str
        Text of the entity.
    spans : list of AnySpan
        List of spans indicating which parts of the entity text correspond to
        which part of the document's full text.
    attrs : EntityAttributeContainer
        Attributes of the entity. Stored in a
        :class:{~medkit.core.EntityAttributeContainer} but can be passed as a list at
        init.
    metadata : dict of str to Any
        The metadata of the entity
    keys : set of str
        Pipeline output keys to which the entity belongs to.
    """

    attrs: EntityAttributeContainer

    def __init__(
        self,
        label: str,
        text: str,
        spans: list[AnySpan],
        attrs: list[Attribute] | None = None,
        metadata: dict[str, Any] | None = None,
        uid: str | None = None,
        store: Store | None = None,
        attr_container_class: type[EntityAttributeContainer] = EntityAttributeContainer,
    ):
        super().__init__(label, text, spans, attrs, metadata, uid, store, attr_container_class)


@dataclasses.dataclass(init=False)
class Relation(TextAnnotation):
    """Relation between two text entities.

    Attributes
    ----------
    uid : str
        The identifier of the relation
    label : str
        The relation label
    source_id : str
        The identifier of the entity from which the relation is defined
    target_id : str
        The identifier of the entity to which the relation is defined
    attrs : AttributeContainer
        The attributes of the relation
    metadata : dict of str to Any
        The metadata of the relation
    keys : set of str
        Pipeline output keys to which the relation belongs to
    """

    source_id: str
    target_id: str

    def __init__(
        self,
        label: str,
        source_id: str,
        target_id: str,
        attrs: list[Attribute] | None = None,
        metadata: dict[str, Any] | None = None,
        uid: str | None = None,
        store: Store | None = None,
        attr_container_class: type[AttributeContainer] = AttributeContainer,
    ):
        super().__init__(
            label=label,
            attrs=attrs,
            metadata=metadata,
            uid=uid,
            attr_container_class=attr_container_class,
        )

        self.source_id = source_id
        self.target_id = target_id

    def to_dict(self) -> dict[str, Any]:
        attrs = [a.to_dict() for a in self.attrs]
        relation_dict = {
            "uid": self.uid,
            "label": self.label,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "attrs": attrs,
            "metadata": self.metadata,
        }
        dict_conv.add_class_name_to_data_dict(self, relation_dict)
        return relation_dict

    @classmethod
    def from_dict(cls, relation_dict: dict[str, Any]) -> Self:
        """Creates a Relation from a dict

        Parameters
        ----------
        relation_dict : dict of str to Any
            A dictionary from a serialized relation as generated by to_dict()
        """
        attrs = [Attribute.from_dict(a) for a in relation_dict["attrs"]]
        return cls(
            uid=relation_dict["uid"],
            label=relation_dict["label"],
            source_id=relation_dict["source_id"],
            target_id=relation_dict["target_id"],
            attrs=attrs,
            metadata=relation_dict["metadata"],
        )
