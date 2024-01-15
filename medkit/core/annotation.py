from __future__ import annotations

__all__ = ["Annotation", "AnnotationType"]

from typing import TYPE_CHECKING, TypeVar, runtime_checkable

from typing_extensions import Protocol

if TYPE_CHECKING:
    from medkit.core.attribute_container import AttributeContainer


@runtime_checkable
class Annotation(Protocol):
    """Base annotation protocol that must be implemented by annotations classes of all
    modalities (text, audio, etc).

    Annotations can be attached to :class:`~medkit.core.document.Document`
    objects and can contain :class:`~medkit.core.attribute.Attribute` objects.

    Attributes
    ----------
    uid : str
        Unique identifier of the annotation
    label : str
        Label of the annotation, can be used to represent the "kind" of
        annotation. (ex: "sentence", "disease", etc)
    keys : set of str
        Pipeline output keys to which the segment belongs to (cf
        :class:`~medkit.core.pipeline.Pipeline`.)
    attrs : AttributeContainer
        Attributes of the annotation, stored in an
        :class:`~medkit.core.attribute_container.AttributeContainer` for easier
        access.
    """

    uid: str
    label: str
    keys: set[str]
    attrs: AttributeContainer


#: Annotation type
AnnotationType = TypeVar("AnnotationType", bound=Annotation)
