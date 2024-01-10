__all__ = ["Document"]

from typing import runtime_checkable

from typing_extensions import Protocol

from medkit.core.annotation import AnnotationType
from medkit.core.annotation_container import AnnotationContainer
from medkit.core.attribute_container import AttributeContainer


@runtime_checkable
class Document(Protocol[AnnotationType]):
    """Base document protocol that must be implemented by document classes of all
    modalities (text, audio, etc).

    Documents can contain :class:`~medkit.core.annotation.Annotation` objects.

    Attributes
    ----------
    uid:
        Unique identifier of the document
    anns:
        Annotations of the document, stored in an
        :class:`~medkit.core.annotation_container.AnnotationContainer` for
        easier access (can be subclassed to add modality-specific features).
    attrs:
        Attributes of the document, stored in an
        :class: `~medkit.core.attribute_container.AttributeContainer` for
        easier access
    raw_segment:
        Auto-generated segment containing the full unprocessed document.
    """

    uid: str
    anns: AnnotationContainer[AnnotationType]
    attrs: AttributeContainer
    raw_segment: AnnotationType
