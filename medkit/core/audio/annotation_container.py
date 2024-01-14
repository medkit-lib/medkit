from __future__ import annotations

__all__ = ["AudioAnnotationContainer"]

from medkit.core.annotation_container import AnnotationContainer
from medkit.core.audio.annotation import Segment


class AudioAnnotationContainer(AnnotationContainer[Segment]):
    """Manage a list of audio annotations belonging to an audio document.

    This behaves more or less like a list: calling `len()` and iterating are
    supported. Additional filtering is available through the `get()` method.

    Also provides handling of raw segment.
    """

    def __init__(self, doc_id: str, raw_segment: Segment):
        super().__init__(doc_id=doc_id)

        # auto-generated RAW_AUDIO segment
        # not stored with other annotations but injected in calls to get()
        # and get_by_id()
        self.raw_segment = raw_segment

    def add(self, ann: Segment):
        if ann.label == self.raw_segment.label:
            msg = f"Cannot add annotation with reserved label {self.raw_segment.label}"
            raise RuntimeError(msg)

        super().add(ann)

    def get(self, *, label: str | None = None, key: str | None = None) -> list[Segment]:
        # inject raw segment
        if label == self.raw_segment.label and key is None:
            return [self.raw_segment]
        return super().get(label=label, key=key)

    def get_by_id(self, uid) -> Segment:
        # inject raw segment
        if uid == self.raw_segment.uid:
            return self.raw_segment
        return super().get_by_id(uid)
