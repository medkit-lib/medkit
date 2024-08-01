from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from medkit._import import import_optional
from medkit.core.text import Entity, span_utils
from medkit.core.text.operation import NEROperation

if TYPE_CHECKING:
    from medkit.core.text import Segment

presidio_analyzer = import_optional("presidio_analyzer", extra="deid")

__all__ = ["PIIDetector"]


class PIIDetector(NEROperation):
    """Classify sensitive text information."""

    def __init__(self, uid: str | None = None, name: str | None = None, **kwargs):
        super().__init__(uid=uid, name=name, **kwargs)
        self._analyzer = presidio_analyzer.AnalyzerEngine()

    def run(self, segments: list[Segment]) -> list[Entity]:
        return [entity for segment in segments for entity in self._run_one(segment)]

    def _run_one(self, segment: Segment) -> Iterator[Entity]:
        for result in self._analyzer.analyze(text=segment.text, language="en"):
            text, spans = span_utils.extract(
                text=segment.text, spans=segment.spans, ranges=[(result.start, result.end)]
            )
            yield Entity(
                label=result.entity_type,
                text=text,
                spans=spans,
                metadata={"score": result.score},
            )
