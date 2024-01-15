from __future__ import annotations

__all__ = ["PreprocessingOperation", "SegmentationOperation"]

import abc
from typing import TYPE_CHECKING

from medkit.core.operation import Operation

if TYPE_CHECKING:
    from medkit.core.audio.annotation import Segment


class PreprocessingOperation(Operation):
    """Abstract operation for pre-processing segments.

    It uses a list of segments as input and produces a list of pre-processed
    segments. Each input segment will have a corresponding output segment.
    """

    @abc.abstractmethod
    def run(self, segments: list[Segment]) -> list[Segment]:
        raise NotImplementedError


class SegmentationOperation(Operation):
    """Abstract operation for segmenting audio.

    It uses a list of segments as input and produces a list of new segments.
    Each input segment will have zero, one or more corresponding output
    segments.
    """

    @abc.abstractmethod
    def run(self, segments: list[Segment]) -> list[Segment]:
        raise NotImplementedError
