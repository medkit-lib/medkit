__all__ = ["PreprocessingOperation", "SegmentationOperation"]

import abc
from typing import List

from medkit.core.audio.annotation import Segment
from medkit.core.operation import Operation


class PreprocessingOperation(Operation):
    """
    Abstract operation for pre-processing segments.

    It uses a list of segments as input and produces a list of pre-processed
    segments. Each input segment will have a corresponding output segment.
    """

    @abc.abstractmethod
    def run(self, segments: List[Segment]) -> List[Segment]:
        raise NotImplementedError


class SegmentationOperation(Operation):
    """
    Abstract operation for segmenting audio.

    It uses a list of segments as input and produces a list of new segments.
    Each input segment will have zero, one or more corresponding output
    segments.
    """

    @abc.abstractmethod
    def run(self, segments: List[Segment]) -> List[Segment]:
        raise NotImplementedError
