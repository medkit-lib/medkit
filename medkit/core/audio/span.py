from __future__ import annotations

__all__ = ["Span"]

from typing import Any, NamedTuple

from medkit.core import dict_conv


class Span(NamedTuple):
    """Boundaries of a slice of audio.

    Attributes
    ----------
    start: float
        Starting point in the original audio, in seconds.
    end: float
        Ending point in the original audio, in seconds.
    """

    start: float
    end: float

    @property
    def length(self):
        """Length of the span, in seconds"""
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        span_dict = {"start": self.start, "end": self.end}
        dict_conv.add_class_name_to_data_dict(self, span_dict)
        return span_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Span:
        return cls(start=data["start"], end=data["end"])


# TODO: support speed variations? ex: speeded up segment with shorter span
# but referencing span longer span in original audio
