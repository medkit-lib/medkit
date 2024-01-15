from __future__ import annotations

__all__ = ["InputConverter", "OutputConverter"]

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medkit.core.document import Document


class InputConverter:
    """Abstract class for converting external document to medkit documents"""

    @abc.abstractmethod
    def load(self, **kwargs) -> list[Document]:
        raise NotImplementedError


class OutputConverter:
    """Abstract class for converting medkit document to external format"""

    @abc.abstractmethod
    def save(self, docs: list[Document], **kwargs) -> list | None:
        raise NotImplementedError
