from __future__ import annotations

__all__ = ["EntityAttributeContainer"]

from typing import TYPE_CHECKING, List, cast

from medkit.core.attribute_container import AttributeContainer
from medkit.core.text.entity_norm_attribute import EntityNormAttribute

if TYPE_CHECKING:
    from medkit.core.attribute import Attribute


class EntityAttributeContainer(AttributeContainer):
    """Manage a list of attributes attached to a text entity.

    This behaves more or less like a list: calling `len()` and iterating are
    supported. Additional filtering is available through the `get()` method.

    Also provides retrieval of normalization attributes.
    """

    def __init__(self, owner_id: str):
        super().__init__(owner_id=owner_id)

        self._norm_ids: list[str] = []

    @property
    def norms(self) -> list[EntityNormAttribute]:
        """Return the list of normalization attributes"""
        return self.get_norms()

    def add(self, attr: Attribute):
        super().add(attr)

        # update norm attributes index
        if isinstance(attr, EntityNormAttribute):
            self._norm_ids.append(attr.uid)

    def get_norms(self) -> list[EntityNormAttribute]:
        """Return a list of the normalization attributes of the annotation"""
        segments = [self.get_by_id(uid) for uid in self._norm_ids]
        return cast(List[EntityNormAttribute], segments)
