from __future__ import annotations

__all__ = ["AttributeContainer"]

import typing
from typing import Iterator

from medkit.core.attribute import Attribute
from medkit.core.store import GlobalStore, Store


class AttributeContainer:
    """Manage a list of attributes attached to another data structure.
    For example, it may be a document or an annotation.

    This behaves more or less like a list: calling `len()` and iterating are
    supported. Additional filtering is available through the `get()` method.

    The attributes will be stored in a :class:`~medkit.core.Store`, which can
    rely on a simple dict or something more complicated like a database.

    This global store may be initialized using :class:~medkit.core.GlobalStore.
    Otherwise, a default one (i.e. dict store) is used.
    """

    def __init__(self, owner_id: str):
        self._store: Store = GlobalStore.get_store()
        self._owner_id = owner_id
        self._attr_ids: list[str] = []
        self._attr_ids_by_label: dict[str, list[str]] = {}

    def __len__(self) -> int:
        """Add support for calling `len()`"""
        return len(self._attr_ids)

    def __iter__(self) -> Iterator[Attribute]:
        """Add support for iterating over an `AttributeContainer` (will yield each
        attribute)
        """
        return iter(self.get_by_id(uid) for uid in self._attr_ids)

    def __getitem__(self, key: int | slice) -> Attribute | list[Attribute]:
        """Add support for subscript access"""
        if isinstance(key, slice):
            return [self.get_by_id(uid) for uid in self._attr_ids[key]]
        return self.get_by_id(self._attr_ids[key])

    def get(self, *, label: str | None = None) -> list[Attribute]:
        """Return a list of the attributes of the annotation, optionally filtering
        by label.

        Parameters
        ----------
        label : str, optional
            Label to use to filter attributes.

        Returns
        -------
        list of Attribute
            The list of all attributes of the annotation, filtered by label if specified.
        """
        if label:
            return [self.get_by_id(uid) for uid in self._attr_ids_by_label.get(label, [])]
        return list(iter(self))

    def add(self, attr: Attribute):
        """Attach an attribute to the annotation.

        Parameters
        ----------
        attr : Attribute
            Attribute to add.

        Raises
        ------
        ValueError
            If the attribute is already attached to the annotation (based on
            `attr.uid`).
        """
        uid = attr.uid
        if uid in self._attr_ids:
            msg = f"Attribute with uid {uid} already attached to annotation"
            raise ValueError(msg)

        self._attr_ids.append(uid)
        self._store.store_data_item(data_item=attr, parent_id=self._owner_id)

        # update label index
        label = attr.label
        if label not in self._attr_ids_by_label:
            self._attr_ids_by_label[label] = []
        self._attr_ids_by_label[label].append(uid)

    def get_by_id(self, uid: str) -> Attribute:
        """Return the attribute corresponding to a specific identifier.

        Parameters
        ----------
        uid : str
            Identifier of the attribute to return.

        Returns
        -------
        Attribute
            The attribute corresponding to the identifier
        """
        attr = self._store.get_data_item(uid)
        if attr is None:
            msg = f"No known attribute with uid '{uid}'"
            raise ValueError(msg)
        return typing.cast(Attribute, attr)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.get() == other.get()

    def __repr__(self) -> str:
        attrs = self.get()
        return f"{self.__class__.__name__}(ann_id={self._owner_id!r}, attrs={attrs!r})"
