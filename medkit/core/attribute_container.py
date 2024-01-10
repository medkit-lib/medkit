__all__ = ["AttributeContainer"]

import typing
from typing import Dict, Iterator, List, Optional, Union

from medkit.core.attribute import Attribute
from medkit.core.store import GlobalStore, Store


class AttributeContainer:
    """
    Manage a list of attributes attached to another data structure.
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
        self._attr_ids: List[str] = []
        self._attr_ids_by_label: Dict[str, List[str]] = {}

    def __len__(self) -> int:
        """Add support for calling `len()`"""
        return len(self._attr_ids)

    def __iter__(self) -> Iterator[Attribute]:
        """
        Add support for iterating over an `AttributeContainer` (will yield each
        attribute)
        """
        return iter(self.get_by_id(uid) for uid in self._attr_ids)

    def __getitem__(self, key: Union[int, slice]) -> Union[Attribute, List[Attribute]]:
        """
        Add support for subscript access
        """

        if isinstance(key, slice):
            return [self.get_by_id(uid) for uid in self._attr_ids[key]]
        else:
            return self.get_by_id(self._attr_ids[key])

    def get(self, *, label: Optional[str] = None) -> List[Attribute]:
        """
        Return a list of the attributes of the annotation, optionally filtering
        by label.

        Parameters
        ----------
        label:
            Label to use to filter attributes.
        """
        if label is None:
            return list(iter(self))
        else:
            return [self.get_by_id(uid) for uid in self._attr_ids_by_label.get(label, [])]

    def add(self, attr: Attribute):
        """
        Attach an attribute to the annotation.

        Parameters
        ----------
        attr:
            Attribute to add.

        Raises
        ------
        ValueError
            If the attribute is already attached to the annotation (based on
            `attr.uid`).
        """

        uid = attr.uid
        if uid in self._attr_ids:
            raise ValueError(f"Attribute with uid {uid} already attached to annotation")

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
        uid:
            Identifier of the attribute to return.
        """

        attr = self._store.get_data_item(uid)
        if attr is None:
            raise ValueError(f"No known attribute with uid '{uid}'")
        return typing.cast(Attribute, attr)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.get() == other.get()

    def __repr__(self) -> str:
        attrs = self.get()
        return f"{self.__class__.__name__}(ann_id={self._owner_id!r}, attrs={attrs!r})"
