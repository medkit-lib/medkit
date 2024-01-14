from __future__ import annotations

__all__ = ["Store", "GlobalStore"]

from typing import TYPE_CHECKING, runtime_checkable

from typing_extensions import Protocol

if TYPE_CHECKING:
    from medkit.core.data_item import IdentifiableDataItem


@runtime_checkable
class Store(Protocol):
    """Store protocol"""

    def store_data_item(self, data_item: IdentifiableDataItem, parent_id: str):
        pass

    def get_data_item(self, data_item_id: str) -> IdentifiableDataItem | None:
        pass

    def get_parent_item(self, data_item) -> IdentifiableDataItem | None:
        pass


class _DictStore:
    def __init__(self) -> None:
        self._data_items_by_id: dict[str, IdentifiableDataItem] = {}
        self._parent_ids_by_id: dict[str, str] = {}

    def store_data_item(self, data_item: IdentifiableDataItem, parent_id: str):
        self._data_items_by_id[data_item.uid] = data_item
        self._parent_ids_by_id[data_item.uid] = parent_id

    def get_data_item(self, data_item_id: str) -> IdentifiableDataItem | None:
        return self._data_items_by_id.get(data_item_id)

    def get_parent_item(self, data_item_id: str) -> IdentifiableDataItem | None:
        parent_id = self._parent_ids_by_id[data_item_id]
        return self._data_items_by_id.get(parent_id)


class GlobalStore:
    """Global store"""

    _store: Store | None = None

    @classmethod
    def init_store(cls, store: Store):
        """Initialize the global store for your application

        Parameters
        ----------
        store : Store
            Store for all the data items

        Raises
        ------
        RuntimeError
            If global store is already set
        """
        if cls._store is None:
            cls._store = store
        else:
            msg = (
                "The global store has already been initialized. If it was not your"
                " intention, please put this line at the beginning of your script to"
                " make sure to set global store before any other initialization"
            )
            raise RuntimeError(msg)

    @classmethod
    def get_store(cls) -> Store:
        """Returns the global store object

        Returns
        -------
        Store
            the global store
        """
        if cls._store is None:
            cls._store = _DictStore()
        return cls._store

    @classmethod
    def del_store(cls):
        """Delete the global store object"""
        cls._store = None
