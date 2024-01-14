from __future__ import annotations

__all__ = ["ProvStore", "create_prov_store"]

from typing import TYPE_CHECKING, runtime_checkable

from typing_extensions import Literal, Protocol

if TYPE_CHECKING:
    from medkit.core.data_item import IdentifiableDataItem
    from medkit.core.operation_desc import OperationDescription


@runtime_checkable
class ProvStore(Protocol):
    def store_data_item(self, data_item: IdentifiableDataItem):
        pass

    def get_data_item(self, data_item_id: str) -> IdentifiableDataItem:
        pass

    def store_op_desc(self, op_desc: OperationDescription):
        pass

    def get_op_desc(self, operation_id: str) -> OperationDescription:
        pass


class _DictStore:
    def __init__(self) -> None:
        self._data_items_by_id: dict[str, IdentifiableDataItem] = {}
        self._op_descs_by_id: dict[str, OperationDescription] = {}

    def store_data_item(self, data_item: IdentifiableDataItem):
        self._data_items_by_id[data_item.uid] = data_item

    def get_data_item(self, data_item_id: str) -> IdentifiableDataItem:
        return self._data_items_by_id[data_item_id]

    def store_op_desc(self, op_desc: OperationDescription):
        self._op_descs_by_id[op_desc.uid] = op_desc

    def get_op_desc(self, operation_id: str) -> OperationDescription:
        return self._op_descs_by_id[operation_id]


StoreType = Literal["dict"]

default_stores = {"dict": _DictStore}


def create_prov_store(store_type: StoreType = "dict"):
    return default_stores.get(store_type, "dict")()
