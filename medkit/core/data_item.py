__all__ = ["IdentifiableDataItem", "IdentifiableDataItemWithAttrs"]

from typing import Iterable, TypeVar, runtime_checkable

from typing_extensions import Protocol


@runtime_checkable
class IdentifiableDataItem(Protocol):
    uid: str


IdentifiableDataItemType = TypeVar("IdentifiableDataItemType", bound=IdentifiableDataItem)


@runtime_checkable
class IdentifiableDataItemWithAttrs(Protocol[IdentifiableDataItemType]):
    attrs: Iterable[IdentifiableDataItemType]
