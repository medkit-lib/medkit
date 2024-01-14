from __future__ import annotations

__all__ = ["AnnotationContainer"]

import typing
from typing import Generic, Iterator

from medkit.core.annotation import AnnotationType
from medkit.core.store import GlobalStore, Store


class AnnotationContainer(Generic[AnnotationType]):
    """Manage a list of annotations belonging to a document.

    This behaves more or less like a list: calling `len()` and iterating are
    supported. Additional filtering is available through the `get()` method.

    The annotations will be stored in a :class:`~medkit.core.Store`, which can
    rely on a simple dict or something more
    complicated like a database.

    This global store may be initialized using :class:~medkit.core.GlobalStore.
    Otherwise, a default one (i.e. dict store) is used.
    """

    def __init__(self, doc_id: str):
        """Instantiate the annotation container

        Parameters
        ----------
        doc_id : str
            The identifier of the document which annotations belong to.
        """
        self._store: Store = GlobalStore.get_store()
        self._doc_id = doc_id
        self._ann_ids: list[str] = []
        self._ann_ids_by_label: dict[str, list[str]] = {}
        self._ann_ids_by_key: dict[str, list[str]] = {}

    def add(self, ann: AnnotationType):
        """Attach an annotation to the document.

        Parameters
        ----------
        ann : AnnotationType
            Annotation to add.

        Raises
        ------
        ValueError
            If the annotation is already attached to the document
            (based on `annotation.uid`)
        """
        uid = ann.uid
        if uid in self._ann_ids:
            msg = f"Impossible to add this annotation. The uid {uid} already exists in the document."
            raise ValueError(msg)

        self._ann_ids.append(uid)
        self._store.store_data_item(data_item=ann, parent_id=self._doc_id)

        # update label index
        label = ann.label
        if label not in self._ann_ids_by_label:
            self._ann_ids_by_label[label] = []
        self._ann_ids_by_label[label].append(uid)

        # update key index
        for key in ann.keys:
            if key not in self._ann_ids_by_key:
                self._ann_ids_by_key[key] = []
            self._ann_ids_by_key[key].append(uid)

    def __len__(self) -> int:
        """Add support for calling `len()`"""
        return len(self._ann_ids)

    def __iter__(self) -> Iterator[AnnotationType]:
        """Add support for iterating over an `AnnotationContainer` (will yield each
        attribute)
        """
        return iter(self.get_by_id(uid) for uid in self._ann_ids)

    def __getitem__(self, key: int | slice) -> AnnotationType | list[AnnotationType]:
        """Add support for subscript access"""
        if isinstance(key, slice):
            return [self.get_by_id(uid) for uid in self._ann_ids[key]]
        return self.get_by_id(self._ann_ids[key])

    def get(self, *, label: str | None = None, key: str | None = None) -> list[AnnotationType]:
        """Return a list of the annotations of the document, optionally filtering
        by label or key.

        Parameters
        ----------
        label : str, optional
            Label to use to filter annotations.
        key : str, optional
            Key to use to filter annotations.
        """
        uids = self.get_ids(label=label, key=key)
        return [self.get_by_id(uid) for uid in uids]

    def get_ids(self, *, label: str | None = None, key: str | None = None) -> Iterator[str]:
        """Return an iterator of the identifiers of the annotations of the
        document, optionally filtering by label or key.

        This method is provided, so it is easier to implement additional
        filtering in subclasses.

        Parameters
        ----------
        label : str, optional
            Label to use to filter annotations.
        key : str, optional
            Key to use to filter annotations.
        """
        uids = iter(self._ann_ids)

        if label is not None:
            uids = (uid for uid in uids if uid in self._ann_ids_by_label.get(label, []))

        if key is not None:
            uids = (uid for uid in uids if uid in self._ann_ids_by_key.get(key, []))

        return uids

    def get_by_id(self, uid: str) -> AnnotationType:
        """Return the annotation corresponding to a specific identifier.

        Parameters
        ----------
        uid : str
            Identifier of the annotation to return.
        """
        ann = self._store.get_data_item(uid)
        if ann is None:
            msg = f"No known annotation with uid '{uid}'"
            raise ValueError(msg)
        return typing.cast(AnnotationType, ann)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.get() == other.get()

    def __repr__(self) -> str:
        anns = self.get()
        return f"{self.__class__.__name__}(doc_id={self._doc_id!r}, anns={anns!r})"
