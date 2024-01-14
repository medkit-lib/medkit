from __future__ import annotations

__all__ = ["Operation", "DocOperation"]

import abc
from typing import TYPE_CHECKING

from medkit.core.id import generate_id
from medkit.core.operation_desc import OperationDescription

if TYPE_CHECKING:
    from medkit.core.document import Document
    from medkit.core.prov_tracer import ProvTracer


class Operation(abc.ABC):
    """Abstract class for all annotator modules"""

    uid: str
    _description: OperationDescription | None = None
    _prov_tracer: ProvTracer | None = None

    @abc.abstractmethod
    def __init__(self, uid: str | None = None, name: str | None = None, **kwargs):
        """Common initialization for all annotators:
          * assigning identifier to operation
          * storing class name, name and config in description

        Parameters
        ----------
        uid: str, optional
            Operation identifier
        name: str, optional
            Operation name (defaults to class name)
        kwargs:
            All other arguments of the child init useful to describe the operation

        Examples
        --------
        In the `__init__` function of your annotator, use:

        >>> init_args = locals()
        >>> init_args.pop("self")
        >>> super().__init__(**init_args)
        """
        if uid is None:
            uid = generate_id()
        if name is None:
            name = self.__class__.__name__

        self.uid = uid
        self._description = OperationDescription(
            uid=self.uid,
            class_name=self.__class__.__name__,
            name=name,
            config=kwargs,
        )

    def set_prov_tracer(self, prov_tracer: ProvTracer):
        """Enable provenance tracing.

        Parameters
        ----------
        prov_tracer: ProvTracer
            The provenance tracer used to trace the provenance.
        """
        self._prov_tracer = prov_tracer

    @property
    def description(self) -> OperationDescription:
        """Contains all the operation init parameters."""
        return self._description

    def check_sanity(self) -> bool:  # noqa: B027
        # TODO: add some checks
        pass


class DocOperation(Operation):
    """Abstract operation directly executed on text documents.
    It uses a list of documents as input for running the operation and creates
    annotations that are directly appended to these documents.
    """

    @abc.abstractmethod
    def run(self, docs: list[Document]) -> None:
        raise NotImplementedError
