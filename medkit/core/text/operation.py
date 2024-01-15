from __future__ import annotations

__all__ = [
    "ContextOperation",
    "NEROperation",
    "SegmentationOperation",
    "CustomTextOpType",
    "create_text_operation",
]

import abc
from collections.abc import Iterable
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable

from medkit.core.operation import Operation

if TYPE_CHECKING:
    from medkit.core.prov_tracer import ProvTracer
    from medkit.core.text.annotation import Entity, Segment


class ContextOperation(Operation):
    """Abstract operation for context detection.
    It uses a list of segments as input for running the operation and creates attributes
    that are directly appended to these segments.
    """

    @abc.abstractmethod
    def run(self, segments: list[Segment]) -> None:
        raise NotImplementedError


class NEROperation(Operation):
    """Abstract operation for detecting entities.
    It uses a list of segments as input and produces a list of detected entities.
    """

    @abc.abstractmethod
    def run(self, segments: list[Segment]) -> list[Entity]:
        raise NotImplementedError


class SegmentationOperation(Operation):
    """Abstract operation for segmenting text.
    It uses a list of segments as input and produces a list of new segments.
    """

    @abc.abstractmethod
    def run(self, segments: list[Segment]) -> list[Segment]:
        raise NotImplementedError


class CustomTextOpType(IntEnum):
    """Supported function types for creating custom text operations."""

    CREATE_ONE_TO_N = 1
    """Take 1 data item, return N new data items."""
    EXTRACT_ONE_TO_N = 2
    """Take 1 data item, return N existing data items"""
    FILTER = 3
    """Take 1 data item, return True or False."""


class _CustomTextOperation(Operation):
    """Internal class representing a custom text operation.

    This class may be only instantiated by `create_text_operation`.

    It uses an user-defined function in the `run` method.
    It handles all provenance settings based on the function type.
    """

    def __init__(self, name: str, uid: str | None = None):
        """Parameters
        ----------
        name : str
            Name of the operation used for provenance info
        uid : str, optional
            Identifier of the operation
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        self._function = None
        self._function_type = None
        self._kwargs = None

    def set_prov_tracer(self, prov_tracer: ProvTracer):
        self._prov_tracer = prov_tracer

    def set_function(self, function: Callable, function_type: CustomTextOpType, **kwargs: Any):
        """Assign a user-defined function to the operation

        Parameters
        ----------
        function : Callable
            User-defined function to be used in `run` method
        function_type : CustomTextOpType
            Type of function.
            Supported values are defined in :class:`~medkit.core.text.CustomTextOpType`
        **kwargs
            Additional arguments of the callable function

        Returns
        -------

        """
        self._function = function
        self._function_type = function_type
        self._kwargs = kwargs
        self.description.config["function_type"] = function_type.name
        # TODO: check signature according to type

    def run(self, all_input_data: list[Any]) -> list[Any]:
        """Run the custom operation on a list of input data and outputs a list of data

        This method uses the user-defined function depending on its type on a
        batch of data.

        Parameters
        ----------
        all_input_data : list of Any
            List of input data

        Returns
        -------
        list of Any
            Flat list of output data
        """
        assert self._function is not None
        assert self._function_type in set(CustomTextOpType)
        if self._function_type in [
            CustomTextOpType.CREATE_ONE_TO_N,
            CustomTextOpType.EXTRACT_ONE_TO_N,
        ]:
            return self._run_one_to_n_function(all_input_data, self._function_type)
        elif self._function_type == CustomTextOpType.FILTER:  # noqa: RET505
            return self._run_filter_function(all_input_data)
        return None

    def _run_one_to_n_function(self, all_input_data: list[Any], function_type: CustomTextOpType) -> list[Any]:
        all_output_data = []
        for input_data in all_input_data:
            output_data = self._function(input_data, **self._kwargs)
            is_iterable = isinstance(output_data, Iterable)
            if is_iterable:
                all_output_data.extend(output_data)
            else:
                all_output_data.append(output_data)
            if function_type == CustomTextOpType.CREATE_ONE_TO_N and self._prov_tracer is not None:
                if is_iterable:
                    for data in output_data:
                        self._prov_tracer.add_prov(
                            data_item=data,
                            op_desc=self.description,
                            source_data_items=[input_data],
                        )
                else:
                    self._prov_tracer.add_prov(
                        data_item=output_data,
                        op_desc=self.description,
                        source_data_items=[input_data],
                    )
        return all_output_data

    def _run_filter_function(self, all_input_data: list[Any]) -> list[Any]:
        all_output_data = []
        for input_data in all_input_data:
            checked = self._function(input_data, **self._kwargs)
            if checked:
                all_output_data.append(input_data)
        return all_output_data


def create_text_operation(
    function: Callable,
    function_type: CustomTextOpType,
    name: str | None = None,
    args: dict | None = None,
) -> _CustomTextOperation:
    """Function for instantiating a custom test operation from a user-defined function

    Parameters
    ----------
    function : Callable
        User-defined function
    function_type : CustomTextOpType
        Type of function.
        Supported values are defined in :class:`~medkit.core.text.CustomTextOpType`
    name : str, optional
        Name of the operation used for provenance info (default: function name)
    args : str, optional
        Dictionary containing the arguments of the function if any.

    Returns
    -------
    _CustomTextOperation
        An instance of a custom text operation
    """
    if name is None:
        name = function.__name__
    if args is None:
        args = {}
    operation = _CustomTextOperation(name=name)
    operation.set_function(function=function, function_type=function_type, **args)
    return operation
