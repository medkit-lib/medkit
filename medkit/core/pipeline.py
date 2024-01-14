from __future__ import annotations

__all__ = [
    "Pipeline",
    "PipelineStep",
    "PipelineCompatibleOperation",
    "DescribableOperation",
    "ProvCompatibleOperation",
]

import dataclasses
from typing import Any, List, cast, runtime_checkable

from typing_extensions import Protocol

from medkit.core.annotation import Annotation
from medkit.core.data_item import IdentifiableDataItem, IdentifiableDataItemWithAttrs
from medkit.core.id import generate_id
from medkit.core.operation_desc import OperationDescription
from medkit.core.prov_tracer import ProvTracer


@runtime_checkable
class PipelineCompatibleOperation(Protocol):
    def run(self, *all_input_data: list[Any]) -> list[Any] | tuple[list[Any], ...] | None:
        """Parameters
        ----------
        all_input_data : list of Any
            One or several list of data items to process
            (according to the number of input the operation needs)

        Returns
        -------
        list of Any or tuple of list, optional
            Tuple of list of all new data items created by the operation.
            Can be None if the operation does not create any new data items
            but rather modify existing items in-place (for instance by
            adding attributes to existing annotations).
            If there is only one list of created data items, it is possible
            to return directly that list without wrapping it in a tuple.
        """


@runtime_checkable
class ProvCompatibleOperation(Protocol):
    def set_prov_tracer(self, prov_tracer: ProvTracer):
        pass


@runtime_checkable
class DescribableOperation(Protocol):
    description: OperationDescription


@dataclasses.dataclass
class PipelineStep:
    """`Pipeline` item describing how a processing operation is connected to other

    Attributes
    ----------
    operation : PipelineCompatibleOperation
        The operation to use at that step
    input_keys : list of str
        For each input of `operation`, the key to use to retrieve the
        corresponding annotations (either retrieved from a document
        or generated by an earlier pipeline step)
    output_keys : list of str
        For each output of `operation`, the key used to pass output annotations
        to the next Pipeline step. Can be empty if `operation` doesn't return
        new annotations.
    aggregate_input_keys : bool, default=False
        If True, all the annotations from multiple input keys are aggregated in a single list. Defaults to False
    """

    operation: PipelineCompatibleOperation
    input_keys: list[str]
    output_keys: list[str]
    aggregate_input_keys: bool = False

    def to_dict(self) -> dict[str, Any]:
        operation = self.operation.description if isinstance(self.operation, DescribableOperation) else None
        return dict(
            operation=operation,
            input_keys=self.input_keys,
            output_keys=self.output_keys,
            aggregate_input_keys=self.aggregate_input_keys,
        )


class Pipeline:
    """Graph of processing operations

    A pipeline is made of pipeline steps, connecting together different processing
    operations by the use of input/output keys. Each operation can be seen as a node
    and the keys are its edge. Two operations can be chained by using the same string
    as an output key for the first operation and as an input key to the second.

    Steps must be added in the order of execution, there isn't any sort of dependency
    detection mechanism.
    """

    def __init__(
        self,
        steps: list[PipelineStep],
        input_keys: list[str],
        output_keys: list[str],
        name: str | None = None,
        uid: str | None = None,
    ):
        """Initialize the pipeline

        Parameters
        ----------
        steps : list of PipelineStep
            List of pipeline steps
            These steps will be executed in the order in which they were added,
            so make sure to add first the steps generating data used by other steps.
        input_keys : list of str
            List of keys corresponding to the inputs passed to `run()`
        output_keys : list of str
            List of keys corresponding to the outputs returned by `run()`
        name : str, optional
            Name describing the pipeline (defaults to the class name)
        uid : str, optional
             Identifier of the pipeline
        """
        if uid is None:
            uid = generate_id()

        self.uid: str = uid
        self.name: str | None = name
        self.steps: list[PipelineStep] = steps
        self.input_keys: list[str] = input_keys
        self.output_keys: list[str] = output_keys

        self._prov_tracer: ProvTracer | None = None
        self._sub_prov_tracer: ProvTracer | None = None

    @property
    def description(self) -> OperationDescription:
        steps = [s.to_dict() for s in self.steps]
        config = dict(
            steps=steps,
            input_keys=self.input_keys,
            output_keys=self.output_keys,
        )
        return OperationDescription(
            uid=self.uid,
            class_name=self.__class__.__name__,
            name=self.__class__.__name__ if self.name is None else self.name,
            config=config,
        )

    def set_prov_tracer(self, prov_tracer: ProvTracer):
        self._prov_tracer = prov_tracer
        self._sub_prov_tracer = ProvTracer(prov_tracer.store)
        for step in self.steps:
            if not isinstance(step.operation, ProvCompatibleOperation):
                msg = "Some operations in the pipeline steps are not provenance-compatible"
                raise TypeError(msg)
            step.operation.set_prov_tracer(self._sub_prov_tracer)

    def run(self, *all_input_data: list[Any]) -> list[Any] | tuple[list[Any], ...] | None:
        """Run the pipeline.

        Parameters
        ----------
        *all_input_data : list of Any
            Input data expected by the pipeline, must be of same length as the
            pipeline `input_keys`.

            For each input key, the corresponding input data must be a list of
            items than can be of any type.

        Returns
        -------
        list of Any or tuple of list, optional
            All output data returned by the pipeline, will be of same length as
            the pipeline `output_keys`.

            For each output key, the corresponding output will be a list of
            items that can be of any type.

            If the pipeline has only one output key, then the corresponding output
            will be directly returned, not wrapped in a tuple. If the pipeline
            doesn't have any output key, nothing (ie `None`) will be returned.
        """
        if len(all_input_data) != len(self.input_keys):
            msg = (
                f"Number of input ({len(all_input_data)}) does not match number of"
                f" input keys ({len(self.input_keys)})"
            )
            raise RuntimeError(msg)

        data_by_key = dict(zip(self.input_keys, all_input_data))
        for step in self.steps:
            self._perform_step(step, data_by_key)

        all_output_data = tuple(data_by_key[key] for key in self.output_keys)

        # Keep keys only for output key
        for data_item in [d for data_tuple in all_output_data for d in data_tuple]:
            if isinstance(data_item, Annotation):
                data_item.keys.intersection_update(self.output_keys)

        if self._prov_tracer is not None:
            self._add_provenance(all_output_data)

        if len(all_output_data) == 0:
            # no output
            return None
        elif len(all_output_data) == 1:
            # unwrap out of tuple if only 1 output
            return all_output_data[0]
        else:
            return all_output_data

    def _perform_step(self, step: PipelineStep, data_by_key: dict[str, Any]):
        # find data to feed to operation
        all_input_data = []
        for input_key in step.input_keys:
            input_data = data_by_key.get(input_key)
            if input_data is None:
                message = f"No data found for input key {input_key}"
                if any(input_key in s.output_keys for s in self.steps):
                    message += "Did you add the steps in the correct order in the pipeline?"
                raise RuntimeError(message)
            all_input_data.append(input_data)
        if step.aggregate_input_keys:
            all_input_data = [[ann for input_key_data in all_input_data for ann in input_key_data]]

        # call operation
        all_output_data = step.operation.run(*all_input_data)

        # wrap output in tuple if necessary
        # (operations performing in-place modifications
        # have no output and return None,
        # operations with single output may return a
        # single list instead of a tuple of lists)
        if all_output_data is None:
            all_output_data = tuple()
        elif not isinstance(all_output_data, tuple):
            all_output_data = (all_output_data,)

        if len(all_output_data) != len(step.output_keys):
            msg = (
                f"Number of outputs ({len(all_output_data)}) does not match number of"
                f" output keys ({len(step.output_keys)})"
            )
            raise RuntimeError(msg)

        # store output data
        for output_key, output_data in zip(step.output_keys, all_output_data):
            if output_key not in data_by_key:
                data_by_key[output_key] = output_data
            else:
                data_by_key[output_key] += output_data
            for data_item in output_data:
                if isinstance(data_item, Annotation):
                    data_item.keys.add(output_key)

    def _add_provenance(self, all_output_data: tuple[list[Any], ...]):
        assert self._prov_tracer is not None and self._sub_prov_tracer is not None

        # flatten all output data to have a list of data items generated by this pipeline
        data_items = [data_item for output_data in all_output_data for data_item in output_data]
        # data items must have identifiers to be provenance-compatible
        data_items = cast(List[IdentifiableDataItem], data_items)

        # ugly hack for attributes generated by pipeline
        # they are not in all_output_data because they are not returned by the operation
        # but directly added to input data items (annotations), but we still want to consider
        # them as an output of the pipeline in terms of provenance
        # find all attributes that were generated by this pipeline,
        # ie that have provenance in the pipeline's sub_prov_tracer
        attrs = [
            attr
            for data_item in data_items
            if isinstance(data_item, IdentifiableDataItemWithAttrs)
            for attr in data_item.attrs
            if (
                self._sub_prov_tracer.has_prov(attr.uid)
                # ignore stub provenance with no operation
                # (were passed as input to the pipeline but not generated by the pipeline)
                and self._sub_prov_tracer.get_prov(attr.uid).op_desc is not None
            )
        ]

        # add them to the list of data items generated by this pipeline
        data_items += attrs
        self._prov_tracer.add_prov_from_sub_tracer(data_items, self.description, self._sub_prov_tracer)

    def check_sanity(self):
        steps_input_keys = [k for s in self.steps for k in s.input_keys]
        for input_key in self.input_keys:
            if input_key not in steps_input_keys:
                msg = f"Pipeline input key {input_key} does not correspond to any step input key"
                raise Exception(msg)

        steps_output_keys = [k for s in self.steps for k in s.output_keys]
        for output_key in self.output_keys:
            if output_key not in steps_output_keys:
                msg = f"Pipeline output key {output_key} does not correspond to any step output key"
                raise Exception(msg)

        for step in self.steps:
            for input_key in step.input_keys:
                if input_key not in steps_output_keys and input_key not in self.input_keys:
                    msg = (
                        f"Step input key {input_key} does not correspond to any step"
                        " output key nor any pipeline input key"
                    )
                    raise Exception(msg)

        available_keys = self.input_keys.copy()
        for step in self.steps:
            for input_key in step.input_keys:
                if input_key not in available_keys:
                    msg = f"Step input key {input_key} is not available yet at this step"
                    raise Exception(msg)
            available_keys += step.output_keys
