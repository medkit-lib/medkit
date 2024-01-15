from __future__ import annotations

__all__ = ["BatchData", "MetricsComputer"]

from typing import Any, runtime_checkable

import torch
from typing_extensions import Protocol, Self


class BatchData(dict):
    """A BatchData pack data allowing both column and row access"""

    def __getitem__(self, index: int) -> dict[str, list[Any] | torch.Tensor]:
        if isinstance(index, str):
            inner_dict = dict(self.items())
            return inner_dict[index]
        return {key: values[index] for key, values in self.items()}

    def to_device(self, device: torch.device) -> Self:
        """Ensure that Tensors in the BatchData object are on the specified `device`

        Parameters
        ----------
        device:
            A `torch.device` object representing the device on which tensors
            will be allocated.

        Returns
        -------
        BatchData
            A new object with the tensors on the proper device.
        """
        inner_batch = BatchData()
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                inner_batch[key] = value.to(device)
            else:
                inner_batch[key] = value
        return inner_batch


@runtime_checkable
class MetricsComputer(Protocol):
    "A MetricsComputer is the base protocol to compute metrics in training"

    def prepare_batch(self, model_output: BatchData, input_batch: BatchData) -> dict[str, list[Any]]:
        """Prepare a batch of data to compute the metrics

        Parameters
        ----------
        model_output: BatchData
            Output data after a model forward pass.
        input_batch: BatchData
            Preprocessed input batch

        Returns
        -------
        dict[str, List[Any]]
            A dictionary with the required data to calculate the metric
        """

    def compute(self, all_data: dict[str, list[Any]]) -> dict[str, float]:
        """Compute metrics using 'all_data'

        Parameters
        ----------
        all_data: dict[str, List[Any]]
            A dictionary to compute the metrics.
            i.e. A dictionary with a list of 'references' and a list of 'predictions'.

        Returns
        -------
        dict[str, float]
            A dictionary with the results
        """
