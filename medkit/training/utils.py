from __future__ import annotations

__all__ = ["BatchData", "MetricsComputer"]

from typing import Any, Dict, List, Union

import torch
from typing_extensions import Protocol, runtime_checkable


class BatchData(dict):
    """A BatchData pack data allowing both column and row access"""

    def __getitem__(self, index: int) -> Dict[str, Union[List[Any], torch.Tensor]]:
        if isinstance(index, str):
            inner_dict = dict(self.items())
            return inner_dict[index]
        return {key: values[index] for key, values in self.items()}

    def to_device(self, device: torch.device) -> BatchData:
        """
        Ensure that Tensors in the BatchData object are on the specified `device`

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

    def prepare_batch(
        self, model_output: BatchData, input_batch: BatchData
    ) -> Dict[str, List[Any]]:
        """Prepare a batch of data to compute the metrics

        Parameters
        ----------
        model_output: BatchData
            Output data after a model forward pass.
        input_batch: BatchData
            Preprocessed input batch

        Returns
        -------
        Dict[str, List[Any]]
            A dictionary with the required data to calculate the metric
        """
        pass

    def compute(self, all_data: Dict[str, List[Any]]) -> Dict[str, float]:
        """Compute metrics using 'all_data'

        Parameters
        ----------
        all_data: Dict[str, List[Any]]
            A dictionary to compute the metrics.
            i.e. A dictionary with a list of 'references' and a list of 'predictions'.

        Returns
        -------
        Dict[str, float]
            A dictionary with the results
        """
        pass
