from __future__ import annotations

__all__ = ["TrainableComponent"]

from typing import TYPE_CHECKING, Any, runtime_checkable

from typing_extensions import Protocol

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from medkit.training.utils import BatchData


@runtime_checkable
class TrainableComponent(Protocol):
    """TrainableComponent is the base protocol to be trainable in medkit"""

    @property
    def device(self) -> torch.device:
        pass

    def configure_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """Create optimizer using the learning rate"""

    def preprocess(self, data_item: Any) -> dict[str, Any]:
        """Preprocess the input data item and return a dictionary with
        everything needed for the forward pass.

        This method is intended to preprocess an input, `self.collate` must be
        used to generate batches for `self.forward` to run properly.
        Preprocess should include `labels` to compute a loss.
        """

    def collate(self, batch: list[dict[str, Any]]) -> BatchData:
        """Collate a list of data processed by `preprocess` to form a batch"""

    def forward(
        self,
        input_batch: BatchData,
        return_loss: bool,
        eval_mode: bool,
    ) -> tuple[BatchData, torch.Tensor | None]:
        """Perform the forward pass on a batch and return the corresponding
        output as well as the loss if `return_loss` is True.

        Before forwarding the model, this method must set the model to training
        or evaluation mode depending on `eval_mode`. In PyTorch models there are
        two methods to set the mode `model.train()` and `model.eval()`
        """

    def save(self, path: str | Path):
        """Save model to disk"""

    def load(self, path: str | Path):
        """Load weights from disk"""
        # model.from_pretrained or torch load
