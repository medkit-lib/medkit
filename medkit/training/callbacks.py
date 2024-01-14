from __future__ import annotations

__all__ = ["TrainerCallback", "DefaultPrinterCallback"]

import logging
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from medkit.training.trainer_config import TrainerConfig


class TrainerCallback:
    """A TrainerCallback is the base class for trainer callbacks"""

    def on_train_begin(self, config: TrainerConfig):
        """Event called at the beginning of training"""

    def on_train_end(self):
        """Event called at the end of training"""

    def on_epoch_begin(self, epoch: int):
        """Event called at the beginning of an epoch"""

    def on_epoch_end(self, metrics: dict[str, float], epoch: int, epoch_time: float):
        """Event called at the end of an epoch"""

    def on_step_begin(self, step_idx: int, nb_batches: int, phase: str):
        """Event called at the beginning of a step in training"""

    def on_step_end(self, step_idx: int, nb_batches: int, phase: str):
        """Event called at the end of a step in training"""

    def on_save(self, checkpoint_dir: str):
        """Event called on saving a checkpoint"""


class DefaultPrinterCallback(TrainerCallback):
    """Default implementation of :class:`~.training.TrainerCallback`"""

    def __init__(self):
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # define handler and formatter
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)

        # ensure a single handler for the logger
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)
        self.logger.addHandler(console_handler)

        self._progress_bar = None

    def on_train_begin(self, config):
        message = (
            "Running training:\n"
            f"\tNum epochs: {config.nb_training_epochs}\n"
            f"\tTrain batch size:{config.batch_size}\n"
            f"\tGradient accum steps: {config.gradient_accumulation_steps}\n"
        )
        self.logger.info(message)

    def on_epoch_end(self, metrics, epoch, epoch_duration):
        message = f"Epoch {epoch} ended (duration: {epoch_duration:.2f}s)\n"

        train_metrics = metrics.get("train", None)
        if train_metrics is not None:
            message += (
                "Training metrics:\n "
                + "\n ".join(f"{metric_key}:{value:8.3f}" for metric_key, value in train_metrics.items())
                + "\n"
            )

        eval_metrics = metrics.get("eval", None)
        if eval_metrics is not None:
            message += (
                "Evaluation metrics:\n "
                + "\n ".join(f"{metric_key}:{value:8.3f}" for metric_key, value in eval_metrics.items())
                + "\n"
            )
        self.logger.info(message)

    def on_train_end(self):
        self.logger.info("Training is completed")

    def on_save(self, checkpoint_dir):
        self.logger.info("Saving checkpoint in %s", checkpoint_dir)

    def on_step_begin(self, step_idx: int, nb_batches: int, phase: str):
        if step_idx == 0:
            assert self._progress_bar is None
            self._progress_bar = tqdm(total=nb_batches)
            self._progress_bar.set_description(phase)

    def on_step_end(self, step_idx: int, nb_batches: int, phase: str):
        self._progress_bar.update()

        if step_idx + 1 == nb_batches:
            self._progress_bar.close()
            self._progress_bar = None
