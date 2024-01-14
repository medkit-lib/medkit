from __future__ import annotations

__all__ = ["Trainer"]

import datetime
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from medkit.training.callbacks import DefaultPrinterCallback, TrainerCallback

if TYPE_CHECKING:
    from medkit.training.trainable_component import TrainableComponent
    from medkit.training.trainer_config import TrainerConfig
    from medkit.training.utils import BatchData, MetricsComputer

# checkpoint constants
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
CONFIG_NAME = "trainer_config.yml"


def set_seed(seed: int = 0):
    """Set seed to keep deterministic operations"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _TrainerDataset(Dataset):
    """A Dataset that preprocesses data using the 'preprocess' defined in a trainable component.
    This class is inspired from the ``PipelineDataset`` class from hugginface transformers library.
    """

    def __init__(self, dataset, component: TrainableComponent):
        self.dataset = dataset
        self.component = component

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        return self.component.preprocess(item)


class Trainer:
    """A trainer is a base training/eval loop for a TrainableComponent that uses PyTorch models
    to create medkit annotations
    """

    def __init__(
        self,
        component: TrainableComponent,
        config: TrainerConfig,
        train_data: Any,
        eval_data: Any,
        metrics_computer: MetricsComputer | None = None,
        lr_scheduler_builder: Callable[[torch.optim.Optimizer], Any] | None = None,
        callback: TrainerCallback | None = None,
    ):
        """Parameters
        ----------
        component:
            The component to train, the component must implement the `TrainableComponent` protocol.
        config:
            A `TrainerConfig` with the parameters for training, the parameter `output_dir` define the
            path of the checkpoints
        train_data:
            The data to use for training. This should be a corpus of medkit objects. The data could be,
            for instance, a `torch.utils.data.Dataset` that returns medkit objects for training.
        eval_data:
            The data to use for evaluation, this is not for testing. This should be a corpus of medkit objects.
            The data can be a `torch.utils.data.Dataset` that returns medkit objects for evaluation.
        metrics_computer:
            Optional `MetricsComputer` object that will be used to compute custom metrics during eval.
            By default, only evaluation metrics will be computed, `do_metrics_in_training` in `config` allows
            metrics in training.
        lr_scheduler_builder:
            Optional function that build a `lr_scheduler` to adjust the learning rate after an epoch. Must take
            an Optimizer and return a `lr_scheduler`. If not provided, the learning rate does not change during
            training.
        callback:
            Optional callback to customize training.
        """
        # enable deterministic operation
        if config.seed is not None:
            set_seed(config.seed)

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.component = component
        self.batch_size = config.batch_size
        self.dataloader_drop_last = False
        self.dataloader_nb_workers = config.dataloader_nb_workers
        self.dataloader_pin_memory = False

        self.device = self.component.device

        self.train_dataloader = self.get_dataloader(train_data, shuffle=True)
        self.eval_dataloader = self.get_dataloader(eval_data, shuffle=False)
        self.nb_training_epochs = config.nb_training_epochs

        self.config = config

        self.optimizer = component.configure_optimizer(self.config.learning_rate)
        self.lr_scheduler = None if lr_scheduler_builder is None else lr_scheduler_builder(self.optimizer)

        self.metrics_computer = metrics_computer

        if callback is None:
            callback = DefaultPrinterCallback()
        self.callback = callback

    def get_dataloader(self, data: any, shuffle: bool) -> DataLoader:
        """Return a DataLoader with transformations defined
        in the component to train
        """
        dataset = _TrainerDataset(data, self.component)
        collate_fn = self.component.collate
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=self.dataloader_drop_last,
            num_workers=self.dataloader_nb_workers,
            pin_memory=self.dataloader_pin_memory,
        )

    def training_epoch(self) -> dict[str, float]:
        """Perform an epoch using the training data.

        When the config enabled metrics in training ('do_metrics_in_training' is True),
        the additional metrics are prepared per batch.

        Return a dictionary with metrics.
        """
        config = self.config
        total_loss_epoch = 0.0
        metrics = {}
        data_for_metrics = defaultdict(list)

        for step, input_batch in enumerate(self.train_dataloader):
            self.callback.on_step_begin(step, nb_batches=len(self.train_dataloader), phase="train")

            model_output, loss = self.make_forward_pass(input_batch, eval_mode=False)

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()

            if ((step + 1) % config.gradient_accumulation_steps == 0) or (step + 1 == len(self.train_dataloader)):
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss_epoch += loss.item()

            if config.do_metrics_in_training and self.metrics_computer is not None:
                prepared_batch = self.metrics_computer.prepare_batch(model_output, input_batch)
                for key, values in prepared_batch.items():
                    data_for_metrics[key].extend(values)

            self.callback.on_step_end(step, nb_batches=len(self.train_dataloader), phase="train")

        total_loss_epoch /= len(self.train_dataloader)
        metrics["loss"] = total_loss_epoch

        if config.do_metrics_in_training and self.metrics_computer is not None:
            metrics.update(self.metrics_computer.compute(dict(data_for_metrics)))
        return metrics

    def evaluation_epoch(self, eval_dataloader) -> dict[str, float]:
        """Perform an epoch using the evaluation data.

        The additional metrics are prepared per batch.
        Return a dictionary with metrics.
        """
        total_loss_epoch = 0.0
        metrics = {}
        data_for_metrics = defaultdict(list)

        with torch.no_grad():
            for step, input_batch in enumerate(eval_dataloader):
                self.callback.on_step_begin(step, nb_batches=len(eval_dataloader), phase="eval")

                model_output, loss = self.make_forward_pass(input_batch, eval_mode=True)
                total_loss_epoch += loss.item()

                if self.metrics_computer is not None:
                    prepared_batch = self.metrics_computer.prepare_batch(model_output, input_batch)
                    for key, values in prepared_batch.items():
                        data_for_metrics[key].extend(values)

                self.callback.on_step_end(step, nb_batches=len(eval_dataloader), phase="eval")

        total_loss_epoch /= len(self.eval_dataloader)
        metrics["loss"] = total_loss_epoch

        if self.metrics_computer is not None:
            metrics.update(self.metrics_computer.compute(dict(data_for_metrics)))
        return metrics

    def make_forward_pass(self, inputs: BatchData, eval_mode: bool) -> tuple[BatchData, torch.Tensor]:
        """Run forward safely, same device as the component"""
        inputs = inputs.to_device(self.device)
        model_output, loss = self.component.forward(inputs, return_loss=True, eval_mode=eval_mode)

        if loss is None:
            msg = "The component did not return a 'loss' from the input."
            raise ValueError(msg)

        return model_output, loss

    def update_learning_rate(self, eval_metrics: dict[str, float]):
        """Call the learning rate scheduler if defined"""
        if self.lr_scheduler is None:
            return

        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
            name_metric_to_track_lr = self.config.metric_to_track_lr
            eval_metric = eval_metrics.get(name_metric_to_track_lr)
            if eval_metric is None:
                msg = (
                    "Learning scheduler needs an eval metric to update the learning"
                    f" rate, '{name_metric_to_track_lr}' was not found"
                )
                raise ValueError(msg)
            self.lr_scheduler.step(eval_metric)
        else:
            self.lr_scheduler.step()

    def train(self) -> list[dict]:
        """Main training method. Call the training / eval loop.

        Return a list with the metrics per epoch.
        """
        self.callback.on_train_begin(config=self.config)
        log_history = []
        last_checkpoint_dir = None
        best_checkpoint_dir = None
        best_checkpoint_metric = None

        for epoch in range(1, self.nb_training_epochs + 1):
            epoch_start_time = time.time()
            self.callback.on_epoch_begin(epoch=epoch)

            train_metrics = self.training_epoch()

            eval_metrics = self.evaluation_epoch(self.eval_dataloader)
            self.update_learning_rate(eval_metrics)

            metrics = {"train": train_metrics, "eval": eval_metrics}
            log_history.append(metrics)

            self.callback.on_epoch_end(
                metrics=metrics,
                epoch=epoch,
                epoch_duration=time.time() - epoch_start_time,
            )

            # save checkpoint every N epochs if N != 0, or at last epoch
            if epoch != self.nb_training_epochs and (
                self.config.checkpoint_period == 0 or epoch % self.config.checkpoint_period != 0
            ):
                continue

            # save last checkpoint
            last_checkpoint_dir = self.save(epoch)
            # track best checkpoint, and remove former best checkpoint if last
            # checkpoint is the new best
            last_checkpoint_metric = metrics["eval"].get(self.config.checkpoint_metric)
            if last_checkpoint_metric is None:
                msg = f"Checkpoint metric '{self.config.checkpoint_metric}' not found"
                raise ValueError(msg)
            if best_checkpoint_dir is None:
                best_checkpoint_dir = last_checkpoint_dir
                best_checkpoint_metric = last_checkpoint_metric
            elif (self.config.minimize_checkpoint_metric and last_checkpoint_metric < best_checkpoint_metric) or (
                not self.config.minimize_checkpoint_metric and last_checkpoint_metric > best_checkpoint_metric
            ):
                shutil.rmtree(best_checkpoint_dir)
                best_checkpoint_dir = last_checkpoint_dir
                best_checkpoint_metric = last_checkpoint_metric

        self.callback.on_train_end()
        return log_history

    def save(self, epoch: int) -> str:
        """Save a checkpoint (trainer configuration, model weights, optimizer and
        scheduler)

        Parameters
        ----------
        epoch : int
            Epoch corresponding of the current training state (will be included
            in the checkpoint name)

        Returns
        -------
        str
            Path of the checkpoint saved
        """
        current_date = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
        name = f"checkpoint_{epoch:03d}_{current_date}"

        checkpoint_dir = Path(self.output_dir) / name
        self.callback.on_save(checkpoint_dir=str(checkpoint_dir))

        checkpoint_dir.mkdir()

        # save config
        config_path = checkpoint_dir / CONFIG_NAME
        with config_path.open(mode="w") as fp:
            yaml.safe_dump(
                self.config.to_dict(),
                fp,
                encoding="utf-8",
                allow_unicode=True,
                sort_keys=False,
            )

        torch.save(self.optimizer.state_dict(), checkpoint_dir / OPTIMIZER_NAME)

        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), checkpoint_dir / SCHEDULER_NAME)

        self.component.save(checkpoint_dir)

        return str(checkpoint_dir)
