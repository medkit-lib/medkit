import logging


class TrainerCallback:
    """A TrainerCallback is the base class for trainer callbacks"""

    def on_init_end(self):
        """Event called at the end of the initialization of a Trainer"""
        pass

    def on_train_begin(self):
        """Event called at the beginning of training"""
        pass

    def on_train_end(self):
        """Event called at the end of training"""
        pass

    def on_epoch_begin(self):
        """Event called at the beginning of an epoch"""
        pass

    def on_epoch_end(self):
        """Event called at the end of an epoch"""
        pass

    def on_step_begin(self):
        """Event called at the beginning of a step in training"""
        pass

    def on_step_end(self):
        """Event called at the end of a step in training"""
        pass


class DefaultPrinterCallback(TrainerCallback):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)

    def on_epoch_end(self, **kwargs):
        logger = self.logger
        metrics = kwargs.pop("metrics", None)
        epoch = kwargs.pop("epoch", None)
        epoch_time = kwargs.pop("epoch_time", None)

        if metrics is not None:
            train_metrics = metrics.get("train", None)
            if train_metrics is not None:
                logger.info("-" * 59)
                msg = "|".join(
                    f"{metric_key}:{value:8.3f}"
                    for metric_key, value in train_metrics.items()
                )
                logger.info(f"Training metrics : {msg}")

            eval_metrics = metrics.get("eval", None)
            if eval_metrics is not None:
                msg = "|".join(
                    f"{metric_key}:{value:8.3f}"
                    for metric_key, value in eval_metrics.items()
                )
                logger.info(f"Evaluation metrics : {msg}")
                logger.info("-" * 59)

        if epoch is not None and epoch_time is not None:
            logger.info(
                "Epoch state: |epoch_id: {:3d} | time: {:5.2f}s".format(
                    epoch, epoch_time
                )
            )
