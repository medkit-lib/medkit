"""This package needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[training]`.
"""

__all__ = [
    "TrainerCallback",
    "DefaultPrinterCallback",
    "Trainer",
    "TrainerConfig",
    "BatchData",
    "MetricsComputer",
    "TrainableComponent",
]

# Verify that torch is installed
from medkit.core.utils import modules_are_available

if not modules_are_available(["torch"]):
    msg = "Requires torch install for importing medkit.training module"
    raise ImportError(msg)

from medkit.training.callbacks import DefaultPrinterCallback, TrainerCallback
from medkit.training.trainable_component import TrainableComponent
from medkit.training.trainer import Trainer
from medkit.training.trainer_config import TrainerConfig
from medkit.training.utils import BatchData, MetricsComputer
