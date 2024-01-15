"""This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[hf-utils]`.
"""
from __future__ import annotations

__all__ = ["check_model_for_task_hf"]


from typing import TYPE_CHECKING

import transformers

if TYPE_CHECKING:
    from pathlib import Path


def check_model_for_task_hf(model: str | Path, task: str, hf_auth_token: str | None = None) -> bool:
    """Check compatibility of a model with a task HuggingFace.
    The model could be in the HuggingFace hub or in local files.

    Parameters
    ----------
    model : str or Path
        Name (on the HuggingFace models hub) or path of the model.
    task : str
        A string representing the HF task to check i.e : 'token-classification'
    hf_auth_token : str, optional
        HuggingFace Authentication token (to access private models on the hub)

    Returns
    -------
    bool
        Model compatibility with the task
    """
    try:
        config = transformers.AutoConfig.from_pretrained(model, token=hf_auth_token)
    except ValueError as err:
        msg = "Impossible to get the task from model"
        raise ValueError(msg) from err

    valid_config_names = [
        config_class.__name__
        for supported_classes in transformers.pipelines.SUPPORTED_TASKS[task]["pt"]
        for config_class in supported_classes._model_mapping
    ]

    return config.__class__.__name__ in valid_config_names
