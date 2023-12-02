"""
This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[hf-utils]`.
"""

__all__ = ["check_model_for_task_HF"]

from pathlib import Path
from typing import Optional, Union

import transformers


def check_model_for_task_HF(
    model: Union[str, Path], task: str, hf_auth_token: Optional[str] = None
) -> bool:
    """Check compatibility of a model with a task HuggingFace.
    The model could be in the HuggingFace hub or in local files.

    Parameters
    ----------
    model:
        Name (on the HuggingFace models hub) or path of the model.

    task:
        A string representing the HF task to check i.e : 'token-classification'

    hf_auth_token:
        HuggingFace Authentication token (to access private models on the
        hub)

    Returns
    -------
    bool
        Model compatibility with the task
    """
    try:
        config = transformers.AutoConfig.from_pretrained(model, token=hf_auth_token)
    except Exception as err:
        raise ValueError("Impossible to get the task from model") from err

    valid_config_names = [
        config_class.__name__
        for supported_classes in transformers.pipelines.SUPPORTED_TASKS[task]["pt"]
        for config_class in supported_classes._model_mapping.keys()
    ]

    return config.__class__.__name__ in valid_config_names
