import json
from pathlib import Path

import pytest

pytest.importorskip(modname="transformers", reason="transformers is not installed")

from medkit.tools.hf_utils import check_model_for_task_hf


def test_with_local_file(tmpdir):
    config_dict = {
        "_name_or_path": "./checkpoint_23-02-2023_19:34",
        "architectures": ["BertForTokenClassification"],
        "model_type": "bert",
    }
    with (Path(tmpdir) / "config.json").open(mode="w") as fp:
        json.dump(config_dict, fp)

    task = check_model_for_task_hf(tmpdir, "token-classification")
    assert task

    task = check_model_for_task_hf(tmpdir, "audio-classification")
    assert not task


@pytest.mark.parametrize(
    ("model", "task", "expected_value"),
    [
        ("samrawal/bert-base-uncased_clinical-ner", "token-classification", True),
        ("samrawal/bert-base-uncased_clinical-ner", "translation", False),
        ("Helsinki-NLP/opus-mt-fr-en", "token-classification", False),
        ("Helsinki-NLP/opus-mt-fr-en", "translation", True),
    ],
)
def test_with_remote_model(model, task, expected_value):
    task = check_model_for_task_hf(model, task)
    assert task == expected_value
