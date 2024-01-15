import logging
from pathlib import Path

import pytest

torch = pytest.importorskip(modname="torch", reason="torch is not installed")
transformers = pytest.importorskip(modname="transformers", reason="transformers is not installed")

from medkit.core.text import Entity, Span, TextDocument
from medkit.text.ner.hf_entity_matcher_trainable import (
    HFEntityMatcherTrainable,
)
from medkit.training.utils import BatchData
from tests.data_utils import get_path_hf_dummy_vocab


@pytest.fixture(autouse=True)
def _create_model_and_tokenizer(tmp_path):
    tokenizer = transformers.BertTokenizerFast(get_path_hf_dummy_vocab(), model_max_length=32)
    config = transformers.BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=20,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=10,
        max_position_embeddings=32,
    )
    # mock a bert model pretrained with 3 labels
    config.label2id = {"B-corporation": 0, "I-corporation": 1, "O": 2}
    config.id2label = {0: "B-corporation", 1: "I-corporation", 2: "O"}
    config.num_labels = 3

    model = transformers.BertForTokenClassification(config=config)
    # save model and tokenizer
    model.save_pretrained(tmp_path / "dummy-bert")
    tokenizer.save_pretrained(tmp_path / "dummy-bert")


@pytest.fixture()
def matcher(tmp_path):
    return HFEntityMatcherTrainable(
        model_name_or_path=tmp_path / "dummy-bert",
        labels=["corporation"],
        tagging_scheme="iob2",
        tokenizer_max_length=8,
    )


@pytest.fixture()
def input_data():
    return TextDocument(
        text="a test medkit",
        anns=[Entity(text="medkit", label="corporation", spans=[Span(7, 13)])],
    )


def test_preprocessing(matcher: HFEntityMatcherTrainable, input_data):
    features = matcher.preprocess(data_item=input_data)
    assert "input_ids" in features
    assert "labels" in features
    assert features["labels"] == [-100, 2, 2, 0, -100, -100]


def test_collate_and_forward(matcher: HFEntityMatcherTrainable, input_data):
    batch = [matcher.preprocess(input_data) for i in range(4)]
    collated_data = matcher.collate(batch)
    assert isinstance(collated_data, BatchData)
    assert isinstance(collated_data["input_ids"], torch.Tensor)
    assert all(tensor.size() == torch.Size([4, 6]) for tensor in collated_data.values())

    # returning loss
    model_output, loss = matcher.forward(collated_data, return_loss=True, eval_mode=True)
    assert isinstance(model_output, BatchData)
    assert "logits" in model_output
    assert isinstance(model_output["logits"], torch.Tensor)
    assert model_output["logits"].size() == torch.Size([4, 6, 3])
    assert loss is not None
    assert isinstance(loss, torch.Tensor)

    # without loss
    model_output, loss = matcher.forward(collated_data, return_loss=False, eval_mode=True)
    assert isinstance(model_output, BatchData)
    assert isinstance(model_output["logits"], torch.Tensor)
    assert loss is None


def test_collate_padding(tmp_path):
    matcher = HFEntityMatcherTrainable(
        model_name_or_path=tmp_path / "dummy-bert",
        labels=["corporation"],
        tagging_scheme="iob2",
        tokenizer_max_length=512,
    )

    docs = [
        TextDocument(
            text="a test medkit",
            anns=[Entity(text="medkit", label="corporation", spans=[Span(7, 13)])],
        ),
        TextDocument(
            text="a test medkit but that is very very much longer",
            anns=[Entity(text="medkit", label="corporation", spans=[Span(7, 13)])],
        ),
    ]
    batch = [matcher.preprocess(d) for d in docs]
    batch_data = matcher.collate(batch)

    # number of tokens in 2d longest text
    expected_padded_length = 13
    assert batch_data["input_ids"].size() == (2, expected_padded_length)
    assert batch_data["attention_mask"].size() == (2, expected_padded_length)
    assert batch_data["labels"].size() == (2, expected_padded_length)


def test_initialization_warnings(tmp_path, caplog):
    # Finetunning with bilou will be show a warning
    # The model has 3 labels and we want finetunning with 5 labels
    # this is ok because we'll do training
    with caplog.at_level(logging.WARNING):
        HFEntityMatcherTrainable(
            model_name_or_path=tmp_path / "dummy-bert",
            labels=["corporation"],
            tagging_scheme="bilou",
        )
    assert "The operation model seems to have different labels" in caplog.text

    # Trying with a model no compatible
    with pytest.raises(ValueError, match="Model .* is not associated to .*"):
        HFEntityMatcherTrainable(
            model_name_or_path="Helsinki-NLP/opus-mt-en-es",
            labels=["corporation"],
            tagging_scheme="bilou",
        )


def test_save(tmp_path, matcher: HFEntityMatcherTrainable):
    output_path = tmp_path / "bert_output"
    matcher.save(output_path)
    expected_files = ["config.json", "model.safetensors", "tokenizer.json", "vocab.txt"]
    assert all(Path(output_path / filename).exists() for filename in expected_files)
    # test loading with the path
    matcher.load(output_path)
