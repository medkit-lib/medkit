import shutil

import pytest

transformers = pytest.importorskip(modname="transformers", reason="transformers is not installed")
torch = pytest.importorskip(modname="torch", reason="torch is not installed")

from medkit.core.text import Entity, Span, TextDocument
from medkit.text.ner.hf_entity_matcher import HFEntityMatcher
from medkit.tools import modules_are_available
from medkit.training import Trainer, TrainerConfig

_TOKENIZER_MAX_LENGTH = 24
_MODEL_NER_CLINICAL = "samrawal/bert-base-uncased_clinical-ner"

TEST_WITH_METRICS = modules_are_available(["seqeval"])
if TEST_WITH_METRICS:
    from medkit.text.metrics.ner import SeqEvalMetricsComputer


# Creating a tiny model with the original vocabulary
@pytest.fixture(autouse=True)
def _create_model_and_tokenizer(tmp_path):
    tokenizer = transformers.BertTokenizerFast.from_pretrained(
        _MODEL_NER_CLINICAL, model_max_length=32
    )  # modify the original config to make a tiny model with the original vocab
    config = transformers.BertConfig.from_pretrained(_MODEL_NER_CLINICAL)
    config.update(
        {
            "vocab_size": tokenizer.vocab_size,
            "hidden_size": 20,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "intermediate_size": 10,
            "max_position_embeddings": 32,
        }
    )

    model = transformers.BertForTokenClassification(config=config)
    # save model and tokenizer
    model.save_pretrained(tmp_path / "tiny_bert")
    tokenizer.save_pretrained(tmp_path / "tiny_bert")


def _create_tiny_data(pairs_text_entities):
    docs = []
    for text, ents in pairs_text_entities:
        entities = [Entity(label=e[0], text=text[e[1] : e[2]], spans=[Span(e[1], e[2])]) for e in ents]
        docs.append(TextDocument(text=text, anns=entities))
    return docs


@pytest.fixture()
def train_data():
    return _create_tiny_data(
        [
            ("Voici un examen", [("test", 9, 15)]),
            ("Le patient vomit ", [("problem", 11, 16)]),
            ("L'IRM a été effectuée", [("test", 2, 5)]),
            ["Sous anesthésie générale", [("treatment", 5, 24)]],
        ]
    )


@pytest.fixture()
def eval_data():
    return _create_tiny_data(
        [
            ("Le scanner a été trouvé", [("test", 9, 15)]),
            ("Douleur", [("problem", 0, 7)]),
        ]
    )


def test_trainer_default(train_data, eval_data, tmp_path):
    matcher = HFEntityMatcher.make_trainable(
        model_name_or_path=tmp_path / "tiny_bert",
        labels=["problem", "treatment", "test"],
        tagging_scheme="iob2",
        tokenizer_max_length=_TOKENIZER_MAX_LENGTH,
    )
    output_dir = tmp_path / "trained_model"
    config = TrainerConfig(
        output_dir=output_dir,
        batch_size=1,
        learning_rate=5e-4,
        nb_training_epochs=3,
        seed=0,
    )

    trainer = Trainer(component=matcher, config=config, train_data=train_data, eval_data=eval_data)

    log_history = trainer.train()
    assert len(log_history) == config.nb_training_epochs
    assert log_history[0]["train"]["loss"] > log_history[-1]["train"]["loss"]
    # we are just overfitting on the so there is no guarantee the eval loss will decrease
    eval_item = next(iter(trainer.eval_dataloader))
    assert list(eval_item["input_ids"].size()) == [1, 10]

    # [FIX] remove model to prevent writing error (cache-pytest)
    shutil.rmtree(output_dir)


@pytest.mark.skipif(not TEST_WITH_METRICS, reason="seqeval is not available")
def test_trainer_with_seqeval(train_data, eval_data, tmp_path):
    matcher = HFEntityMatcher.make_trainable(
        model_name_or_path=tmp_path / "tiny_bert",
        labels=["problem", "treatment", "test"],
        tagging_scheme="iob2",
        tokenizer_max_length=_TOKENIZER_MAX_LENGTH,
    )
    output_dir = tmp_path / "trained_model_metrics"
    config = TrainerConfig(
        output_dir=output_dir,
        batch_size=1,
        learning_rate=5e-4,
        nb_training_epochs=1,
        seed=0,
    )

    # define a metrics computer with seq eval
    metrics_computer = SeqEvalMetricsComputer(id_to_label=matcher.id_to_label, tagging_scheme=matcher.tagging_scheme)
    trainer = Trainer(
        component=matcher,
        config=config,
        train_data=train_data,
        eval_data=eval_data,
        metrics_computer=metrics_computer,
    )

    log_history = trainer.train()
    assert len(log_history) == config.nb_training_epochs

    # check key metrics
    assert "macro_precision" in log_history[0]["eval"]
    assert "accuracy" in log_history[0]["eval"]
    assert "problem_precision" in log_history[0]["eval"]

    # [FIX] remove model to prevent writing error (cache-pytest)
    shutil.rmtree(output_dir)
