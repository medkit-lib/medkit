import pytest
from numpy.testing import assert_almost_equal

pytest.importorskip(modname="seqeval", reason="seqeval is not installed")
pytest.importorskip(modname="transformers", reason="transformers is not installed")

from transformers import BertTokenizerFast

from medkit.core.text import Entity, ModifiedSpan, Span, TextDocument
from medkit.text.metrics.ner import SeqEvalEvaluator
from tests.data_utils import get_path_hf_dummy_vocab


@pytest.fixture()
def document():
    return TextDocument(
        text="medkit is a python library",
        anns=[
            Entity(label="corporation", spans=[Span(start=0, end=6)], text="medkit"),
            Entity(label="language", spans=[Span(start=12, end=18)], text="python"),
        ],
    )


_PREDICTED_ENTS_BY_CASE = {
    "perfect_prediction": [
        Entity(label="corporation", spans=[Span(start=0, end=6)], text="medkit"),
        Entity(label="language", spans=[Span(start=12, end=18)], text="python"),
    ],
    "one_missing": [
        Entity(label="corporation", spans=[Span(start=0, end=6)], text="medkit"),
        Entity(label="language", spans=[Span(start=10, end=16)], text="a pyth"),
    ],
    "incorrect_prediction": [
        Entity(label="misc", spans=[Span(start=19, end=23)], text="lib "),
    ],
}


TEST_DATA = [
    (
        _PREDICTED_ENTS_BY_CASE["perfect_prediction"],
        {
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1-score": 1.0,
            "accuracy": 1.0,
            "support": 2,
        },
    ),
    (
        _PREDICTED_ENTS_BY_CASE["one_missing"],
        {
            "macro_precision": 0.5,
            "macro_recall": 0.5,
            "macro_f1-score": 0.5,
            "accuracy": 0.8,
            "support": 2,
        },
    ),
    (
        _PREDICTED_ENTS_BY_CASE["incorrect_prediction"],
        {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1-score": 0.0,
            "accuracy": (0.38),  # there is 14 'O' in GT, 4 were tagged with 'misc' so, 10/26
            "support": 2,
        },
    ),
]


@pytest.mark.parametrize(
    ("predicted_entities", "expected_metrics"),
    TEST_DATA,
    ids=[
        "perfect_prediction",
        "one_missing",
        "incorrect_prediction",
    ],
)
def test_evaluator_bio(document, predicted_entities, expected_metrics):
    # define an evaluator with IOB2 scheme, no entities metrics
    tagging_scheme = "iob2"
    evaluator = SeqEvalEvaluator(tokenizer=None, tagging_scheme=tagging_scheme, return_metrics_by_label=False)
    metrics = evaluator.compute(documents=[document], predicted_entities=[predicted_entities])
    assert len(metrics.keys()) == len(expected_metrics.keys())
    for metric_key, value in expected_metrics.items():
        assert metric_key in metrics
        assert_almost_equal(metrics[metric_key], value, decimal=2)


@pytest.mark.parametrize(
    ("tagging_scheme", "expected_accuracy"),
    [("iob2", 0.80), ("bilou", 0.76)],
)
def test_evaluator_with_entities_all_schemes(document, tagging_scheme, expected_accuracy):
    # only accuracy changes with the scheme
    # testing with two entities, one incorrect
    predicted_entities = _PREDICTED_ENTS_BY_CASE["one_missing"]

    evaluator = SeqEvalEvaluator(tokenizer=None, tagging_scheme=tagging_scheme, return_metrics_by_label=True)
    metrics = evaluator.compute(documents=[document], predicted_entities=[predicted_entities])
    expected_metrics = {
        "macro_precision": 0.5,
        "macro_recall": 0.5,
        "macro_f1-score": 0.5,
        "support": 2,
        "accuracy": expected_accuracy,
        "corporation_precision": 1.0,
        "corporation_recall": 1.0,
        "corporation_f1-score": 1.0,
        "corporation_support": 1,
        "language_precision": 0.0,
        "language_recall": 0.0,
        "language_f1-score": 0.0,
        "language_support": 1,
    }
    assert len(metrics.keys()) == len(expected_metrics.keys())
    for metric_key, value in expected_metrics.items():
        assert metric_key in metrics
        assert_almost_equal(metrics[metric_key], value, decimal=2)


@pytest.mark.parametrize(
    ("tagging_scheme", "expected_accuracy"),
    [("iob2", 0.75), ("bilou", 0.75)],
)
def test_evaluator_with_bert_tokenizer(document, tagging_scheme, expected_accuracy):
    # testing with a bert tokenizer two entities, one incorrect
    predicted_entities = _PREDICTED_ENTS_BY_CASE["one_missing"]
    tokenizer = BertTokenizerFast(get_path_hf_dummy_vocab())
    evaluator = SeqEvalEvaluator(
        tokenizer=tokenizer,
        tagging_scheme=tagging_scheme,
        return_metrics_by_label=True,
    )
    metrics = evaluator.compute(documents=[document], predicted_entities=[predicted_entities])
    expected_metrics = {
        "macro_precision": 0.5,
        "macro_recall": 0.5,
        "macro_f1-score": 0.5,
        "support": 2,
        "accuracy": expected_accuracy,
        "corporation_precision": 1.0,
        "corporation_recall": 1.0,
        "corporation_f1-score": 1.0,
        "corporation_support": 1,
        "language_precision": 0.0,
        "language_recall": 0.0,
        "language_f1-score": 0.0,
        "language_support": 1,
    }
    assert len(metrics.keys()) == len(expected_metrics.keys())
    for metric_key, value in expected_metrics.items():
        assert metric_key in metrics
        assert_almost_equal(metrics[metric_key], value, decimal=2)


def test_modified_spans():
    """Behavior when encountering predicted entities with only modified spans
    No tag can be added to the document's raw text in that case
    """
    doc = TextDocument(text="Je souffre d'asthme.")
    entity = Entity(label="disorder", text="asthme", spans=[Span(13, 19)])
    doc.anns.add(entity)
    # entity on ModifiedSpan not referring to any spans in raw text
    # (can happen when using a translator and the alignment model fails to realign
    # some words)
    predicted_entity = Entity(
        label="disorder",
        text="asthma",
        spans=[ModifiedSpan(length=6, replaced_spans=[])],
    )
    evaluator = SeqEvalEvaluator(return_metrics_by_label=False)
    # should not crash
    metrics = evaluator.compute(documents=[doc], predicted_entities=[[predicted_entity]])
    assert metrics == {
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1-score": 0.0,
        "support": 1,
        "accuracy": 0.7,
    }


def test_labels_remapping(document):
    # identical to reference entities but with abbreviated labels
    predicted_entities = [
        Entity(label="CORP", spans=[Span(start=0, end=6)], text="medkit"),
        Entity(label="LANG", spans=[Span(start=12, end=18)], text="python"),
    ]

    expected_metrics = {
        "macro_precision": 1.0,
        "macro_recall": 1.0,
        "macro_f1-score": 1.0,
        "support": 2,
        "accuracy": 1.0,
    }

    # remap only predicted entities
    evaluator = SeqEvalEvaluator(
        labels_remapping={"CORP": "corporation", "LANG": "language"},
        return_metrics_by_label=False,
    )
    metrics = evaluator.compute(documents=[document], predicted_entities=[predicted_entities])
    assert metrics == expected_metrics

    # remap all entities (predicted and reference) to unique label
    evaluator = SeqEvalEvaluator(
        labels_remapping={
            "CORP": "ent",
            "LANG": "ent",
            "corporation": "ent",
            "language": "ent",
        },
        return_metrics_by_label=False,
    )
    metrics = evaluator.compute(documents=[document], predicted_entities=[predicted_entities])
    assert metrics == expected_metrics
