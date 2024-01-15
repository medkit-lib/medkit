import pytest
from numpy.testing import assert_almost_equal

from medkit.core import Attribute
from medkit.core.text import TextDocument

classification = pytest.importorskip("medkit.text.metrics.classification")

_TXT_1 = "Compte de rendu"
_TXT_2 = "Report d'urgences"
_LABEL_ATTR = "type_note"


@pytest.fixture()
def true_documents():
    return [
        TextDocument(_TXT_1, attrs=[Attribute(label=_LABEL_ATTR, value="CR")]),
        TextDocument(_TXT_2, attrs=[Attribute(label=_LABEL_ATTR, value="URG")]),
    ]


@pytest.fixture()
def evaluator():
    return classification.TextClassificationEvaluator(attr_label=_LABEL_ATTR)


_PREDICTED_VALUES_BY_CASE = {
    "perfect_prediction": [
        TextDocument(_TXT_1, attrs=[Attribute(label=_LABEL_ATTR, value="CR")]),
        TextDocument(_TXT_2, attrs=[Attribute(label=_LABEL_ATTR, value="URG")]),
    ],
    "one_missing": [
        TextDocument(_TXT_1, attrs=[Attribute(label=_LABEL_ATTR, value="URG")]),
        TextDocument(_TXT_2, attrs=[Attribute(label=_LABEL_ATTR, value="URG")]),
    ],
    "incorrect_prediction": [
        TextDocument(_TXT_1, attrs=[Attribute(label=_LABEL_ATTR, value="URG")]),
        TextDocument(_TXT_2, attrs=[Attribute(label=_LABEL_ATTR, value="CR")]),
    ],
}

TEST_DATA = [
    (
        _PREDICTED_VALUES_BY_CASE["perfect_prediction"],
        {
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1-score": 1.0,
            "accuracy": 1.0,
            "support": 2,
        },
    ),
    (
        _PREDICTED_VALUES_BY_CASE["one_missing"],
        {
            "macro_precision": 0.25,
            "macro_recall": 0.5,
            "macro_f1-score": 0.33,
            "accuracy": 0.5,
            "support": 2,
        },
    ),
    (
        _PREDICTED_VALUES_BY_CASE["incorrect_prediction"],
        {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1-score": 0.0,
            "accuracy": 0.0,
            "support": 2,
        },
    ),
]


@pytest.mark.parametrize(
    ("predicted_docs", "expected_metrics"),
    TEST_DATA,
    ids=[
        "perfect_prediction",
        "one_missing",
        "incorrect_prediction",
    ],
)
def test_classification_report(
    evaluator,
    true_documents,
    predicted_docs,
    expected_metrics,
):
    metrics = evaluator.compute_classification_report(
        true_docs=true_documents,
        predicted_docs=predicted_docs,
        metrics_by_attr_value=False,
    )

    assert metrics.keys() == expected_metrics.keys()
    for metric_key, value in expected_metrics.items():
        assert_almost_equal(metrics[metric_key], value, decimal=2)


def test_classification_report_by_attr_value(evaluator, true_documents):
    predicted_docs = _PREDICTED_VALUES_BY_CASE["one_missing"]
    metrics = evaluator.compute_classification_report(
        true_docs=true_documents,
        predicted_docs=predicted_docs,
        metrics_by_attr_value=True,
    )
    expected_metrics = {
        "macro_precision": 0.25,
        "macro_recall": 0.5,
        "macro_f1-score": 0.33,
        "support": 2,
        "accuracy": 0.5,
        "CR_precision": 0,
        "CR_recall": 0,
        "CR_f1-score": 0,
        "CR_support": 1,
        "URG_precision": 0.5,
        "URG_recall": 1.0,
        "URG_f1-score": 0.67,
        "URG_support": 1,
    }
    assert metrics.keys() == expected_metrics.keys()
    for metric_key, value in expected_metrics.items():
        assert_almost_equal(metrics[metric_key], value, decimal=2)


TEST_DATA_CK = [
    (_PREDICTED_VALUES_BY_CASE["perfect_prediction"], {"cohen_kappa": 1, "support": 2}),
    (_PREDICTED_VALUES_BY_CASE["one_missing"], {"cohen_kappa": 0, "support": 2}),
    (
        _PREDICTED_VALUES_BY_CASE["incorrect_prediction"],
        {"cohen_kappa": -1, "support": 2},
    ),
]


@pytest.mark.parametrize(
    ("predicted_docs", "expected_metrics"),
    TEST_DATA_CK,
    ids=[
        "total_agreement",
        "partial_agreement",
        "chance_agreement",
    ],
)
def test_cohen_kappa(
    evaluator,
    true_documents,
    predicted_docs,
    expected_metrics,
):
    metrics = evaluator.compute_cohen_kappa(true_documents, predicted_docs)
    assert metrics.keys() == expected_metrics.keys()
    for metric_key, value in expected_metrics.items():
        assert_almost_equal(metrics[metric_key], value, decimal=2)


TEST_DATA_KA = [
    (
        _PREDICTED_VALUES_BY_CASE["perfect_prediction"],
        {"krippendorff_alpha": 1, "nb_annotators": 2, "support": 2},
    ),
    (
        _PREDICTED_VALUES_BY_CASE["one_missing"],
        {"krippendorff_alpha": 0, "nb_annotators": 2, "support": 2},
    ),
    (
        _PREDICTED_VALUES_BY_CASE["incorrect_prediction"],
        {"krippendorff_alpha": -0.5, "nb_annotators": 2, "support": 2},
    ),
]


@pytest.mark.parametrize(
    ("predicted_docs", "expected_metrics"),
    TEST_DATA_KA,
    ids=[
        "total_agreement",
        "partial_agreement",
        "chance_agreement",
    ],
)
def test_krippendorff_alpha(
    evaluator,
    true_documents,
    predicted_docs,
    expected_metrics,
):
    metrics = evaluator.compute_krippendorff_alpha([true_documents, predicted_docs])
    assert metrics.keys() == expected_metrics.keys()
    for metric_key, value in expected_metrics.items():
        assert_almost_equal(metrics[metric_key], value, decimal=2)


def test_assertions_krippendorff(evaluator, true_documents):
    # test number of annotators
    with pytest.raises(ValueError, match="'docs_annotators' should contain .*"):
        evaluator.compute_krippendorff_alpha([true_documents])

    with pytest.raises(ValueError, match="'docs_annotators' should contain .*"):
        evaluator.compute_krippendorff_alpha(true_documents)


def test_assertions_docs(true_documents):
    # test number of annotators
    evaluator = classification.TextClassificationEvaluator(attr_label="other")
    with pytest.raises(ValueError, match="No attribute with label .*"):
        evaluator._extract_attr_values(true_documents)

    doc_test = true_documents[0]
    doc_test.attrs.add(Attribute(label="other", value=[0, 1, 2]))
    evaluator = classification.TextClassificationEvaluator(attr_label="other")
    with pytest.raises(ValueError, match="The type of the attr value .*"):
        evaluator._extract_attr_values([doc_test])
