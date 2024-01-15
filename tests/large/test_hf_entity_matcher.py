import pytest

pytest.importorskip(modname="torch", reason="torch is not installed")
pytest.importorskip(modname="transformers", reason="transformers is not installed")

from medkit.core.text import Segment, Span
from medkit.text.ner.hf_entity_matcher import HFEntityMatcher

_MODEL = "samrawal/bert-base-uncased_clinical-ner"
_MODEL_NO_VALID = "Helsinki-NLP/opus-mt-en-es"


def _get_sentence_segment(text):
    return Segment(
        label="sentence",
        spans=[Span(0, len(text))],
        text=text,
    )


def test_basic():
    """Basic behavior"""
    sentence_1 = _get_sentence_segment("The patient has asthma and is using ventoline.")
    sentence_2 = _get_sentence_segment("The patient has diabetes.")
    sentences = [sentence_1, sentence_2]
    matcher = HFEntityMatcher(model=_MODEL)
    entities = matcher.run(sentences)
    assert len(entities) == 3

    # 1st entity
    entity_1 = entities[0]
    assert entity_1.label == "problem"
    assert entity_1.text == "asthma"
    assert entity_1.spans == [Span(16, 22)]

    # 2nd entity
    entity_2 = entities[1]
    assert entity_2.label == "treatment"
    assert entity_2.text == "ventoline"
    assert entity_2.spans == [Span(36, 45)]

    # 3rd entity
    entity_3 = entities[2]
    assert entity_3.label == "problem"
    assert entity_3.text == "diabetes"
    assert entity_3.spans == [Span(16, 24)]


def test_model_error():
    with pytest.raises(ValueError, match="Model .* is not associated to .*"):
        HFEntityMatcher(model=_MODEL_NO_VALID)
