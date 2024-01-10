import pytest

from medkit.core.text.annotation import Segment
from medkit.core.text.span import Span

pytest.importorskip(modname="nlstruct", reason="nlstruct is not installed")
pytest.importorskip(modname="torch", reason="torch is not installed")
pytest.importorskip(modname="transformers", reason="transformers is not installed")


from medkit.text.ner.nlstruct_entity_matcher import NLStructEntityMatcher

_MODEL = "NesrineBannour/CAS-privacy-preserving-model"
_MODEL_NO_VALID = "hf-internal-testing/tiny-bert-pt-safetensors-bf16"


def test_basic():
    """Basic behavior"""
    matcher = NLStructEntityMatcher(model_name_or_dirpath=_MODEL)
    text = "Je lui prescris du lorazepam."
    segment = Segment(text=text, spans=[Span(0, len(text))], label="test")

    entities = matcher.run(segments=[segment])
    assert len(entities) == 2

    # 1st entity
    entity_1 = entities[0]
    assert entity_1.label == "LIVB"
    assert entity_1.text == "lui"
    assert entity_1.spans == [Span(3, 6)]

    # attribute
    attrs = entity_1.attrs.get(label="confidence")
    assert len(attrs) == 1
    assert attrs[0].value == 1.0

    # 2nd entity
    entity_2 = entities[1]
    assert entity_2.label == "CHEM"
    assert entity_2.text == "lorazepam"
    assert entity_2.spans == [Span(19, 28)]


def test_raises_wrong_model_error():
    with pytest.raises(FileNotFoundError, match="There was no PyTorch file with a NLstruct.*"):
        NLStructEntityMatcher(model_name_or_dirpath=_MODEL_NO_VALID)
