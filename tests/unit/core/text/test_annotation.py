import pytest

from medkit.core.text.annotation import Entity
from medkit.core.text.entity_norm_attribute import EntityNormAttribute
from medkit.core.text.span import ModifiedSpan, Span


def test_spans_for_segment():
    Entity(label="disease", text="asthme", spans=[Span(0, 6)])  # correct span
    Entity(
        label="disease",
        text="asthme",
        spans=[Span(0, 1), ModifiedSpan(length=5, replaced_spans=[Span(1, 6)])],
    )  # correct mixed spans
    with pytest.raises(AssertionError):  # wrong span
        Entity(label="disease", text="asthme", spans=[Span(0, 1)])
    with pytest.raises(AssertionError):  # wrong mixed span
        Entity(
            label="disease",
            text="asthme",
            spans=[Span(0, 1), ModifiedSpan(length=1, replaced_spans=[Span(1, 3)])],
        )


def test_normalization():
    """Test normalization helper"""
    entity = Entity(
        label="disease",
        spans=[Span(739, 755)],
        text="neurofibromatose",
    )

    norm = EntityNormAttribute(kb_name="ICD", kb_id="9540/0", kb_version="10")
    entity.attrs.add(norm)

    # EntityNormAttribute object should be returned by entity.attrs.get_norms()
    norms = entity.attrs.get_norms()
    assert len(norms) == 1
    assert norms[0] == norm
