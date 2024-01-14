import pytest

pytest.importorskip(modname="edsnlp", reason="edsnlp is not installed")

from spacy.tokens.underscore import Underscore

from medkit.core import Attribute, ProvTracer
from medkit.core.text import Segment, Span
from medkit.text.ner import (
    DateAttribute,
    DurationAttribute,
    RelativeDateAttribute,
    RelativeDateDirection,
)
from medkit.text.ner.edsnlp_date_matcher import EDSNLPDateMatcher


# EDSNLP uses spacy which might add new extensions globally
@pytest.fixture(autouse=True)
def _reset_spacy_extensions():
    yield
    Underscore.doc_extensions = {}
    Underscore.span_extensions = {}
    Underscore.token_extensions = {}


_SPAN_OFFSET = 10


def _get_segment(text):
    return Segment(
        label="sentence",
        text=text,
        spans=[Span(_SPAN_OFFSET, _SPAN_OFFSET + len(text))],
    )


def test_absolute_date():
    """Matching of absolute date"""
    matcher = EDSNLPDateMatcher()
    seg = _get_segment("Hospitalisé le 25/10/2012")
    entities = matcher.run([seg])

    assert len(entities) == 1
    entity = entities[0]
    assert entity.label == "date"
    assert entity.text == "25/10/2012"
    assert entity.spans == [Span(_SPAN_OFFSET + 15, _SPAN_OFFSET + 25)]

    attrs = entity.attrs.get(label="date")
    assert len(attrs) == 1
    attr = attrs[0]
    assert isinstance(attr, DateAttribute)
    assert attr.value == "2012-10-25"
    assert attr.year == 2012
    assert attr.month == 10
    assert attr.day == 25


def test_relative_date():
    """Matching of relative date"""
    matcher = EDSNLPDateMatcher()
    seg = _get_segment("Hospitalisé il y a 2 mois")
    entities = matcher.run([seg])

    assert len(entities) == 1
    entity = entities[0]
    assert entity.text == "il y a 2 mois"

    attrs = entity.attrs.get(label="date")
    assert len(attrs) == 1
    attr = attrs[0]
    assert isinstance(attr, RelativeDateAttribute)
    assert attr.direction == RelativeDateDirection.PAST
    assert attr.value == "- 2 months"
    assert attr.months == 2


def test_duration():
    """Matching of duration"""
    matcher = EDSNLPDateMatcher()
    seg = _get_segment("Hospitalisé pendant 2 mois")
    entities = matcher.run([seg])

    assert len(entities) == 1
    entity = entities[0]
    assert entity.text == "pendant 2 mois"

    attrs = entity.attrs.get(label="duration")
    assert len(attrs) == 1
    attr = attrs[0]
    assert isinstance(attr, DurationAttribute)
    assert attr.value == "2 months"
    assert attr.months == 2


def test_attrs_to_copy():
    """Copying of selected attributes from input segment to created entity"""
    seg = _get_segment("Hospitalisé le 25/10/2012")
    # copied attribute
    section_attr = Attribute(label="section", value="HISTORY")
    seg.attrs.add(section_attr)
    # uncopied attribute
    seg.attrs.add(Attribute(label="negation", value=True))

    matcher = EDSNLPDateMatcher(attrs_to_copy=["section"])
    entity = matcher.run([seg])[0]

    # only section attribute was copied
    assert len(entity.attrs.get(label="section")) == 1
    assert len(entity.attrs.get(label="negation")) == 0

    # copied attribute has same value but new id
    copied_section_attr = entity.attrs.get(label="section")[0]
    assert copied_section_attr.value == section_attr.value
    assert copied_section_attr.uid != section_attr.uid


def test_prov():
    """Generated provenance nodes"""
    seg = _get_segment("Hospitalisé le 25/10/2012")

    matcher = EDSNLPDateMatcher()

    prov_tracer = ProvTracer()
    matcher.set_prov_tracer(prov_tracer)
    entities = matcher.run([seg])

    entity = entities[0]
    entity_prov = prov_tracer.get_prov(entity.uid)
    assert entity_prov.data_item == entity
    assert entity_prov.op_desc == matcher.description
    assert entity_prov.source_data_items == [seg]

    attr = entity.attrs.get(label="date")[0]
    attr_prov = prov_tracer.get_prov(attr.uid)
    assert attr_prov.data_item == attr
    assert attr_prov.op_desc == matcher.description
    assert attr_prov.source_data_items == [seg]
