import pytest

pytest.importorskip(modname="edsnlp", reason="edsnlp is not installed")

from spacy.tokens.underscore import Underscore

from medkit.core import Attribute, ProvTracer
from medkit.core.text import Segment, Span
from medkit.text.ner.edsnlp_tnm_matcher import EDSNLPTNMMatcher
from medkit.text.ner.tnm_attribute import TNMAttribute


# EDSNLP uses spacy which might add new extensions globally
@pytest.fixture(autouse=True)
def _reset_spacy_extensions():
    yield
    Underscore.doc_extensions = {}
    Underscore.span_extensions = {}
    Underscore.token_extensions = {}


_SPAN_OFFSET = 10


def _get_segment(text="TNM: pTx N1 M1"):
    return Segment(
        label="sentence",
        text=text,
        spans=[Span(_SPAN_OFFSET, _SPAN_OFFSET + len(text))],
    )


def test_basic():
    """Basic behavior"""
    matcher = EDSNLPTNMMatcher()
    seg = _get_segment()
    entities = matcher.run([seg])

    assert len(entities) == 1
    entity = entities[0]
    assert entity.label == "TNM"
    assert entity.text == "pTx N1 M1"
    assert entity.spans == [Span(_SPAN_OFFSET + 5, _SPAN_OFFSET + 14)]

    attrs = entity.attrs.get(label="TNM")
    assert len(attrs) == 1
    attr = attrs[0]
    assert isinstance(attr, TNMAttribute)
    assert attr.value == "pTxN1M1"
    # testing for values shown at https://aphp.github.io/edsnlp/v0.9.1/pipelines/ner/tnm/
    assert attr.prefix.value == "p"
    assert attr.tumour is None
    assert attr.tumour_specification.value == "x"
    assert attr.node.value == "1"
    assert attr.node_specification is None
    assert attr.metastasis.value == "1"
    assert attr.resection_completeness is None
    assert attr.version is None
    assert attr.version_year is None


def test_attrs_to_copy():
    """Copying of selected attributes from input segment to created entity"""
    seg = _get_segment()
    # copied attribute
    section_attr = Attribute(label="section", value="HISTORY")
    seg.attrs.add(section_attr)
    # uncopied attribute
    seg.attrs.add(Attribute(label="negation", value=True))

    matcher = EDSNLPTNMMatcher(attrs_to_copy=["section"])
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
    seg = _get_segment()

    matcher = EDSNLPTNMMatcher()

    prov_tracer = ProvTracer()
    matcher.set_prov_tracer(prov_tracer)
    entities = matcher.run([seg])

    entity = entities[0]
    entity_prov = prov_tracer.get_prov(entity.uid)
    assert entity_prov.data_item == entity
    assert entity_prov.op_desc == matcher.description
    assert entity_prov.source_data_items == [seg]

    attr = entity.attrs.get(label="TNM")[0]
    attr_prov = prov_tracer.get_prov(attr.uid)
    assert attr_prov.data_item == attr
    assert attr_prov.op_desc == matcher.description
    assert attr_prov.source_data_items == [seg]
