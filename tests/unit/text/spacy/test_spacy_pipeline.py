import pytest

spacy = pytest.importorskip(modname="spacy", reason="spacy is not installed")

from spacy.tokens import Doc
from spacy.tokens import Span as SpacySpan

from medkit.core import ProvTracer
from medkit.core.text import Entity, Segment, Span
from medkit.text.spacy import SpacyPipeline


@spacy.Language.component(
    "_attribute_adder_v2",
    requires=["doc.ents"],
    retokenizes=False,
)
def _custom_component(spacy_doc: Doc) -> Doc:
    """Mock spacy component, this component adds two entities including 'has_numbers' as extension"""
    # set an attribute in spacy
    if not SpacySpan.has_extension("has_numbers"):
        SpacySpan.set_extension("has_numbers", default=None)
    # add entities in spacy doc
    ents = [spacy_doc.char_span(0, 12, "PERSON"), spacy_doc.char_span(58, 62, "DATE")]
    spacy_doc.ents = list(spacy_doc.ents) + ents

    for ent in spacy_doc.ents:
        # modify the value of the attr
        value = any(token.is_digit for token in ent)
        ent._.set("has_numbers", value)
    return spacy_doc


@pytest.fixture(scope="module")
def nlp_spacy_modified():
    # use an empty spacy nlp object
    nlp = spacy.blank("en")
    nlp.add_pipe("_attribute_adder_v2", last=True)
    return nlp


TEXT_SPACY = "Marie Dupont started treatment at the central hospital in 2012"


def _get_segment():
    return Segment(text=TEXT_SPACY, spans=[Span(0, len(TEXT_SPACY))], label="test")


def test_default_spacy_pipeline(nlp_spacy_modified):
    # by default, spacyPipeline converts all spacy entities and spans
    # to medkit entities and segments
    segment = _get_segment()
    pipe = SpacyPipeline(nlp_spacy_modified)
    new_segments = pipe.run([segment])

    # original segment does not have entities, nlp from spacy adds 2 entities
    assert len(new_segments) == 2
    assert all(isinstance(seg, Entity) for seg in new_segments)
    assert all(len(seg.attrs) == 1 for seg in new_segments)

    ent = new_segments[0]
    assert ent.label == "PERSON"
    assert ent.text == "Marie Dupont"
    attr = ent.attrs.get(label="has_numbers")[0]
    assert not attr.value

    ent = new_segments[1]
    assert ent.label == "DATE"
    assert ent.text == "2012"
    attr = ent.attrs.get(label="has_numbers")[0]
    assert attr.value


def test_prov(nlp_spacy_modified):
    prov_tracer = ProvTracer()

    segment = _get_segment()
    # set provenance tracer
    pipe = SpacyPipeline(nlp=nlp_spacy_modified)
    pipe.set_prov_tracer(prov_tracer)

    # execute the pipeline
    new_segments = pipe.run([segment])

    # check new entity
    entity = new_segments[0]
    entity_prov = prov_tracer.get_prov(entity.uid)
    assert entity_prov.data_item == entity
    assert entity_prov.op_desc == pipe.description
    assert entity_prov.source_data_items == [segment]

    attribute = entity.attrs.get()[0]
    attr_prov = prov_tracer.get_prov(attribute.uid)
    assert attr_prov.data_item == attribute
    assert attr_prov.op_desc == pipe.description
    assert attr_prov.source_data_items == [segment]
