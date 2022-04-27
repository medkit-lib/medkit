import pytest
import spacy.cli
from medkit.core import ProvBuilder, DictStore
from medkit.core.text import Span as MedkitSpan
from medkit.core.text.annotation import Entity, Segment
from medkit.text.spacy import SpacyPipeline

import spacy
from spacy.tokens import Span, Doc


@pytest.fixture(scope="module")
def nlp_spacy_modified():
    # download spacy models to test sents transfer
    if not spacy.util.is_package("en_core_web_sm"):
        spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("_attribute_adder_v2", last=True)
    return nlp


@spacy.Language.component(
    "_attribute_adder_v2",
    requires=["doc.ents"],
    retokenizes=False,
)
def _custom_component(spacy_doc: Doc) -> Doc:
    """Mock spacy component, this component adds 'has_numbers' extension
    in each entity"""
    # set an attribute in spacy
    if not Span.has_extension("has_numbers"):
        Span.set_extension("has_numbers", default=None)

    for ent in spacy_doc.ents:
        # check if any token is a number
        value = any(token.is_digit for token in ent)
        ent._.set("has_numbers", value)
    return spacy_doc


TEXT_SPACY = (
    " The patient was in the hospital of Paris in 2005 for an unknown degree of"
    " influenza."
)


def _get_segment():
    return Segment(
        text=TEXT_SPACY, spans=[MedkitSpan(0, len(TEXT_SPACY))], label="test"
    )


def test_default_spacy_pipeline(nlp_spacy_modified):
    segment = _get_segment()
    # define spacy pipeline and execute it
    pipe = SpacyPipeline(nlp_spacy_modified)
    new_segments = pipe.run([segment])

    assert len(new_segments) == 2
    assert all(isinstance(seg, Entity) for seg in new_segments)
    assert all(len(seg.attrs) == 1 for seg in new_segments)

    ent = new_segments[0]
    assert ent.label == "GPE"
    assert ent.text == "Paris"
    assert ent.attrs[0].label == "has_numbers"
    assert not ent.attrs[0].value

    ent = new_segments[1]
    assert ent.label == "DATE"
    assert ent.text == "2005"
    assert ent.attrs[0].label == "has_numbers"
    assert ent.attrs[0].value


def test_prov(nlp_spacy_modified):

    store = DictStore()
    prov_builder = ProvBuilder(store=store)

    segment = _get_segment()
    # set provenance builder
    pipe = SpacyPipeline(nlp=nlp_spacy_modified)
    pipe.set_prov_builder(prov_builder)

    # execute the pipeline
    new_segments = pipe.run([segment])

    graph = prov_builder.graph

    # check new entity
    entity = new_segments[0]
    node = graph.get_node(entity.id)
    assert node.data_item_id == entity.id
    assert node.operation_id == pipe.id
    assert node.source_ids == [segment.id]

    attribute = entity.attrs[0]
    attr = graph.get_node(attribute.id)
    assert attr.data_item_id == attribute.id
    assert attr.operation_id == pipe.id
    assert attr.source_ids == [entity.id]
