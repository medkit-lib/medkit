import pytest

spacy = pytest.importorskip(modname="spacy", reason="spacy is not installed")

from spacy.tokens import Span as SpacySpan

from medkit.core import ProvTracer
from medkit.core.text import Entity, Span, TextDocument
from medkit.io.spacy import SpacyInputConverter
from medkit.text.spacy.spacy_utils import _define_attrs_extensions


@pytest.fixture(scope="module")
def nlp_spacy():
    # use an empty spacy nlp object
    return spacy.blank("en")


def _get_doc_spacy(nlp_spacy):
    """Spacy doc with entities and spans"""
    spacy_doc = nlp_spacy("Marie Dupont started treatment at the central hospital in 2012")
    # add entities in spacy doc
    ents = [spacy_doc.char_span(0, 12, "PERSON"), spacy_doc.char_span(58, 62, "DATE")]
    spacy_doc.ents = list(spacy_doc.ents) + ents
    # add span groups in spacy doc
    spacy_doc.spans["NOUN_CHUNKS"] = [spacy_doc[0:2], spacy_doc[3:4], spacy_doc[5:8]]
    spacy_doc.spans["SENTENCES"] = [spacy_doc[0:10]]
    return spacy_doc


TEST_ENTS_FROM_SPACY = [(None, 2), ([], 0), (["PERSON"], 1)]


@pytest.mark.parametrize(
    ("labels_ents_to_transfer", "expected_nb_ents"),
    TEST_ENTS_FROM_SPACY,
    ids=["default", "no_annotations", "entity_by_label"],
)
def test_input_converter_entity_transfer(
    nlp_spacy,
    labels_ents_to_transfer,
    expected_nb_ents,
):
    # create a spacy doc containing 2 entities and two span groups
    doc = _get_doc_spacy(nlp_spacy)

    # define a spacy input converter without spans or attributes
    spacy_converter = SpacyInputConverter(
        entities=labels_ents_to_transfer,
        span_groups=[],
        attrs=[],
    )
    # get a medkit doc from a spacy doc
    medkit_docs = spacy_converter.load([doc])

    assert len(medkit_docs) == 1
    medkit_doc = medkit_docs[0]

    assert isinstance(medkit_doc, TextDocument)
    # all entities should be included
    assert len(medkit_doc.anns.get_entities()) == expected_nb_ents
    assert len(medkit_doc.anns) == expected_nb_ents

    ents = medkit_doc.anns.get_entities()
    assert all(isinstance(ent, Entity) for ent in ents)
    # check entity
    if ents:
        entity_0 = medkit_doc.anns.get(label="PERSON")[0]
        assert entity_0.label == "PERSON"
        assert entity_0.text == "Marie Dupont"
        assert entity_0.spans == [Span(0, 12)]
        assert len(entity_0.attrs) == 0


TEST_ATTR_FROM_SPACY = [
    (None, [1, 2], [1, True]),
    ([], [0, 0], []),
    (["is_selected_ent"], [0, 1], [True]),
]


@pytest.mark.parametrize(
    ("attrs_to_transfer", "expected_nb_attrs", "expected_values_attr_date"),
    TEST_ATTR_FROM_SPACY,
    ids=["default", "no_attributes", "attr_by_label"],
)
def test_input_converter_attribute_transfer(nlp_spacy, attrs_to_transfer, expected_nb_attrs, expected_values_attr_date):
    # define spacy extensions to test
    if not SpacySpan.has_extension("nb_tokens_in"):
        SpacySpan.set_extension("nb_tokens_in", default=None)

    if not SpacySpan.has_extension("is_selected_ent"):
        SpacySpan.set_extension("is_selected_ent", default=None)

    # create a spacy doc containing 2 entities and two span groups
    doc = _get_doc_spacy(nlp_spacy)

    # simulates a component spacy that adds one attribute to all entities
    # to test transferring attributes by name, the fake component adds another
    # attribute in the DATE entity
    for e in doc.ents:
        e._.set("nb_tokens_in", len(list(e)))
        if e.label_ == "DATE":
            e._.set("is_selected_ent", True)

    # define a spacy input converter to transfer all entities, no spans and selected attrs
    spacy_converter = SpacyInputConverter(
        entities=None,
        span_groups=[],
        attrs=attrs_to_transfer,
    )
    # use the input converter to get a spacy doc
    medkit_docs = spacy_converter.load([doc])

    assert len(medkit_docs) == 1
    medkit_doc = medkit_docs[0]

    assert isinstance(medkit_doc, TextDocument)
    assert medkit_doc.text == doc.text
    assert len(medkit_doc.anns.get_entities()) == 2

    ents = medkit_doc.anns.get_entities()
    # verify the number of attrs for each entity
    assert [len(ent.attrs) for ent in ents] == expected_nb_attrs

    # check DATE entity
    date_entity = medkit_doc.anns.get(label="DATE")[0]
    assert [a.value for a in date_entity.attrs] == expected_values_attr_date


def test_input_converter_medkit_attribute_transfer_all_anns(nlp_spacy):
    # define an extension and a mock_attr_medkit
    if not SpacySpan.has_extension("nb_tokens_in"):
        SpacySpan.set_extension("nb_tokens_in", default=None)

    label_mock_attr_medkit = "mock_attr_medkit"
    _define_attrs_extensions([label_mock_attr_medkit])

    # create a spacy doc containing 2 entities
    doc = _get_doc_spacy(nlp_spacy)

    # simulates a component spacy that adds two attributes to all entities
    # and one attribute to sentences spans
    for e in doc.ents:
        e._.set("nb_tokens_in", len(list(e)))
        # to test transfer of medkit attributes, we add a medkit attribute manually
        # each medkit attr is transferred into spacy as two extensions
        # label_mock_attr_medkit and label_mock_attr_medkit_medkit_id
        e._.set(label_mock_attr_medkit, "value_for_entities")
        e._.set(f"{label_mock_attr_medkit}_medkit_id", "12345")

    for sp in doc.spans["SENTENCES"]:
        # simulates a different medkit attribute value for sentences
        sp._.set(label_mock_attr_medkit, "value_for_sentences")
        sp._.set(f"{label_mock_attr_medkit}_medkit_id", "12345")

    # define a spacy input converter to transfer all entities,attrs and spans
    spacy_converter = SpacyInputConverter(
        entities=None,
        span_groups=None,
        attrs=None,
    )
    # use the input converter to get a spacy doc
    medkit_docs = spacy_converter.load([doc])

    assert len(medkit_docs) == 1
    medkit_doc = medkit_docs[0]

    assert isinstance(medkit_doc, TextDocument)
    assert medkit_doc.text == doc.text
    assert len(medkit_doc.anns.get_entities()) == 2

    ents = medkit_doc.anns.get_entities()
    # verify the number of attrs for each entity
    assert [len(ent.attrs) for ent in ents] == [2, 2]
    # check value for medkit attr transferred
    entity_0 = medkit_doc.anns.get(label="PERSON")[0]
    mock_medkit_attr = entity_0.attrs.get(label=label_mock_attr_medkit)[0]
    assert mock_medkit_attr.value == "value_for_entities"

    # verify segments
    segments = medkit_doc.anns.get_segments()
    # three nouns and one sentence
    assert {s.label for s in segments} == {"NOUN_CHUNKS", "SENTENCES"}
    assert len(segments) == 4

    sentence = medkit_doc.anns.get(label="SENTENCES")[0]
    assert len(sentence.attrs) == 1
    attr = sentence.attrs.get()[0]
    assert attr.label == label_mock_attr_medkit
    assert attr.value == "value_for_sentences"


TEST_SEGMENTS_FROM_SPACY = [
    (None, {"SENTENCES", "NOUN_CHUNKS"}, 4),
    ([], set(), 0),
    (["NOUN_CHUNKS"], {"NOUN_CHUNKS"}, 3),
]


@pytest.mark.parametrize(
    ("name_spans_to_transfer", "expected_labels_in_segments", "expected_total_annotations"),
    TEST_SEGMENTS_FROM_SPACY,
    ids=["default", "no_attributes", "attr_by_label"],
)
def test_input_converter_segments_transfer(
    nlp_spacy,
    name_spans_to_transfer,
    expected_labels_in_segments,
    expected_total_annotations,
):
    # in this test, we transfer sentences (doc.sents) and noun-chunks (doc.noun_chunks)
    # to create two span groups. Noun chunks are flat phrases that have a noun as their head.

    # create a spacy doc containing 2 entities and two span groups
    doc = _get_doc_spacy(nlp_spacy)

    # define a spacy input converter to transfer selected spans, no entities nor attributes
    spacy_converter = SpacyInputConverter(
        entities=[],
        span_groups=name_spans_to_transfer,
        attrs=[],
    )
    # get a medkit doc from a spacy doc
    medkit_docs = spacy_converter.load([doc])

    assert len(medkit_docs) == 1
    medkit_doc = medkit_docs[0]
    assert medkit_doc.text == doc.text
    assert isinstance(medkit_doc, TextDocument)

    # each span group was transferred as a segment using its name as label
    assert {s.label for s in medkit_doc.anns.get_segments()} == expected_labels_in_segments

    # all selected spans should be included
    assert len(medkit_doc.anns) == expected_total_annotations

    # check segments
    if len(medkit_doc.anns):
        segments = medkit_doc.anns.get(label="NOUN_CHUNKS")
        # all noun-chunks were transferred
        assert len(segments) == 3
        expected_texts = sorted(["Marie Dupont", "treatment", "the central hospital"])
        texts = sorted([seg.text for seg in segments])
        assert texts == expected_texts


def test_prov(nlp_spacy):
    # create a spacy doc containing 2 entities and two span groups
    doc = _get_doc_spacy(nlp_spacy)

    spacy_converter = SpacyInputConverter()
    prov_tracer = ProvTracer()
    spacy_converter.set_prov_tracer(prov_tracer)

    docs = spacy_converter.load([doc])

    medkit_doc = docs[0]
    entity = medkit_doc.anns.get(label="PERSON")[0]

    prov = prov_tracer.get_prov(entity.uid)
    assert prov.data_item == entity
    assert prov.op_desc == spacy_converter.description
    assert len(prov.source_data_items) == 0
