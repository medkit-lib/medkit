import pytest

spacy = pytest.importorskip(modname="spacy", reason="spacy is not installed")

from spacy.tokens import Doc

from medkit.core import Attribute
from medkit.core.text import Entity, Segment, Span, TextDocument
from medkit.io.spacy import SpacyOutputConverter


@pytest.fixture(scope="module")
def nlp_spacy():
    # use an empty nlp object
    return spacy.blank("en")


TEXT = (
    "The patient's father was prescribed Lisinopril because he was suffering from"
    " severe hypertension.\nThe patient is taking Levothyroxine"
)


def _get_doc():
    medkit_doc = TextDocument(text=TEXT)

    # entities
    ent_1 = Entity(label="medication", spans=[Span(36, 46)], text="Lisinopril", attrs=[])
    medkit_doc.anns.add(ent_1)

    ent_2_attr = Attribute(label="severity", value="high")
    ent_2 = Entity(label="disease", spans=[Span(84, 96)], text="hypertension", attrs=[ent_2_attr])
    medkit_doc.anns.add(ent_2)

    ent_3 = Entity(label="medication", spans=[Span(120, 133)], text="Levothyroxine", attrs=[])
    medkit_doc.anns.add(ent_3)

    # segments
    seg_1_attr = Attribute(label="family", value=True)
    seg_1 = Segment(
        label="PEOPLE",
        spans=[Span(0, 20)],
        text="The patient's father",
        attrs=[seg_1_attr],
    )
    medkit_doc.anns.add(seg_1)

    seg_2_attr = Attribute(label="family", value=False)
    seg_2 = Segment(label="PEOPLE", spans=[Span(98, 109)], text="The patient", attrs=[seg_2_attr])
    medkit_doc.anns.add(seg_2)
    return medkit_doc


TEST_ENTS_TO_SPACY = [
    (None, 3, ["medication", "disease"]),
    ([], 0, []),
    (["medication"], 2, ["medication"]),
]


@pytest.mark.parametrize(
    ("labels_ents_to_transfer", "expected_nb_ents", "expected_label_ents"),
    TEST_ENTS_TO_SPACY,
    ids=["all_ents", "no_annotations", "entity_by_label"],
)
def test_output_converter_entity_transfer(nlp_spacy, labels_ents_to_transfer, expected_nb_ents, expected_label_ents):
    # get medkit document to test
    medkit_doc = _get_doc()

    # testing output converter
    spacy_output_converter = SpacyOutputConverter(
        nlp_spacy,
        apply_nlp_spacy=False,
        labels_anns=labels_ents_to_transfer,
        attrs=[],
    )
    spacy_docs = spacy_output_converter.convert([medkit_doc])
    assert len(spacy_docs) == 1

    spacy_doc = spacy_docs[0]
    assert isinstance(spacy_doc, Doc)
    assert spacy_doc._.get("medkit_id") is None

    assert len(spacy_doc.ents) == expected_nb_ents
    assert all(e._.get("medkit_id") is None for e in spacy_doc.ents)


def _span_spacy_by_text(spans, text):
    return [sp for sp in spans if sp.text == text]


TEST_ATTR_TRANSFER = [
    (None, True, True),
    ([], False, False),
    (["family"], False, True),
]


@pytest.mark.parametrize(
    ("attrs_to_transfer", "should_have_severity_attr", "should_have_family_attr"),
    TEST_ATTR_TRANSFER,
    ids=["default", "no_attrs", "only_family_attr"],
)
def test_output_converter_attr_transfer(
    nlp_spacy, attrs_to_transfer, should_have_severity_attr, should_have_family_attr
):
    medkit_doc = _get_doc()

    spacy_output_converter = SpacyOutputConverter(
        nlp_spacy,
        apply_nlp_spacy=False,
        labels_anns=None,
        attrs=attrs_to_transfer,
    )

    spacy_doc = spacy_output_converter.convert([medkit_doc])[0]
    assert isinstance(spacy_doc, Doc)
    assert spacy_doc.text == medkit_doc.text
    assert len(spacy_doc.ents) == 3
    assert len(spacy_doc.spans) == 1
    assert len(spacy_doc.spans["PEOPLE"]) == 2

    ent_desease = _span_spacy_by_text(spacy_doc.ents, "hypertension")[0]
    if should_have_severity_attr:
        assert ent_desease._.get("severity") == "high"
        # medkit_id is not transferred
        assert ent_desease._.get("severity_medkit_id") is None
    else:
        assert ent_desease._.get("severity") is None

    span_people = _span_spacy_by_text(spacy_doc.spans["PEOPLE"], "The patient's father")[0]
    if should_have_family_attr:
        assert span_people._.get("family") is not None
        # medkit_id is not transferred
        assert span_people._.get("family_medkit_id") is None
    else:
        assert span_people._.get("family") is None
