import pytest

pytest.importorskip(modname="spacy", reason="spacy is not installed")

from spacy import displacy

from medkit.core.text import (
    Entity,
    EntityNormAttribute,
    ModifiedSpan,
    Segment,
    Span,
    TextDocument,
)
from medkit.text.spacy.displacy_utils import (
    entities_to_displacy,
    medkit_doc_to_displacy,
)

_TEXT = "The patient has asthma and a diabetes of type 1."


def _custom_entity_formatter(entity):
    label = entity.label
    attrs_strings = []
    for attr in entity.attrs:
        if isinstance(attr, EntityNormAttribute):
            attrs_strings.append(f"{attr.kb_name}={attr.kb_id}")
        else:
            attrs_strings.append(f"{attr.label}={attr.value}")
    label += " (" + ", ".join(attrs_strings) + ")"
    return label


_TEST_DATA = [
    # basic
    (
        [
            Entity(label="disease", spans=[Span(16, 22)], text="asthma"),
            Entity(label="disease", spans=[Span(27, 47)], text="a diabetes of type 1"),
        ],
        None,
        {
            "ents": [
                {"label": "disease", "start": 16, "end": 22},
                {"label": "disease", "start": 27, "end": 47},
            ],
            "text": _TEXT,
        },
    ),
    # entity with modified span
    (
        [
            Entity(
                label="disease",
                spans=[ModifiedSpan(length=15, replaced_spans=[Span(27, 47)])],
                text="type 1 diabetes",
            )
        ],
        None,
        {
            "ents": [
                {"label": "disease", "start": 27, "end": 47},
            ],
            "text": _TEXT,
        },
    ),
    # entity with modified span containing several replaced spans with small gaps
    (
        [
            Entity(
                label="disease",
                spans=[ModifiedSpan(length=15, replaced_spans=[Span(29, 36), Span(39, 47)])],
                text="type 1 diabetes",
            )
        ],
        None,
        {
            "ents": [
                {"label": "disease", "start": 29, "end": 47},
            ],
            "text": _TEXT,
        },
    ),
    # custom formatter displaying entity attributes
    (
        [
            Entity(
                label="disease",
                spans=[Span(27, 47)],
                text="a diabetes of type 1",
                attrs=[EntityNormAttribute(kb_name="umls", kb_id="C0011854", kb_version="2021AB")],
            ),
        ],
        _custom_entity_formatter,
        {
            "ents": [
                {"label": "disease (umls=C0011854)", "start": 27, "end": 47},
            ],
            "text": _TEXT,
        },
    ),
]


@pytest.mark.parametrize(
    ("entities", "entity_formatter", "expected_displacy_data"),
    _TEST_DATA,
)
def test_entities_to_displacy(entities, entity_formatter, expected_displacy_data):
    displacy_data = entities_to_displacy(entities, _TEXT, entity_formatter)

    assert displacy_data == expected_displacy_data
    displacy.render(displacy_data, manual=True, style="ent")


def _get_doc():
    doc = TextDocument(text=_TEXT)
    entity_1 = Entity(label="subject", spans=[Span(4, 11)], text="patient")
    doc.anns.add(entity_1)
    entity_2 = Entity(label="disease", spans=[Span(16, 22)], text="asthma")
    doc.anns.add(entity_2)
    entity_3 = Entity(label="disease", spans=[Span(27, 47)], text="a diabetes of type 1")
    doc.anns.add(entity_3)
    segment = Segment(label="segment", spans=[Span(0, 19)], text="This is a sentence.")
    doc.anns.add(segment)
    return doc


def test_medkit_doc_to_displacy_default():
    doc = _get_doc()

    # by default, display all entities but not segments
    displacy_data = medkit_doc_to_displacy(doc)

    # should have same result as directly calling entities_to_displacy() with all entities
    entities = doc.anns.get_entities()
    expected_displacy_data = entities_to_displacy(entities, _TEXT)
    assert displacy_data == expected_displacy_data


def test_medkit_doc_to_displacy_filtered():
    doc = _get_doc()

    # keep only entities with "disease" label
    displacy_data = medkit_doc_to_displacy(doc, entity_labels=["disease"])

    # should have same result as directly calling entities_to_displacy() with selected entities
    entities = doc.anns.get(label="disease")
    expected_displacy_data = entities_to_displacy(entities, _TEXT)
    assert displacy_data == expected_displacy_data
