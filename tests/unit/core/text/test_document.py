from pathlib import Path

import pytest

from medkit.core import Attribute, generate_id
from medkit.core.text.annotation import Entity, Relation, Segment
from medkit.core.text.document import TextDocument
from medkit.core.text.span import Span
from tests.data_utils import get_text_document

# TODO remove tests redundant with test_annotation_container


@pytest.fixture()
def init_data():
    doc = TextDocument(text="")
    attribute = Attribute(label="Negation")
    ent1 = Entity(label="ent1", spans=[Span(0, 0)], text="", attrs=[attribute])
    ent2 = Entity(label="ent2", spans=[Span(0, 0)], text="")
    segment = Segment(label="seg1", spans=[Span(0, 0)], text="")
    relation = Relation(label="toto", source_id=ent1.uid, target_id=ent2.uid)
    return doc, ent1, ent2, segment, relation, attribute


def test_add_annotation(init_data):
    doc, ent1, ent2, segment, relation, attribute = init_data
    # Test entity addition in entity list
    doc.anns.add(ent1)
    assert ent1 in doc.anns.get_entities()
    # Test exception when adding the same annotation
    with pytest.raises(ValueError, match="Impossible to add this annotation."):
        doc.anns.add(ent1)
    # Test segment addition
    doc.anns.add(segment)
    assert segment in doc.anns.get_segments()
    # Test relation addition in annotations list
    doc.anns.add(ent2)
    doc.anns.add(relation)
    assert doc.anns.get_by_id(relation.uid) == relation


def test_get_annotations_by_key(init_data):
    doc, ent1, ent2, segment, relation, attribute = init_data
    ent1.keys.add("superkey")
    doc.anns.add(ent1)
    assert doc.anns.get(key="superkey") == [ent1]
    assert doc.anns.get(key="ello") == []


def test_get_annotations_by_label(init_data):
    doc, ent1, ent2, segment, relation, attribute = init_data
    doc.anns.add(ent1)
    doc.anns.add(ent2)

    assert doc.anns.get(label=ent1.label) == [ent1]
    assert doc.anns.get(label=ent1.label)[0].attrs.get() == [attribute]
    assert doc.anns.get(label=ent2.label) == [ent2]

    # add 2d annotation for same label and make sure we find all annotations
    # for that label
    ent3 = Entity(label=ent1.label, spans=[Span(0, 0)], text="")
    doc.anns.add(ent3)
    assert doc.anns.get(label=ent1.label) == [ent1, ent3]


def test_raw_segment():
    # raw text segment automatically generated
    text = "This is the raw text."
    doc = TextDocument(text=text)
    seg = doc.raw_segment
    assert seg is not None
    assert seg.label == TextDocument.RAW_LABEL
    assert seg.text == text
    assert seg.spans == [Span(0, len(text))]

    # also reachable through label and uid
    assert doc.anns.get(label=TextDocument.RAW_LABEL) == [seg]
    assert doc.anns.get_by_id(seg.uid) is seg
    # but not included in full annotation list
    assert seg not in doc.anns

    # docs with same ids should have raw text segments with same uid
    uid = generate_id()
    doc_1 = TextDocument(uid=uid, text=text)
    ann_1 = doc_1.anns.get(label=TextDocument.RAW_LABEL)[0]
    doc_2 = TextDocument(uid=uid, text=text)
    ann_2 = doc_2.anns.get(label=TextDocument.RAW_LABEL)[0]
    assert ann_1.uid == ann_2.uid

    # manually adding annotation with reserved label RAW_LABEL is forbidden
    doc = TextDocument(text=text)
    seg = Segment(label=TextDocument.RAW_LABEL, spans=[Span(0, len(text))], text=text)
    with pytest.raises(RuntimeError, match=r"Cannot add annotation with reserved label .*"):
        doc.anns.add(seg)


def test_add_attribute():
    doc = TextDocument("Les patients présentant un asthme chronique.")
    attr = Attribute(label="article", value=True)
    doc.attrs.add(attr)
    assert doc.attrs.get(label="article")[0].value is True


def test_snippet():
    doc = get_text_document("doc1")
    entity = Segment(
        label="disease",
        spans=[Span(739, 755)],
        text="neurofibromatose",
    )
    doc.anns.add(entity)

    snippet = doc.get_snippet(entity, max_extend_length=49)
    expected = "tats de la suspicion de neurofibromatose, je proposerai ou pas un"
    assert snippet == expected


def test_from_dir():
    dir_ = Path("tests/data/text")
    docs = TextDocument.from_dir(dir_, pattern="doc[1-3].txt")
    assert len(docs) == 3
    assert docs[0].metadata["path_to_text"] == str((dir_ / "doc1.txt").absolute())
    assert docs[0].text == (dir_ / "doc1.txt").read_text()
    assert docs[1].text == (dir_ / "doc2.txt").read_text()
    assert docs[2].text == (dir_ / "doc3.txt").read_text()
