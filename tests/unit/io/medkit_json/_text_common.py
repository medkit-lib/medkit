__all__ = [
    "build_doc",
    "build_docs",
    "build_anns",
    "DOC_JSON_FILE",
    "DOCS_JSONL_FILE",
    "ANNS_JSONL_FILE",
    "SPLIT_DOC_JSON_FILE",
    "SPLIT_DOC_ANNS_JSONL_FILE",
]

from pathlib import Path

from medkit.core import Attribute
from medkit.core.text import Entity, Segment, Span, TextDocument

DOC_JSON_FILE = Path("tests/data/medkit_json/text_doc.json")
DOCS_JSONL_FILE = Path("tests/data/medkit_json/text_docs.jsonl")
ANNS_JSONL_FILE = Path("tests/data/medkit_json/text_anns.jsonl")
SPLIT_DOC_JSON_FILE = Path("tests/data/medkit_json/split_text_doc.json")
SPLIT_DOC_ANNS_JSONL_FILE = Path("tests/data/medkit_json/split_text_doc_anns.jsonl")


def build_doc():
    """Build a text doc with 2 entities and 1 attribute on the 2d entity"""
    doc = TextDocument(uid="d1", text="I have diabetes and asthma.")
    entity_1 = Entity(uid="e1", label="disease", spans=[Span(7, 15)], text="diabetes")
    doc.anns.add(entity_1)
    entity_2 = Entity(uid="e2", label="disease", spans=[Span(20, 26)], text="asthma")
    entity_2.attrs.add(Attribute(uid="a1", label="is_negated", value=False))
    doc.anns.add(entity_2)

    return doc


def build_docs():
    """Build 2 text docs with 1 entity each"""
    doc_1 = TextDocument(uid="d1", text="I have diabetes.")
    entity_1 = Entity(uid="e1", label="disease", spans=[Span(7, 15)], text="diabetes")
    doc_1.anns.add(entity_1)

    doc_2 = TextDocument(uid="d2", text="I have asthma.")
    entity_2 = Entity(uid="e2", label="disease", spans=[Span(7, 13)], text="asthma")
    doc_2.anns.add(entity_2)

    return [doc_1, doc_2]


def build_anns():
    """Build 1 segment and 2 entities with 1 attribute on the 2d entity"""
    segment = Segment(
        uid="s1",
        label="sentence",
        text="I have diabetes and asthma.",
        spans=[Span(0, 27)],
    )
    entity_1 = Entity(uid="e1", label="disease", spans=[Span(7, 15)], text="diabetes")
    entity_2 = Entity(uid="e2", label="disease", spans=[Span(20, 26)], text="asthma")
    entity_2.attrs.add(Attribute(uid="a1", label="is_negated", value=False))
    return [segment, entity_1, entity_2]
