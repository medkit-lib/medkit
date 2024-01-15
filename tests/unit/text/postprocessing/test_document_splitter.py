import pytest

from medkit.core import Attribute, ProvTracer
from medkit.core.text import Entity, ModifiedSpan, Relation, Segment, Span, TextDocument
from medkit.text.postprocessing.document_splitter import DocumentSplitter


@pytest.fixture()
def doc():
    text = "The medkit library. This is a    large      entity"

    # a normal segment
    segment_1 = Segment(
        label="normal_sentence",
        spans=[Span(0, 18)],
        text="The medkit library",
        attrs=[Attribute(label="segment_attr", value=0)],
        metadata={"sent_id": "001"},
    )
    # modified segment
    segment_2 = Segment(
        label="modified_sentence",
        spans=[
            Span(start=20, end=29),
            ModifiedSpan(length=1, replaced_spans=[Span(start=29, end=33)]),
            Span(start=33, end=38),
            ModifiedSpan(length=1, replaced_spans=[Span(start=38, end=44)]),
            Span(start=44, end=50),
        ],
        text="This is a large entity",
        metadata={"sent_id": "002"},
    )
    entity_1 = Entity(
        uid="e1",
        label="ORG",
        text="medkit",
        spans=[Span(4, 10)],
        attrs=[Attribute(label="entity_attr", value=0)],
    )

    entity_2 = Entity(
        uid="e2",
        label="ENTITY",
        spans=[
            Span(start=33, end=38),
            ModifiedSpan(length=1, replaced_spans=[Span(start=38, end=44)]),
            Span(start=44, end=50),
        ],
        text="large entity",
    )

    entity_3 = Entity(
        uid="e3",
        label="MISC",
        spans=[Span(20, 24)],
        text="This",
    )
    # relations
    relation_1 = Relation(label="not_related", source_id="e1", target_id="e2")
    relation_2 = Relation(label="related", source_id="e2", target_id="e3")

    anns = [segment_1, segment_2, entity_1, entity_2, entity_3, relation_1, relation_2]
    return TextDocument(text=text, anns=anns, metadata={"doc_id": "001"})


def test_document_splitter_no_attrs(doc):
    splitter = DocumentSplitter(
        segment_label="normal_sentence",
        entity_labels=["ORG"],
        attr_labels=[],
        relation_labels=[],
    )
    new_docs = splitter.run([doc])
    assert len(new_docs) == 1

    new_doc = new_docs[0]
    assert isinstance(new_doc, TextDocument)

    assert new_doc.text == "The medkit library"
    assert new_doc.metadata == {"sent_id": "001", "doc_id": "001"}
    entities_doc = new_doc.anns.get_entities()
    assert len(entities_doc) == 1
    assert entities_doc[0].spans == [Span(4, 10)]


def test_document_splitter_attrs(doc):
    splitter = DocumentSplitter(
        segment_label="normal_sentence",
        entity_labels=["ORG"],
        attr_labels=None,
        relation_labels=[],
    )
    new_docs = splitter.run([doc])
    assert len(new_docs) == 1

    new_doc = new_docs[0]
    assert new_doc.attrs.get(label="segment_attr")
    assert len(new_doc.attrs.get(label="entity_attr")) == 0

    entity_0 = new_doc.anns.get(label="ORG")[0]
    assert entity_0.attrs.get(label="entity_attr")
    assert len(entity_0.attrs.get(label="segment_attr")) == 0


def test_with_modified_spans(doc):
    splitter = DocumentSplitter(
        segment_label="modified_sentence",
        entity_labels=["ENTITY"],
        attr_labels=[],
        relation_labels=[],
    )
    new_docs = splitter.run([doc])
    assert len(new_docs) == 1

    new_doc = new_docs[0]
    assert isinstance(new_doc, TextDocument)

    assert new_doc.text == "This is a    large      entity"
    assert new_doc.metadata == {"sent_id": "002", "doc_id": "001"}

    entities_doc = new_doc.anns.get_entities()
    assert len(entities_doc) == 1

    # spans should be offset
    entity_1 = entities_doc[0]
    assert entity_1.spans == [
        Span(start=13, end=18),
        ModifiedSpan(length=1, replaced_spans=[Span(start=18, end=24)]),
        Span(start=24, end=30),
    ]
    assert entity_1.text == "large entity"


def test_with_relations(doc):
    splitter = DocumentSplitter(
        segment_label="modified_sentence",
        entity_labels=["ENTITY", "MISC"],
        attr_labels=[],
        relation_labels=["related"],
    )
    new_docs = splitter.run([doc])
    assert len(new_docs) == 1

    new_doc = new_docs[0]
    relations = new_doc.anns.get_relations()
    assert len(relations) == 1

    relation = relations[0]
    entity_1 = new_doc.anns.get(label="ENTITY")[0]
    entity_2 = new_doc.anns.get(label="MISC")[0]
    assert relation.source_id == entity_1.uid
    assert relation.target_id == entity_2.uid


def test_prov(doc):
    splitter = DocumentSplitter(
        segment_label="normal_sentence",
        entity_labels=None,  # include all entities
        attr_labels=None,
        relation_labels=[],
    )
    prov_tracer = ProvTracer()
    splitter.set_prov_tracer(prov_tracer)
    new_docs = splitter.run([doc])
    new_doc = new_docs[0]

    sentence_1 = doc.anns.get(label="normal_sentence")[0]
    prov_1 = prov_tracer.get_prov(new_doc.uid)
    assert prov_1.data_item == new_doc
    assert prov_1.op_desc == splitter.description
    assert prov_1.source_data_items == [sentence_1]

    # check prov doc attr
    segment_attr = sentence_1.attrs.get(label="segment_attr")[0]
    doc_attr = new_doc.attrs.get(label="segment_attr")[0]
    prov_2 = prov_tracer.get_prov(doc_attr.uid)
    assert prov_2.data_item == doc_attr
    assert prov_2.op_desc == splitter.description
    assert prov_2.source_data_items == [segment_attr]

    entity_1 = doc.anns.get(label="ORG")[0]
    entity_1_new_doc = new_doc.anns.get(label="ORG")[0]
    prov_3 = prov_tracer.get_prov(entity_1_new_doc.uid)
    assert prov_3.data_item == entity_1_new_doc
    assert prov_3.op_desc == splitter.description
    assert prov_3.source_data_items == [entity_1]

    # check prov entity attr
    entity_attr = entity_1.attrs.get(label="entity_attr")[0]
    new_entity_attr = entity_1_new_doc.attrs.get(label="entity_attr")[0]
    prov_4 = prov_tracer.get_prov(new_entity_attr.uid)
    assert prov_4.data_item == new_entity_attr
    assert prov_4.op_desc == splitter.description
    assert prov_4.source_data_items == [entity_attr]


def test_prov_with_relations(doc):
    splitter = DocumentSplitter(
        segment_label="modified_sentence",
        entity_labels=None,  # include all entities
        attr_labels=["segment_attr"],
        relation_labels=None,
    )

    prov_tracer = ProvTracer()
    splitter.set_prov_tracer(prov_tracer)
    new_docs = splitter.run([doc])
    new_doc = new_docs[0]

    # check provenance in the new relation
    relation = doc.anns.get(label="related")[0]
    new_relation = new_doc.anns.get(label="related")[0]
    prov_1 = prov_tracer.get_prov(new_relation.uid)
    assert prov_1.data_item == new_relation
    assert prov_1.op_desc == splitter.description
    assert prov_1.source_data_items == [relation]
