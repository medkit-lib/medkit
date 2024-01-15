from pathlib import Path

from medkit.core import ProvTracer
from medkit.core.text import ModifiedSpan, Span, UMLSNormAttribute
from medkit.io.brat import BratInputConverter


def test_load():
    brat_converter = BratInputConverter()
    assert brat_converter.description.name == "BratInputConverter"
    docs = brat_converter.load(dir_path="tests/data/brat/")
    # 2d .ann file in dir ignored because it has no corresponding .txt
    assert len(docs) == 1

    doc = docs[0]

    assert "path_to_text" in doc.metadata
    assert "path_to_ann" in doc.metadata

    path_to_text = doc.metadata["path_to_text"]
    with Path(path_to_text).open() as file:
        text = file.read()
    assert doc.text == text

    # all expected annotations should be present
    anns = doc.anns.get()
    assert len(anns) == 9
    assert len(doc.anns.get(label="medication")) == 2
    assert len(doc.anns.get(label="disease")) == 2
    assert len(doc.anns.get(label="vitamin")) == 3
    # relations are now handled by TextDocument
    assert len(doc.anns.get_relations()) == 2

    # check relation for T1
    entity_t1 = doc.anns.get(label="medication")[0]
    entity_t3 = doc.anns.get(label="disease")[0]
    relations_ent_t1 = doc.anns.get_relations(source_id=entity_t1.uid)
    assert len(relations_ent_t1) == 1

    assert relations_ent_t1[0].label == "treats"
    assert relations_ent_t1[0].target_id == entity_t3.uid
    assert not doc.anns.get_relations(source_id=entity_t3.uid)

    # check entity T4 disease
    entity_1 = doc.anns.get(label="disease")[1]
    assert entity_1.label == "disease"
    assert entity_1.text == "Hypothyroidism"
    assert entity_1.spans == [Span(147, 161)]
    assert entity_1.metadata.get("brat_id") == "T4"

    # check attribute
    assert len(entity_1.attrs) == 1
    attr = entity_1.attrs.get()[0]
    assert attr.label == "antecedent"
    assert attr.value is None
    assert attr.metadata.get("brat_id") == "A3"

    # check attribute from note
    note_attrs = entity_t3.attrs.get(label="brat_note")
    assert len(note_attrs) == 1
    assert note_attrs[0].value == "To be reviewed"

    # check multi-span entity
    entity_2 = doc.anns.get(label="vitamin")[1]
    assert entity_2.spans == [
        Span(251, 259),
        # brat inserts a with space between discountinuous spans
        ModifiedSpan(length=1, replaced_spans=[]),
        Span(263, 264),
    ]


def test_detect_cuis_in_notes():
    brat_converter = BratInputConverter(detect_cuis_in_notes=True)
    docs = brat_converter.load(dir_path="tests/data/brat/")
    doc = docs[0]
    # retrieve entity with CUI in note
    entity = doc.anns.get(label="medication")[0]
    # check umls norm attribute
    assert len(entity.attrs.norms) == 1
    norm_attr = entity.attrs.norms[0]
    assert isinstance(norm_attr, UMLSNormAttribute)
    assert norm_attr.cui == "C0011849"

    # retrieve entity with multiple CUIs in note
    entity = doc.anns.get(label="medication")[1]
    # check umls norm attribute
    assert len(entity.attrs.norms) == 2
    assert entity.attrs.norms[0].cui == "C3021755"
    assert entity.attrs.norms[1].cui == "C3021757"

    # disable CUI detection
    brat_converter = BratInputConverter(detect_cuis_in_notes=False)
    docs = brat_converter.load(dir_path="tests/data/brat/")
    doc = docs[0]
    # retrieve entity with CUI in note
    entity = doc.anns.get(label="medication")[0]
    assert len(entity.attrs.norms) == 0


def test_relations():
    brat_converter = BratInputConverter()
    doc = brat_converter.load(dir_path="tests/data/brat/")[0]
    relations = doc.anns.get_relations()
    assert len(relations) == 2

    assert relations[0].source_id == doc.anns.get(label="medication")[0].uid
    assert relations[0].target_id == doc.anns.get(label="disease")[0].uid
    assert relations[0].label == "treats"

    assert relations[1].source_id == doc.anns.get(label="medication")[1].uid
    assert relations[1].target_id == doc.anns.get(label="disease")[1].uid
    assert relations[1].label == "treats"


def test_load_no_anns():
    brat_converter = BratInputConverter()
    docs = brat_converter.load(dir_path="tests/data/text")
    for doc in docs:
        assert len(doc.anns) == 0


def test_prov():
    brat_converter = BratInputConverter()
    prov_tracer = ProvTracer()
    brat_converter.set_prov_tracer(prov_tracer)
    docs = brat_converter.load(dir_path="tests/data/brat")

    doc = docs[0]
    entity = doc.anns.get(label="disease")[1]
    prov = prov_tracer.get_prov(entity.uid)
    assert prov.data_item == entity
    assert prov.op_desc == brat_converter.description
    assert len(prov.source_data_items) == 0
