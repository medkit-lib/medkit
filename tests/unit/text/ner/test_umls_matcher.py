from pathlib import Path

import pytest

from medkit.core import Attribute, ProvTracer
from medkit.core.text import Segment, Span, UMLSNormAttribute
from medkit.text.ner.umls_matcher import UMLSMatcher

_UMLS_DIR = Path(__file__).parent / "sample_umls_data/2021AB"

_ASTHMA_CUI = "C0004096"
_DIABETES_CUI = "C0011854"


def _get_sentence_segment(text):
    return Segment(
        label="sentence",
        spans=[Span(0, len(text))],
        text=text,
    )


def test_basic(tmpdir):
    sentence = _get_sentence_segment("The patient has asthma.")

    umls_matcher = UMLSMatcher(umls_dir=_UMLS_DIR, language="ENG", cache_dir=tmpdir)
    entities = umls_matcher.run([sentence])

    # entity
    assert len(entities) == 1
    entity = entities[0]
    assert entity is not None
    assert entity.text == "asthma"
    assert entity.spans == [Span(16, 22)]
    assert entity.label == "disorder"

    # normalization attribute
    norm_attrs = entity.attrs.get_norms()
    assert len(norm_attrs) == 1
    norm_attr = norm_attrs[0]
    assert isinstance(norm_attr, UMLSNormAttribute)
    assert norm_attr.cui == _ASTHMA_CUI
    assert norm_attr.umls_version == "2021AB"
    assert norm_attr.term == "Asthma"
    assert norm_attr.score == 1.0


def test_multiple_matches(tmpdir):
    sentence = _get_sentence_segment("The patient has asthma and type 1 diabetes.")

    umls_matcher = UMLSMatcher(umls_dir=_UMLS_DIR, language="ENG", cache_dir=tmpdir)
    entities = umls_matcher.run([sentence])

    assert len(entities) == 2

    # 1st entity (asthma)
    entity_1 = entities[0]
    assert entity_1.label == "disorder"
    assert entity_1.text == "asthma"
    assert entity_1.spans == [Span(16, 22)]

    norm_attr_1 = entity_1.attrs.get_norms()[0]
    assert norm_attr_1.cui == _ASTHMA_CUI
    assert norm_attr_1.term == "Asthma"

    # 1st entity (diabetes)
    entity_2 = entities[1]
    assert entity_2.label == "disorder"
    assert entity_2.text == "type 1 diabetes"
    assert entity_2.spans == [Span(27, 42)]

    norm_attr_2 = entity_2.attrs.get_norms()[0]
    assert norm_attr_2.cui == _DIABETES_CUI
    assert norm_attr_2.term == "Type 1 Diabetes"


def test_cache(tmpdir):
    """Make sure we don't rebuild the database when cache can be used"""
    _ = UMLSMatcher(umls_dir=_UMLS_DIR, language="FRE", cache_dir=tmpdir)
    db_file = tmpdir / "simstring"
    print(dir(db_file.stat()))
    modification_time = db_file.stat().mtime

    # same params, cached should be used
    _ = UMLSMatcher(umls_dir=_UMLS_DIR, language="FRE", cache_dir=tmpdir)
    assert db_file.stat().mtime == modification_time

    # different params, cache can't be used, an error should be thrown
    with pytest.raises(
        Exception,
        match=("Cache directory .* contains database pre-computed with different params: .*"),
    ):
        _ = UMLSMatcher(umls_dir=_UMLS_DIR, language="FRE", cache_dir=tmpdir, normalize_unicode=True)


def test_language(tmpdir):
    sentence = _get_sentence_segment("Le patient fait de l'Asthme.")

    umls_matcher = UMLSMatcher(umls_dir=_UMLS_DIR, language="FRE", cache_dir=tmpdir)
    entities = umls_matcher.run([sentence])

    # entity
    entity = entities[0]
    assert entity.label == "disorder"
    assert entity.text == "Asthme"

    # normalization attribute, same CUI as in english
    norm_attr = entity.attrs.get_norms()[0]
    assert norm_attr.cui == _ASTHMA_CUI
    assert norm_attr.term == "Asthme"


def test_lowercase(tmpdir):
    sentence = _get_sentence_segment("Le patient fait de l'asthme.")

    # no match without lowercase flag because concept is only
    # available with leading uppercase in french
    umls_matcher = UMLSMatcher(
        umls_dir=_UMLS_DIR,
        language="FRE",
        lowercase=False,
        cache_dir=tmpdir / "nolower",
    )
    entities = umls_matcher.run([sentence])
    assert len(entities) == 0

    # with lowercase flag, entity is found
    umls_matcher = UMLSMatcher(umls_dir=_UMLS_DIR, language="FRE", lowercase=True, cache_dir=tmpdir / "lower")
    entities = umls_matcher.run([sentence])
    assert len(entities) == 1
    entity = entities[0]
    assert entity.label == "disorder"
    assert entity.text == "asthme"

    norm_attr = entity.attrs.get_norms()[0]
    assert norm_attr.cui == _ASTHMA_CUI
    assert norm_attr.term == "Asthme"


def test_ambiguous_match(tmpdir):
    sentence = _get_sentence_segment("The patient has diabetes.")

    umls_matcher = UMLSMatcher(umls_dir=_UMLS_DIR, language="ENG", cache_dir=tmpdir)
    entities = umls_matcher.run([sentence])

    # "diabetes" is a term of several CUIs but only 1 entity with
    # 1 normalization attribute is created
    assert len(entities) == 1
    entity = entities[0]
    norm_attrs = entity.attrs.get_norms()
    assert len(norm_attrs) == 1


def test_acronym(tmpdir):
    sentence = _get_sentence_segment("Le patient a fait un AVC.")

    umls_matcher = UMLSMatcher(umls_dir=_UMLS_DIR, language="FRE", cache_dir=tmpdir)
    entities = umls_matcher.run([sentence])

    assert len(entities) == 1
    entity = entities[0]
    assert entity.text == "AVC"
    assert entity.spans == [Span(21, 24)]

    norm_attrs = entity.attrs.get_norms()
    norm_attr = norm_attrs[0]
    assert norm_attr.term == "AVC (Accident Vasculaire Cérébral)"
    assert norm_attr.score == 1.0


def test_attrs_to_copy(tmpdir):
    sentence = _get_sentence_segment("The patient has asthma.")
    # copied attribute
    neg_attr = Attribute(label="negation", value=True)
    sentence.attrs.add(neg_attr)
    # uncopied attribute
    sentence.attrs.add(Attribute(label="hypothesis", value=True))

    umls_matcher = UMLSMatcher(
        umls_dir=_UMLS_DIR,
        language="ENG",
        cache_dir=tmpdir,
        attrs_to_copy=["negation"],
    )
    entity = umls_matcher.run([sentence])[0]

    norm_attrs = entity.attrs.get_norms()
    assert len(norm_attrs) == 1
    # only negation attribute was copied
    neg_attrs = entity.attrs.get(label="negation")
    assert len(neg_attrs) == 1
    assert len(entity.attrs.get(label="hypothesis")) == 0

    # copied attribute has same value but new id
    copied_neg_attr = neg_attrs[0]
    assert copied_neg_attr.value == neg_attr.value
    assert copied_neg_attr.uid != neg_attr.uid


def test_prov(tmpdir):
    sentence = _get_sentence_segment("The patient has asthma.")

    umls_matcher = UMLSMatcher(umls_dir=_UMLS_DIR, language="ENG", cache_dir=tmpdir)

    prov_tracer = ProvTracer()
    umls_matcher.set_prov_tracer(prov_tracer)
    entities = umls_matcher.run([sentence])

    entity = entities[0]
    entity_prov = prov_tracer.get_prov(entity.uid)
    assert entity_prov.data_item == entity
    assert entity_prov.op_desc == umls_matcher.description
    assert entity_prov.source_data_items == [sentence]

    attr = entity.attrs.get_norms()[0]
    attr_prov = prov_tracer.get_prov(attr.uid)
    assert attr_prov.data_item == attr
    assert attr_prov.op_desc == umls_matcher.description
    assert attr_prov.source_data_items == [sentence]


TEST_OUTPUT_LABELS = [
    (None, "disorder"),
    ("disease", "disease"),
    ({"DISO": "problem"}, "problem"),
]


@pytest.mark.parametrize(
    ("output_labels", "expected_label"),
    TEST_OUTPUT_LABELS,
    ids=["default_label", "label_str", "label_dict"],
)
def test_output_label(output_labels, expected_label, tmpdir):
    sentence = _get_sentence_segment("The patient has asthma and type 1 diabetes.")

    umls_matcher = UMLSMatcher(
        umls_dir=_UMLS_DIR,
        language="ENG",
        output_labels_by_semgroup=output_labels,
        cache_dir=tmpdir,
    )
    entities = umls_matcher.run([sentence])

    assert len(entities) == 2
    assert all(ent.label == expected_label for ent in entities)
