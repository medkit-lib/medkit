try:
    import spacy

    if not spacy.util.is_package("fr_core_news_sm"):
        spacy.cli.download("fr_core_news_sm")
except ImportError:
    spacy = None

import pytest

from medkit.core import Attribute, ProvTracer
from medkit.core.text import Segment, Span, UMLSNormAttribute
from medkit.text.ner._base_simstring_matcher import (
    _build_candidate_ranges_with_regexp,
    _build_candidate_ranges_with_spacy,
)
from medkit.text.ner.simstring_matcher import (
    SimstringMatcher,
    SimstringMatcherNormalization,
    SimstringMatcherRule,
)

_TEXT = "Le patient souffre de diabète et d'asthme."


def _get_sentence_segment(text=_TEXT):
    return Segment(
        label="sentence",
        spans=[Span(10, 10 + len(text))],
        text=text,
    )


def test_basic():
    """Basic behavior"""
    sentence = _get_sentence_segment()

    rule = SimstringMatcherRule(term="diabète", label="problem")
    matcher = SimstringMatcher(rules=[rule])
    entities = matcher.run([sentence])

    assert len(entities) == 1
    entity = entities[0]
    assert entity.text == "diabète"
    assert entity.spans == [Span(32, 39)]
    assert entity.label == "problem"


def test_multiple_matches():
    """Multiple matches in same segment"""
    sentence = _get_sentence_segment()

    rule_1 = SimstringMatcherRule(term="diabète", label="problem")
    rule_2 = SimstringMatcherRule(term="asthme", label="problem")
    rule_3 = SimstringMatcherRule(term="cancer", label="problem")
    rule_4 = SimstringMatcherRule(term="patient", label="person")
    matcher = SimstringMatcher(rules=[rule_1, rule_2, rule_3, rule_4])
    entities = matcher.run([sentence])

    assert len(entities) == 3

    # 1st entity (patient)
    entity_1 = entities[0]
    assert entity_1.text == "patient"
    assert entity_1.spans == [Span(13, 20)]
    assert entity_1.label == "person"

    # 2d entity (diabète)
    entity_2 = entities[1]
    assert entity_2.text == "diabète"
    assert entity_2.spans == [Span(32, 39)]
    assert entity_2.label == "problem"

    # 3d entity (asthme)
    entity_3 = entities[2]
    assert entity_3.text == "asthme"
    assert entity_3.spans == [Span(45, 51)]
    assert entity_3.label == "problem"


def test_approximate_match():
    """Fuzzy matching"""
    rule = SimstringMatcherRule(term="diabète", label="problem")
    matcher = SimstringMatcher(rules=[rule], threshold=0.1)

    sentence = _get_sentence_segment(text="Le patient souffre de diabèt et d'asthme.")
    entities = matcher.run([sentence])

    assert len(entities) == 1
    entity = entities[0]
    assert entity.text == "diabèt"


def test_empty_segment():
    """Handling of segment with zero tokens"""
    sentence = _get_sentence_segment(text="")

    rule = SimstringMatcherRule(term="diabète", label="problem")
    matcher = SimstringMatcher(rules=[rule])
    entities = matcher.run([sentence])

    assert len(entities) == 0


def test_normalization():
    """Normalization attribute added to entity"""
    sentence = _get_sentence_segment()

    rule = SimstringMatcherRule(
        term="diabète",
        label="problem",
        normalizations=[SimstringMatcherNormalization(kb_name="umls", kb_version="2021AB", kb_id="C0011849")],
    )
    matcher = SimstringMatcher(rules=[rule])
    entities = matcher.run([sentence])

    entity = entities[0]
    norm_attrs = entity.attrs.get_norms()
    assert len(norm_attrs) == 1
    norm_attr_1 = norm_attrs[0]
    assert type(norm_attr_1) is UMLSNormAttribute
    assert norm_attr_1.kb_name == "umls"
    assert norm_attr_1.umls_version == "2021AB"
    assert norm_attr_1.cui == "C0011849"


def test_case_sensitive():
    """Rule term and entity in text have different case, but are matched anyway"""
    sentence = _get_sentence_segment()
    sentence.text = sentence.text.replace("asthme", "Asthme")

    # no matches with case sensitivity
    rule = SimstringMatcherRule(term="asthme", label="problem", case_sensitive=True)
    matcher = SimstringMatcher(rules=[rule])
    entities = matcher.run([sentence])
    assert len(entities) == 0

    # without case sensitivity, one match is found
    rule = SimstringMatcherRule(term="asthme", label="problem", case_sensitive=False)
    matcher = SimstringMatcher(rules=[rule])
    entities = matcher.run([sentence])
    assert len(entities) == 1
    entity = entities[0]
    # lowercase not applied to entity text
    assert entity.text == "Asthme"


def test_unicode_sensitive():
    """Rule term and entity in text have different non-ascii chars, but are matched anyway"""
    sentence = _get_sentence_segment()
    sentence.text = sentence.text.replace("diabète", "dïabete")

    # no matches with unicode sensitivity
    rule = SimstringMatcherRule(term="diabète", label="problem", unicode_sensitive=True)
    matcher = SimstringMatcher(rules=[rule], threshold=1.0)
    entities = matcher.run([sentence])
    assert len(entities) == 0

    # without unicode sensitivity, one match is found
    rule = SimstringMatcherRule(term="diabète", label="problem", unicode_sensitive=False)
    matcher = SimstringMatcher(rules=[rule], threshold=1.0)
    entities = matcher.run([sentence])
    assert len(entities) == 1
    entity = entities[0]
    # unicode normalization not applied to entity text
    assert entity.text == "dïabete"


@pytest.mark.skipif(spacy is None, reason="spacy not available")
def test_spacy_tokenization():
    """Tokenize with spacy and use POS tag to avoid false positive"""
    sentence = _get_sentence_segment("Les symptomes se sont atténués")

    rule = SimstringMatcherRule(term="LES", label="problem")

    # one match with default tokenization
    matcher = SimstringMatcher(rules=[rule])
    entities = matcher.run([sentence])
    assert len(entities) == 1

    # zero match with spacy tokenization
    matcher = SimstringMatcher(rules=[rule], spacy_tokenization_language="fr")
    entities = matcher.run([sentence])
    assert len(entities) == 0


def test_blacklist():
    """Ignore blacklisted exact matches"""
    sentence = _get_sentence_segment("Il est possible que le patient ait subi un AVC")

    rule = SimstringMatcherRule(term="AIT", label="problem")

    # 1 match without blacklist
    matcher = SimstringMatcher(rules=[rule])
    entities = matcher.run([sentence])
    assert len(entities) == 1
    assert entities[0].text == "ait"

    # 0 match with blacklist
    matcher = SimstringMatcher(rules=[rule], blacklist=["ait"])
    entities = matcher.run([sentence])
    assert len(entities) == 0


def test_same_beginning():
    """Ignore matches with different start"""
    sentence = _get_sentence_segment("On constate une inactivation virale")

    rule = SimstringMatcherRule(term="activation virale", label="phenomena")

    # 1 match without same beginning flag
    matcher = SimstringMatcher(rules=[rule], threshold=0.8, same_beginning=False)
    entities = matcher.run([sentence])
    assert len(entities) == 1
    assert entities[0].text == "inactivation virale"

    # 0 match with flag
    matcher = SimstringMatcher(rules=[rule], threshold=0.8, same_beginning=True)
    entities = matcher.run([sentence])
    assert len(entities) == 0


def test_candidates_with_regexp():
    """Test internal function tokenizing the text and building candidates"""
    ranges = _build_candidate_ranges_with_regexp(_TEXT, min_length=3, max_length=15)
    candidates = [_TEXT[start:end] for start, end in ranges]
    assert candidates == [
        "Le patient",
        "patient",
        "patient souffre",
        "souffre",
        "souffre de",
        "de diabète",
        "de diabète et",
        "de diabète et d",
        "diabète",
        "diabète et",
        "diabète et d",
        "et d",
        "et d'asthme",
        "d'asthme",
        "asthme",
    ]


@pytest.mark.skipif(spacy is None, reason="spacy not available")
def test_candidates_with_spacy():
    """Test internal function tokenizing the text and building candidates"""
    spacy_lang = spacy.load(
        "fr_core_news_sm",
        # only keep tok2vec and morphologizer to get POS tags
        disable=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
    )
    doc = spacy_lang(_TEXT)

    ranges = _build_candidate_ranges_with_spacy(doc, min_length=3, max_length=15)
    candidates = [_TEXT[start:end] for start, end in ranges]
    assert candidates == [
        "patient",
        "patient souffre",
        "souffre",
        "diabète",
        "asthme",
    ]


def test_attrs_to_copy():
    sentence = _get_sentence_segment()
    # copied attribute
    neg_attr = Attribute(label="negation", value=True)
    sentence.attrs.add(neg_attr)
    # uncopied attribute
    sentence.attrs.add(Attribute(label="hypothesis", value=True))

    rule = SimstringMatcherRule(term="diabète", label="problem")

    matcher = SimstringMatcher(rules=[rule], attrs_to_copy=["negation"])
    entity = matcher.run([sentence])[0]

    # only negation attribute was copied
    neg_attrs = entity.attrs.get(label="negation")
    assert len(neg_attrs) == 1
    assert len(entity.attrs.get(label="hypothesis")) == 0

    # copied attribute has same value but new id
    copied_neg_attr = neg_attrs[0]
    assert copied_neg_attr.value == neg_attr.value
    assert copied_neg_attr.uid != neg_attr.uid


def test_prov():
    sentence = _get_sentence_segment()

    rule = SimstringMatcherRule(
        term="diabète",
        label="problem",
        normalizations=[SimstringMatcherNormalization(kb_name="umls", kb_version="2021AB", kb_id="C0011849")],
    )
    matcher = SimstringMatcher(rules=[rule])

    prov_tracer = ProvTracer()
    matcher.set_prov_tracer(prov_tracer)
    entities = matcher.run([sentence])

    entity = entities[0]
    entity_prov = prov_tracer.get_prov(entity.uid)
    assert entity_prov.data_item == entity
    assert entity_prov.op_desc == matcher.description
    assert entity_prov.source_data_items == [sentence]

    attr = entity.attrs.get_norms()[0]
    attr_prov = prov_tracer.get_prov(attr.uid)
    assert attr_prov.data_item == attr
    assert attr_prov.op_desc == matcher.description
    assert attr_prov.source_data_items == [sentence]


def test_load_save_rules(tmpdir):
    rules_file = tmpdir / "rules.yml"
    rules = [
        SimstringMatcherRule(term="asthme", label="problem"),
        SimstringMatcherRule(
            term="diabète",
            label="problem",
            case_sensitive=True,
            unicode_sensitive=True,
            normalizations=[SimstringMatcherNormalization(kb_name="umls", kb_version="2021AB", kb_id="C0011849")],
        ),
    ]

    SimstringMatcher.save_rules(rules, rules_file)
    assert SimstringMatcher.load_rules(rules_file) == rules
