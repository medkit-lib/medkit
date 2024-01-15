import logging

import pytest

from medkit.core import Attribute, ProvTracer
from medkit.core.text import EntityNormAttribute, Segment, Span, UMLSNormAttribute
from medkit.text.ner.regexp_matcher import (
    _PATH_TO_DEFAULT_RULES,
    RegexpMatcher,
    RegexpMatcherNormalization,
    RegexpMatcherRule,
)

_TEXT = "The patient has asthma and type 1 diabetes."


def _get_sentence_segment(text=_TEXT):
    return Segment(
        label="sentence",
        spans=[Span(0, len(text))],
        text=text,
    )


def test_single_rule():
    sentence = _get_sentence_segment()

    rule = RegexpMatcherRule(
        id="id_regexp_diabetes",
        label="Diabetes",
        regexp="diabetes",
        version="1",
    )
    matcher = RegexpMatcher(rules=[rule])
    entities = matcher.run([sentence])

    assert len(entities) == 1
    entity = entities[0]
    assert entity.label == "Diabetes"
    assert entity.text == "diabetes"
    assert entity.spans == [Span(34, 42)]
    assert entity.metadata["rule_id"] == "id_regexp_diabetes"
    assert entity.metadata["version"] == "1"


def test_multiple_rules():
    sentence = _get_sentence_segment()

    rule_1 = RegexpMatcherRule(
        id="id_regexp_diabetes",
        label="Diabetes",
        regexp="diabetes",
        version="1",
    )
    rule_2 = RegexpMatcherRule(
        id="id_regexp_asthma",
        label="Asthma",
        regexp="asthma",
        version="1",
    )
    matcher = RegexpMatcher(rules=[rule_1, rule_2])
    entities = matcher.run([sentence])

    assert len(entities) == 2

    # 1st entity (diabetes)
    entity_1 = entities[0]
    assert entity_1.label == "Diabetes"
    assert entity_1.text == "diabetes"
    assert entity_1.spans == [Span(34, 42)]
    assert entity_1.metadata["rule_id"] == "id_regexp_diabetes"
    assert entity_1.metadata["version"] == "1"

    # 2d entity (asthma)
    entity_2 = entities[1]
    assert entity_2.label == "Asthma"
    assert entity_2.text == "asthma"
    assert entity_2.spans == [Span(16, 22)]
    assert entity_2.metadata["rule_id"] == "id_regexp_asthma"
    assert entity_2.metadata["version"] == "1"


def test_multiple_rules_no_id():
    sentence = _get_sentence_segment()

    rule_1 = RegexpMatcherRule(label="Diabetes", regexp="diabetes")
    rule_2 = RegexpMatcherRule(label="Asthma", regexp="asthma")
    matcher = RegexpMatcher(rules=[rule_1, rule_2])
    entities = matcher.run([sentence])

    assert len(entities) == 2

    # entities have corresponding rule index as rule_id metadta
    entity_1 = entities[0]
    assert entity_1.metadata["rule_id"] == 0
    entity_2 = entities[1]
    assert entity_2.metadata["rule_id"] == 1


def test_normalization():
    sentence = _get_sentence_segment()

    rule = RegexpMatcherRule(
        label="Diabetes",
        regexp="diabetes",
        normalizations=[
            RegexpMatcherNormalization(kb_name="icd", kb_version="10", kb_id="E10-E14"),
            RegexpMatcherNormalization(kb_name="umls", kb_version="2020AB", kb_id="C0011849"),
        ],
    )
    matcher = RegexpMatcher(rules=[rule])
    entities = matcher.run([sentence])

    entity = entities[0]
    assert entity.label == "Diabetes"

    norm_attrs = entity.attrs.get_norms()
    assert len(norm_attrs) == 2
    norm_attr_1 = norm_attrs[0]
    assert type(norm_attr_1) is EntityNormAttribute
    assert norm_attr_1.kb_name == "icd"
    assert norm_attr_1.kb_version == "10"
    assert norm_attr_1.kb_id == "E10-E14"
    assert norm_attr_1.term is None

    norm_attr_2 = norm_attrs[1]
    assert type(norm_attr_2) is UMLSNormAttribute
    assert norm_attr_2.umls_version == "2020AB"
    assert norm_attr_2.cui == "C0011849"
    assert norm_attr_2.term is None


def test_exclusion_regex():
    sentence = _get_sentence_segment()

    rule = RegexpMatcherRule(label="Diabetes", regexp="diabetes", exclusion_regexp="type 1 diabetes")
    matcher = RegexpMatcher(rules=[rule])
    entities = matcher.run([sentence])

    assert len(entities) == 0


def test_case_sensitivity_off():
    sentence = _get_sentence_segment()

    rule = RegexpMatcherRule(label="Diabetes", regexp="DIABETES", case_sensitive=False)
    matcher = RegexpMatcher(rules=[rule])
    entities = matcher.run([sentence])

    assert len(entities) == 1
    entity = entities[0]
    assert entity.label == "Diabetes"


def test_case_sensitivity_on():
    sentence = _get_sentence_segment()

    rule = RegexpMatcherRule(label="Diabetes", regexp="DIABETES")
    matcher = RegexpMatcher(rules=[rule])
    entities = matcher.run([sentence])

    assert len(entities) == 0


def test_case_sensitivity_exclusion_on():
    sentence = _get_sentence_segment()

    rule = RegexpMatcherRule(
        label="Diabetes",
        regexp="diabetes",
        exclusion_regexp="TYPE 1 DIABETES",
        case_sensitive=True,
    )
    matcher = RegexpMatcher(rules=[rule])
    entities = matcher.run([sentence])

    assert len(entities) == 1
    entity = entities[0]
    assert entity.label == "Diabetes"


def test_unicode_sensitive_off(caplog):
    sentence = _get_sentence_segment("Le patient fait du diabète")

    rule = RegexpMatcherRule(label="Diabetes", regexp="diabete", unicode_sensitive=False)
    matcher = RegexpMatcher(rules=[rule])
    entities = matcher.run([sentence])

    assert len(entities) == 1
    entity = entities[0]
    assert entity.label == "Diabetes"

    sentence_with_ligatures = _get_sentence_segment(
        "Il a une sœur atteinte de diabète et pensait que sa mère avait peut-être aussi le diabète. "
    )
    with caplog.at_level(logging.INFO, logger="medkit.text.ner.regexp_matcher"):
        matcher.run([sentence_with_ligatures])
        assert len(caplog.messages) == 1


def test_unicode_sensitive_on():
    sentence = _get_sentence_segment("Le patient fait du diabète")

    rule = RegexpMatcherRule(label="Diabetes", regexp="diabete", unicode_sensitive=True)
    matcher = RegexpMatcher(rules=[rule])
    entities = matcher.run([sentence])

    assert len(entities) == 0


def test_attrs_to_copy():
    sentence = _get_sentence_segment()
    # copied attribute
    neg_attr = Attribute(label="negation", value=True)
    sentence.attrs.add(neg_attr)
    # uncopied attribute
    sentence.attrs.add(Attribute(label="hypothesis", value=True))

    rule = RegexpMatcherRule(label="Diabetes", regexp="diabetes")

    matcher = RegexpMatcher(
        rules=[rule],
        attrs_to_copy=["negation"],
    )
    entity = matcher.run([sentence])[0]

    # only negation attribute was copied
    neg_attrs = entity.attrs.get(label="negation")
    assert len(neg_attrs) == 1
    assert len(entity.attrs.get(label="hypothesis")) == 0

    # copied attribute has same value but new id
    copied_neg_attr = neg_attrs[0]
    assert copied_neg_attr.value == neg_attr.value
    assert copied_neg_attr.uid != neg_attr.uid


def test_match_at_start_of_segment():
    """Make sure we are able to match entities starting at beginning of a segment"""
    text = "Diabetes and asthma"
    sentence = Segment(label="sentence", text=text, spans=[Span(0, len(text))])

    rule = RegexpMatcherRule(label="Diabetes", regexp="diabetes", case_sensitive=False)
    matcher = RegexpMatcher(rules=[rule])
    entities = matcher.run([sentence])

    assert len(entities) == 1
    entity = entities[0]
    assert entity.label == "Diabetes"


def test_default_rules():
    sentence = _get_sentence_segment()

    # make sure default rules can be loaded and executed
    matcher = RegexpMatcher()
    _ = matcher.run([sentence])


def test_prov():
    sentence = _get_sentence_segment()

    normalization = RegexpMatcherNormalization("umls", "2020AB", "C0011849")
    rule = RegexpMatcherRule(label="Diabetes", regexp="diabetes", normalizations=[normalization])
    matcher = RegexpMatcher(rules=[rule])

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
        RegexpMatcherRule(
            label="Diabetes",
            regexp="diabetes",
            normalizations=[
                RegexpMatcherNormalization("icd", "10", "E10-E14"),
                RegexpMatcherNormalization("umls", "2020AB", "C0011849"),
            ],
        ),
        RegexpMatcherRule(
            id="id_regexp_asthma",
            label="Asthma",
            regexp="asthma",
            version="1",
        ),
    ]

    RegexpMatcher.save_rules(rules, rules_file)
    assert RegexpMatcher.load_rules(rules_file) == rules


def test_rules_file_encoding_error():
    with pytest.raises(UnicodeError):
        RegexpMatcher.load_rules(path_to_rules=_PATH_TO_DEFAULT_RULES, encoding="ascii")
