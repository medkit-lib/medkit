import logging

import pytest

from medkit.core import ProvTracer
from medkit.core.text import Segment, Span
from medkit.text.context.family_detector import (
    _PATH_TO_DEFAULT_RULES,
    FamilyDetector,
    FamilyDetectorRule,
)

_OUTPUT_LABEL = "family"


def _get_syntagma_segments(syntagma_texts):
    return [
        Segment(
            label="syntagma",
            spans=[Span(0, len(text))],
            text=text,
        )
        for text in syntagma_texts
    ]


def test_single_rule():
    syntagmas = _get_syntagma_segments(["Father died of cancer", "Patient died of cancer"])

    rule = FamilyDetectorRule(id="id_fam_father", regexp=r"\bfather\b")
    detector = FamilyDetector(output_label=_OUTPUT_LABEL, rules=[rule])
    detector.run(syntagmas)

    # 1st syntagma has family ref
    attrs_1 = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)
    assert len(attrs_1) == 1
    attr_1 = attrs_1[0]
    assert attr_1.label == _OUTPUT_LABEL
    assert attr_1.value is True
    assert attr_1.metadata["rule_id"] == "id_fam_father"

    # 2nd syntagma has no family ref
    attrs_2 = syntagmas[1].attrs.get(label=_OUTPUT_LABEL)
    assert len(attrs_2) == 1
    attr_2 = attrs_2[0]
    assert attr_2.value is False
    assert not attr_2.metadata


def test_multiple_rules():
    syntagmas = _get_syntagma_segments(["Father died of cancer", "Mother died of cancer"])
    rule_1 = FamilyDetectorRule(id="id_fam_father", regexp=r"\bfather\b")
    rule_2 = FamilyDetectorRule(id="id_fam_mother", regexp=r"\bmother\b")
    detector = FamilyDetector(output_label=_OUTPUT_LABEL, rules=[rule_1, rule_2])
    detector.run(syntagmas)

    # 1st syntagma has family ref, matched by 1st rule
    attr_1 = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_1.value is True
    assert attr_1.metadata["rule_id"] == "id_fam_father"

    # 2nd syntagma also has family ref, matched by 2nd rule
    attr_2 = syntagmas[1].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_2.value is True
    assert attr_2.metadata["rule_id"] == "id_fam_mother"


def test_multiple_rules_no_id():
    syntagmas = _get_syntagma_segments(["Father died of cancer", "Mother died of cancer"])
    rule_1 = FamilyDetectorRule(regexp=r"\bfather\b")
    rule_2 = FamilyDetectorRule(regexp=r"\bmother\b")
    detector = FamilyDetector(output_label=_OUTPUT_LABEL, rules=[rule_1, rule_2])
    detector.run(syntagmas)

    # attributes have corresponding rule index as rule_id metadata
    attr_1 = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_1.metadata["rule_id"] == 0
    attr_2 = syntagmas[1].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_2.metadata["rule_id"] == 1


def test_exclusions():
    syntagmas = _get_syntagma_segments(
        [
            "Family has history of cancer",
            "Decision was taken after discussing with the family",
        ]
    )

    rule = FamilyDetectorRule(regexp=r"\bfamily\b", exclusion_regexps=[r"\bwith\s+the\s+family"])
    detector = FamilyDetector(output_label=_OUTPUT_LABEL, rules=[rule])
    detector.run(syntagmas)

    # 1st syntagma has family ref
    attr_1 = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_1.value is True

    # 2nd syntagma doesn't have family ref because of exclusion
    attr_2 = syntagmas[1].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_2.value is False


def test_case_sensitive_on_off():
    syntagmas = _get_syntagma_segments(["Father died of cancer", "father died of cancer"])

    rule = FamilyDetectorRule(regexp=r"\bfather\b", case_sensitive=False)
    detector = FamilyDetector(output_label=_OUTPUT_LABEL, rules=[rule])
    detector.run(syntagmas)

    # both syntagmas have family refs
    attr_1 = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_1.value is True
    attr_2 = syntagmas[1].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_2.value is True


def test_case_sensitive_on():
    syntagmas = _get_syntagma_segments(["Father died of cancer", "father died of cancer"])

    rule = FamilyDetectorRule(regexp=r"\bfather\b", case_sensitive=True)
    detector = FamilyDetector(output_label=_OUTPUT_LABEL, rules=[rule])
    detector.run(syntagmas)

    # only 2nd syntagma has negation
    attr_1 = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_1.value is False
    attr_2 = syntagmas[1].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_2.value is True


def test_unicode_sensitive_off(caplog):
    syntagmas = _get_syntagma_segments(["Pere decede d'un cancer", "Père décédé d'un cancer"])

    rule = FamilyDetectorRule(regexp=r"\bpere\b", unicode_sensitive=False)
    detector = FamilyDetector(output_label=_OUTPUT_LABEL, rules=[rule])
    detector.run(syntagmas)

    # both syntagmas have family refs
    attr_1 = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_1.value is True
    attr_2 = syntagmas[1].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_2.value is True

    syntagmas_with_ligatures = _get_syntagma_segments(["Sœur non covidée"])
    with caplog.at_level(logging.INFO, logger="medkit.text.context.family_detector"):
        detector.run(syntagmas_with_ligatures)
        assert len(caplog.messages) == 1


def test_unicode_sensitive_on():
    syntagmas = _get_syntagma_segments(["Pere decede d'un cancer", "Père décédé d'un cancer"])

    rule = FamilyDetectorRule(regexp=r"\bpère\b", unicode_sensitive=True)
    detector = FamilyDetector(output_label=_OUTPUT_LABEL, rules=[rule])
    detector.run(syntagmas)

    # only 2nd syntagma has family ref
    attr_1 = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_1.value is False
    attr_2 = syntagmas[1].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_2.value is True


def test_empty_segment():
    """Make sure an attribute is created with False value even for empty segments"""
    syntagmas = _get_syntagma_segments(["", " .", "21."])
    rule = FamilyDetectorRule(id="id_fam_father", regexp=r"\bfather\b")
    detector = FamilyDetector(output_label=_OUTPUT_LABEL, rules=[rule])
    detector.run(syntagmas)
    for syntagma in syntagmas:
        attrs = syntagma.attrs.get(label=_OUTPUT_LABEL)
        assert len(attrs) == 1
        assert attrs[0].value is False


def test_prov():
    syntagmas = _get_syntagma_segments(["Father died of cancer"])

    rule = FamilyDetectorRule(regexp=r"^no\b")
    detector = FamilyDetector(output_label=_OUTPUT_LABEL, rules=[rule])

    prov_tracer = ProvTracer()
    detector.set_prov_tracer(prov_tracer)
    detector.run(syntagmas)

    attr = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)[0]
    prov = prov_tracer.get_prov(attr.uid)
    assert prov.data_item == attr
    assert prov.op_desc == detector.description
    assert prov.source_data_items == [syntagmas[0]]


# fmt: off
# text, is_family
_TEST_DATA = [
    ("Antécédents familiaux de cancer", True),
    ("Un antécédent familial de cancer", True),
    ("ATCD familiaux: cancer", True),
    ("Plusieurs cancer dans l'histoire familiale", True),
    ("Mère décédée d'un cancer", True),
    ("Père décédé d'un cancer", True),
    ("Deux cousins décédés d'un cancer", True),
    ("Une cousine décédée d'un cancer", True),
    ("Une tante décédée d'un cancer", True),
    ("Oncle décécé d'un cancer", True),
    ("Un cas de cancer chez la soeur", True),
    ("Son papa est décédé d'un cancer", True),
    ("Sa maman est décédée d'un cancer", True),
    ("Frère décédé d'un cancer", True),
    ("Cancers chez les grand-parents", True),
    ("Un neveu atteint d'un cancer", True),
    ("Son fils est atteint d'un cancer", True),
    ("Une nièce souffrant d'un cancer", True),
    ("Plusieurs cancers coté paternel", True),
    ("Plusieurs cancers coté maternel", True),
    ("Plusieurs cancers dans la famille", True),
    ("La décision a été prise en concertation avec la famille", False),
    ("Terrain familial propice au cancer", True),
]
# fmt: on


def test_default_rules():
    syntagma_texts = [d[0] for d in _TEST_DATA]
    syntagmas = _get_syntagma_segments(syntagma_texts)

    detector = FamilyDetector(output_label=_OUTPUT_LABEL)
    detector.run(syntagmas)

    for i in range(len(_TEST_DATA)):
        _, is_family_ref = _TEST_DATA[i]
        syntagma = syntagmas[i]
        attr = syntagma.attrs.get(label=_OUTPUT_LABEL)[0]

        if is_family_ref:
            assert attr.value is True, f"Syntagma '{syntagma.text}' should have been detected as family"
        else:
            assert attr.value is False, f"Syntagma '{syntagma.text}' shouldn't have been detected as family"


def test_load_save_rules(tmpdir):
    rules_file = tmpdir / "rules.yml"
    rules = [
        FamilyDetectorRule(regexp=r"\bfather\b"),
        FamilyDetectorRule(regexp=r"\bmother\b"),
    ]

    FamilyDetector.save_rules(rules, rules_file)
    assert FamilyDetector.load_rules(rules_file) == rules


def test_rules_file_encoding_error():
    with pytest.raises(UnicodeError):
        FamilyDetector.load_rules(path_to_rules=_PATH_TO_DEFAULT_RULES, encoding="utf-16")
