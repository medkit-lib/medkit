import logging

import pytest

from medkit.core import ProvTracer
from medkit.core.text import Segment, Span
from medkit.text.context.negation_detector import (
    _PATH_TO_DEFAULT_RULES,
    NegationDetector,
    NegationDetectorRule,
)

_OUTPUT_LABEL = "negation"


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
    syntagmas = _get_syntagma_segments(["No sign of covid", "Patient has asthma"])

    rule = NegationDetectorRule(id="id_neg_no", regexp=r"^no\b")
    detector = NegationDetector(output_label=_OUTPUT_LABEL, rules=[rule])
    detector.run(syntagmas)

    # 1st syntagma has negation
    attrs_1 = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)
    assert len(attrs_1) == 1
    attr_1 = attrs_1[0]
    assert attr_1.label == _OUTPUT_LABEL
    assert attr_1.value is True
    assert attr_1.metadata["rule_id"] == "id_neg_no"

    # 2d syntagma has no negation
    attrs_2 = syntagmas[1].attrs.get(label=_OUTPUT_LABEL)
    assert len(attrs_2) == 1
    attr_2 = attrs_2[0]
    assert attr_2.value is False
    assert not attr_2.metadata


def test_multiple_rules():
    syntagmas = _get_syntagma_segments(["No sign of covid", "Diabetes is discarded"])

    rule_1 = NegationDetectorRule(id="id_neg_no", regexp=r"^no\b")
    rule_2 = NegationDetectorRule(id="id_neg_discard", regexp=r"\bdiscard(s|ed)?\b")
    detector = NegationDetector(output_label=_OUTPUT_LABEL, rules=[rule_1, rule_2])
    detector.run(syntagmas)

    # 1st syntagma has negation, matched by 1st rule
    attr_1 = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_1.value is True
    assert attr_1.metadata["rule_id"] == "id_neg_no"

    # 2d syntagma also has negation, matched by 2d rule
    attr_2 = syntagmas[1].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_2.value is True
    assert attr_2.metadata["rule_id"] == "id_neg_discard"


def test_multiple_rules_no_id():
    syntagmas = _get_syntagma_segments(["No sign of covid", "Diabetes is discarded"])

    rule_1 = NegationDetectorRule(regexp=r"^no\b")
    rule_2 = NegationDetectorRule(regexp=r"\bdiscard(s|ed)?\b")
    detector = NegationDetector(output_label=_OUTPUT_LABEL, rules=[rule_1, rule_2])
    detector.run(syntagmas)

    # attributes have corresponding rule index as rule_id metadata
    attr_1 = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_1.metadata["rule_id"] == 0
    attr_2 = syntagmas[1].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_2.metadata["rule_id"] == 1


def test_exclusions():
    syntagmas = _get_syntagma_segments(["Diabetes is discarded", "Results have not discarded covid"])

    rule = NegationDetectorRule(
        id="id_neg_discard",
        regexp=r"\bdiscard(s|ed)?\b",
        exclusion_regexps=[r"\bnot\s*\bdiscard"],
    )
    detector = NegationDetector(output_label=_OUTPUT_LABEL, rules=[rule])
    detector.run(syntagmas)

    # 1st syntagma has negation
    attr_1 = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_1.value is True

    # 2d syntagma doesn't have negation because of exclusion
    attr_2 = syntagmas[1].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_2.value is False


def test_case_sensitive_off():
    syntagmas = _get_syntagma_segments(["No sign of covid", "no sign of covid"])

    rule = NegationDetectorRule(regexp=r"^no\b", case_sensitive=False)
    detector = NegationDetector(output_label=_OUTPUT_LABEL, rules=[rule])
    detector.run(syntagmas)

    # both syntagmas have negation
    attr_1 = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_1.value is True
    attr_2 = syntagmas[1].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_2.value is True


def test_case_sensitive_on():
    syntagmas = _get_syntagma_segments(["No sign of covid", "no sign of covid"])

    rule = NegationDetectorRule(regexp=r"^no\b", case_sensitive=True)
    detector = NegationDetector(output_label=_OUTPUT_LABEL, rules=[rule])
    detector.run(syntagmas)

    # only 2d syntagma has negation
    attr_1 = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_1.value is False
    attr_2 = syntagmas[1].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_2.value is True


def test_case_sensitive_exclusions():
    syntagmas = _get_syntagma_segments(["NOT DISCARDED: covid", "not discarded: covid"])

    rule = NegationDetectorRule(
        id="id_neg_discard",
        regexp=r"\bdiscard(s|ed)?\b",
        exclusion_regexps=[r"\bNOT DISCARDED:"],
        case_sensitive=True,
    )
    detector = NegationDetector(output_label=_OUTPUT_LABEL, rules=[rule])
    detector.run(syntagmas)

    # 1st syntagma doesn't have negation because of exclusion
    attr_1 = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_1.value is False

    # 2d syntagma has negation
    attr_2 = syntagmas[1].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_2.value is True


def test_unicode_sensitive_off(caplog):
    syntagmas = _get_syntagma_segments(["Elimine: covid", "Éliminé: covid"])

    rule = NegationDetectorRule(regexp=r"elimine: ", unicode_sensitive=False)
    detector = NegationDetector(output_label=_OUTPUT_LABEL, rules=[rule])
    detector.run(syntagmas)

    # both syntagmas have negation
    attr_1 = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_1.value is True
    attr_2 = syntagmas[1].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_2.value is True

    syntagmas_with_ligatures = _get_syntagma_segments(["Sœur non covidée"])
    with caplog.at_level(logging.INFO, logger="medkit.text.context.negation_detector"):
        detector.run(syntagmas_with_ligatures)
        assert len(caplog.messages) == 1


def test_unicode_sensitive_on():
    syntagmas = _get_syntagma_segments(["Elimine: covid", "Éliminé: covid"])

    rule = NegationDetectorRule(regexp=r"éliminé: ", unicode_sensitive=True)
    detector = NegationDetector(output_label=_OUTPUT_LABEL, rules=[rule])
    detector.run(syntagmas)

    # only 2d syntagma has negation
    attr_1 = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_1.value is False
    attr_2 = syntagmas[1].attrs.get(label=_OUTPUT_LABEL)[0]
    assert attr_2.value is True


def test_empty_segment():
    """Make sure an attribute is created with False value even for empty segments"""
    syntagmas = _get_syntagma_segments(["", " .", "21."])
    rule = NegationDetectorRule(id="id_neg_no", regexp=r"^no\b")
    detector = NegationDetector(output_label="negation", rules=[rule])
    detector.run(syntagmas)
    for syntagma in syntagmas:
        attrs = syntagma.attrs.get(label=_OUTPUT_LABEL)
        assert len(attrs) == 1
        assert attrs[0].value is False


def test_prov():
    syntagmas = _get_syntagma_segments(["No sign of covid"])

    rule = NegationDetectorRule(regexp=r"^no\b")
    detector = NegationDetector(output_label=_OUTPUT_LABEL, rules=[rule])

    prov_tracer = ProvTracer()
    detector.set_prov_tracer(prov_tracer)
    detector.run(syntagmas)

    attr = syntagmas[0].attrs.get(label=_OUTPUT_LABEL)[0]
    prov = prov_tracer.get_prov(attr.uid)
    assert prov.data_item == attr
    assert prov.op_desc == detector.description
    assert prov.source_data_items == [syntagmas[0]]


# fmt: off
_TEST_DATA = [
    # pas * d
    ("pas de covid", True, "id_neg_pas_d"),
    ("Pas de covid", True, "id_neg_pas_d"),  # case insensitive
    # pas * d, exclusions
    ("pas du tout de covid", True, "id_neg_pas_d"),
    ("pas de doute, le patient est atteint", False, None),
    ("Covid pas éliminée", False, None),
    ("Covid pas exclue", False, None),
    ("pas de soucis lors du traitement", False, None),
    ("pas d'objection au traitement", False, None),
    ("Je ne reviens pas sur ce point", False, None),
    # pas * pour
    ("pas suffisant pour un covid", True, "id_neg_pas_pour"),
    # pas * pour, exclusions
    ("pas suffisant pour éliminer un covid", False, None),
    ("pas suffisant pour exclure un covid", False, None),
    # (ne|n') (l'|la|le * pas
    ("L'examen ne montre pas cette lésion", True, "id_neg_n_l_pas"),
    ("L'examen n'a pas montré cette lésion", True, "id_neg_n_l_pas"),
    ("L'examen ne la montre pas", True, "id_neg_n_l_pas"),
    ("L'examen ne le montre pas", True, "id_neg_n_l_pas"),
    ("L'examen ne l'a pas montré", True, "id_neg_n_l_pas"),
    # (ne|n') (l'|la|le * pas, exclusions
    ("L'examen ne laisse pas de doute sur la présence d'une lésion", False, None),
    ("L'examen ne permet pas d'éliminer le diagnostic covid", False, None),
    ("L'examen ne permet pas d'exclure le diagnostic covid", False, None),
    ("Le traitement n'entraîne pas de soucis", False, None),
    ("La proposition de traitement n'entraîne pas d'objection'", False, None),
    # sans
    ("sans symptome", True, "id_neg_sans"),
    # sans, exclusions
    ("sans doute souffrant du covid", False, None),
    ("sans éliminer le diagnostic covid", False, None),
    ("Traitement accepté sans problème", False, None),
    ("Traitement accepté sans soucis", False, None),
    ("Traitement accepté sans objection", False, None),
    ("Traitement accepté sans difficulté", False, None),
    # aucun
    ("aucun symptome", True, "id_neg_aucun"),
    # aucun, exclusions
    ("aucun doute sur la présence d'une lésion", False, None),
    ("Le traitement n'entraine aucun problème", False, None),
    ("aucune objection au traitement", False, None),
    # élimine
    ("Covid éliminé", True, "id_neg_elimine"),
    # élimine, exclusions
    ("Covid pas éliminé", False, None),
    ("Covid pas complètement éliminé", False, None),
    ("sans éliminer la possibilité d'un covid", False, None),
    # éliminant
    ("éliminant le covid", True, "id_neg_eliminant"),
    # éliminant, exclusions
    ("n'éliminant pas le covid", False, None),
    # infirme
    ("Covid infirmé", True, "id_neg_infirme"),
    # infirme, exclusions
    ("Ne permet pas d'infirmer le covid", False, None),
    ("Ne permet pas d'infirmer totalement le covid", False, None),
    ("sans infirmer la possibilité d'un covid", False, None),
    # infirmant
    ("infirmant le covid", True, "id_neg_infirmant"),
    # infirmant, exclusions
    ("n'infirmant pas le covid", False, None),
    # exclu
    ("Le covid est exclu", True, "id_neg_exclu"),
    ("La maladie est exclue", True, "id_neg_exclu"),
    # exclu, exclusions
    ("Il ne faut pas exclure le covid", False, None),
    ("sans exclure le covid", False, None),
    # misc
    ("Jamais de covid", True, "id_neg_jamais"),
    ("Orientant pas vers un covid", True, "id_neg_orientant_pas_vers"),
    ("Ni covid ni trouble respiration", True, "id_neg_ni"),
    ("Covid: non", True, "id_neg_column_non"),
    ("Covid: aucun", True, "id_neg_aucun"),  # FIXME: not matched by expected rule
    ("Covid: exclu", True, "id_neg_exclu"),  # FIXME: not matched by expected rule
    ("Lésions: absentes", True, "id_neg_column_absen"),
    ("Covid: négatif", True, "id_neg_negati"),
    ("Glycémie: normale", True, "id_neg_normal"),
    ("Glycémie: pas normale", False, None),
]
# fmt: on


def test_default_rules():
    syntagma_texts = [d[0] for d in _TEST_DATA]
    syntagmas = _get_syntagma_segments(syntagma_texts)

    detector = NegationDetector(output_label=_OUTPUT_LABEL)
    detector.run(syntagmas)

    for i in range(len(_TEST_DATA)):
        _, is_negated, rule_id = _TEST_DATA[i]
        syntagma = syntagmas[i]
        attr = syntagma.attrs.get(label=_OUTPUT_LABEL)[0]

        if is_negated:
            assert attr.value is True, (
                f"Syntagma '{syntagma.text}' should have been matched by '{rule_id}' " "but wasn't"
            )
            assert attr.metadata["rule_id"] == rule_id, (
                f"Syntagma '{syntagma.text}' should have been matched by '{rule_id}' "
                f"but was matched by '{attr.metadata['rule_id']}' instead"
            )
        else:
            assert (
                attr.value is False
            ), f"Syntagma '{syntagma.text}' was matched by '{attr.metadata['rule_id']}' but shouldn't have been"


def test_load_save_rules(tmpdir):
    rules_file = tmpdir / "rules.yml"
    rules = [
        NegationDetectorRule(id="id_neg_no", regexp=r"^no\b"),
        NegationDetectorRule(id="id_neg_discard", regexp=r"\bdiscard(s|ed)?\b"),
    ]

    NegationDetector.save_rules(rules, rules_file)
    assert NegationDetector.load_rules(rules_file) == rules


def test_rules_file_encoding_error():
    with pytest.raises(UnicodeError):
        NegationDetector.load_rules(path_to_rules=_PATH_TO_DEFAULT_RULES, encoding="utf-16")
