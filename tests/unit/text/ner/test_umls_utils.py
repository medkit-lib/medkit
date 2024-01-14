from pathlib import Path

import pytest

from medkit.text.ner.umls_utils import (
    UMLSEntry,
    guess_umls_version,
    load_umls_entries,
    preprocess_acronym,
    preprocess_term_to_match,
)

_PATH_TO_MR_CONSO_FILE = Path(__file__).parent / "sample_umls_data/2021AB/MRCONSO.RRF"

# fmt: off
_LOAD_TEST_PARAMS = [
    # no filter
    (
        None, None,
        [
            UMLSEntry(cui="C0004096", term="Asthma"),
            UMLSEntry(cui="C0004096", term="Asthme"),
            UMLSEntry(cui="C0011849", term="Diabetes Mellitus"),
            UMLSEntry(cui="C0011849", term="diabetes"),
            UMLSEntry(cui="C0011849", term="Diabète"),
            UMLSEntry(cui="C0011854", term="Type 1 Diabetes"),
            UMLSEntry(cui="C0011854", term="Diabète de type 1"),
            UMLSEntry(cui="C0011860", term="Diabetes Mellitus, Non-Insulin-Dependent"),
            UMLSEntry(cui="C0011860", term="Diabète de type 2"),
            UMLSEntry(cui="C0038454", term="AVC (Accident Vasculaire Cérébral)"),
        ],
    ),
    # filter on language
    (
        None, ["ENG"],
        [
            UMLSEntry(cui="C0004096", term="Asthma"),
            UMLSEntry(cui="C0011849", term="Diabetes Mellitus"),
            UMLSEntry(cui="C0011849", term="diabetes"),
            UMLSEntry(cui="C0011854", term="Type 1 Diabetes"),
            UMLSEntry(cui="C0011860", term="Diabetes Mellitus, Non-Insulin-Dependent"),
        ],
    ),
    # filter on source
    (
        ["ICD10"], ["ENG"],
        [
            UMLSEntry(cui="C0011849", term="Diabetes mellitus")
        ],
    ),
]
# fmt: on


@pytest.mark.parametrize(
    ("sources", "languages", "expected_entries"),
    _LOAD_TEST_PARAMS,
)
def test_load_umls_entries(sources, languages, expected_entries):
    entries_iter = load_umls_entries(
        _PATH_TO_MR_CONSO_FILE,
        sources=sources,
        languages=languages,
    )
    assert list(entries_iter) == expected_entries


_PREPROCESS_TEST_PARAMS = [
    ("Diabète", "Diabète", False, False),
    ("diabète", "diabète", True, False),
    ("Diabète", "Diabete", False, True),
]


@pytest.mark.parametrize(
    ("term", "expected_result", "lowercase", "normalize_unicode"),
    _PREPROCESS_TEST_PARAMS,
)
def test_preprocess_term_to_match(term, expected_result, lowercase, normalize_unicode):
    result = preprocess_term_to_match(
        term,
        lowercase,
        normalize_unicode,
    )
    assert result == expected_result


_ACRONYM_TEST_PARAMS = [
    ("AVC (Accident Vasculaire Cérébral)", "AVC"),
    ("ECG (ÉlectroCardioGramme)", "ECG"),
    ("Nutrition (physiologie)", None),
]


@pytest.mark.parametrize(
    ("term", "expected_result"),
    _ACRONYM_TEST_PARAMS,
)
def test_acronym(term, expected_result):
    assert preprocess_acronym(term) == expected_result


_VERSION_TEST_PARAMS = [
    "/home/user/umls/2021AB/META/MRCONSO.RRF",
    "/home/user/umls/2021AB/META/",
    "/home/user/umls/2021AB/",
]


@pytest.mark.parametrize("path", _VERSION_TEST_PARAMS)
def test_guess_umls_version(path):
    version = guess_umls_version(path)
    assert version == "2021AB"
