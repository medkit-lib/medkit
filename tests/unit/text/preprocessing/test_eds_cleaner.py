import pytest

from medkit.core.prov_tracer import ProvTracer
from medkit.core.text import ModifiedSpan, Segment, Span
from medkit.text.preprocessing.eds_cleaner import EDSCleaner

TEST_DEFAULT_CONFIG = [
    (
        ("Nom patient : la Mme. Marie Dupont, date le     .24 avril .    pour un probleme"),
        "Nom patient : la Mme  Marie Dupont, date le 24 avril pour un probleme",
        [
            Span(start=0, end=20),
            ModifiedSpan(length=1, replaced_spans=[Span(start=20, end=21)]),
            Span(start=21, end=43),
            ModifiedSpan(
                length=1,
                replaced_spans=[Span(start=43, end=48), Span(start=48, end=49)],
            ),
            Span(start=49, end=57),
            ModifiedSpan(
                length=1,
                replaced_spans=[Span(start=57, end=59), Span(start=59, end=63)],
            ),
            Span(start=63, end=79),
        ],
    ),
    (
        (
            "Traitement : (N.B. : absence de notion de la prescription d'une"
            " HBPM)\n\n\n à dose curative dès cet appel."
        ),
        ("Traitement : à dose curative dès cet appel ; N B. : absence de notion de la prescription d'une HBPM."),
        [
            Span(start=0, end=12),
            ModifiedSpan(length=1, replaced_spans=[]),
            Span(start=73, end=102),
            ModifiedSpan(length=3, replaced_spans=[]),
            Span(start=14, end=15),
            ModifiedSpan(length=1, replaced_spans=[Span(start=15, end=16)]),
            Span(start=16, end=68),
            ModifiedSpan(length=1, replaced_spans=[]),
        ],
    ),
    (
        "Résultats : examen reviens (+) le (5 juin)",
        "Résultats : examen reviens  positif  le ,5 juin,",
        [
            Span(start=0, end=27),
            ModifiedSpan(length=9, replaced_spans=[Span(start=27, end=30)]),
            Span(start=30, end=34),
            ModifiedSpan(length=1, replaced_spans=[Span(start=34, end=35)]),
            Span(start=35, end=41),
            ModifiedSpan(length=1, replaced_spans=[Span(start=41, end=42)]),
        ],
    ),
    (
        "Le patient\n\n\n       reviens\n\n le 4 d\n\n\n\navril",
        "Le patient reviens le 4 d avril",
        [
            Span(start=0, end=10),
            ModifiedSpan(
                length=1,
                replaced_spans=[Span(start=10, end=13), Span(start=13, end=20)],
            ),
            Span(start=20, end=27),
            ModifiedSpan(
                length=1,
                replaced_spans=[Span(start=27, end=29), Span(start=29, end=30)],
            ),
            Span(start=30, end=36),
            ModifiedSpan(length=1, replaced_spans=[Span(start=36, end=40)]),
            Span(start=40, end=45),
        ],
    ),
]


def _get_raw_segment(text):
    return Segment(
        label="RAW_TEXT",
        spans=[Span(0, len(text))],
        text=text,
    )


@pytest.mark.parametrize(
    ("text", "expected_text", "expected_spans"),
    TEST_DEFAULT_CONFIG,
    ids=[
        "replace_points_keywords_fr",
        "replace_large_parentheses",
        "replace_small_parentheses",
        "remove_multiple_newlines",
    ],
)
def test_default_cleaner(text, expected_text, expected_spans):
    # default config: this configuration allows to obtain the
    # same results as in the original version of the endlines function
    # Note: EDSCleaner replaces now all whitespaces by a single whitespace
    # this is different from the original implementation but it's more coherent
    cleaner = EDSCleaner(keep_endlines=False, handle_parentheses_eds=True, handle_points_eds=True)
    raw_segment = _get_raw_segment(text)
    clean_ann = cleaner.run([raw_segment])[0]
    assert clean_ann.text == expected_text
    assert clean_ann.spans == expected_spans


TEST_PARAMS_CONFIG = [
    (
        EDSCleaner(keep_endlines=True),
        ("Le patient\n\n\n  reviens. Nom patient probleme.\n\nTraitement :\n\n\n à dose curative dès cet appel."),
        ("Le patient reviens. Nom patient probleme..\nTraitement : à dose curative dès cet appel."),
        [
            Span(start=0, end=10),
            ModifiedSpan(
                length=1,
                replaced_spans=[Span(start=10, end=13), Span(start=13, end=15)],
            ),
            Span(start=15, end=45),
            ModifiedSpan(length=2, replaced_spans=[Span(start=45, end=47)]),
            Span(start=47, end=59),
            ModifiedSpan(
                length=1,
                replaced_spans=[Span(start=59, end=62), Span(start=62, end=63)],
            ),
            Span(start=63, end=93),
        ],
    ),
    (
        EDSCleaner(keep_endlines=False),
        ("Le patient\n\n\nreviens. Nom patient probleme.\n\nTraitement :\n\n\n à dose curative dès cet appel."),
        ("Le patient reviens. Nom patient probleme.. Traitement : à dose curative dès cet appel."),
        [
            Span(start=0, end=10),
            ModifiedSpan(
                length=1,
                replaced_spans=[Span(start=10, end=13)],
            ),
            Span(start=13, end=43),
            ModifiedSpan(length=2, replaced_spans=[Span(start=43, end=45)]),
            Span(start=45, end=57),
            ModifiedSpan(
                length=1,
                replaced_spans=[Span(start=57, end=60), Span(start=60, end=61)],
            ),
            Span(start=61, end=91),
        ],
    ),
    (
        EDSCleaner(handle_parentheses_eds=False),
        (
            "Traitement : (N.B. : absence de notion de la prescription d'une"
            " HBPM)\n\n\n à dose curative dès cet appel."
        ),
        ("Traitement : (N B. : absence de notion de la prescription d'une HBPM) à dose curative dès cet appel."),
        [
            Span(start=0, end=15),
            ModifiedSpan(length=1, replaced_spans=[Span(start=15, end=16)]),
            Span(start=16, end=69),
            ModifiedSpan(
                length=1,
                replaced_spans=[Span(start=69, end=72), Span(start=72, end=73)],
            ),
            Span(start=73, end=103),
        ],
    ),
    (  # whitespaces are removed now
        EDSCleaner(handle_points_eds=False),
        ("Nom patient : la Mme. Marie Du \n\npont, date le     .24 avril .    pour un probleme"),
        "Nom patient : la Mme. Marie Du pont, date le .24 avril . pour un probleme",
        [
            Span(start=0, end=30),
            ModifiedSpan(
                length=1,
                replaced_spans=[Span(start=30, end=31), Span(start=31, end=33)],
            ),
            Span(start=33, end=46),
            ModifiedSpan(length=1, replaced_spans=[Span(start=46, end=51)]),
            Span(start=51, end=62),
            ModifiedSpan(length=1, replaced_spans=[Span(start=62, end=66)]),
            Span(start=66, end=82),
        ],
    ),
]


@pytest.mark.parametrize(
    ("cleaner", "text", "expected_text", "expected_spans"),
    TEST_PARAMS_CONFIG,
    ids=[
        "keep_endlines_true",
        "keep_endlines_false",
        "no_change_parentheses_eds",
        "no_change_points_keywords_fr",
    ],
)
def test_cleaner_params(cleaner, text, expected_text, expected_spans):
    raw_segment = _get_raw_segment(text)
    clean_ann = cleaner.run([raw_segment])[0]
    assert clean_ann.text == expected_text
    assert clean_ann.spans == expected_spans


def test_prov():
    raw_segment = _get_raw_segment("Traitement :\n\n\n à dose curative dès cet appel.")

    cleaner = EDSCleaner()
    prov_tracer = ProvTracer()
    cleaner.set_prov_tracer(prov_tracer)
    clean_segments = cleaner.run([raw_segment])

    clean_segment = clean_segments[0]
    prov_1 = prov_tracer.get_prov(clean_segment.uid)
    assert prov_1.data_item == clean_segment
    assert prov_1.op_desc == cleaner.description
    assert prov_1.source_data_items == [raw_segment]
