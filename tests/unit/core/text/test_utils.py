from medkit.core.text import ModifiedSpan, Span
from medkit.core.text.utils import (
    _replace_big_parentheses,
    _replace_small_parentheses,
    clean_multiple_whitespaces_in_sentence,
    clean_newline_character,
    clean_parentheses_eds,
    replace_multiple_newline_after_sentence,
    replace_newline_inside_sentence,
    replace_point_after_keywords,
    replace_point_before_keywords,
    replace_point_in_numbers,
    replace_point_in_uppercase,
)


def test_replace_point_after_keywords_strict():
    text = (
        "We want to remove the dots that indicate a person's title."
        "Normally, these dots are close to the title of interest. "
        "That is why we set the mode to strict to change Dr.Dupont correctly."
        "Dr   . maybe is not a person's title"
    )
    spans = [Span(0, 219)]
    text, spans = replace_point_after_keywords(text, spans, keywords=["Dr"], strict=True)
    assert text == (
        "We want to remove the dots that indicate a person's title."
        "Normally, these dots are close to the title of interest. "
        "That is why we set the mode to strict to change Dr Dupont correctly."
        "Dr   . maybe is not a person's title"
    )
    assert spans == [
        Span(start=0, end=165),
        ModifiedSpan(length=1, replaced_spans=[Span(start=165, end=166)]),
        Span(start=166, end=219),
    ]


def test_replace_point_after_keywords_non_strict():
    text = "When Dr. Dupont wrote this, a dot appeared by mistake after the keyword   . we should remove it."
    spans = [Span(0, 96)]
    text, spans = replace_point_after_keywords(text, spans, keywords=["keyword"], strict=False)
    assert text == "When Dr. Dupont wrote this, a dot appeared by mistake after the keyword  we should remove it."
    assert spans == [
        Span(start=0, end=71),
        ModifiedSpan(length=1, replaced_spans=[Span(start=71, end=75)]),
        Span(start=75, end=96),
    ]


def test_replace_multiple_endlines():
    text = "This is\n\ntest\n\n\n  Another\nphrase."
    spans = [Span(0, 33)]
    text, spans = replace_multiple_newline_after_sentence(text, spans)
    assert text == "This is\n\ntest\n  Another\nphrase."
    assert spans == [
        Span(start=0, end=13),
        ModifiedSpan(length=1, replaced_spans=[Span(start=13, end=16)]),
        Span(start=16, end=33),
    ]


def test_replace_newline_inside_sentence():
    text = "This is\n\n\ta test\n\nAnother\nphrase."
    spans = [Span(0, 33)]
    text, spans = replace_newline_inside_sentence(text, spans)
    assert text == "This is \ta test\n\nAnother phrase."
    assert spans == [
        Span(start=0, end=7),
        ModifiedSpan(length=1, replaced_spans=[Span(start=7, end=9)]),
        Span(start=9, end=25),
        ModifiedSpan(length=1, replaced_spans=[Span(start=25, end=26)]),
        Span(start=26, end=33),
    ]


def test_clean_newline_character_keep_endline():
    text = "This is\n\n\ttest\nAnother\nphrase\n\n?"
    spans = [Span(0, 32)]
    text, spans = clean_newline_character(text, spans, keep_endlines=True)
    assert text == "This is \ttest.\nAnother phrase ?"
    assert spans == [
        Span(start=0, end=7),
        ModifiedSpan(length=1, replaced_spans=[Span(start=7, end=9)]),
        Span(start=9, end=14),
        ModifiedSpan(length=2, replaced_spans=[Span(start=14, end=15)]),
        Span(start=15, end=22),
        ModifiedSpan(length=1, replaced_spans=[Span(start=22, end=23)]),
        Span(start=23, end=29),
        ModifiedSpan(length=1, replaced_spans=[Span(start=29, end=31)]),
        Span(start=31, end=32),
    ]


def test_clean_newline_character_non_keep_endline():
    text = "This is\n\n\ttest\nAnother\nphrase\n\n?"
    spans = [Span(0, 32)]
    text, spans = clean_newline_character(text, spans, keep_endlines=False)
    assert text == "This is \ttest. Another phrase ?"
    assert spans == [
        Span(start=0, end=7),
        ModifiedSpan(length=1, replaced_spans=[Span(start=7, end=9)]),
        Span(start=9, end=14),
        ModifiedSpan(length=2, replaced_spans=[Span(start=14, end=15)]),
        Span(start=15, end=22),
        ModifiedSpan(length=1, replaced_spans=[Span(start=22, end=23)]),
        Span(start=23, end=29),
        ModifiedSpan(length=1, replaced_spans=[Span(start=29, end=31)]),
        Span(start=31, end=32),
    ]


def test_big_parentheses():
    text = "All tests(biological, metabolic and imaging)    are negative."
    spans = [Span(0, 61)]
    text, spans = _replace_big_parentheses(text, spans)
    assert text == "All tests are negative ; biological, metabolic and imaging."
    assert spans == [
        Span(start=0, end=9),
        ModifiedSpan(length=1, replaced_spans=[]),  # insert space
        Span(start=48, end=60),  # text after parentheses
        ModifiedSpan(length=3, replaced_spans=[]),  # insert ' ; '
        Span(start=10, end=43),  # text inside parentheses
        ModifiedSpan(length=1, replaced_spans=[]),  # insert '.'
    ]

    # multiple matches
    text = (
        "All tests(biological, metabolic and imaging)    are negative."
        "Blood tests   (done in public laboratories, private laboratories and in"
        " pharmacies)are positive."
    )
    spans = [Span(0, 157)]
    text, spans = _replace_big_parentheses(text, spans)

    assert text == (
        "All tests are negative ; biological, metabolic and imaging."
        "Blood tests are positive ; done in public laboratories, private laboratories"
        " and in pharmacies."
    )

    assert spans == [
        Span(start=0, end=9),  # the first match
        ModifiedSpan(length=1, replaced_spans=[]),
        Span(start=48, end=60),
        ModifiedSpan(length=3, replaced_spans=[]),
        Span(start=10, end=43),
        ModifiedSpan(length=1, replaced_spans=[]),
        Span(start=61, end=72),  # the second match
        ModifiedSpan(length=1, replaced_spans=[]),
        Span(start=144, end=156),
        ModifiedSpan(length=3, replaced_spans=[]),
        Span(start=76, end=143),
        ModifiedSpan(length=1, replaced_spans=[]),
    ]


def test_small_parentheses():
    text = "All tests (biological, metabolic)   are (negative)"
    spans = [Span(start=0, end=50)]
    text, spans = _replace_small_parentheses(text, spans)

    assert text == "All tests ,biological, metabolic,   are ,negative,"
    assert spans == [
        Span(start=0, end=10),
        ModifiedSpan(length=1, replaced_spans=[Span(start=10, end=11)]),
        Span(start=11, end=32),
        ModifiedSpan(length=1, replaced_spans=[Span(start=32, end=33)]),
        Span(start=33, end=40),
        ModifiedSpan(length=1, replaced_spans=[Span(start=40, end=41)]),
        Span(start=41, end=49),
        ModifiedSpan(length=1, replaced_spans=[Span(start=49, end=50)]),
    ]


def test_clean_multiple_whitespaces():
    # Now this function replaces all whitespaces, without restriction
    text = "All tests     biological,    and metabolic are negative THIS     maybe   NOT"
    spans = [Span(start=0, end=76)]

    text, spans = clean_multiple_whitespaces_in_sentence(text, spans)
    assert text == "All tests biological, and metabolic are negative THIS maybe NOT"
    assert spans == [
        Span(start=0, end=9),
        ModifiedSpan(length=1, replaced_spans=[Span(start=9, end=14)]),
        Span(start=14, end=25),
        ModifiedSpan(length=1, replaced_spans=[Span(start=25, end=29)]),
        Span(start=29, end=60),
        ModifiedSpan(length=1, replaced_spans=[Span(start=60, end=65)]),
        Span(start=65, end=70),
        ModifiedSpan(length=1, replaced_spans=[Span(start=70, end=73)]),
        Span(start=73, end=76),
    ]


def test_clean_parentheses_eds():
    text = (
        "Le test PCR est  (-), pas de nouvelles."
        "L'examen d'aujourd'hui est (+)."
        "Les bilans réalisés (biologique, métabolique en particulier à la recherche"
        " de GAMT et X fragile) sont revenus négatifs. Le patient a un traitement (debuté le 3/02)."
    )
    spans = [Span(start=0, end=234)]

    text, spans = clean_parentheses_eds(text, spans)

    assert text == (
        "Le test PCR est   negatif , pas de nouvelles."
        "L'examen d'aujourd'hui est  positif ."
        "Les bilans réalisés sont revenus négatifs ; biologique, métabolique en"
        " particulier à la recherche de GAMT et X fragile. Le patient a un traitement ,debuté le 3/02,."
    )
    assert spans == [
        Span(start=0, end=17),
        ModifiedSpan(length=9, replaced_spans=[Span(start=17, end=20)]),  # replace by ' negatif '
        Span(start=20, end=66),
        ModifiedSpan(length=9, replaced_spans=[Span(start=66, end=69)]),  # replace by ' positif '
        Span(start=69, end=89),  # start big parentheses
        ModifiedSpan(length=1, replaced_spans=[]),  # insert space
        Span(start=167, end=188),  # end of phrase
        ModifiedSpan(length=3, replaced_spans=[]),  # insert ' ; '
        Span(start=91, end=165),  # text inside big parentheses
        ModifiedSpan(length=1, replaced_spans=[]),  # insert '.'
        Span(start=189, end=217),  # start small parentheses
        ModifiedSpan(length=1, replaced_spans=[Span(start=217, end=218)]),  # insert ','
        Span(start=218, end=232),  # text inside small parentheses
        ModifiedSpan(length=1, replaced_spans=[Span(start=232, end=233)]),  # insert ','
        Span(start=233, end=234),
    ]


def test_no_text_after_parenthesis():
    text = "Les bilans réalisés (biologique, métabolique en particulier à la recherche de GAMT et X fragile)."
    spans = [Span(start=0, end=97)]

    text, spans = clean_parentheses_eds(text, spans)
    assert text == "Les bilans réalisés ; biologique, métabolique en particulier à la recherche de GAMT et X fragile."
    assert spans == [
        Span(start=0, end=19),  # text before parentheses
        ModifiedSpan(length=3, replaced_spans=[]),  # insert ' ; '
        Span(start=21, end=95),  # text inside big parentheses
        ModifiedSpan(length=1, replaced_spans=[]),  # insert '.' (no text after parenthseses)
    ]


def test_point_between_characters():
    text = "We found a point ING.DRG between."
    spans = [Span(0, 33)]
    text, spans = replace_point_in_uppercase(text, spans)
    assert text == "We found a point ING DRG between."
    assert spans == [
        Span(start=0, end=20),
        ModifiedSpan(length=1, replaced_spans=[Span(start=20, end=21)]),
        Span(start=21, end=33),
    ]

    text = "We found a point 55.66 between."
    spans = [Span(0, 31)]
    text, spans = replace_point_in_numbers(text, spans)
    assert text == "We found a point 55,66 between."
    assert spans == [
        Span(start=0, end=19),
        ModifiedSpan(length=1, replaced_spans=[Span(start=19, end=20)]),
        Span(start=20, end=31),
    ]

    text = "We found a point before the keyword: .  pour  and before .avec, we should remove them."
    spans = [Span(0, 86)]
    text, spans = replace_point_before_keywords(text, spans, keywords=["pour", "avec"])
    assert text == "We found a point before the keyword: pour  and before avec, we should remove them."
    assert spans == [
        Span(start=0, end=36),
        ModifiedSpan(length=1, replaced_spans=[Span(start=36, end=40)]),
        Span(start=40, end=56),
        ModifiedSpan(length=1, replaced_spans=[Span(start=56, end=58)]),
        Span(start=58, end=86),
    ]
