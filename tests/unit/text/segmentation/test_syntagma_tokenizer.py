import pytest

from medkit.core import ProvTracer
from medkit.core.text import Segment, Span
from medkit.text.segmentation.syntagma_tokenizer import (
    _PATH_TO_DEFAULT_RULES,
    SyntagmaTokenizer,
)

_TEXT = (
    " Elle avait été améliorée par l'intervention pratiquée par le chirurgien mais"
    " présentait des petites douleurs résiduelles sur la face interne du genou"
    " droit"
)
TEST_CONFIG = [
    # basic
    (
        SyntagmaTokenizer(),
        _TEXT,
        [
            {
                "spans": [Span(start=1, end=72)],
                "text": ("Elle avait été améliorée par l'intervention pratiquée par le chirurgien"),
            },
            {
                "spans": [Span(start=73, end=156)],
                "text": ("mais présentait des petites douleurs résiduelles sur la face interne du genou droit"),
            },
        ],
    ),
    # multiline syntagmas (don't split on \r and \n)
    (
        SyntagmaTokenizer(separators=(r"\bmais\b",)),
        "Elle avait été\naméliorée mais présentait des douleurs",
        [
            {
                "spans": [Span(start=0, end=24)],
                "text": "Elle avait été\naméliorée",
            },
            {
                "spans": [Span(start=25, end=53)],
                "text": "mais présentait des douleurs",
            },
        ],
    ),
]


def _get_segment_from_text(text):
    return Segment(
        label="segment",
        spans=[Span(0, len(text))],
        text=text,
    )


@pytest.mark.parametrize(("syntagma_tokenizer", "text", "expected_syntagmas"), TEST_CONFIG)
def test_run(syntagma_tokenizer, text, expected_syntagmas):
    segment = _get_segment_from_text(text)

    syntagmas = syntagma_tokenizer.run([segment])
    assert len(syntagmas) == len(expected_syntagmas)
    for i, expected in enumerate(expected_syntagmas):
        assert syntagmas[i].label == syntagma_tokenizer._DEFAULT_LABEL
        assert syntagmas[i].spans == expected["spans"]
        assert syntagmas[i].text == expected["text"]


def test_prov():
    segment = _get_segment_from_text(_TEXT)

    tokenizer = SyntagmaTokenizer()
    prov_tracer = ProvTracer()
    tokenizer.set_prov_tracer(prov_tracer)
    syntagmas = tokenizer.run([segment])

    syntagma_1 = syntagmas[0]
    prov_1 = prov_tracer.get_prov(syntagma_1.uid)
    assert prov_1.data_item == syntagma_1
    assert prov_1.op_desc == tokenizer.description
    assert prov_1.source_data_items == [segment]

    syntagma_2 = syntagmas[1]
    prov_2 = prov_tracer.get_prov(syntagma_2.uid)
    assert prov_2.data_item == syntagma_2
    assert prov_2.op_desc == tokenizer.description
    assert prov_2.source_data_items == [segment]


def test_syntagma_def_file_encoding_error():
    with pytest.raises(UnicodeError):
        SyntagmaTokenizer.load_syntagma_definition(filepath=_PATH_TO_DEFAULT_RULES, encoding="utf-16")


def test_syntagma_def_file(tmp_path):
    filepath = tmp_path.joinpath("syntagma.yml")
    separators = (
        r"(?<=\. )[\w\d]+",  # Separateur: début de phrase (après un point-espace)
        r"(?<=\n)[\w\d]+",  # Separateur: début de phrase (après une nouvelle ligne)
        r"(?<=: )\w+",  # Separateur: debut de syntagme (après un :)
        r"(?<= )mais\s+(?=\w)",
        # Separateur: mais (précédé d'un espace et suivi d'un espace et mot
        r"(?<= )sans\s+(?=\w)",
        # Separateur: sans (précédé d'un espace et suivi d'un espace et mot
        r"(?<= )donc\s+(?=\w)",
        # Separateur: donc (précédé d'un espace et suivi d'un espace et mot
    )
    SyntagmaTokenizer.save_syntagma_definition(syntagma_seps=separators, filepath=filepath, encoding="utf-8")

    loaded_seps = SyntagmaTokenizer.load_syntagma_definition(filepath=filepath, encoding="utf-8")
    assert loaded_seps == separators


def test_tokenizer_behavior():
    syntagma_tokenizer = SyntagmaTokenizer()

    # use cases with '*qu'il/qu'elle/que'
    segment = _get_segment_from_text(
        "J'ai recommandé environ six à huit semaines de Medifast pour que le patient"
        " obtienne une perte de poids préopératoire de 10 %. "  # pour que
        "Elle a récemment eu des travaux de laboratoire et du cholestérol pour une"
        " demande d'assurance-vie et va m'envoyer ces résultats lorsqu'ils seront"
        " disponibles."  # lorsqu
        "Cet après-midi, elle m'a appelé parce que la fréquence cardiaque était entre"
        " 120 et 140. "  # parce que
        "Elle souffre également d'une maladie de la thyroïde dans le passé bien que"
        " cela ne soit pas clair. "  # bien que
        "La patiente déclare qu'elle est en surpoids depuis environ 35 ans et qu'elle a"
        " essayé plusieurs modalités de perte de poids dans le passé. "  # et qu
        "Cet après-midi, alors que je vois le patient, l'infirmière m'informe que le"
        " rythme a finalement été contrôlé avec de l'esmolol. "  # alors qu
        "Alternativement, je pourrais l'amener à l'hôpital pour quatre jours de"
        " drainage du LCR. "  # pour quatre - no match
    )

    syntagmas = syntagma_tokenizer.run([segment])
    assert any(syntagma.text.startswith("pour qu") for syntagma in syntagmas)
    assert any(syntagma.text.startswith("lorsqu") for syntagma in syntagmas)
    assert any(syntagma.text.startswith("parce que") for syntagma in syntagmas)
    assert any(syntagma.text.startswith("bien qu") for syntagma in syntagmas)
    assert any(syntagma.text.startswith("et qu") for syntagma in syntagmas)
    assert any(syntagma.text.startswith("alors qu") for syntagma in syntagmas)
    assert not all(syntagma.text.startswith("quatre jours") for syntagma in syntagmas)

    # use cases with '.'

    segment = _get_segment_from_text(
        "Repos à domicile. Absence de fièvre, frissons, hallucinations ou sueurs nocturnes"
    )

    syntagmas = syntagma_tokenizer.run([segment])
    assert syntagmas[0].text == "Repos à domicile."
    assert syntagmas[1].text == "Absence de fièvre, frissons, hallucinations ou sueurs nocturnes"
