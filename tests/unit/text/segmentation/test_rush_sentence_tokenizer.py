import pytest

from medkit.core import ProvTracer
from medkit.core.text import ModifiedSpan, Segment, Span

pytest.importorskip(modname="PyRuSH", reason="PyRuSH is not installed")

from medkit.text.segmentation.rush_sentence_tokenizer import (
    RushSentenceTokenizer,
)

TEST_CONFIG = [
    (
        (
            "Sentence testing the dot. Newlines\ndo not split. Test"
            " interrogation? Semicolon; and exclamation marks! do not split"
        ),
        RushSentenceTokenizer(),
        [
            (
                "Sentence testing the dot.",
                [Span(start=0, end=25)],
            ),
            (
                "Newlines\ndo not split.",
                [Span(start=26, end=48)],
            ),
            (
                "Test interrogation?",
                [Span(start=49, end=68)],
            ),
            (
                "Semicolon; and exclamation marks! do not split",
                [Span(start=69, end=115)],
            ),
        ],
    ),
    (
        "Newlines\ncan be\nreplaced.",
        RushSentenceTokenizer(keep_newlines=False),
        [
            (
                "Newlines can be replaced.",
                [
                    Span(start=0, end=8),
                    ModifiedSpan(length=1, replaced_spans=[Span(start=8, end=9)]),
                    Span(start=9, end=15),
                    ModifiedSpan(length=1, replaced_spans=[Span(start=15, end=16)]),
                    Span(start=16, end=25),
                ],
            ),
        ],
    ),
]


def _get_clean_text_segment(text):
    return Segment(
        label="clean_text",
        spans=[Span(0, len(text))],
        text=text,
    )


@pytest.mark.parametrize(
    ("text", "sentence_tokenizer", "expected_sentences"),
    TEST_CONFIG,
    ids=["default", "keep_newlines"],
)
def test_default_rules(text, sentence_tokenizer, expected_sentences):
    clean_text_segment = _get_clean_text_segment(text)

    sentences = sentence_tokenizer.run([clean_text_segment])
    assert len(sentences) == len(expected_sentences)
    for i, (text, spans) in enumerate(expected_sentences):
        assert sentences[i].text == text
        assert sentences[i].spans == spans


def test_prov():
    clean_text_segment = _get_clean_text_segment("This is a sentence. This is another sentence. ")

    tokenizer = RushSentenceTokenizer()
    prov_tracer = ProvTracer()
    tokenizer.set_prov_tracer(prov_tracer)
    sentences = tokenizer.run([clean_text_segment])

    sentence_1 = sentences[0]
    prov_1 = prov_tracer.get_prov(sentence_1.uid)
    assert prov_1.data_item == sentence_1
    assert prov_1.op_desc == tokenizer.description
    assert prov_1.source_data_items == [clean_text_segment]

    sentence_2 = sentences[1]
    prov_2 = prov_tracer.get_prov(sentence_2.uid)
    assert prov_2.data_item == sentence_2
    assert prov_2.op_desc == tokenizer.description
    assert prov_2.source_data_items == [clean_text_segment]
