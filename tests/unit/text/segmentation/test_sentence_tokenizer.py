import pytest

from medkit.core import ProvTracer
from medkit.core.text import Segment, Span
from medkit.text.segmentation import SentenceTokenizer

_TEXT = (
    "Sentence testing the dot. We are testing the carriage return\rthis is the"
    " newline\n Test interrogation ? Now, testing semicolon;Exclamation! Several"
    " punctuation characters?!..."
)


def _get_clean_text_segment(text):
    return Segment(
        label="clean_text",
        spans=[Span(0, len(text))],
        text=text,
    )


TEST_CONFIG = [
    # basic
    (
        SentenceTokenizer(),
        _TEXT,
        [
            ("Sentence testing the dot", [Span(start=0, end=24)]),
            ("We are testing the carriage return", [Span(start=26, end=60)]),
            ("this is the newline", [Span(start=61, end=80)]),
            ("Test interrogation ", [Span(start=82, end=101)]),
            ("Now, testing semicolon", [Span(start=103, end=125)]),
            ("Exclamation", [Span(start=126, end=137)]),
            ("Several punctuation characters", [Span(start=139, end=169)]),
        ],
    ),
    # keep punct chars in sentence
    (
        SentenceTokenizer(keep_punct=True),
        _TEXT,
        [
            ("Sentence testing the dot.", [Span(start=0, end=25)]),
            ("We are testing the carriage return", [Span(start=26, end=60)]),
            ("this is the newline", [Span(start=61, end=80)]),
            ("Test interrogation ?", [Span(start=82, end=102)]),
            ("Now, testing semicolon;", [Span(start=103, end=126)]),
            ("Exclamation!", [Span(start=126, end=138)]),
            ("Several punctuation characters?!...", [Span(start=139, end=174)]),
        ],
    ),
    # don't split on newlines
    (
        SentenceTokenizer(split_on_newlines=False),
        _TEXT,
        [
            ("Sentence testing the dot", [Span(start=0, end=24)]),
            (
                ("We are testing the carriage return\rthis is the newline\n Test interrogation "),
                [Span(start=26, end=101)],
            ),
            ("Now, testing semicolon", [Span(start=103, end=125)]),
            ("Exclamation", [Span(start=126, end=137)]),
            ("Several punctuation characters", [Span(start=139, end=169)]),
        ],
    ),
    # trailing sentence with no final punct char
    (
        SentenceTokenizer(),
        "This is a sentence. This is a trailing sentence with no punct",
        [
            ("This is a sentence", [Span(start=0, end=18)]),
            ("This is a trailing sentence with no punct", [Span(start=20, end=61)]),
        ],
    ),
    # sentence ending with both punct and newline, keep only punct
    (
        SentenceTokenizer(keep_punct=True),
        "This is a sentence ending with punct and newline.\nThis is another sentence.",
        [
            (
                "This is a sentence ending with punct and newline.",
                [Span(start=0, end=49)],
            ),
            ("This is another sentence.", [Span(start=50, end=75)]),
        ],
    ),
    # empty sentence shall not be returned
    (
        SentenceTokenizer(),
        "This is a sentence.    ",
        [
            (
                "This is a sentence",
                [Span(start=0, end=18)],
            ),
        ],
    ),
    # empty sentence shall not be returned (keep_punct:true)
    (
        SentenceTokenizer(keep_punct=True),
        "This is a sentence.  !  ",
        [
            (
                "This is a sentence.",
                [Span(start=0, end=19)],
            ),
        ],
    ),
]


@pytest.mark.parametrize(
    ("sentence_tokenizer", "text", "expected_sentences"),
    TEST_CONFIG,
    ids=[
        "default",
        "keep_punct",
        "multiline",
        "trailing",
        "punct_and_newline",
        "empty_sentence",
        "empty_sentence_with_punct",
    ],
)
def test_run(sentence_tokenizer, text, expected_sentences):
    clean_text_segment = _get_clean_text_segment(text)
    sentences = sentence_tokenizer.run([clean_text_segment])

    assert len(sentences) == len(expected_sentences)
    for i, (text, spans) in enumerate(expected_sentences):
        assert sentences[i].text == text
        assert sentences[i].spans == spans


def test_prov():
    clean_text_segment = _get_clean_text_segment("This is a sentence. This is another sentence.")

    tokenizer = SentenceTokenizer()
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
