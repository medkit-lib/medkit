import pytest

pytest.importorskip(modname="torch", reason="torch is not installed")
pytest.importorskip(modname="transformers", reason="transformers is not installed")

from medkit.core.text import ModifiedSpan, Segment, Span, span_utils
from medkit.text.ner import RegexpMatcher, RegexpMatcherRule
from medkit.text.translation.hf_translator import HFTranslator, _Aligner

_TEXT_FR = "Je souffre d'insuffisance cardiaque depuis 10 ans."
_TEXT_EN = "I've been suffering from heart failure for 10 years."
_TEXT_ALIGNMENTS = [
    ("I", "Je"),
    ("'ve been ", None),
    ("suffering", "souffre"),
    (" ", None),
    ("from", "d"),
    (" ", None),
    ("heart", "cardiaque"),
    (" ", None),
    ("failure", "insuffisance"),
    (" ", None),
    ("for", "depuis"),
    (" ", None),
    ("10", "10"),
    (" ", None),
    ("years", "ans"),
    (".", "."),
]


def _get_raw_text_segment(text):
    return Segment(
        label="raw_text",
        spans=[Span(0, len(text))],
        text=text,
    )


@pytest.fixture(scope="module")
def translator():
    return HFTranslator()


@pytest.fixture(scope="module")
def translator_en_to_fr():
    return HFTranslator(translation_model="Helsinki-NLP/opus-mt-en-fr")


def test_translator_fr_to_en(translator):
    segment = _get_raw_text_segment(_TEXT_FR)
    translated_segment = translator.run([segment])[0]
    assert translated_segment.text == _TEXT_EN


def test_translator_en_to_fr(translator_en_to_fr):
    segment = _get_raw_text_segment(_TEXT_EN)
    translated_segment = translator_en_to_fr.run([segment])[0]
    assert translated_segment.text == _TEXT_FR


def _get_text_alignments(original_text, translated_segment):
    """Return a list of tuple associating each word in the translated text
    to its corresponding word in the original text (if any).
    This is to visualize alignment in an easier way than with spans
    """
    text_alignments = []
    start = 0
    for span in translated_segment.spans:
        end = start + span.length
        translated_sub_text = translated_segment.text[start:end]
        if isinstance(span, ModifiedSpan):
            if span.replaced_spans:
                original_sub_text = " ".join(original_text[s.start : s.end] for s in span.replaced_spans)
            else:
                original_sub_text = None
        else:
            original_sub_text = original_text[span.start : span.end]
        text_alignments.append((translated_sub_text, original_sub_text))
        start = end
    return text_alignments


def test_translator_with_matcher(translator):
    """Make sure we are able to link an entity matched on translated text back to original
    text
    """
    rule = RegexpMatcherRule(
        regexp="heart failure",
        label="heart failure",
        id="id_heart_failure",
        version="1",
    )
    matcher = RegexpMatcher(rules=[rule])

    segment = _get_raw_text_segment(_TEXT_FR)
    translated_segment = translator.run([segment])[0]

    entities = matcher.run([translated_segment])
    assert len(entities) == 1
    entity = entities[0]
    assert entity.label == "heart failure"
    assert entity.text == "heart failure"

    text_alignments = _get_text_alignments(_TEXT_FR, translated_segment)
    assert text_alignments == _TEXT_ALIGNMENTS

    spans = span_utils.normalize_spans(entity.spans)
    spans = span_utils.clean_up_gaps_in_normalized_spans(spans, _TEXT_FR)
    matched_original_text = " ".join(segment.text[s.start : s.end] for s in spans)
    assert matched_original_text == "insuffisance cardiaque"


@pytest.mark.parametrize(
    ("input_size", "batch_size"),
    [(12, 1), (1, 12), (24, 12)],
)
def test_batch(input_size, batch_size):
    # generate different texts by changing number of years in ref sentence,
    segments = []
    for i in range(input_size):
        text = _TEXT_FR.replace("10", str(i + 1))
        segment = _get_raw_text_segment(text)
        segments.append(segment)

    # append a longer text at the end
    text = _TEXT_FR[-1] + ", " + _TEXT_FR[-1]
    segment = _get_raw_text_segment(text)
    segments.append(segment)

    # translate all texts with translator with specific batch size
    translator = HFTranslator(batch_size=batch_size)
    translated_segments = translator.run(segments)
    assert len(translated_segments) == len(segments)

    # check that result is identical to translating one by one
    for segment, translated_segment in zip(segments, translated_segments):
        expected_translated_segment = translator.run([segment])[0]
        assert translated_segment.text == expected_translated_segment.text
        assert translated_segment.spans == expected_translated_segment.spans


def test_ranges_sorting():
    """Alignment ranges are properly sorted even with model outputting non-monotonic
    token alignment
    """
    aligner = _Aligner(model="aneuraz/awesome-align-with-co")
    range_alignment = aligner.align(
        source_texts=["CHIRURGICAL ANTICEDENTS: surgery"],
        target_texts=["ANTÉCÉDENT CHIRURGICAUX: chirurgie"],
    )[0]
    assert all(sorted(r) == r for r in range_alignment.values())
