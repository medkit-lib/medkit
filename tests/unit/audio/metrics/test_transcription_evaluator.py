import pytest

pytest.importorskip(modname="speechbrain", reason="speechbrain is not installed")

from medkit.audio.metrics.transcription import (
    TranscriptionEvaluator,
    TranscriptionEvaluatorResult,
)
from medkit.core import Attribute
from medkit.core.audio import AudioDocument, MemoryAudioBuffer, Segment, Span
from tests.audio_utils import generate_silence

# dummy 4.0 seconds audio signal
_FULL_AUDIO = MemoryAudioBuffer(
    signal=generate_silence(duration=4.0, sample_rate=4000),
    sample_rate=4000,
)


def _get_doc():
    # reference document: 2 transcribed speech segments
    doc = AudioDocument(audio=_FULL_AUDIO)

    turn_seg_1 = Segment(
        label="speech",
        audio=_FULL_AUDIO.trim_duration(start_time=0.0, end_time=2.0),
        span=Span(start=0.0, end=2.0),
        attrs=[Attribute(label="transcription", value="Bonjour ça va bien ?")],
    )
    doc.anns.add(turn_seg_1)

    turn_seg_2 = Segment(
        label="speech",
        audio=_FULL_AUDIO.trim_duration(start_time=2.0, end_time=4.0),
        span=Span(2.0, 4.0),
        attrs=[Attribute(label="transcription", value="Ça va et vous ?")],
    )
    doc.anns.add(turn_seg_2)

    return doc


_TEST_DATA = {
    # identical to reference
    "identical": (
        [
            {"start": 0.0, "end": 2.0, "transcription": "Bonjour ça va bien ?"},
            {"start": 3.0, "end": 4.0, "transcription": "Ça va et vous ?"},
        ],
        {},
        TranscriptionEvaluatorResult(
            wer=0.0,
            word_insertions=0.0,
            word_deletions=0.0,
            word_substitutions=0.0,
            word_support=8,
            cer=0.0,
            char_insertions=0.0,
            char_deletions=0.0,
            char_substitutions=0.0,
            char_support=32,
        ),
    ),
    # a few transcription errors
    "errors": (
        [
            {"start": 0.0, "end": 2.0, "transcription": "Bonjour ça va ?"},
            {"start": 3.0, "end": 4.0, "transcription": "Bien et vous ?"},
        ],
        {},
        TranscriptionEvaluatorResult(
            wer=0.25,
            word_insertions=0.0,
            word_deletions=0.25,
            word_substitutions=0.0,
            word_support=8,
            cer=0.1875,
            char_insertions=0.0,
            char_deletions=0.1875,
            char_substitutions=0.0,
            char_support=32,
        ),
    ),
    # identical to extra whitespaces between words
    "whitespace_insensitive": (
        [
            {"start": 0.0, "end": 2.0, "transcription": " Bonjour    ça va bien ?"},
            {"start": 2.0, "end": 4.0, "transcription": "Ça va   et vous ?  "},
        ],
        {},
        TranscriptionEvaluatorResult(
            wer=0.0,
            word_insertions=0.0,
            word_deletions=0.0,
            word_substitutions=0.0,
            word_support=8,
            cer=0.0,
            char_insertions=0.0,
            char_deletions=0.0,
            char_substitutions=0.0,
            char_support=32,
        ),
    ),
    # insensitivity to case and punctuation
    "case_and_punct_insensitive": (
        [
            {"start": 0.0, "end": 2.0, "transcription": "BONJOUR ÇA VA BIEN"},
            {"start": 2.0, "end": 4.0, "transcription": "ÇA VA ET VOUS"},
        ],
        {
            "case_sensitive": False,
            "remove_punctuation": True,
        },
        TranscriptionEvaluatorResult(
            wer=0.0,
            word_insertions=0.0,
            word_deletions=0.0,
            word_substitutions=0.0,
            word_support=8,
            cer=0.0,
            char_insertions=0.0,
            char_deletions=0.0,
            char_substitutions=0.0,
            char_support=32,
        ),
    ),
    # insensitivity to special chars
    "replace_unicode": (
        [
            {"start": 0.0, "end": 2.0, "transcription": "Bonjour ca va bien ?"},
            {"start": 2.0, "end": 4.0, "transcription": "Ca va et vous ?"},
        ],
        {
            "replace_unicode": True,
        },
        TranscriptionEvaluatorResult(
            wer=0.0,
            word_insertions=0.0,
            word_deletions=0.0,
            word_substitutions=0.0,
            word_support=8,
            cer=0.0,
            char_insertions=0.0,
            char_deletions=0.0,
            char_substitutions=0.0,
            char_support=32,
        ),
    ),
    # don't assume segments are sorted by time
    "unsorted_segs": (
        [
            {"start": 2.0, "end": 4.0, "transcription": "Ça va et vous ?"},
            {"start": 0.0, "end": 2.0, "transcription": "Bonjour ça va bien ?"},
        ],
        {},
        TranscriptionEvaluatorResult(
            wer=0.0,
            word_insertions=0.0,
            word_deletions=0.0,
            word_substitutions=0.0,
            word_support=8,
            cer=0.0,
            char_insertions=0.0,
            char_deletions=0.0,
            char_substitutions=0.0,
            char_support=32,
        ),
    ),
}


@pytest.mark.parametrize(("speech_data", "params", "expected_result"), _TEST_DATA.values(), ids=_TEST_DATA.keys())
def test_transcription_evaluator(speech_data, params, expected_result):
    pred_segs = [
        Segment(
            label="turn",
            audio=_FULL_AUDIO.trim_duration(s["start"], s["end"]),
            span=Span(s["start"], s["end"]),
            attrs=[Attribute(label="transcription", value=s["transcription"])],
        )
        for s in speech_data
    ]

    doc = _get_doc()
    evaluator = TranscriptionEvaluator(speech_label="speech", transcription_label="transcription", **params)
    result = evaluator.compute([doc], [pred_segs])
    assert result == expected_result
