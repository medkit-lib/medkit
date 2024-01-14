import pytest

# must import pandas first, cf workaround description in metrics/diarization.py
pytest.importorskip(modname="pandas", reason="pandas (therefore pyannote) is not installed")
pytest.importorskip(modname="pyannote.audio", reason="pyannote.audio is not installed")

from medkit.audio.metrics.diarization import (
    DiarizationEvaluator,
    DiarizationEvaluatorResult,
)
from medkit.core import Attribute
from medkit.core.audio import AudioDocument, MemoryAudioBuffer, Segment, Span
from tests.audio_utils import generate_silence

# dummy 6.0 seconds audio signal
_FULL_AUDIO = MemoryAudioBuffer(
    signal=generate_silence(duration=6.0, sample_rate=4000),
    sample_rate=4000,
)


def _get_doc():
    # reference document: 2 speech turns with 2 speakers
    doc = AudioDocument(audio=_FULL_AUDIO)

    turn_seg_1 = Segment(
        label="turn",
        audio=_FULL_AUDIO.trim_duration(start_time=0.0, end_time=4.0),
        span=Span(start=0.0, end=4.0),
        attrs=[Attribute(label="speaker", value="Alice")],
    )
    doc.anns.add(turn_seg_1)

    turn_seg_2 = Segment(
        label="turn",
        audio=_FULL_AUDIO.trim_duration(start_time=5.0, end_time=6.0),
        span=Span(5.0, 6.0),
        attrs=[Attribute(label="speaker", value="Bob")],
    )
    doc.anns.add(turn_seg_2)

    return doc


_TEST_DATA = {
    # identical to reference
    "identical": (
        [
            {"start": 0.0, "end": 4.0, "speaker": "Alice"},
            {"start": 5.0, "end": 6.0, "speaker": "Bob"},
        ],
        {"collar": 0.0},
        DiarizationEvaluatorResult(
            der=0.0,
            confusion=0.0,
            false_alarm=0.0,
            missed_detection=0.0,
            total_speech=5.0,
            support=6.0,
        ),
    ),
    # speaker confusion
    "confusion": (
        [
            {"start": 0.0, "end": 4.0, "speaker": "Alice"},
            {"start": 5.0, "end": 6.0, "speaker": "Alice"},  # should be Bob
        ],
        {"collar": 0.0},
        DiarizationEvaluatorResult(
            der=0.2,
            confusion=0.2,
            false_alarm=0.0,
            missed_detection=0.0,
            total_speech=5.0,
            support=6.0,
        ),
    ),
    # missed detection
    "missed_detection": (
        [
            {"start": 0.0, "end": 4.0, "speaker": "Alice"},
            # missing 2d turn
        ],
        {"collar": 0.0},
        DiarizationEvaluatorResult(
            der=0.2,
            confusion=0.0,
            false_alarm=0.0,
            missed_detection=0.2,
            total_speech=5.0,
            support=6.0,
        ),
    ),
    # false alarm
    "false_alarm": (
        [
            {"start": 0.0, "end": 4.5, "speaker": "Alice"},  # should end at 4.0
            {"start": 5.0, "end": 6.0, "speaker": "Bob"},
        ],
        {"collar": 0.0},
        DiarizationEvaluatorResult(
            der=0.1,
            confusion=0.0,
            false_alarm=0.1,
            missed_detection=0.0,
            total_speech=5.0,
            support=6.0,
        ),
    ),
    # nearly identical to reference, error margin covered by collar
    "collar": (
        [
            # +/-0.1 error
            {"start": 0.1, "end": 3.9, "speaker": "Alice"},
            {"start": 5.1, "end": 5.9, "speaker": "Bob"},
        ],
        {"collar": 0.5},
        DiarizationEvaluatorResult(
            der=0.0,
            confusion=0.0,
            false_alarm=0.0,
            missed_detection=0.0,
            total_speech=4.0,  # collar is deducted from reference speech
            support=6.0,
        ),
    ),
}


@pytest.mark.parametrize(("turns_data", "params", "expected_result"), _TEST_DATA.values(), ids=_TEST_DATA.keys())
def test_diarization_evaluator(turns_data, params, expected_result):
    pred_segs = [
        Segment(
            label="turn",
            audio=_FULL_AUDIO.trim_duration(t["start"], t["end"]),
            span=Span(t["start"], t["end"]),
            attrs=[Attribute(label="speaker", value=t["speaker"])],
        )
        for t in turns_data
    ]

    doc = _get_doc()
    evaluator = DiarizationEvaluator(turn_label="turn", speaker_label="speaker", **params)
    result = evaluator.compute([doc], [pred_segs])
    assert result == expected_result
