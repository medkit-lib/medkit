import pytest

# must import pandas first, cf workaround description in pa_speaker_detector.py
pytest.importorskip(modname="pandas", reason="pandas (therefore pyannote) is not installed")
pytest.importorskip(modname="pyannote.audio", reason="pyannote.audio is not installed")

from pathlib import Path

from medkit.audio.segmentation.pa_speaker_detector import (
    PASpeakerDetector,
)
from medkit.core.audio import FileAudioBuffer, Segment, Span

_PIPELINE_MODEL = Path(__file__).parent / "diar_pipeline_config.yaml"
_AUDIO = FileAudioBuffer("tests/data/audio/dialog_long.ogg")
_SPEAKER_CHANGE_TIME = 4.0
_MARGIN = 1.0


def _get_segment():
    return Segment(
        label="RAW_AUDIO",
        span=Span(start=0.0, end=_AUDIO.duration),
        audio=_AUDIO,
    )


def test_basic():
    speaker_detector = PASpeakerDetector(
        model=_PIPELINE_MODEL,
        output_label="turn",
        min_nb_speakers=2,
        max_nb_speakers=2,
    )
    segment = _get_segment()
    turns = speaker_detector.run([segment])
    assert len(turns) == 2

    # span of 1st segment should be from beginning to speaker change time
    turn_1 = turns[0]
    span_1 = turn_1.span
    assert 0.0 <= span_1.start <= _MARGIN
    assert _SPEAKER_CHANGE_TIME - _MARGIN <= span_1.end <= _SPEAKER_CHANGE_TIME + _MARGIN

    # span of 2nd segment should be from speaker change time to end
    turn_2 = turns[1]
    span_2 = turn_2.span
    assert _SPEAKER_CHANGE_TIME - _MARGIN <= span_2.start <= _SPEAKER_CHANGE_TIME + _MARGIN
    assert _AUDIO.duration - _MARGIN <= span_2.end <= _AUDIO.duration

    # segments must have different speakers
    speaker_1 = turn_1.attrs.get(label="speaker")[0].value
    speaker_2 = turn_2.attrs.get(label="speaker")[0].value
    assert speaker_1 != speaker_2
