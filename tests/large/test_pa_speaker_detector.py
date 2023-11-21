import pytest

# must import pandas first, cf workaround description in pa_speaker_detector.py
pytest.importorskip(
    modname="pandas", reason="pandas (therefore pyannote) is not installed"
)
pytest.importorskip(modname="pyannote.audio", reason="pyannote.audio is not installed")

from pathlib import Path  # noqa: E402

from medkit.core.audio import FileAudioBuffer, Segment, Span  # noqa: E402
from medkit.audio.segmentation.pa_speaker_detector import (
    PASpeakerDetector,
)  # noqa: E402

# model weights provided by pyannote and speechbrain on huggingface hub
_TEST_DATA_DIR = Path(__file__).parent.parent / "large_data"
_SEGMENTATION_MODEL = _TEST_DATA_DIR / "pyannote" / "segmentation" / "pytorch_model.bin"
_EMBEDDING_MODEL = _TEST_DATA_DIR / "speechbrain" / "spkrec-ecapa-voxceleb"
# simple params that will work with our test file
_CLUSTERING = "AgglomerativeClustering"
_PIPELINE_PARAMS = {
    "segmentation": {
        "min_duration_off": 0.0,
    },
    "clustering": {
        "method": "centroid",
        "min_cluster_size": 12,
        "threshold": 0.7,
    },
}


_AUDIO = FileAudioBuffer("tests/data/audio/dialog_long.ogg")
_SPEAKER_CHANGE_TIME = 4.0
_MARGIN = 1.0


def _get_segment():
    return Segment(
        label="RAW_AUDIO",
        span=Span(start=0.0, end=_AUDIO.duration),
        audio=_AUDIO,
    )


@pytest.mark.xfail
def test_basic():
    speaker_detector = PASpeakerDetector(
        segmentation_model=_SEGMENTATION_MODEL,
        embedding_model=_EMBEDDING_MODEL,
        clustering=_CLUSTERING,
        output_label="turn",
        pipeline_params=_PIPELINE_PARAMS,
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
    assert (
        _SPEAKER_CHANGE_TIME - _MARGIN <= span_1.end <= _SPEAKER_CHANGE_TIME + _MARGIN
    )

    # span of 2nd segment should be from speaker change time to end
    turn_2 = turns[1]
    span_2 = turn_2.span
    assert (
        _SPEAKER_CHANGE_TIME - _MARGIN <= span_2.start <= _SPEAKER_CHANGE_TIME + _MARGIN
    )
    assert _AUDIO.duration - _MARGIN <= span_2.end <= _AUDIO.duration

    # segments must have different speakers
    speaker_1 = turn_1.attrs.get(label="speaker")[0].value
    speaker_2 = turn_2.attrs.get(label="speaker")[0].value
    assert speaker_1 != speaker_2
