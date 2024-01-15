import pytest

# must import pandas first, cf workaround description in pa_speaker_detector.py
pytest.importorskip(modname="pandas", reason="pandas (therefore pyannote) is not installed")
pytest.importorskip(modname="pyannote.audio", reason="pyannote.audio is not installed")

import math
from typing import NamedTuple

from medkit.audio.segmentation.pa_speaker_detector import (
    PASpeakerDetector,
)
from medkit.core import ProvTracer
from medkit.core.audio import MemoryAudioBuffer, Segment, Span
from tests.audio_utils import generate_sin_signal, signals_are_equal

_SAMPLE_RATE = 16000
_OUTPUT_LABEL = "turn"
_SPAN_OFFSET = 10.0


class _MockedPASegment(NamedTuple):
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class _MockedPAAnnotation:
    def __init__(self, segments, labels):
        self.segments = segments
        self.labels = labels

    def itertracks(self, yield_label):
        # the real Annotation.itertracks return segments, track names and speaker labels
        # but we don't use the track names
        track_names = [None for _ in self.segments]
        return zip(self.segments, track_names, self.labels)


# mock of SpeakerDiarization class used by PASpeakerDetector
class _MockedPipeline:
    def __init__(self):
        self.segmentation_batch_size = 1
        self.embedding_batch_size = 1

    def to(self, device):
        return self

    def apply(self, file, **kwargs):
        # return hard coded results (always split in half)
        duration = file["waveform"].shape[-1] / file["sample_rate"]
        segments = [
            _MockedPASegment(0.0, duration / 2),
            _MockedPASegment(duration / 2, duration),
        ]
        labels = ["speaker_0", "speaker_1"]
        return _MockedPAAnnotation(segments, labels)


@pytest.fixture(scope="module", autouse=True)
def _mocked_pipeline(module_mocker):
    module_mocker.patch(
        "medkit.audio.segmentation.pa_speaker_detector.SpeakerDiarization",
        _MockedPipeline,
    )
    module_mocker.patch(
        "medkit.audio.segmentation.pa_speaker_detector.Pipeline.from_pretrained",
        return_value=_MockedPipeline(),
    )


def _get_segment(duration):
    signal = generate_sin_signal(duration, _SAMPLE_RATE)
    audio = MemoryAudioBuffer(signal=signal, sample_rate=_SAMPLE_RATE)

    return Segment(
        label="RAW_AUDIO",
        span=Span(start=_SPAN_OFFSET, end=_SPAN_OFFSET + duration),
        audio=audio,
    )


def test_basic():
    """Basic behavior"""
    speaker_detector = PASpeakerDetector(
        model="mock-pipeline",
        output_label=_OUTPUT_LABEL,
        min_nb_speakers=2,
        max_nb_speakers=2,
    )

    duration = 2
    input_seg = _get_segment(duration)
    turn_segs = speaker_detector.run([input_seg])
    assert len(turn_segs) == 2

    # check 1st segment
    turn_seg_1 = turn_segs[0]
    assert turn_seg_1.label == _OUTPUT_LABEL
    span_1 = turn_seg_1.span
    speaker_change_time = duration / 2.0
    # span should go from start to 1st half
    assert span_1.start == input_seg.span.start
    assert span_1.end == input_seg.span.start + speaker_change_time
    # audio duration must be consistent with span length
    audio_1 = turn_seg_1.audio
    assert math.isclose(audio_1.duration, span_1.length)
    # audio must have same characteristics as input audio
    input_audio = input_seg.audio
    assert audio_1.nb_channels == input_audio.nb_channels
    assert audio_1.sample_rate == input_audio.sample_rate
    # segment audio must correspond to 1st half of input audio
    expected_signal = input_audio.trim_duration(end_time=speaker_change_time).read()
    assert signals_are_equal(audio_1.read(), expected_signal)

    # check 2nd segment
    turn_seg_2 = turn_segs[1]
    assert turn_seg_2.label == _OUTPUT_LABEL
    span_2 = turn_seg_2.span
    # span should go from 1st half to end
    assert span_2.start == input_seg.span.start + speaker_change_time
    assert span_2.end == input_seg.span.end
    audio_2 = turn_seg_2.audio
    assert math.isclose(audio_2.duration, span_2.length)
    # segment audio must correspond to 2d half of input audio
    expected_signal = input_audio.trim_duration(start_time=speaker_change_time).read()
    assert signals_are_equal(turn_seg_2.audio.read(), expected_signal)


def test_multiple():
    """Several segments passed as input"""
    speaker_detector = PASpeakerDetector(
        model="mock-pipeline",
        output_label=_OUTPUT_LABEL,
        min_nb_speakers=2,
        max_nb_speakers=2,
    )

    duration_1 = 2.0
    input_seg_1 = _get_segment(duration_1)
    duration_2 = 4.0
    input_seg_2 = _get_segment(duration_2)
    turn_segs = speaker_detector.run([input_seg_1, input_seg_2])
    assert len(turn_segs) == 4

    speaker_change_time_1 = duration_1 / 2.0
    turn_seg_1 = turn_segs[0]
    span_1 = turn_seg_1.span
    assert span_1.start == input_seg_1.span.start
    assert span_1.end == input_seg_1.span.start + speaker_change_time_1
    expected_signal = input_seg_1.audio.trim_duration(end_time=speaker_change_time_1).read()
    assert signals_are_equal(turn_seg_1.audio.read(), expected_signal)

    turn_seg_2 = turn_segs[1]
    span_2 = turn_seg_2.span
    assert span_2.start == input_seg_1.span.start + speaker_change_time_1
    assert span_2.end == input_seg_1.span.end
    expected_signal = input_seg_1.audio.trim_duration(start_time=speaker_change_time_1).read()
    assert signals_are_equal(turn_seg_2.audio.read(), expected_signal)

    speaker_change_time_2 = duration_2 / 2.0
    turn_seg_3 = turn_segs[2]
    span_3 = turn_seg_3.span
    assert span_3.start == input_seg_2.span.start
    assert span_3.end == input_seg_2.span.start + speaker_change_time_2
    expected_signal = input_seg_2.audio.trim_duration(end_time=speaker_change_time_2).read()
    assert signals_are_equal(turn_seg_3.audio.read(), expected_signal)

    turn_seg_4 = turn_segs[3]
    span_4 = turn_seg_4.span
    assert span_4.start == input_seg_2.span.start + speaker_change_time_2
    assert span_4.end == input_seg_2.span.end
    expected_signal = input_seg_2.audio.trim_duration(start_time=speaker_change_time_2).read()
    assert signals_are_equal(turn_seg_4.audio.read(), expected_signal)


def test_prov():
    """Generated provenance nodes"""
    speaker_detector = PASpeakerDetector(
        model="mock-pipeline",
        output_label=_OUTPUT_LABEL,
        min_nb_speakers=2,
        max_nb_speakers=2,
    )
    prov_tracer = ProvTracer()
    speaker_detector.set_prov_tracer(prov_tracer)

    input_seg_1 = _get_segment(duration=2.0)
    input_seg_2 = _get_segment(duration=4.0)
    turn_segs = speaker_detector.run([input_seg_1, input_seg_2])

    # data item uid and operation uid are correct
    turn_seg_1 = turn_segs[0]
    prov_1 = prov_tracer.get_prov(turn_seg_1.uid)
    assert prov_1.data_item == turn_seg_1
    assert prov_1.op_desc == speaker_detector.description

    # 1st and 2d voices segments have 1st input segment as source
    assert prov_1.source_data_items == [input_seg_1]

    turn_seg_2 = turn_segs[1]
    prov_2 = prov_tracer.get_prov(turn_seg_2.uid)
    assert prov_2.source_data_items == [input_seg_1]

    # 3d and 4th voices segments have 2d input segment as source
    turn_seg_3 = turn_segs[2]
    prov_3 = prov_tracer.get_prov(turn_seg_3.uid)
    assert prov_3.source_data_items == [input_seg_2]

    turn_seg_4 = turn_segs[3]
    prov_4 = prov_tracer.get_prov(turn_seg_4.uid)
    assert prov_4.source_data_items == [input_seg_2]
