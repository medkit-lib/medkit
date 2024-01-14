import pytest

pytest.importorskip(modname="webrtcvad", reason="webrtcvad is not installed")

import math
from pathlib import Path

import numpy as np
import soundfile as sf

from medkit.audio.segmentation.webrtc_voice_detector import (
    WebRTCVoiceDetector,
)
from medkit.core import ProvTracer
from medkit.core.audio import MemoryAudioBuffer, Segment, Span
from tests.audio_utils import generate_silence, signals_are_equal

_PATH_TO_VOICE_FILE = Path("tests/data/audio/voice.ogg")
_OUTPUT_LABEL = "voice"
_SPAN_OFFSET = 10.0
_DETECTION_MARGIN = 0.5


def _get_segment(voice_signal, sample_rate, silence_duration):
    nb_channels = voice_signal.shape[0]
    silence = generate_silence(silence_duration, sample_rate, nb_channels)
    signal = np.concatenate((silence, voice_signal, silence, voice_signal, silence), axis=1)
    audio = MemoryAudioBuffer(signal, sample_rate)
    return Segment(label="raw", span=Span(_SPAN_OFFSET, _SPAN_OFFSET + audio.duration), audio=audio)


def _check_voice_segment(voice_seg, original_seg, expected_start, expected_end):
    assert voice_seg.label == _OUTPUT_LABEL
    # span should be within original span
    assert original_seg.span.start <= voice_seg.span.start <= original_seg.span.end
    assert original_seg.span.start <= voice_seg.span.end <= original_seg.span.end
    # nb_channels and sample_rate should be identical to original audio
    voice_audio = voice_seg.audio
    original_audio = original_seg.audio
    assert voice_audio.nb_channels == original_audio.nb_channels
    assert voice_audio.sample_rate == original_audio.sample_rate
    # duration shouldn't be longer than original
    assert voice_audio.duration <= original_audio.duration
    # audio duration should be consistent with span length
    assert math.isclose(voice_audio.duration, voice_seg.span.length)

    # detected spans should be close enough to expected
    assert expected_start - _DETECTION_MARGIN < voice_seg.span.start < expected_start + _DETECTION_MARGIN
    assert expected_end - _DETECTION_MARGIN < voice_seg.span.end < expected_end + _DETECTION_MARGIN

    # voice signal should correspond to original signal span
    expected_signal = original_audio.trim_duration(
        start_time=voice_seg.span.start - _SPAN_OFFSET,
        end_time=voice_seg.span.end - _SPAN_OFFSET,
    ).read()
    assert signals_are_equal(voice_audio.read(), expected_signal)


def test_basic():
    """Basic behavior"""
    # use file containing voice signal
    voice_signal, sample_rate = sf.read(_PATH_TO_VOICE_FILE, always_2d=True, dtype=np.float32)
    voice_signal = voice_signal.T
    voice_duration = voice_signal.shape[1] / sample_rate
    # interleave it with silences (twice per segment)
    silence_duration_1 = 3.0
    seg_1 = _get_segment(voice_signal, sample_rate, silence_duration_1)
    silence_duration_2 = 5.0
    seg_2 = _get_segment(voice_signal, sample_rate, silence_duration_2)
    segs = [seg_1, seg_2]

    detector = WebRTCVoiceDetector(output_label=_OUTPUT_LABEL)
    voice_segs = detector.run(segs)
    assert len(voice_segs) == 4

    # voice segments detected in 1st input segment
    voice_seg_1 = voice_segs[0]
    expected_start_1 = _SPAN_OFFSET + silence_duration_1
    expected_end_1 = expected_start_1 + voice_duration
    _check_voice_segment(voice_seg_1, seg_1, expected_start_1, expected_end_1)

    voice_seg_2 = voice_segs[1]
    expected_start_2 = expected_end_1 + silence_duration_1
    expected_end_2 = expected_start_2 + voice_duration
    _check_voice_segment(voice_seg_2, seg_1, expected_start_2, expected_end_2)

    # voice segments detected in 2d input segment
    voice_seg_3 = voice_segs[2]
    expected_start_3 = _SPAN_OFFSET + silence_duration_2
    expected_end_3 = expected_start_3 + voice_duration
    _check_voice_segment(voice_seg_3, seg_2, expected_start_3, expected_end_3)

    voice_seg_4 = voice_segs[3]
    expected_start_4 = expected_end_3 + silence_duration_2
    expected_end_4 = expected_start_4 + voice_duration
    _check_voice_segment(voice_seg_4, seg_2, expected_start_4, expected_end_4)


def test_prov():
    """Generated provenance nodes"""
    # use file containing voice signal
    voice_signal, sample_rate = sf.read(_PATH_TO_VOICE_FILE, always_2d=True, dtype=np.float32)
    voice_signal = voice_signal.T
    # interleave it with silences (twice per segment)
    silence_duration_1 = 3.0
    seg_1 = _get_segment(voice_signal, sample_rate, silence_duration_1)
    silence_duration_2 = 5.0
    seg_2 = _get_segment(voice_signal, sample_rate, silence_duration_2)
    segs = [seg_1, seg_2]

    detector = WebRTCVoiceDetector(output_label=_OUTPUT_LABEL)
    prov_tracer = ProvTracer()
    detector.set_prov_tracer(prov_tracer)
    voice_segs = detector.run(segs)
    assert len(voice_segs) == 4

    # data item uid and operation uid are correct
    voice_seg_1 = voice_segs[0]
    prov_1 = prov_tracer.get_prov(voice_seg_1.uid)
    assert prov_1.data_item == voice_seg_1
    assert prov_1.op_desc == detector.description

    # 1st and 2d voices segments have 1st input segment as source
    assert prov_1.source_data_items == [seg_1]

    voice_seg_2 = voice_segs[1]
    prov_2 = prov_tracer.get_prov(voice_seg_2.uid)
    assert prov_2.source_data_items == [seg_1]

    # 3d and 4th voices segments have 2d input segment as source
    voice_seg_3 = voice_segs[2]
    prov_3 = prov_tracer.get_prov(voice_seg_3.uid)
    assert prov_3.source_data_items == [seg_2]

    voice_seg_4 = voice_segs[3]
    prov_4 = prov_tracer.get_prov(voice_seg_4.uid)
    assert prov_4.source_data_items == [seg_2]
