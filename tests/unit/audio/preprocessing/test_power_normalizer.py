import math

import numpy as np

from medkit.audio.preprocessing.power_normalizer import PowerNormalizer
from medkit.core import ProvTracer
from medkit.core.audio import MemoryAudioBuffer, Segment, Span
from tests.audio_utils import generate_sin_signal

_SAMPLE_RATE = 4000
_OUTPUT_LABEL = "normalized"
_SPAN_OFFSET = 10.0
_TOLERANCE = 1e-7


def _get_segment(signal):
    duration = signal.shape[1] / _SAMPLE_RATE
    audio = MemoryAudioBuffer(signal=signal, sample_rate=_SAMPLE_RATE)
    return Segment(label="raw", span=Span(_SPAN_OFFSET, _SPAN_OFFSET + duration), audio=audio)


def _check_normalized_segment(normalized_seg, original_seg):
    assert normalized_seg.label == _OUTPUT_LABEL
    # span is identical
    assert normalized_seg.span == original_seg.span
    # audio has same number of channels, duration and sample rate
    normalized_audio = normalized_seg.audio
    original_audio = original_seg.audio
    assert normalized_audio.nb_channels == original_audio.nb_channels
    assert normalized_audio.duration == original_audio.duration
    assert normalized_audio.sample_rate == original_audio.sample_rate


def _powers_are_close(power_1, power_2):
    return math.isclose(power_1, power_2, abs_tol=_TOLERANCE)


def test_basic():
    """Basic behavior"""
    # build 1st segment, 2 channels with sin waves
    amps_1 = [0.1, 0.2]
    signal_1 = generate_sin_signal(
        duration=0.5,
        sample_rate=_SAMPLE_RATE,
        nb_channels=2,
        amplitudes=amps_1,
    )
    seg_1 = _get_segment(signal_1)

    # build 2d segment, 1 channel with sin wave
    signal_2 = generate_sin_signal(
        duration=1.0,
        sample_rate=_SAMPLE_RATE,
        nb_channels=1,
        amplitudes=[0.3],
    )
    seg_2 = _get_segment(signal_2)

    segs = [seg_1, seg_2]

    normalizer = PowerNormalizer(output_label=_OUTPUT_LABEL)
    normalized_segs = normalizer.run(segs)
    assert len(normalized_segs) == len(segs)

    # check attributes of 1st normalized segment
    normalized_seg_1 = normalized_segs[0]
    _check_normalized_segment(normalized_seg_1, seg_1)
    # check audio, power must be close to 1.0
    normalized_signal_1 = normalized_seg_1.audio.read()
    normalized_power_1 = np.std(normalized_signal_1)
    assert _powers_are_close(normalized_power_1, 1.0)
    # power ratio between channels must be preserved
    normalized_power_ratio = np.std(normalized_signal_1[0]) / np.std(normalized_signal_1[1])
    assert _powers_are_close(normalized_power_ratio, amps_1[0] / amps_1[1])

    # check attributes of 2d normalized segment
    normalized_seg_2 = normalized_segs[1]
    _check_normalized_segment(normalized_seg_1, seg_1)
    # check audio, power must be close to 1.0
    normalized_signal_2 = normalized_seg_2.audio.read()
    normalized_power_2 = np.std(normalized_signal_2)
    assert _powers_are_close(normalized_power_2, 1.0)


def test_target_value():
    """Custom target_value"""
    signal = generate_sin_signal(duration=0.5, sample_rate=_SAMPLE_RATE, amplitudes=[0.1])
    seg = _get_segment(signal)

    target_value = 0.9
    normalizer = PowerNormalizer(output_label=_OUTPUT_LABEL, target_value=target_value)
    normalized_seg = normalizer.run([seg])[0]

    # power must be close to 0.9
    normalized_signal = normalized_seg.audio.read()
    normalized_power = np.std(normalized_signal)
    assert _powers_are_close(normalized_power, target_value)


def test_channel_wise():
    """Channel-wise normalization, balance across channel is not preserved"""
    signal = generate_sin_signal(duration=0.5, sample_rate=_SAMPLE_RATE, nb_channels=2, amplitudes=[0.1, 0.5])
    seg = _get_segment(signal)

    normalizer = PowerNormalizer(output_label=_OUTPUT_LABEL, channel_wise=True)
    normalized_seg = normalizer.run([seg])[0]

    # power of each channel must be close to 1.0
    normalized_signal = normalized_seg.audio.read()
    normalized_power_1, normalized_power_2 = np.std(normalized_signal, axis=1)
    assert _powers_are_close(normalized_power_1, 1.0)
    assert _powers_are_close(normalized_power_2, 1.0)


def test_prov():
    """Generated provenance nodes"""
    signal_1 = generate_sin_signal(duration=0.25, sample_rate=_SAMPLE_RATE, nb_channels=2)
    seg_1 = _get_segment(signal_1)
    signal_2 = generate_sin_signal(duration=0.25, sample_rate=_SAMPLE_RATE, nb_channels=2)
    seg_2 = _get_segment(signal_2)
    segs = [seg_1, seg_2]

    normalizer = PowerNormalizer(output_label=_OUTPUT_LABEL)
    prov_tracer = ProvTracer()
    normalizer.set_prov_tracer(prov_tracer)
    normalized_segs = normalizer.run(segs)

    # data item uid and operation uid are correct
    normalized_seg_1 = normalized_segs[0]
    prov_1 = prov_tracer.get_prov(normalized_seg_1.uid)
    assert prov_1.data_item == normalized_seg_1
    assert prov_1.op_desc == normalizer.description
    # each text segment has corresponding voice segment as source
    normalized_seg_2 = normalized_segs[1]
    assert prov_1.source_data_items == [seg_1]
    prov_2 = prov_tracer.get_prov(normalized_seg_2.uid)
    assert prov_2.source_data_items == [seg_2]
