import pytest

pytest.importorskip(modname="resampy", reason="resampy is not installed")

import numpy as np

from medkit.audio.preprocessing.resampler import Resampler
from medkit.core import ProvTracer
from medkit.core.audio import MemoryAudioBuffer, Segment, Span
from tests.audio_utils import generate_sin_signal

_OUTPUT_LABEL = "downmixed"
_TARGET_SAMPLE_RATE = 4000
_SPAN_OFFSET = 10.0


def _get_segment(signal, sample_rate):
    duration = signal.shape[1] / sample_rate
    audio = MemoryAudioBuffer(signal=signal, sample_rate=sample_rate)
    return Segment(label="raw", span=Span(_SPAN_OFFSET, _SPAN_OFFSET + duration), audio=audio)


def _check_resampled_segment(resampled_seg, original_seg):
    assert resampled_seg.label == _OUTPUT_LABEL
    # span is identical
    assert resampled_seg.span == original_seg.span
    # audio has same duration and number of channels
    resampled_audio = resampled_seg.audio
    original_audio = original_seg.audio
    assert resampled_audio.duration == original_audio.duration
    assert resampled_audio.nb_channels == original_audio.nb_channels
    # sample rate changed to target
    assert resampled_audio.sample_rate == _TARGET_SAMPLE_RATE


def _get_freqs(signal, sample_rate):
    fft = np.fft.fft(signal, axis=1)
    fft_freqs = np.fft.fftfreq(signal.shape[1], d=1 / sample_rate)
    return [abs(fft_freqs[i]) for i in np.argmax(fft, axis=1)]


def test_basic():
    """Basic behavior"""
    # build 1st segment, 2 channels with lower sample rate
    freqs_1 = [110, 220]
    sample_rate_1 = _TARGET_SAMPLE_RATE / 2
    signal_1 = generate_sin_signal(
        duration=0.5,
        sample_rate=sample_rate_1,
        nb_channels=2,
        freqs=freqs_1,
    )
    seg_1 = _get_segment(signal_1, sample_rate_1)

    # build 2d segment, 1 channel with higher sample rate
    freqs_2 = [330]
    sample_rate_2 = _TARGET_SAMPLE_RATE * 2
    signal_2 = generate_sin_signal(
        duration=0.5,
        sample_rate=sample_rate_2,
        nb_channels=1,
        freqs=freqs_2,
    )
    seg_2 = _get_segment(signal_2, sample_rate_2)

    segs = [seg_1, seg_2]

    resampler = Resampler(output_label=_OUTPUT_LABEL, sample_rate=_TARGET_SAMPLE_RATE)
    resampled_segs = resampler.run(segs)
    assert len(resampled_segs) == len(segs)

    # check attributes of 1st resampled segment
    resampled_seg_1 = resampled_segs[0]
    _check_resampled_segment(resampled_seg_1, seg_1)
    # check audio signal, frequencies should be the same
    resampled_audio_1 = resampled_seg_1.audio
    resampled_freqs_1 = _get_freqs(resampled_audio_1.read(), resampled_audio_1.sample_rate)
    assert resampled_freqs_1 == freqs_1

    # check attributes of 2d downmixed segment
    resampled_seg_2 = resampled_segs[1]
    _check_resampled_segment(resampled_seg_2, seg_2)
    # check audio signal, frequency should be the same
    resampled_audio_2 = resampled_seg_2.audio
    resampled_freqs_2 = _get_freqs(resampled_audio_2.read(), resampled_audio_2.sample_rate)
    assert resampled_freqs_2 == freqs_2


def test_fast():
    """Fast param set to True"""
    freqs = [110, 220]
    sample_rate_1 = _TARGET_SAMPLE_RATE / 2
    signal_1 = generate_sin_signal(
        duration=0.5,
        sample_rate=sample_rate_1,
        nb_channels=2,
        freqs=freqs,
    )
    seg = _get_segment(signal_1, sample_rate_1)

    resampler = Resampler(output_label=_OUTPUT_LABEL, sample_rate=_TARGET_SAMPLE_RATE)
    resampled_seg = resampler.run([seg])[0]

    # check audio signal, frequencies should be the same
    resampled_audio = resampled_seg.audio
    resampled_freqs = _get_freqs(resampled_audio.read(), resampled_audio.sample_rate)
    assert resampled_freqs == freqs


def test_prov():
    """Generated provenance nodes"""
    signal_1 = generate_sin_signal(duration=0.25, sample_rate=_TARGET_SAMPLE_RATE * 2, nb_channels=2)
    seg_1 = _get_segment(signal_1, _TARGET_SAMPLE_RATE * 2)
    signal_2 = generate_sin_signal(duration=0.25, sample_rate=_TARGET_SAMPLE_RATE * 2, nb_channels=2)
    seg_2 = _get_segment(signal_2, _TARGET_SAMPLE_RATE * 2)
    segs = [seg_1, seg_2]

    resampler = Resampler(output_label=_OUTPUT_LABEL, sample_rate=_TARGET_SAMPLE_RATE)
    prov_tracer = ProvTracer()
    resampler.set_prov_tracer(prov_tracer)
    resampled_segs = resampler.run(segs)

    resampled_seg_1, resampled_seg_2 = resampled_segs
    # data item uid and operation uid are correct
    prov_1 = prov_tracer.get_prov(resampled_seg_1.uid)
    assert prov_1.data_item == resampled_seg_1
    assert prov_1.op_desc == resampler.description
    # each text segment has corresponding voice segment as source
    assert prov_1.source_data_items == [seg_1]
    prov_2 = prov_tracer.get_prov(resampled_seg_2.uid)
    assert prov_2.source_data_items == [seg_2]
