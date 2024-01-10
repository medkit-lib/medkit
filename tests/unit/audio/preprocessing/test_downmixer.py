import numpy as np

from medkit.audio.preprocessing.downmixer import Downmixer
from medkit.core import ProvTracer
from medkit.core.audio import MemoryAudioBuffer, Segment, Span
from tests.audio_utils import generate_dc_signal, signals_are_equal

_SAMPLE_RATE = 4000
_OUTPUT_LABEL = "downmixed"
_SPAN_OFFSET = 10.0


def _get_segment(signal):
    duration = signal.shape[1] / _SAMPLE_RATE
    audio = MemoryAudioBuffer(signal=signal, sample_rate=_SAMPLE_RATE)
    return Segment(label="raw", span=Span(_SPAN_OFFSET, _SPAN_OFFSET + duration), audio=audio)


def _check_downmixed_segment(downmixed_seg, original_seg):
    assert downmixed_seg.label == _OUTPUT_LABEL
    # span is identical
    assert downmixed_seg.span == original_seg.span
    # audio has same duration and sample rate but only 1 channel
    downmixed_audio = downmixed_seg.audio
    original_audio = original_seg.audio
    assert downmixed_audio.duration == original_audio.duration
    assert downmixed_audio.sample_rate == original_audio.sample_rate
    assert downmixed_audio.nb_channels == 1


def test_basic():
    """Basic behavior"""
    # build 1st segment, constant value 0.1 on 1st channel and 0.2 on 2d channel
    duration_1 = 0.5
    amplitudes_1 = [0.1, 0.2]
    signal_1 = generate_dc_signal(
        duration=duration_1,
        sample_rate=_SAMPLE_RATE,
        nb_channels=2,
        amplitudes=amplitudes_1,
    )
    seg_1 = _get_segment(signal_1)

    # build 2d segment, constant value 0.3 on 1st channel and 0.4 on 2d channel
    duration_2 = 1.0
    amplitudes_2 = [0.3, 0.4]
    signal_2 = generate_dc_signal(
        duration=duration_2,
        sample_rate=_SAMPLE_RATE,
        nb_channels=2,
        amplitudes=amplitudes_2,
    )
    seg_2 = _get_segment(signal_2)

    segs = [seg_1, seg_2]

    downmixer = Downmixer(output_label=_OUTPUT_LABEL)
    downmixed_segs = downmixer.run(segs)
    assert len(downmixed_segs) == len(segs)

    # check attributes of 1st downmixed segment
    downmixed_seg_1 = downmixed_segs[0]
    _check_downmixed_segment(downmixed_seg_1, seg_1)
    # check audio signal, should be normalized sum of both channels
    downmixed_signal_1 = downmixed_seg_1.audio.read()
    expected_signal_1 = generate_dc_signal(
        duration=duration_1,
        sample_rate=_SAMPLE_RATE,
        nb_channels=1,
        amplitudes=[sum(amplitudes_1) / 2],
    )
    assert signals_are_equal(downmixed_signal_1, expected_signal_1)

    # check attributes of 2d downmixed segment
    downmixed_seg_2 = downmixed_segs[1]
    _check_downmixed_segment(downmixed_seg_1, seg_1)
    # check audio signal, should be normalized sum of both channels
    downmixed_signal_2 = downmixed_seg_2.audio.read()
    expected_signal_2 = generate_dc_signal(
        duration=duration_2,
        sample_rate=_SAMPLE_RATE,
        nb_channels=1,
        amplitudes=[sum(amplitudes_2) / 2],
    )
    assert signals_are_equal(downmixed_signal_2, expected_signal_2)


def test_prevent_clipping():
    """Behavior with and without clipping prevention"""
    duration = 0.5
    signal = generate_dc_signal(
        duration=duration,
        sample_rate=_SAMPLE_RATE,
        nb_channels=2,
        amplitudes=[0.9, 0.4],
    )
    seg = _get_segment(signal)

    # downmixer without clipping prevention
    downmixer_1 = Downmixer(output_label=_OUTPUT_LABEL, prevent_clipping=False)
    downmixed_seg_1 = downmixer_1.run([seg])[0]
    downmixed_signal_1 = downmixed_seg_1.audio.read()
    # downmixed signal is the sum of all channels
    assert np.max(downmixed_signal_1) == 0.9 + 0.4

    # downmixer with clipping prevention
    downmixer_2 = Downmixer(output_label=_OUTPUT_LABEL, prevent_clipping=True)
    downmixed_seg_2 = downmixer_2.run([seg])[0]
    downmixed_signal_2 = downmixed_seg_2.audio.read()
    # downmixed signal is the sum of all channels divided by the number of channels
    assert np.max(downmixed_signal_2) == (0.9 + 0.4) / 2


def test_prov():
    """Generated provenance nodes"""
    signal_1 = generate_dc_signal(duration=0.25, sample_rate=_SAMPLE_RATE, nb_channels=2)
    seg_1 = _get_segment(signal_1)
    signal_2 = generate_dc_signal(duration=0.25, sample_rate=_SAMPLE_RATE, nb_channels=2)
    seg_2 = _get_segment(signal_2)
    segs = [seg_1, seg_2]

    downmixer = Downmixer(output_label=_OUTPUT_LABEL)
    prov_tracer = ProvTracer()
    downmixer.set_prov_tracer(prov_tracer)
    downmixed_segs = downmixer.run(segs)

    # data item uid and operation uid are correct
    downmixed_seg_1 = downmixed_segs[0]
    prov_1 = prov_tracer.get_prov(downmixed_seg_1.uid)
    assert prov_1.data_item == downmixed_seg_1
    assert prov_1.op_desc == downmixer.description
    # each text segment has corresponding voice segment as source
    assert prov_1.source_data_items == [seg_1]
    downmixed_seg_2 = downmixed_segs[1]
    prov_2 = prov_tracer.get_prov(downmixed_seg_2.uid)
    assert prov_2.source_data_items == [seg_2]
