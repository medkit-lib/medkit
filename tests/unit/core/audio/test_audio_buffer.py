import pytest
import soundfile as sf

from medkit.core.audio.audio_buffer import FileAudioBuffer, MemoryAudioBuffer
from tests.audio_utils import generate_sin_signal, signals_are_equal


def _test_read(audio, original_signal):
    # simple read
    signal_1 = audio.read()
    assert signals_are_equal(signal_1, original_signal)

    # read a copy than can safely be mutated without side effects
    signal_2 = audio.read(copy=True)
    assert signals_are_equal(signal_2, original_signal)
    signal_2[:, 100:200] = 0.0
    assert signals_are_equal(audio.read(), original_signal)


def _test_trim(audio, start=100, end=200):
    full_signal = audio.read()

    # trim slice
    trimmed_audio_1 = audio.trim(start, end)
    assert signals_are_equal(
        trimmed_audio_1.read(),
        full_signal[:, start:end],
    )

    # trim from start of file up to specific sample
    trimmed_audio_2 = audio.trim(end=end)
    assert signals_are_equal(
        trimmed_audio_2.read(),
        full_signal[:, :end],
    )
    # trim from specific sample up to end of file
    trimmed_audio_3 = audio.trim(start=start)
    assert signals_are_equal(
        trimmed_audio_3.read(),
        full_signal[:, start:],
    )

    # same tests, on audio previously already trimmed
    if start == 100 and end == 200:
        _test_trim(trimmed_audio_1, start=10, end=50)


def _test_trim_duration(audio, start_time=0.1, end_time=0.2):
    full_signal = audio.read()
    start_sample = round(start_time * audio.sample_rate)
    end_sample = round(end_time * audio.sample_rate)

    # trim slice
    trimmed_audio_1 = audio.trim_duration(start_time, end_time)
    assert signals_are_equal(
        trimmed_audio_1.read(),
        full_signal[:, start_sample:end_sample],
    )

    # trim from start of file up to specific time
    trimmed_audio_2 = audio.trim_duration(end_time=end_time)
    assert signals_are_equal(
        trimmed_audio_2.read(),
        full_signal[:, :end_sample],
    )
    # trim from specific time up to end of file
    trimmed_audio_3 = audio.trim_duration(start_time=start_time)
    assert signals_are_equal(
        trimmed_audio_3.read(),
        full_signal[:, start_sample:],
    )


# nb_channels, duration, sample_rate
_TEST_PARAMS = [
    (2, 0.25, 4000),
    (1, 0.25, 4000),
    (2, 1.0, 4000),
    (2, 0.25, 16000),
]


@pytest.mark.parametrize(("nb_channels", "duration", "sample_rate"), _TEST_PARAMS)
def test_memory_buffer_basic(nb_channels, duration, sample_rate):
    """Basic behavior of MemoryAudioBuffer"""
    signal = generate_sin_signal(duration, sample_rate, nb_channels)
    audio = MemoryAudioBuffer(signal=signal.copy(), sample_rate=sample_rate)

    assert audio.nb_channels == nb_channels
    assert audio.duration == duration
    assert audio.sample_rate == sample_rate
    assert audio.nb_samples == round(duration * sample_rate)

    _test_read(audio, signal)


def test_memory_buffer_trim():
    """MemoryAudioBuffer trim methods"""
    sample_rate = 4000
    signal = generate_sin_signal(duration=0.25, sample_rate=sample_rate, nb_channels=2)
    audio = MemoryAudioBuffer(signal=signal.copy(), sample_rate=sample_rate)

    _test_trim(audio)
    _test_trim_duration(audio)


@pytest.mark.parametrize(("nb_channels", "duration", "sample_rate"), _TEST_PARAMS)
def test_file_buffer_basic(tmp_path, nb_channels, duration, sample_rate):
    """Basic behavior of FileAudioBuffer"""
    signal = generate_sin_signal(duration, sample_rate, nb_channels)
    path = tmp_path / "audio.wav"
    sf.write(path, signal.T, sample_rate, subtype="FLOAT")
    audio = FileAudioBuffer(path)

    assert audio.nb_channels == nb_channels
    assert audio.duration == duration
    assert audio.sample_rate == sample_rate
    assert audio.nb_samples == round(duration * sample_rate)

    _test_read(audio, signal)


def test_file_buffer_trim(tmp_path):
    """FileAudioBuffer read methods"""
    sample_rate = 4000
    signal = generate_sin_signal(duration=0.25, sample_rate=sample_rate, nb_channels=2)
    path = tmp_path / "audio.wav"
    sf.write(path, signal.T, sample_rate, subtype="FLOAT")
    audio = FileAudioBuffer(path)

    _test_trim(audio)
    _test_trim_duration(audio)
