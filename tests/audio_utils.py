__all__ = [
    "generate_sin_signal",
    "generate_dc_signal",
    "generate_silence",
    "signals_are_equal",
]

import numpy as np

_BASE_FREQ = 110


def generate_sin_signal(duration, sample_rate, nb_channels=1, amplitudes=None, freqs=None):
    """Generate a sin wave signal with a different frequency for each channel"""
    # default amplitudes to 1.0 for all channels if none provided
    if amplitudes is None:
        amplitudes = [1.0] * nb_channels
    else:
        assert len(amplitudes) == nb_channels
    # default freqs to multiples of _BASE_FREQ if none provided
    if freqs is None:
        freqs = [_BASE_FREQ * (i + 1) for i in range(nb_channels)]
    else:
        assert len(freqs) == nb_channels

    nb_samples = round(duration * sample_rate)
    time = np.arange(nb_samples, dtype=np.float32) / sample_rate

    # generate sin waves for each channel
    signal = np.sin(2 * np.pi * np.array(freqs, dtype=np.float32).reshape(-1, 1) * time.reshape(1, -1))
    # apply amplitudes
    signal *= np.array(amplitudes).reshape(-1, 1)

    return signal


def generate_dc_signal(duration, sample_rate, nb_channels=1, amplitudes=None):
    """Generate a constant signal"""
    # default amplitudes to 1.0 for all channels if none provided
    if amplitudes is None:
        amplitudes = [1.0] * nb_channels
    else:
        assert len(amplitudes) == nb_channels

    nb_samples = round(duration * sample_rate)
    return np.tile(np.array(amplitudes, ndmin=2).T, nb_samples)


def generate_silence(duration, sample_rate, nb_channels=1):
    nb_samples = round(duration * sample_rate)
    return np.zeros((nb_channels, nb_samples))


def signals_are_equal(signal_1, signal_2):
    return np.allclose(signal_1, signal_2, atol=1e-6)
