import pytest

pytest.importorskip(modname="torchaudio", reason="torchaudio is not installed")
pytest.importorskip(modname="transformers", reason="transformers is not installed")
pytest.importorskip(modname="speechbrain", reason="speechbrain is not installed")

import numpy as np

from medkit.audio.transcription.sb_transcriber import SBTranscriber
from medkit.core.audio import (
    FileAudioBuffer,
    MemoryAudioBuffer,
    Segment,
    Span,
)

_MODEL = "speechbrain/asr-wav2vec2-commonvoice-en"
_AUDIO = FileAudioBuffer("tests/data/audio/voice.ogg")
_EXPECTED_TEXT = "Hello this is my voice i m speaking to you."


def test_basic():
    """Basic behavior"""
    span = Span(0.0, _AUDIO.duration)
    seg = Segment(label="turn", audio=_AUDIO, span=span)

    transcriber = SBTranscriber(model=_MODEL, output_label="transcribed_text", needs_decoder=True)
    transcriber.run([seg])

    attrs = seg.attrs.get(label="transcribed_text")
    assert len(attrs) == 1
    attr = attrs[0]
    assert attr.value == _EXPECTED_TEXT


@pytest.mark.parametrize("batch_size", [1, 5, 10, 15])
def test_batch(batch_size):
    """Various batch sizes (smallest, half, exact number of items, more than)"""
    transcriber = SBTranscriber(model=_MODEL, needs_decoder=True)

    # generate batch of different audios by duplicating signal every other time
    audios = []
    short_signal = _AUDIO.read()
    long_signal = np.concatenate((short_signal, short_signal), axis=1)
    for i in range(batch_size):
        signal = short_signal if i % 2 else long_signal
        audio = MemoryAudioBuffer(signal, _AUDIO.sample_rate)
        audios.append(audio)

    # transcribe batch of audios
    texts = transcriber._transcribe_audios(audios)
    assert len(texts) == len(audios)

    for audio, text in zip(audios, texts):
        expected_text = transcriber._transcribe_audios([audio])[0]
        assert text == expected_text
