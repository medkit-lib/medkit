import pytest

pytest.importorskip(modname="torchaudio", reason="torchaudio is not installed")
pytest.importorskip(modname="transformers", reason="transformers is not installed")
pytest.importorskip(modname="speechbrain", reason="speechbrain is not installed")

import numpy as np
import torch

from medkit.audio.transcription.sb_transcriber import SBTranscriber
from medkit.core import ProvTracer
from medkit.core.audio import MemoryAudioBuffer, Segment, Span

_MOCK_MODEL_NAME = "mock-model"
_SAMPLE_RATE = 16000
_TEXT_TEMPLATE = "AUDIO HAS {} SAMPLES"


# mock of speechbrain.pretrained.EncoderASR and
# speechbrain.pretrained.EncoderDecoderASR classes used by SBTranscriber
class _MockSpeechbrainASR:
    def __init__(self):
        # original class stores AudioNormalizer instance that we use to check
        # the sample rate
        self.audio_normalizer = _MockAudioNormalizer()

    @classmethod
    def from_hparams(cls, source, savedir, run_opts):
        assert source == _MOCK_MODEL_NAME
        return cls()

    def transcribe_batch(self, wavs, wav_lengths):
        assert isinstance(wavs, torch.Tensor)
        assert wavs.ndim == 2
        assert isinstance(wav_lengths, torch.Tensor)
        assert wav_lengths.ndim == 1

        # for each wav in batch, return string containing sample count (allows
        # us to check that each audio input has corresponding output)
        texts = []
        for wav_length in wav_lengths:
            # convert speechbrain relative length to absolute sample count
            nb_samples = int(wavs.shape[-1] * wav_length)
            text = _TEXT_TEMPLATE.format(nb_samples)
            texts.append(text)

        # original class returns tuple of texts and tokens but we only use
        # texts
        return texts, None


class _MockAudioNormalizer:
    def __init__(self):
        self.sample_rate = _SAMPLE_RATE


@pytest.fixture(scope="module", autouse=True)
def _mocked_asr(module_mocker):
    module_mocker.patch("speechbrain.pretrained.EncoderASR", _MockSpeechbrainASR)
    module_mocker.patch("speechbrain.pretrained.EncoderDecoderASR", _MockSpeechbrainASR)


def _gen_segment(nb_samples) -> Segment:
    audio = MemoryAudioBuffer(signal=np.zeros((1, nb_samples)), sample_rate=_SAMPLE_RATE)
    return Segment(label="turn", audio=audio, span=Span(0, audio.duration))


def test_basic():
    """Basic behavior"""
    transcriber = SBTranscriber(model=_MOCK_MODEL_NAME, output_label="transcribed_text", needs_decoder=False)

    seg_1 = _gen_segment(1000)
    seg_2 = _gen_segment(2000)
    transcriber.run([seg_1, seg_2])

    attrs_1 = seg_1.attrs.get(label="transcribed_text")
    assert len(attrs_1) == 1
    assert attrs_1[0].value == "Audio has 1000 samples."

    attrs_2 = seg_2.attrs.get(label="transcribed_text")
    assert len(attrs_2) == 1
    assert attrs_2[0].value == "Audio has 2000 samples."


def test_no_formatting():
    """No reformatting of transcribed text (raw text as returned by speechbrain ASR)"""
    transcriber = SBTranscriber(
        model=_MOCK_MODEL_NAME,
        output_label="transcribed_text",
        needs_decoder=False,
        add_trailing_dot=False,
        capitalize=False,
    )
    seg = _gen_segment(1000)
    transcriber.run([seg])
    attr = seg.attrs.get(label="transcribed_text")[0]
    assert attr.value == "AUDIO HAS 1000 SAMPLES"


def test_prov():
    seg = _gen_segment(1000)

    transcriber = SBTranscriber(model=_MOCK_MODEL_NAME, output_label="transcribed_text", needs_decoder=False)
    prov_tracer = ProvTracer()
    transcriber.set_prov_tracer(prov_tracer)
    transcriber.run([seg])

    attr = seg.attrs.get(label="transcribed_text")[0]
    prov = prov_tracer.get_prov(attr.uid)
    assert prov.data_item == attr
    assert prov.op_desc == transcriber.description
    assert prov.source_data_items == [seg]
