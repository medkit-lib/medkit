import pytest

pytest.importorskip(modname="torchaudio", reason="torchaudio is not installed")
pytest.importorskip(modname="transformers", reason="transformers is not installed")

import numpy as np

from medkit.audio.transcription.hf_transcriber import HFTranscriber
from medkit.core import ProvTracer
from medkit.core.audio import MemoryAudioBuffer, Segment, Span

_SAMPLE_RATE = 16000
_TEXT_TEMPLATE = "AUDIO HAS {} SAMPLES"


# mock of AutomaticSpeechRecognitionPipeline class used by HFTranscriber
class _MockPipeline:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, audio_dicts):
        # for each audio dict in batch, return string containing sample count
        # (allows us to check that each audio input has corresponding output)
        text_dicts = []
        for audio_dict in audio_dicts:
            nb_samples = int(audio_dict["raw"].shape[-1])
            text = _TEXT_TEMPLATE.format(nb_samples)
            text_dicts.append({"text": text})

        return text_dicts


@pytest.fixture(scope="module", autouse=True)
def _mocked_pipeline(module_mocker):
    module_mocker.patch(
        "transformers.pipelines.get_task",
        return_value="automatic-speech-recognition",
    )
    module_mocker.patch("transformers.pipeline", _MockPipeline)


def _gen_segment(nb_samples) -> Segment:
    audio = MemoryAudioBuffer(signal=np.zeros((1, nb_samples)), sample_rate=_SAMPLE_RATE)
    return Segment(label="turn", audio=audio, span=Span(0, audio.duration))


def test_basic():
    """Basic behavior"""
    transcriber = HFTranscriber(model="mock-model", output_label="transcribed_text")

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
    """No reformatting of transcribed text (raw text as returned by transformers pipeline)"""
    transcriber = HFTranscriber(
        model="mock-model",
        output_label="transcribed_text",
        add_trailing_dot=False,
        capitalize=False,
    )
    seg = _gen_segment(1000)
    transcriber.run([seg])
    attr = seg.attrs.get(label="transcribed_text")[0]
    assert attr.value == "AUDIO HAS 1000 SAMPLES"


def test_prov():
    seg = _gen_segment(1000)

    transcriber = HFTranscriber(model="mock-model", output_label="transcribed_text")

    prov_tracer = ProvTracer()
    transcriber.set_prov_tracer(prov_tracer)
    transcriber.run([seg])

    attr = seg.attrs.get(label="transcribed_text")[0]
    prov = prov_tracer.get_prov(attr.uid)
    assert prov.data_item == attr
    assert prov.op_desc == transcriber.description
    assert prov.source_data_items == [seg]
